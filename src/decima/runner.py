"""Run DECIMA over a big catalog in chunks, in worker processes, resumably.

    decima.run("stars.fits", out="run/", columns={...}, config=Config(), workers=4)

Layout of `out`:
    config.json  job.json  inputs.pt     settings, chunk size, and the inputs of the stars that can be run
    rejected.ecsv                        input rows that cannot be run, with the reason
    chunk_XXXXXX.npz                     per-chunk results, written atomically as each chunk finishes
    summary.<format>                     one row per star (percentiles, diagnostics, flags), in input order
    samples.h5                           "samples" (n_stars, n_keep, n_q) posterior draws, "id", "quantity_names"

Chunk c (stars c*chunk ...) is seeded config.seed + c*chunk, so results do not depend on the number of workers, and re-running the same call resumes: finished chunks are skipped. A different config or chunk size needs a new `out`.

Each worker uses one CPU thread; with config.device="cuda" every worker shares the GPU.
"""
import json
import multiprocessing as mp
import os
import time
import traceback
from pathlib import Path

import h5py
import numpy as np

from decima.pipeline import Config, StarInputs, concat, quantity_names, summary_table


def _chunk_file(out, c):
    return Path(out) / f"chunk_{c:06d}.npz"


def _log(out, msg):
    with open(Path(out) / "progress.log", "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}\n")


def _prepare(source, out, columns, cfg, chunk):
    """Write config/job/inputs once; on resume check they match."""
    from decima.io import build_inputs, read_table
    out.mkdir(parents=True, exist_ok=True)
    job = {"chunk": chunk}
    if (out / "inputs.pt").exists() and (out / "config.json").exists(): # config.json is written last: without it the preparation was interrupted, so redo it
        if Config.from_json((out / "config.json").read_text()).to_json() != cfg.to_json() \
                or json.loads((out / "job.json").read_text()) != job:
            raise ValueError(f"{out} holds a run with different settings; use a new `out` or delete it")
        return
    inputs, rejected = build_inputs(read_table(source, columns), cfg)
    rejected.write(out / "rejected.ecsv", overwrite=True)
    inputs.save(out / "inputs.pt")
    (out / "job.json").write_text(json.dumps(job))
    (out / "config.json").write_text(cfg.to_json()) # last: its presence with inputs.pt marks a prepared run
    _log(out, f"PREPARED  {len(inputs)} stars, {len(rejected)} rejected, chunks of {chunk}")


def _worker(out, k, workers):
    import torch
    from decima.models import load_models
    from decima.pipeline import run_chunk
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    out = Path(out)
    cfg = Config.from_json((out / "config.json").read_text())
    chunk = json.loads((out / "job.json").read_text())["chunk"]
    inputs = StarInputs.load(out / "inputs.pt")
    models = None # loaded lazily so a resumed worker with nothing left to do exits quickly
    for c in range(k, -(-len(inputs) // chunk), workers):
        path = _chunk_file(out, c)
        if path.exists():
            continue
        models = models or load_models(cfg.device)
        lo = c * chunk
        t = time.time()
        try:
            res = run_chunk(inputs.slice(lo, lo + chunk), models, cfg, seed=cfg.seed + lo)
        except Exception as e: # logged and skipped; re-running retries it
            traceback.print_exc()
            _log(out, f"FAILED  chunk {c} on worker {k}: {type(e).__name__}: {e}")
            continue
        tmp = Path(str(path) + ".tmp")
        with open(tmp, "wb") as f: # a file handle stops numpy appending ".npz" to the name
            np.savez_compressed(f, **res)
        os.replace(tmp, path)
        _log(out, f"chunk {c} (stars {lo}-{lo + len(res['ids'])}) worker {k} {time.time() - t:.0f}s")


def _merge(out, cfg, chunk, summary_format):
    n = len(StarInputs.load(out / "inputs.pt"))
    files = [_chunk_file(out, c) for c in range(-(-n // chunk))]
    missing = [p.name for p in files if not p.exists()]
    if missing:
        raise RuntimeError(f"{len(missing)} chunk(s) failed or are missing (see {out / 'progress.log'}); "
                           f"re-run the same call to retry them")
    parts = []
    for p in files:
        with np.load(p) as d:
            parts.append({k: d[k] for k in d.files})
    res = concat(parts)
    summary_table(res, cfg).write(out / f"summary.{summary_format}", overwrite=True)
    with h5py.File(out / "samples.h5", "w") as h:
        h["samples"] = res["samples"]
        ids = res["ids"]
        h["id"] = np.char.encode(ids, "utf-8") if ids.dtype.kind == "U" else ids # h5py has no unicode arrays: strings are stored as UTF-8 bytes
        h["quantity_names"] = np.array(quantity_names(cfg), dtype="S")
    _log(out, f"MERGED  {n} stars")


def run(source, out, columns: dict = None, config: Config = None, workers: int = 1, chunk: int = 250,
        summary_format: str = "fits"):
    """Run DECIMA on `source` (path, astropy Table or pandas DataFrame) into the directory `out`; see the module docstring. `summary_format` is the file extension astropy writes the summary table with (fits, ecsv, csv, ...).
    Returns the path of the summary table."""
    out, cfg = Path(out), config or Config()
    _prepare(source, out, columns, cfg, chunk)
    ctx = mp.get_context("spawn") # fresh interpreters: no forked torch / CUDA state
    procs = [ctx.Process(target=_worker, args=(str(out), k, workers)) for k in range(workers)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    _merge(out, cfg, chunk, summary_format)
    return out / f"summary.{summary_format}"
