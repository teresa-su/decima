# Run a large catalogue

Use `decima.infer` for up to a few thousand stars in a notebook ([tutorial](../tutorial.ipynb)). For anything bigger use **`decima run`**: it works through the catalogue in chunks, saves each chunk as it finishes, spreads them over worker processes, and **resumes** if it is interrupted.

## Run it

```bash
decima check catalogue.fits --columns cols.json --config settings.json     # first: catches mistakes in seconds
decima run catalogue.fits --out run/ --columns cols.json --config settings.json --workers 16
```

The same from Python:

```python notest
import decima

if __name__ == "__main__":      # needed in a script: workers are started as fresh processes
    decima.run("catalogue.fits", out="run/", columns=cols, config=cfg, workers=16)
```

`--columns` is the [columns file](config-files.ipynb) (your names → DECIMA's) and `--config` the settings file; both are optional.

| Option | Default | |
|---|---|---|
| `--workers` | 1 | Worker processes. Each uses one CPU thread and loads its own copy of the models (about 0.25 GB). |
| `--chunk` | 250 | Stars per checkpoint. Also sets the random draws (below). |
| `--format` | `fits` | Format of the summary table: any extension astropy writes (`ecsv`, `csv`, …). |

## If it stops: run the same command again

If the run stops for any reason without finishing: run the **same command** again. Finished chunks are skipped and the rest are done. Two things are checked first:

- the settings and `--chunk` must be the same, or the run is refused (`holds a run with different settings`); start a new `--out` instead;
- DECIMA does **not** check that the table and columns file are the same as before. Keep them unchanged.

If some chunks fail (look at `progress.log` for the error), the run ends with `N chunk(s) failed or are missing`. Fix the cause and run the same command again to retry only those.

## What is in the run directory

| File | Contents |
|---|---|
| `summary.fits` | One row per star that was run: percentiles, `failed`, `ess_frac`, … ([columns](../reference/outputs.md)) |
| `samples.h5` | `samples` (n_stars, n_keep, 7), `id`, `quantity_names` |
| `rejected.ecsv` | Rows that could not be run, and why |
| `config.json`, `job.json` | The settings and chunk size of this run |
| `progress.log` | A line per finished or failed chunk, with timings |
| `inputs.pt`, `chunk_XXXXXX.npz` | Working files that make resuming possible. Delete them when the run is finished. |

Rejected stars are not in the summary, so its rows do not line up with your table's. Map an `id` column and match on it.

## Reproducible results

Chunk `c` is seeded `seed + c × chunk`, so results do not depend on the number of workers, but they **do** depend on `--chunk`. `decima.infer(..., chunk=250)` gives the same draws as `decima run` with the same `--chunk`.

## GPU

Put `"device": "cuda"` in the settings file (or `cuda:1` for a particular GPU):

```bash
decima default-config | sed 's/"cpu"/"cuda"/' > settings.json
```

- All workers share the GPU, so a few workers are enough to keep it busy.
- Raise `stars_per_batch` (default 100) on a GPU; lower it if memory runs out.
- CUDA and CPU draw different random numbers, so their results differ within the posterior's noise.

