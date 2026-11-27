"""decima.run: the result does not depend on the number of workers, survives an interrupted run, and equals infer."""
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from astropy.table import Table

import decima
from decima.models import data_dir
from test_regression import XIANG_COLUMNS

pytestmark = pytest.mark.skipif(data_dir() is None, reason="needs the models (DECIMA_DATA_DIR)")
STARS = Path(__file__).parent / "data" / "reference_stars.ecsv"
CFG = decima.Config(n_latent_x=20, n_theta=10, n_keep=5)


def _read(out):
    with h5py.File(out / "samples.h5") as h:
        return Table.read(out / "summary.ecsv"), h["samples"][()]


def test_workers_resume_and_infer(tmp_path):
    kw = dict(columns=XIANG_COLUMNS, config=CFG, chunk=2, summary_format="ecsv")
    decima.run(STARS, tmp_path / "w1", workers=1, **kw)
    decima.run(STARS, tmp_path / "w2", workers=2, **kw)
    (tmp_path / "w2" / "chunk_000001.npz").unlink() # as if interrupted before chunk 1 finished
    decima.run(STARS, tmp_path / "w2", workers=2, **kw)
    (s1, x1), (s2, x2) = _read(tmp_path / "w1"), _read(tmp_path / "w2")
    assert len(s1) == 5 and x1.shape == (5, 5, 7)
    assert np.array_equal(x1, x2) and all(np.array_equal(s1[c], s2[c]) for c in s1.colnames)

    torch.set_num_threads(1) # as the workers
    s3, x3, rejected = decima.infer(STARS, XIANG_COLUMNS, CFG, chunk=2)
    assert len(rejected) == 0 and np.array_equal(x1, x3)

    with pytest.raises(ValueError): # another config in the same directory
        decima.run(STARS, tmp_path / "w1", workers=1, **dict(kw, config=decima.Config(n_theta=20)))


def test_string_ids_reach_samples_h5(tmp_path):
    t = Table.read(STARS)
    t["GAIAEDR3_SOURCE_ID_1"] = [f"star{i}" for i in range(len(t))]  # CSV / pandas ids are strings, which h5py cannot store as is
    decima.run(t, tmp_path, columns=XIANG_COLUMNS, config=CFG, chunk=3, summary_format="ecsv")
    with h5py.File(tmp_path / "samples.h5") as h:
        assert [i.decode() for i in h["id"][()]] == [f"star{i}" for i in range(5)]


def test_run_interrupted_during_preparation_restarts(tmp_path):
    kw = dict(columns=XIANG_COLUMNS, config=CFG, chunk=3, summary_format="ecsv")
    decima.run(STARS, tmp_path, **kw)
    (tmp_path / "config.json").unlink()  # config.json is written last: a run stopped before it has inputs.pt only
    for f in tmp_path.glob("chunk_*.npz"):
        f.unlink()
    decima.run(STARS, tmp_path, **kw)  # prepares again instead of failing on the missing config.json
    assert len(Table.read(tmp_path / "summary.ecsv")) == 5
