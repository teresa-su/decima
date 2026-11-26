"""DECIMA against the isosbi pipeline it was extracted from: identical stage-1 + stage-2 outputs, bit for bit, on CPU.

The reference (tests/data) is written by isosbi's export_decima.py. Needs DECIMA_DATA_DIR (the exported models).
"""
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from decima.io import build_inputs, read_table
from decima.models import load_models
from decima.pipeline import Advanced, Config, run_chunk, stage1, stage2

pytestmark = pytest.mark.skipif(not os.environ.get("DECIMA_DATA_DIR"), reason="needs DECIMA_DATA_DIR")

DATA = Path(__file__).parent / "data"
SEED = 22
XIANG_COLUMNS = {
    "id": "GAIAEDR3_SOURCE_ID_1", "ra": "RAJ2000", "dec": "DEJ2000",
    "G": "GAIA_G", "BP": "GAIA_BP", "RP": "GAIA_RP", "G_error": "E_GAIA_G", "BP_error": "E_GAIA_BP",
    "RP_error": "E_GAIA_RP", "J": "TWOMASS_J", "H": "TWOMASS_H", "Ks": "TWOMASS_K", "J_error": "j_msigcom",
    "H_error": "h_msigcom", "Ks_error": "ks_msigcom", "parallax": "PARALLAX", "parallax_error": "E_PARALLAX",
    "teff": "TEFF", "teff_error": "E_TEFF", "logg": "LOGG", "logg_error": "E_LOGG", "feh": "FEH", "feh_error": "E_FEH",
    "alpha_fe": "ALPHA_FE", "alpha_fe_error": "E_ALPHA_FE",
}
ALL = ("G", "BP", "RP", "J", "H", "Ks")
# the configs of isosbi's export_decima.REF_CONFIGS; the reference draws are per star (stars_per_batch=1) unless set,
# and without the A_K inflation unless set
B1 = dict(stars_per_batch=1, ak_inflate=False)
CONFIGS = {
    "default": Config(**B1),
    "a0_ss2018_logg_trunc": Config(extinction="SS2018", use_logg=True, **B1, advanced=Advanced(mixture_alpha=0.0, truncate=True)),
    "a05_f19teff_bands_batch": Config(extinction="F19_TEFF", bands=("G", "J", "Ks"), stars_per_batch=5, ak_inflate=False,
                                      advanced=Advanced(mixture_alpha=0.5)),
    "a1_wc2019_noimf_ceiling_ak": Config(extinction="WC2019", **dict(B1, ak_inflate=True),
                                         advanced=Advanced(mixture_alpha=1.0, imf_prior=False, npe_ratio_ceiling=0.95)),
    "none_grey": Config(extinction="none", **B1),
    "f19_grey": Config(bands=("BP", "RP", "H", "Ks"), **B1, advanced=Advanced(grey=0.1)),
}


@pytest.fixture(scope="module")
def models():
    torch.set_num_threads(1)
    return load_models()


@pytest.fixture(scope="module")
def ref():
    with np.load(DATA / "reference.npz") as d:
        return {k: d[k] for k in d.files}


@pytest.mark.parametrize("name", CONFIGS)
def test_matches_isosbi(name, models, ref):
    cfg = CONFIGS[name]
    cfg.n_latent_x, cfg.n_theta = 50, 20
    inputs, rejected = build_inputs(read_table(DATA / "reference_stars.ecsv", XIANG_COLUMNS), cfg)
    assert len(rejected) == 0
    torch.manual_seed(SEED)  # as run_chunk
    s1 = stage1(inputs, models, cfg, SEED)
    (d, e, theta, w), _, diag = stage2(s1, inputs, models, cfg, torch_seed=SEED)
    got = dict(distance=d, ebv=cfg.law().to_ebv(e), theta=theta, weights=w, ess_norm=s1["ess_norm"],
               **{k: diag[k] for k in ("par_valid", "frac_valid_samples_pre", "ess_frac", "ess_frac_raw",
                                       "n_negative_ebv_per_star", "frac_colour_clipped")})
    for k, v in got.items():
        np.testing.assert_array_equal(v.numpy(), ref[f"{name}/{k}"], err_msg=f"{name}/{k}")


def test_run_chunk_summary(models, ref):
    """run_chunk's percentiles are those of the reference draws (theta, age, distance, A0)."""
    cfg = Config(n_latent_x=50, n_theta=20, n_keep=7, **B1)
    inputs, _ = build_inputs(read_table(DATA / "reference_stars.ecsv", XIANG_COLUMNS), cfg)
    res = run_chunk(inputs, models, cfg, seed=SEED)
    theta = ref["default/theta"].astype(np.float64)
    q = np.concatenate([theta, 10 ** (theta[..., 2:3] - 9), ref["default/distance"][..., None].astype(np.float64)], -1)
    np.testing.assert_array_equal(res["percentiles"][..., :6],
                                  np.percentile(q, cfg.percentiles, axis=1).transpose(1, 0, 2).astype(np.float32))
    assert res["samples"].shape == (5, 7, 7)
