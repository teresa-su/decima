"""Fast checks that need no models."""
from math import lgamma, log

import healpy as hp
import numpy as np
import pytest
import torch
from astropy.table import Table

from decima.distance import BJ_DEFAULT_PARAMS, _bj_ggd_table, bailer_jones, bj_prior_params
from decima.extinction import F19_BPRP, FixedLaw, PolynomialLaw, SS2018, get_law
from decima.io import build_inputs, read_table
from decima.pipeline import Advanced, Config

ROWS = Table({"teff": [5800.0, 6000.0, 5500.0], "teff_error": [80.0, 80.0, 80.0], "feh": [0.0, -0.3, np.nan],
              "feh_error": [0.05] * 3, "parallax": [2.0, 1.0, 1.5], "parallax_error": [0.02, 0.0, 0.02],
              "Ks": [9.0, 10.0, 11.0], "Ks_error": [0.02, np.nan, 0.02], "ra": [10.0, 200.0, 300.0],
              "dec": [-20.0, 40.0, np.nan], "BP": [11.0, 12.0, 13.0], "BP_error": [0.001, 0.001, 0.001]})


def test_custom_laws_reproduce_builtins():
    shape = (2, 3)
    iso = {"BP": torch.rand(6) + 1.0, "RP": torch.rand(6)}
    A0 = torch.rand(shape)
    bands = ["G", "Ks"]
    for law, custom in ((get_law("SS2018"), FixedLaw(SS2018)), (get_law("F19"), PolynomialLaw(F19_BPRP))):
        for a, b in zip(law(bands, iso, shape, "cpu", A0=A0), custom(bands, iso, shape, "cpu", A0=A0)):
            assert torch.equal(a, b)


def test_config_json_roundtrip_with_custom_law():
    cfg = Config(bands=("J", "H", "Ks"), extinction=FixedLaw({"J": 0.7, "H": 0.4, "Ks": 0.2}, grey=0.01),
                 advanced=Advanced(mixture_alpha=0.3))
    back = Config.from_json(cfg.to_json())
    assert back.to_json() == cfg.to_json() and back.law().R == {"J": 0.7, "H": 0.4, "Ks": 0.2}
    assert Config.from_json(Config(extinction=get_law("F19_TEFF")).to_json()).law().x == "teff"


def test_bands_need_ks():
    with pytest.raises(ValueError):
        Config(bands=("G", "BP"))


@pytest.mark.parametrize("ext", ["csv", "ecsv", "fits"])
def test_read_formats_and_column_map(tmp_path, ext):
    t = ROWS.copy()
    t.rename_column("teff", "TEFF")
    t["unused"] = 1
    t.write(tmp_path / f"s.{ext}")
    got = read_table(tmp_path / f"s.{ext}", {"teff": "TEFF"})
    assert "teff" in got.colnames and "unused" not in got.colnames
    assert np.allclose(got["teff"], ROWS["teff"])


def test_build_inputs_rejects_and_fills():
    inputs, rejected = build_inputs(ROWS, Config(bands=("BP", "Ks")))
    assert list(rejected["row"]) == [1, 2] and list(rejected["reason"]) == ["parallax_error <= 0", "no feh"]
    assert len(inputs) == 1 and inputs.x_obs[0, 1] == 0.0 and np.isclose(inputs.x_obs_err[0, 1], 0.15)
    assert inputs.m_stds[0, 0] == pytest.approx(0.02)  # BP error floored
    assert not inputs.ak_inflated.any()


def test_bj_override_and_lookup():
    t = ROWS[:1].copy()
    inputs, _ = build_inputs(t, Config())
    L, a, b = bj_prior_params([10.0], [-20.0])
    assert inputs.L[0] == L[0] and inputs.alpha[0] == a[0]
    t["L"], t["alpha"], t["beta"] = [500.0], [1.5], [3.0]
    inputs, _ = build_inputs(t, Config())
    assert (inputs.L[0], inputs.alpha[0], inputs.beta[0]) == (500.0, 1.5, 3.0)


def test_bj_table_lookup_and_ggd():
    # exact equatorial pixel centres map back to their own pixel's parameters; no position -> the default
    ra, dec = hp.pix2ang(32, np.arange(12288), nest=True, lonlat=True)
    got = np.stack([x.numpy() for x in bj_prior_params(ra, dec)], 1)
    assert np.allclose(got, _bj_ggd_table(), rtol=1e-6)
    assert [x.item() for x in bj_prior_params([np.nan], [10.0])] == list(BJ_DEFAULT_PARAMS)
    # GGD log_prob matches the closed form, including an extreme pixel
    for L, a, b in [(1000.0, 1.0, 2.0), (2.5e-6, 0.19, 7.1), (620.0, 0.79, 5.7)]:
        for r in (50.0, 500.0, 5000.0):
            want = log(a) - lgamma((b + 1) / a) - (b + 1) * log(L) + b * log(r) - (r / L) ** a
            assert abs(bailer_jones(L, a, b).log_prob(torch.tensor(r)).item() - want) < 1e-3
