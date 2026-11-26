"""Fast checks that need no models."""
from math import lgamma, log

import healpy as hp
import numpy as np
import pytest
import torch
from astropy.table import Table

import decima

from decima.distance import BJ_DEFAULT_PARAMS, _bj_ggd_table, bailer_jones, bj_prior_params
from decima.extinction import F19_BPRP, FixedLaw, PolynomialLaw, SS2018, get_law
from decima.io import build_inputs, read_table
from decima.io import parallax_zero_point
from decima.pipeline import Config, failed_stars, summary_table

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
                 mixture_alpha=0.3)
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
    inputs, rejected = build_inputs(ROWS, Config(bands=("BP", "Ks"), pseudo_mag_inflate=False))
    assert list(rejected["row"]) == [1, 2] and list(rejected["reason"]) == ["parallax_error <= 0", "no feh"]
    assert len(inputs) == 1 and inputs.x_obs[0, 1] == 0.0 and np.isclose(inputs.x_obs_err[0, 1], 0.2)
    assert inputs.m_stds[0, 0] == pytest.approx(0.02) # BP error floored
    assert not inputs.pseudo_mag_inflated.any() and not inputs.zero_point_applied.any()


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


def test_mixture_fraction_must_be_whole():
    Config(mixture_alpha=0.3, n_theta=10)
    with pytest.raises(ValueError):
        Config(mixture_alpha=0.05, n_theta=10)


def test_stars_without_position_rejected():
    t = ROWS[:1].copy()
    t["ra"], t["dec"] = [np.nan], [np.nan]
    inputs, rejected = build_inputs(t, Config())
    assert len(inputs) == 0 and list(rejected["reason"]) == ["no ra, dec (or L, alpha, beta)"]
    t["L"], t["alpha"], t["beta"] = [500.0], [1.5], [3.0]  # a per-star prior is enough
    inputs, _ = build_inputs(t, Config())
    assert len(inputs) == 1


def test_failed_stars():
    rng = np.random.default_rng(0)
    age = rng.uniform(1, 10, (3, 100))
    age[1] = 5.0                # collapsed: p16 == p50 == p84
    age[2, :60] = 3.0           # collapsed on one side
    assert list(failed_stars(age, np.array([False, False, False]))) == [False, True, True]
    assert list(failed_stars(age[:1], np.array([True]))) == [True]


def test_summary_meta_roundtrip(tmp_path):
    cfg = Config(bands=("J", "Ks"))
    n, n_q = 2, 7
    res = {"ids": np.arange(n), "percentiles": np.zeros((n, len(cfg.percentiles), n_q), np.float32),
           "band_used": np.ones((n, 2), bool), "band_err_missing": np.zeros((n, 2), bool),
           **{k: np.zeros(n) for k in ("failed", "uniform_fallback", "ess_frac", "ess_norm_stage1", "frac_valid_samples_pre",
                                       "frac_colour_clipped", "n_negative_extinction", "par_valid", "pseudo_mag_inflated",
                                       "zero_point_applied")}}
    t = summary_table(res, cfg)
    assert "feh_init_p50" in t.colnames and "failed" in t.colnames
    for ext in ("fits", "ecsv"):
        t.write(tmp_path / f"s.{ext}")
        back = Table.read(tmp_path / f"s.{ext}")
        assert Config.from_json(back.meta["CONFIG"]).to_json() == cfg.to_json()
        assert back.meta["DECIMA"] == decima.__version__


def test_parallax_zero_point():
    zpt = pytest.importorskip("zero_point.zpt")
    zpt.load_tables()
    # a 5-parameter and a 6-parameter solution, then one outside the recipe (3-parameter): NaN
    g, nu, pc, lat, sol = [15.0, 17.0, 15.0], [1.5, np.nan, 1.5], [np.nan, 1.45, np.nan], [10.0, -30.0, 10.0], [31, 95, 3]
    got = parallax_zero_point(g, nu, pc, lat, sol)
    a = lambda *v: np.array(v)  # the package needs arrays under numpy 2
    want = zpt.get_zpt(a(15.0, 17.0), a(1.5, 1.5), a(1.5, 1.45), a(10.0, -30.0), a(31, 95))
    assert np.allclose(got[:2], want) and np.isnan(got[2])
    t = ROWS[:1].copy()
    t["G"], t["nu_eff_used_in_astrometry"], t["pseudocolour"], t["ecl_lat"], t["astrometric_params_solved"] = [15.0], [1.5], [np.nan], [10.0], [31]
    inputs, _ = build_inputs(t, Config(parallax_zero_point=True))
    assert inputs.zero_point_applied[0] and np.isclose(inputs.par_obs[0].item(), 2.0 - want[0], rtol=1e-6)
