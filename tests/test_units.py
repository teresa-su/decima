"""Fast checks that need no models."""
from math import lgamma, log
from pathlib import Path

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
from decima.pipeline import Config, failed_stars, percentiles, summary_table

XIANG_COLUMNS = {
    "id": "GAIAEDR3_SOURCE_ID_1", "ra": "RAJ2000", "dec": "DEJ2000",
    "G": "GAIA_G", "BP": "GAIA_BP", "RP": "GAIA_RP", "G_error": "E_GAIA_G", "BP_error": "E_GAIA_BP",
    "RP_error": "E_GAIA_RP", "J": "TWOMASS_J", "H": "TWOMASS_H", "Ks": "TWOMASS_K", "J_error": "j_msigcom",
    "H_error": "h_msigcom", "Ks_error": "ks_msigcom", "parallax": "PARALLAX", "parallax_error": "E_PARALLAX",
    "teff": "TEFF", "teff_error": "E_TEFF", "logg": "LOGG", "logg_error": "E_LOGG", "feh": "FEH", "feh_error": "E_FEH",
    "alpha_fe": "ALPHA_FE", "alpha_fe_error": "E_ALPHA_FE",
}
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


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_percentiles_match_numpy_bit_for_bit(device):
    rng = np.random.default_rng(0)
    for n in (1, 2, 7, 8000):
        q = rng.normal(size=(20, n, 3)) * 10 ** rng.uniform(-3, 4, size=(20, 1, 3))
        q[:5] = np.round(q[:5]) # ties
        q[7, rng.integers(0, n), 1] = np.nan
        q[8, :, 2] = 3.0
        for pcts in ((5, 16, 50, 84, 95), (0, 2.5, 50, 97.5, 100), (33.3,)):
            assert np.array_equal(percentiles(q, pcts, device), np.percentile(q, pcts, axis=1), equal_nan=True)


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


# --- settings errors say how to fix them -----------------------------------------------------------------------------

def test_config_errors_say_how_to_fix():
    with pytest.raises(ValueError, match="0.1, 0.2"):
        Config(mixture_alpha=0.05, n_theta=10)
    with pytest.raises(ValueError, match="did you mean 'F19'"):
        Config(extinction="F99")
    with pytest.raises(ValueError, match="use 'cuda'"):
        Config(device="gpu")
    Config(device="cuda:1")
    with pytest.raises(ValueError, match="did you mean 'n_theta'"):
        Config.from_json('{"n_thetas": 5}')
    assert Config.from_json('{"n_keep": 7}').n_keep == 7  # missing settings keep their defaults


# --- column metadata, suggest_columns, check_inputs ------------------------------------------------------------------

def test_column_info_is_consistent():
    from decima.io import COLUMN_INFO, COLUMNS, REQUIRED
    assert tuple(COLUMN_INFO) == COLUMNS
    assert {c for c, i in COLUMN_INFO.items() if i.group == "required"} == set(REQUIRED) | {"Ks_error"}
    assert all(i.description for i in COLUMN_INFO.values())


def test_suggest_columns_reproduces_the_reference_mapping():
    sug = decima.suggest_columns(Table.read(Path(__file__).parent / "data" / "reference_stars.ecsv"))
    assert dict(sug) == XIANG_COLUMNS and not sug.ambiguous and not sug.missing and not sug.unused


@pytest.mark.parametrize("names,expect", [
    ({"parallax": "Plx", "parallax_error": "e_Plx", "teff": "Teff", "teff_error": "e_Teff"},
     {"parallax": "Plx", "parallax_error": "e_Plx", "teff": "Teff", "teff_error": "e_Teff"}),
    ({"Ks": "Kmag", "Ks_error": "e_Kmag", "feh": "fe_h", "feh_error": "e_fe_h"},
     {"Ks": "Kmag", "Ks_error": "e_Kmag", "feh": "fe_h", "feh_error": "e_fe_h"}),
    ({"Ks": "ks_m", "Ks_error": "ks_msigcom", "J": "j_m", "J_error": "j_msigcom"},
     {"Ks": "ks_m", "Ks_error": "ks_msigcom", "J": "j_m", "J_error": "j_msigcom"}),
])
def test_suggest_columns_patterns(names, expect):
    t = Table({v: [1.0] for v in names.values()})
    assert dict(decima.suggest_columns(t)) == expect


def test_suggest_columns_does_not_guess_between_candidates():
    t = Table({"TEFF": [5800.0], "teff_spec": [5700.0], "FEH": [0.0]})
    sug = decima.suggest_columns(t)
    assert "teff" not in sug and sorted(sug.ambiguous["teff"]) == ["TEFF", "teff_spec"]
    assert sug["feh"] == "FEH"


def test_suggest_columns_already_canonical_and_missing():
    sug = decima.suggest_columns(ROWS)
    assert dict(sug) == {} and "teff" in sug.exact
    sug = decima.suggest_columns(Table({"TEFF": [1.0]}))
    assert "feh" in sug.missing and "ra" in sug.missing


def _broken():
    t = ROWS.copy()
    t["ra"], t["dec"] = [10.0, 200.0, 300.0], [-20.0, 40.0, 10.0]
    t["feh"][2] = 0.1
    t["parallax_error"] = 0.02
    return t


def test_check_inputs_ok_and_counts_rejections():
    rep = decima.check_inputs(_broken())
    assert rep.ok and rep.n_run == 3 and rep.rejected == {}
    t = _broken()
    t["feh"][1] = np.nan
    rep = decima.check_inputs(t)
    assert rep.ok and rep.n_run == 2 and rep.rejected == {"no feh": 1}
    assert "2 of 3 stars can be run" in str(rep)


def test_check_inputs_names_each_problem():
    t = _broken()
    t["teff"] = np.log10(t["teff"])                      # log Teff
    t["parallax"] = t["parallax"] / 1000                 # arcsec
    t["Ks"][0] = 99.0                                    # flag
    t["ra"], t["dec"] = np.radians(t["ra"]), np.radians(t["dec"])  # radians
    t.rename_column("feh", "FEH_x")
    rep = decima.check_inputs(t)                          # no columns file: feh is missing
    assert not rep.ok
    assert any("FEH_x" in e for e in rep.errors)
    t.rename_column("FEH_x", "feh")
    rep = decima.check_inputs(t)
    text = " | ".join(rep.warnings)
    assert rep.ok
    for word in ("Kelvin", "arcseconds", "99", "radians"):
        assert word in text


def test_check_inputs_columns_file_errors():
    t = _broken()
    rep = decima.check_inputs(t, {"tef": "teff"})
    assert any("did you mean 'teff'" in e for e in rep.errors)
    rep = decima.check_inputs(t, {"teff": "TEF"})
    assert any("teff" in e and "'TEF'" in e and "did you mean 'teff'" in e for e in rep.errors)


def test_check_inputs_settings_and_unreadable_table(tmp_path):
    (tmp_path / "bad.json").write_text('{"n_thetas": 5}')
    rep = decima.check_inputs(_broken(), config=tmp_path / "bad.json")
    assert not rep.ok and any("n_theta" in e for e in rep.errors)
    assert not decima.check_inputs(tmp_path / "nope.fits").ok


def test_read_table_names_the_canonical_key():
    with pytest.raises(ValueError, match=r"teff -> 'TEF'.*did you mean 'teff'"):
        read_table(ROWS, {"teff": "TEF"})
    with pytest.raises(ValueError, match="did you mean 'teff'"):
        read_table(ROWS, {"tef": "teff"})


def test_fetch_uses_a_copy_in_decima_data_dir_as_is(tmp_path, monkeypatch):
    from decima.models import fetch
    monkeypatch.setenv("DECIMA_DATA_DIR", str(tmp_path))
    (tmp_path / "npe.pt").write_bytes(b"not the released file")
    assert fetch("npe.pt") == tmp_path / "npe.pt"  # no download, and the copy is not overwritten
    assert (tmp_path / "npe.pt").read_bytes() == b"not the released file"
