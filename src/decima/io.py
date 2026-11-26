"""Read a table of stars and turn it into `StarInputs`.

Any format astropy reads (FITS, CSV, ECSV, VOTable, HDF5, Parquet, ...), an astropy Table or a pandas DataFrame.
Canonical column names; pass `columns={"teff": "TEFF", ...}` (canonical -> yours) for a table that names them differently:

    required: parallax, parallax_error [mas]; teff, teff_error [K]; feh, feh_error; Ks, Ks_error [mag]
    position: ra, dec [deg, ICRS] for the Bailer-Jones+21 distance prior, unless L, alpha, beta are given per star; stars with neither are rejected
    optional: id; alpha_fe, alpha_fe_error; logg, logg_error; G, BP, RP, J, H and their _error column
    zero-point: nu_eff_used_in_astrometry, pseudocolour, ecl_lat, astrometric_params_solved (Gaia archive names), only for Config.parallax_zero_point

Parallaxes are used as given, unless Config.parallax_zero_point subtracts the Lindegren+21 zero-point (`parallax_zero_point`). Magnitudes are Gaia (E)DR3 G, BP, RP and 2MASS J, H, Ks; masked or NaN means not measured, and that band is skipped for the star. Errors are floored (`Config.gaia_err_floor`, `twomass_err_floor`); a magnitude without an error gets `Config.missing_err`. A missing [alpha/Fe] (or its error) is filled with `Config.alpha_fe_fill` +- `alpha_fe_err_fill`. Stars without one of the required values are not run; `build_inputs` returns them with the reason.
"""

import numpy as np
import torch
from astropy.table import Table

from decima.ak import estimate_a_k, inflate_pseudo_mag_err
from decima.distance import bj_prior_params
from decima.extinction import BANDS
from decima.pipeline import Config, StarInputs

REQUIRED = ("Ks", "feh", "feh_error", "teff", "teff_error", "parallax", "parallax_error")
COLUMNS = ("id", "ra", "dec", "parallax", "parallax_error", "teff", "teff_error", "feh", "feh_error", "alpha_fe", "alpha_fe_error", "logg", "logg_error", "L", "alpha", "beta", *BANDS, *(f"{b}_error" for b in BANDS), "nu_eff_used_in_astrometry", "pseudocolour", "ecl_lat", "astrometric_params_solved")
ASTROMETRY = ("nu_eff_used_in_astrometry", "pseudocolour", "ecl_lat", "astrometric_params_solved")
_GAIA = ("G", "BP", "RP")


def read_table(source, columns: dict = None) -> Table:
    """The canonical columns (`COLUMNS`) of `source` (path, Table or DataFrame), renamed from `columns`."""
    if isinstance(source, Table):
        t = source
    elif hasattr(source, "iloc"): # pandas DataFrame
        t = Table.from_pandas(source)
    else:
        t = Table.read(source)
    columns = columns or {}
    unknown = set(columns) - set(COLUMNS)
    if unknown:
        raise ValueError(f"unknown canonical column(s) {sorted(unknown)}; choose from {COLUMNS}")
    absent = [v for v in columns.values() if v not in t.colnames]
    if absent:
        raise ValueError(f"column(s) {absent} not in the table")
    return Table({c: t[columns.get(c, c)] for c in COLUMNS if columns.get(c, c) in t.colnames})


def _f(tab, name):
    """Float column with masked entries as NaN (np.asarray alone would return the masked values' fill data); all NaN
    if the table has no such column."""
    if name not in tab.colnames:
        return np.full(len(tab), np.nan)
    return np.ma.filled(np.ma.asarray(tab[name]).astype(float), np.nan)


def _t(a):
    return torch.tensor(np.asarray(a, dtype=np.float32))


def parallax_zero_point(g, nu_eff, pseudocolour, ecl_lat, params_solved):
    """Lindegren+21 Gaia (E)DR3 parallax zero-point [mas] per star, from the gaiadr3-zeropoint package; subtract it from the parallax.

    NaN where an input is missing or the star is outside the recipe's validity (not a 5- or 6-parameter solution, G outside 6-21, nu_eff or pseudocolour out of range).
    """
    try:
        from zero_point import zpt
    except ImportError as e:
        raise ImportError("Config.parallax_zero_point needs the gaiadr3-zeropoint package: pip install 'decima-stars[zeropoint]'") from e
    zpt.load_tables()
    g, nu_eff, pseudocolour, ecl_lat, params_solved = (np.asarray(v, float) for v in (g, nu_eff, pseudocolour, ecl_lat, params_solved))
    six = params_solved == 95
    nu_eff, pseudocolour = np.where(six, 1.5, nu_eff), np.where(six, pseudocolour, 1.5) # the input the solution type does not use
    ok = np.isin(params_solved, (31, 95)) & np.isfinite(g + nu_eff + pseudocolour + ecl_lat)
    out = np.full(len(g), np.nan)
    if ok.any():
        out[ok] = zpt.get_zpt(g[ok], nu_eff[ok], pseudocolour[ok], ecl_lat[ok], params_solved[ok].astype(int), _warnings=False)
    return out


def build_inputs(tab: Table, cfg: Config):
    """(StarInputs for the stars that can be run, Table of the rest: row, id, reason). `tab` has canonical names."""
    missing = [k for k in REQUIRED + ("Ks_error",) if k not in tab.colnames]
    if missing:
        raise ValueError(f"required column(s) {missing} missing")
    if not ({"ra", "dec"} <= set(tab.colnames) or {"L", "alpha", "beta"} <= set(tab.colnames)):
        raise ValueError("need ra, dec (for the Bailer-Jones+21 prior) or per-star L, alpha, beta")

    reason = np.full(len(tab), "", dtype=object)
    for k in REQUIRED:
        reason[(reason == "") & ~np.isfinite(_f(tab, k))] = f"no {k}"
    reason[(reason == "") & ~(_f(tab, "parallax_error") > 0)] = "parallax_error <= 0"
    has_pos = (np.isfinite(_f(tab, "ra")) & np.isfinite(_f(tab, "dec"))) | (np.isfinite(_f(tab, "L")) & np.isfinite(_f(tab, "alpha")) & np.isfinite(_f(tab, "beta")))
    reason[(reason == "") & ~has_pos] = "no ra, dec (or L, alpha, beta)"
    ok = reason == ""
    rows = np.flatnonzero(ok)
    rejected = Table({"row": np.flatnonzero(~ok), "reason": reason[~ok].astype(str)})
    if "id" in tab.colnames:
        rejected.add_column(np.asarray(tab["id"])[~ok], name="id", index=1)
    stars = tab[ok]

    c = lambda k: _f(stars, k)

    def err(band):
        e = c(f"{band}_error")
        out = np.fmax(e, cfg.gaia_err_floor if band in _GAIA else cfg.twomass_err_floor)
        return out if not cfg.missing_err or cfg.missing_err <= 0 else np.where(np.isfinite(e), out, cfg.missing_err)

    teff = c("teff")
    log_teff = np.log10(teff)
    log_teff_err = c("teff_error") / (teff * np.log(10))

    # prepare pseudo absolute magnitude M_pseudo,K = parallax * 10^(0.2 Ks - 2) = 10^(-0.2 M_K) [parallax in mas], with its error from the parallax and Ks
    parallax, par_err = c("parallax"), c("parallax_error")
    zero_point_applied = np.zeros(len(stars), bool)
    if cfg.parallax_zero_point:
        missing = [k for k in ("G",) + ASTROMETRY if k not in stars.colnames]
        if missing:
            raise ValueError(f"parallax_zero_point needs the column(s) {missing}")
        zp = parallax_zero_point(c("G"), *(c(k) for k in ASTROMETRY))
        zero_point_applied = np.isfinite(zp)
        parallax = parallax - np.where(zero_point_applied, zp, 0.0) # no zero-point (outside its validity): parallax as given
    ks, ks_err = c("Ks"), err("Ks")
    pseudo_mag = parallax * 10 ** (0.2 * ks - 2)
    pseudo_mag_err = np.sqrt((10 ** (0.2 * ks - 2) * par_err) ** 2 + (0.2 * np.log(10) * pseudo_mag) ** 2 * ks_err ** 2)

    # inflate error in the pseudo absolute magnitude by the Casagrande+21 A_K estimate, if used, to account for uncertainty in correcting for extinction.
    pseudo_mag_inflated = np.zeros(len(stars), bool)
    if cfg.pseudo_mag_inflate:
        a_k = estimate_a_k(c("BP"), ks, teff, c("logg"), c("feh"), g=c("G"))["a_k"]
        pseudo_mag_err = inflate_pseudo_mag_err(pseudo_mag, pseudo_mag_err, a_k)
        pseudo_mag_inflated = np.isfinite(a_k)

    no_alpha = ~np.isfinite(c("alpha_fe")) | ~np.isfinite(c("alpha_fe_error"))
    alpha_fe = np.where(no_alpha, cfg.alpha_fe_fill, c("alpha_fe"))
    alpha_fe_err = np.where(no_alpha, cfg.alpha_fe_err_fill, c("alpha_fe_error"))

    x_obs = np.stack([c("feh"), alpha_fe, log_teff, pseudo_mag], axis=1)
    x_obs_err = np.stack([c("feh_error"), alpha_fe_err, log_teff_err, pseudo_mag_err], axis=1)

    m_obs = np.stack([c(b) for b in cfg.bands], axis=1)
    m_stds = np.stack([err(b) for b in cfg.bands], axis=1)
    m_stds = np.where(np.isfinite(m_obs), m_stds, np.nan) # no magnitude, no error
    err_missing = np.isfinite(m_obs) & ~np.isfinite(np.stack([c(f"{b}_error") for b in cfg.bands], axis=1))

    L, alpha, beta = bj_prior_params(c("ra"), c("dec"))
    given = np.isfinite(c("L")) & np.isfinite(c("alpha")) & np.isfinite(c("beta")) # per-star prior overrides the map
    for t, k in ((L, "L"), (alpha, "alpha"), (beta, "beta")):
        t[torch.from_numpy(given)] = _t(c(k)[given])

    inputs = StarInputs(
        ids=np.asarray(stars["id"]) if "id" in stars.colnames else rows,
        x_obs=_t(x_obs), x_obs_err=_t(x_obs_err),
        m_obs=_t(m_obs), m_stds=_t(m_stds), err_missing=torch.tensor(err_missing),
        par_obs=_t(parallax), par_error=_t(par_err),
        log_g=_t(c("logg")), log_g_err=_t(c("logg_error")),
        L=L, alpha=alpha, beta=beta, pseudo_mag_inflated=torch.tensor(pseudo_mag_inflated),
        zero_point_applied=torch.tensor(zero_point_applied),
    )
    return inputs, rejected
