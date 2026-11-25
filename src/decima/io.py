"""Read a table of stars and turn it into `StarInputs`.

Any format astropy reads (FITS, CSV, ECSV, VOTable, HDF5, Parquet, ...), an astropy Table or a pandas DataFrame.
Canonical column names; pass `columns={"teff": "TEFF", ...}` (canonical -> yours) for a table that names them differently:

    required   parallax, parallax_error [mas]; teff, teff_error [K]; feh, feh_error; Ks, Ks_error [mag]
    position   ra, dec [deg, ICRS] for the Bailer-Jones+21 distance prior, unless L, alpha, beta are given per star
    optional   id; alpha_fe, alpha_fe_error; logg, logg_error; G, BP, RP, J, H and their _error columns

Parallaxes are used as given: correct them for the Gaia zero-point first. Magnitudes are Gaia (E)DR3 G, BP, RP and 2MASS
J, H, Ks; masked or NaN means not measured, and that band is skipped for the star. Errors are floored
(`Advanced.gaia_err_floor`, `twomass_err_floor`); a magnitude without an error gets `Advanced.missing_err`. A missing
[alpha/Fe] (or its error) is filled with `Advanced.alpha_fe_fill` +- `alpha_fe_err_fill`. Stars without one of the
required values are not run; `build_inputs` returns them with the reason.
"""
import numpy as np
import torch
from astropy.table import Table

from decima.ak import estimate_a_k, inflate_pseudo_mag_err
from decima.distance import bj_prior_params
from decima.extinction import BANDS
from decima.pipeline import Config, StarInputs

REQUIRED = ("Ks", "feh", "feh_error", "teff", "teff_error", "parallax", "parallax_error")
COLUMNS = ("id", "ra", "dec", "parallax", "parallax_error", "teff", "teff_error", "feh", "feh_error", "alpha_fe",
           "alpha_fe_error", "logg", "logg_error", "L", "alpha", "beta",
           *BANDS, *(f"{b}_error" for b in BANDS))
_GAIA = ("G", "BP", "RP")


def read_table(source, columns: dict = None) -> Table:
    """The canonical columns (`COLUMNS`) of `source` (path, Table or DataFrame), renamed from `columns`."""
    if isinstance(source, Table):
        t = source
    elif hasattr(source, "iloc"):  # pandas DataFrame
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
    ok = reason == ""
    rows = np.flatnonzero(ok)
    rejected = Table({"row": np.flatnonzero(~ok), "reason": reason[~ok].astype(str)})
    if "id" in tab.colnames:
        rejected.add_column(np.asarray(tab["id"])[~ok], name="id", index=1)
    stars = tab[ok]

    adv = cfg.advanced
    c = lambda k: _f(stars, k)

    def err(band):
        e = c(f"{band}_error")
        out = np.fmax(e, adv.gaia_err_floor if band in _GAIA else adv.twomass_err_floor)
        return out if not adv.missing_err or adv.missing_err <= 0 else np.where(np.isfinite(e), out, adv.missing_err)

    teff = c("teff")
    log_teff = np.log10(teff)
    log_teff_err = c("teff_error") / (teff * np.log(10))

    # Pseudo absolute magnitude: parallax * 10^(0.2 Ks - 2) = 10^(-0.2 M_K) [parallax in mas], with its error from the
    # parallax and Ks. Kept in numpy until the final cast (torch * numpy would silently upcast to float64).
    parallax, par_err = c("parallax"), c("parallax_error")
    ks, ks_err = c("Ks"), err("Ks")
    pseudo_mag = parallax * 10 ** (0.2 * ks - 2)
    pseudo_mag_err = np.sqrt((10 ** (0.2 * ks - 2) * par_err) ** 2 + (0.2 * np.log(10) * pseudo_mag) ** 2 * ks_err ** 2)

    ak_inflated = np.zeros(len(stars), bool)
    if cfg.ak_inflate:
        a_k = estimate_a_k(c("BP"), ks, teff, c("logg"), c("feh"), g=c("G"))["a_k"]
        pseudo_mag_err = inflate_pseudo_mag_err(pseudo_mag, pseudo_mag_err, a_k)
        ak_inflated = np.isfinite(a_k)

    no_alpha = ~np.isfinite(c("alpha_fe")) | ~np.isfinite(c("alpha_fe_error"))
    alpha_fe = np.where(no_alpha, adv.alpha_fe_fill, c("alpha_fe"))
    alpha_fe_err = np.where(no_alpha, adv.alpha_fe_err_fill, c("alpha_fe_error"))

    x_obs = np.stack([c("feh"), alpha_fe, log_teff, pseudo_mag], axis=1)
    x_obs_err = np.stack([c("feh_error"), alpha_fe_err, log_teff_err, pseudo_mag_err], axis=1)

    m_obs = np.stack([c(b) for b in cfg.bands], axis=1)
    m_stds = np.stack([err(b) for b in cfg.bands], axis=1)
    m_stds = np.where(np.isfinite(m_obs), m_stds, np.nan)  # no magnitude, no error
    err_missing = np.isfinite(m_obs) & ~np.isfinite(np.stack([c(f"{b}_error") for b in cfg.bands], axis=1))

    L, alpha, beta = bj_prior_params(c("ra"), c("dec"))
    given = np.isfinite(c("L")) & np.isfinite(c("alpha")) & np.isfinite(c("beta"))  # per-star prior overrides the map
    for t, k in ((L, "L"), (alpha, "alpha"), (beta, "beta")):
        t[torch.from_numpy(given)] = _t(c(k)[given])

    inputs = StarInputs(
        ids=np.asarray(stars["id"]) if "id" in stars.colnames else rows,
        x_obs=_t(x_obs), x_obs_err=_t(x_obs_err),
        m_obs=_t(m_obs), m_stds=_t(m_stds), err_missing=torch.tensor(err_missing),
        par_obs=_t(parallax), par_error=_t(par_err),
        log_g=_t(c("logg")), log_g_err=_t(c("logg_error")),
        L=L, alpha=alpha, beta=beta, ak_inflated=torch.tensor(ak_inflated),
    )
    return inputs, rejected
