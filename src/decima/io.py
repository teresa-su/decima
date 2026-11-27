"""Read a table of stars and turn it into `StarInputs`.

Any format astropy reads (FITS, CSV, ECSV, VOTable, HDF5, Parquet, ...), an astropy Table or a pandas DataFrame.
Canonical column names; pass `columns={"teff": "TEFF", ...}` (canonical -> yours) for a table that names them differently:

    required: parallax, parallax_error [mas]; teff, teff_error [K]; feh, feh_error; Ks, Ks_error [mag]
    position: ra, dec [deg, ICRS] for the Bailer-Jones+21 distance prior, unless L, alpha, beta are given per star; stars with neither are rejected
    optional: id; alpha_fe, alpha_fe_error; logg, logg_error; G, BP, RP, J, H and their _error column
    zero-point: nu_eff_used_in_astrometry, pseudocolour, ecl_lat, astrometric_params_solved (Gaia archive names), only for Config.parallax_zero_point

Parallaxes are used as given, unless Config.parallax_zero_point subtracts the Lindegren+21 zero-point (`parallax_zero_point`). Magnitudes are Gaia (E)DR3 G, BP, RP and 2MASS J, H, Ks; masked or NaN means not measured, and that band is skipped for the star. Errors are floored (`Config.gaia_err_floor`, `twomass_err_floor`); a magnitude without an error gets `Config.missing_err`. A missing [alpha/Fe] (or its error) is filled with `Config.alpha_fe_fill` +- `alpha_fe_err_fill`. Stars without one of the required values are not run; `build_inputs` returns them with the reason.
"""

import difflib
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

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


class ColumnInfo(NamedTuple):
    group: str # "required", "position", "optional" or "zero-point"
    unit: str
    description: str
    aliases: tuple = () # lower-case names (punctuation ignored) that `suggest_columns` accepts for the column; errors get e_x, x_err, ... automatically


def _build_column_info():
    base = {
        "id": ("optional", "", "Star identifier. Without it the summary's `id` is the 0-based row of the input table.",
               ("id", "sourceid", "gaiasourceid", "gaiaedr3sourceid", "gaiaedr3sourceid1", "gaiadr3sourceid", "sobjectid", "apogeeid", "objid", "designation")),
        "ra": ("position", "deg", "Right ascension (ICRS), for the Bailer-Jones+21 distance prior.", ("ra", "raj2000", "raicrs", "radeg")),
        "dec": ("position", "deg", "Declination (ICRS), for the Bailer-Jones+21 distance prior.", ("dec", "dej2000", "decj2000", "decicrs", "decdeg")),
        "parallax": ("required", "mas", "Gaia parallax (zero-point corrected, or see `parallax_zero_point`).", ("parallax", "plx", "gaiaparallax", "gaiaedr3parallax", "gaiadr3parallax")),
        "teff": ("required", "K", "Spectroscopic effective temperature.", ("teff", "teffk", "teffspec", "effectivetemperature")),
        "feh": ("required", "dex", "Spectroscopic surface [Fe/H].", ("feh", "fehspec")),
        "alpha_fe": ("optional", "dex", "Spectroscopic [alpha/Fe]. Missing values are filled with `alpha_fe_fill` (default 0) +- `alpha_fe_err_fill` (default 0.2).", ("alphafe", "afe")),
        "logg": ("optional", "dex", "Spectroscopic log g (cgs). Used for the pseudo-magnitude inflation, and in stage 2 only with `use_logg`.", ("logg", "loggspec", "surfacegravity")),
    }
    info = {}
    for k in ("id", "ra", "dec"):
        info[k] = ColumnInfo(*base[k])
    for k, nice in (("parallax", "parallax"), ("teff", "T_eff"), ("feh", "[Fe/H]"), ("alpha_fe", "[alpha/Fe]"), ("logg", "log g")):
        group, unit, desc, aliases = base[k]
        info[k] = ColumnInfo(group, unit, desc, aliases)
        info[f"{k}_error"] = ColumnInfo(group, unit, f"Uncertainty on {nice}." + (" Used as given (no floor)." if k in ("teff", "feh", "alpha_fe") else ""))
    info["L"] = ColumnInfo("position", "pc", "Per-star Bailer-Jones+21 prior length scale; with `alpha` and `beta` it replaces the lookup from ra, dec.")
    info["alpha"] = ColumnInfo("position", "", "Per-star Bailer-Jones+21 prior shape parameter (with `L`, `beta`).")
    info["beta"] = ColumnInfo("position", "", "Per-star Bailer-Jones+21 prior shape parameter (with `L`, `alpha`).")
    bands = {"G": ("Gaia G", ("g", "gaiag", "gmag", "photgmeanmag", "gaiaedr3photgmeanmag", "gaiadr3photgmeanmag")),
             "BP": ("Gaia BP", ("bp", "gaiabp", "bpmag", "photbpmeanmag", "gaiaedr3photbpmeanmag", "gaiadr3photbpmeanmag")),
             "RP": ("Gaia RP", ("rp", "gaiarp", "rpmag", "photrpmeanmag", "gaiaedr3photrpmeanmag", "gaiadr3photrpmeanmag")),
             "J": ("2MASS J", ("j", "jmag", "jm", "twomassj", "2massj")),
             "H": ("2MASS H", ("h", "hmag", "hm", "twomassh", "2massh")),
             "Ks": ("2MASS Ks", ("ks", "k", "kmag", "ksmag", "ksm", "km", "twomassk", "twomassks", "2massk", "2massks"))}
    for b in BANDS:
        nice, aliases = bands[b]
        info[b] = ColumnInfo("required" if b == "Ks" else "optional", "mag", f"{nice} apparent magnitude. NaN or masked: the band is skipped for that star." if b != "Ks" else f"{nice} apparent magnitude (required: it gives the pseudo-magnitude).", aliases)
    for b in BANDS:
        floor = "gaia_err_floor" if b in _GAIA else "twomass_err_floor"
        info[f"{b}_error"] = ColumnInfo("required" if b == "Ks" else "optional", "mag", f"Uncertainty on {bands[b][0]}, floored at `{floor}`; a magnitude without one gets `missing_err`.")
    info["nu_eff_used_in_astrometry"] = ColumnInfo("zero-point", "1/um", "Gaia archive column; only for `parallax_zero_point` (which also needs `G`).")
    info["pseudocolour"] = ColumnInfo("zero-point", "1/um", "Gaia archive column; only for `parallax_zero_point`.")
    info["ecl_lat"] = ColumnInfo("zero-point", "deg", "Gaia archive column (ecliptic latitude); only for `parallax_zero_point`.")
    info["astrometric_params_solved"] = ColumnInfo("zero-point", "", "Gaia archive column (31 = 5-parameter, 95 = 6-parameter solution); only for `parallax_zero_point`.")
    return info


COLUMN_INFO = _build_column_info() # canonical name -> ColumnInfo; the single source for suggest_columns, check_inputs and the generated reference page
assert tuple(COLUMN_INFO) == COLUMNS


def _norm(name) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _close(name, options):
    """The option closest to `name`, ignoring case, or None."""
    options = list(options)
    m = difflib.get_close_matches(str(name).lower(), [str(o).lower() for o in options], n=1)
    return options[[str(o).lower() for o in options].index(m[0])] if m else None


def _load(source) -> Table:
    if isinstance(source, Table):
        return source
    if hasattr(source, "iloc"): # pandas DataFrame
        return Table.from_pandas(source)
    return Table.read(source)


def read_table(source, columns: dict = None) -> Table:
    """The canonical columns (`COLUMNS`) of `source` (path, Table or DataFrame), renamed from `columns` (canonical -> yours)."""
    t = _load(source)
    columns = columns or {}
    unknown = set(columns) - set(COLUMNS)
    if unknown:
        hint = "".join(f" ({k!r}: did you mean {m!r}?)" for k in sorted(unknown) if (m := _close(k, COLUMNS)))
        raise ValueError(f"unknown canonical column(s) {sorted(unknown)}{hint}; the columns file maps DECIMA's names to yours, {{\"teff\": \"YOUR_TEFF\"}}. DECIMA's names: {COLUMNS}")
    absent = {k: v for k, v in columns.items() if v not in t.colnames}
    if absent:
        raise ValueError("column(s) not in the table: " + "; ".join(f"{k} -> {v!r}" + (f" (did you mean {m!r}?)" if (m := _close(v, t.colnames)) else "") for k, v in absent.items()))
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
        raise ValueError(f"required column(s) {missing} missing (a columns file maps DECIMA's names to yours: see `decima columns`)")
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


# ---------------------------------------------------------------------------------------------------------------------
# Finding and checking the columns of a table: `decima columns` and `decima check`
# ---------------------------------------------------------------------------------------------------------------------

_ERR_PREFIXES = ("e", "err", "error", "sig", "sigma", "unc")
_ERR_SUFFIXES = ("err", "error", "e", "sig", "sigma", "unc", "msigcom") # msigcom: 2MASS


def _aliases(canonical) -> set:
    """Normalised names under which a table might call `canonical`: its own name, the listed aliases, and for an error column the usual e_x / x_err / x_error forms of its value column's aliases."""
    if canonical.endswith("_error"):
        value = canonical[: -len("_error")]
        bases = {_norm(value), *COLUMN_INFO[value].aliases}
        out = {_norm(canonical)}
        for a in bases:
            out |= {p + a for p in _ERR_PREFIXES} | {a + x for x in _ERR_SUFFIXES}
        return out
    return {_norm(canonical), *COLUMN_INFO[canonical].aliases}


class ColumnSuggestion(dict):
    """canonical name -> the table's column, for the columns that need renaming; this is the content of a columns file (`json.dump` it).

    Also: `.exact` (columns that already carry a canonical name), `.ambiguous` (canonical -> several candidate columns, nothing
    guessed), `.missing` (required columns that were not found) and `.unused` (table columns that were not matched).
    """
    exact: list
    ambiguous: dict
    missing: list
    unused: list
    close: dict # missing column -> the unused column whose name is closest to it (a guess, not used)

    def __str__(self):
        lines = []
        for c in COLUMNS:
            if c in self.exact:
                lines.append(f"  ok  {c:<26} (already named {c})")
            elif c in self:
                lines.append(f"  ok  {c:<26} <- {self[c]!r}")
            elif c in self.ambiguous:
                lines.append(f"  ??  {c:<26} several candidates {self.ambiguous[c]}: choose one")
            elif c in self.missing:
                lines.append(f"  --  {c:<26} NOT FOUND (required: {COLUMN_INFO[c].description})" + (f"; closest column: {self.close[c]!r}" if c in self.close else ""))
        return "\n".join(lines)


def suggest_columns(source) -> ColumnSuggestion:
    """Suggest the columns file for a table: which of its columns hold DECIMA's inputs.

    Matches column names ignoring case and punctuation against the usual names (`COLUMN_INFO`): Gaia archive names, `E_X` / `X_ERR`
    / `X_error` error columns, `*_msigcom` for 2MASS, and so on. When two columns could both be `x`, nothing is guessed: the
    candidates are in `.ambiguous`. Check the result (units, in particular): `decima.check_inputs`.
    """
    t = _load(source)
    names = list(t.colnames)
    out = ColumnSuggestion()
    out.exact = [c for c in COLUMNS if c in names]
    taken = set(out.exact)
    by_norm = {}
    for n in names:
        if n not in taken:
            by_norm.setdefault(_norm(n), []).append(n)
    claims = {}
    for c in COLUMNS:
        if c in out.exact:
            continue
        cand = [n for a in sorted(_aliases(c)) for n in by_norm.get(a, [])]
        cand = list(dict.fromkeys(cand))
        if cand:
            claims[c] = cand
    out.ambiguous = {c: v for c, v in claims.items() if len(v) > 1}
    for c, v in claims.items():
        if len(v) == 1:
            out[c] = v[0]
    for n in {v for v in out.values()}: # one column claimed by two names: guess neither
        owners = [c for c, v in out.items() if v == n]
        if len(owners) > 1:
            for c in owners:
                out.ambiguous[c] = [n]
                del out[c]
    have = set(out.exact) | set(out)
    out.missing = [c for c in COLUMNS if COLUMN_INFO[c].group == "required" and c not in have]
    if not ({"ra", "dec"} <= have or {"L", "alpha", "beta"} <= have):
        out.missing += [c for c in ("ra", "dec") if c not in have]
    used = set(out.exact) | set(out.values())
    out.unused = [n for n in names if n not in used]
    unused_norm = {_norm(n): n for n in out.unused}
    out.close = {}
    for c in out.missing:
        for a in [_norm(c), *COLUMN_INFO[c].aliases]:
            if (m := difflib.get_close_matches(a, list(unused_norm), n=1, cutoff=0.6)):
                out.close[c] = unused_norm[m[0]]
                break
    return out


@dataclass
class Report:
    """What `check_inputs` found. `ok` is False if a run would stop or no star could be run; warnings need a look."""
    source: str
    n_rows: int = 0
    mapping: dict = field(default_factory=dict) # canonical -> the table column used
    n_run: int = 0
    rejected: dict = field(default_factory=dict) # reason -> number of stars
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def __str__(self):
        out = [f"DECIMA input check: {self.source} ({self.n_rows} rows)"]
        if self.mapping:
            out.append("Columns used:")
            for c in COLUMNS:
                if c in self.mapping:
                    out.append(f"  {c:<26} <- {self.mapping[c]!r}" if self.mapping[c] != c else f"  {c:<26}")
        for title, items, mark in (("Errors (a run would stop)", self.errors, "x"), ("Warnings (look at these)", self.warnings, "!"), ("Notes", self.notes, "-")):
            if items:
                out.append(f"{title}:")
                out += [f"  {mark} {m}" for m in items]
        if self.n_rows and self.ok:
            rej = ", ".join(f"{r}: {n}" for r, n in self.rejected.items())
            out.append(f"{self.n_run} of {self.n_rows} stars can be run; {self.n_rows - self.n_run} rejected" + (f" ({rej})" if rej else "") + ".")
        out.append("Result: " + ("OK" if self.ok else "FIX THE ERRORS ABOVE BEFORE RUNNING"))
        return "\n".join(out)


def check_inputs(source, columns: dict = None, config=None) -> Report:
    """Check a table, a columns file and settings before a run. Needs no models.

    `source`: path, astropy Table or pandas DataFrame. `columns`: the columns mapping (canonical -> yours) or None.
    `config`: a `Config`, the path of a settings file, or None for the defaults. Reports missing or misnamed columns, values that look
    like the wrong unit, and how many stars would be rejected (a dry run of `build_inputs`).
    """
    label = str(source) if isinstance(source, (str, os.PathLike)) else type(source).__name__
    rep = Report(label)
    try:
        t0 = _load(source)
    except Exception as e:
        rep.errors.append(f"could not read the table: {type(e).__name__}: {e}")
        return rep
    rep.n_rows = len(t0)
    columns = dict(columns or {})

    try:
        cfg = config if isinstance(config, Config) else Config.from_json(Path(config).read_text()) if config is not None else Config()
    except Exception as e:
        rep.errors.append(f"settings: {type(e).__name__}: {e}")
        cfg = Config()

    sug = suggest_columns(t0)
    for k in sorted(set(columns) - set(COLUMNS)):
        m = _close(k, COLUMNS)
        rep.errors.append(f"columns file: {k!r} is not one of DECIMA's column names" + (f" (did you mean {m!r}?)" if m else "")
                          + ". The file maps DECIMA's names to yours: {\"teff\": \"YOUR_TEFF\"}")
    valid = {}
    for k, v in columns.items():
        if k not in COLUMNS:
            continue
        if v in t0.colnames:
            valid[k] = v
        else:
            m = _close(v, t0.colnames)
            rep.errors.append(f"columns file: {k} -> {v!r}, but the table has no column {v!r}" + (f" (did you mean {m!r}?)" if m else ""))
    tab = read_table(t0, valid)
    rep.mapping = {c: valid.get(c, c) for c in tab.colnames}

    def hint(c):
        if c in sug:
            return f"; {sug[c]!r} looks right: add \"{c}\": \"{sug[c]}\" to the columns file (or run `decima columns`)"
        if c in sug.ambiguous:
            return f"; candidates {sug.ambiguous[c]}: pick one"
        if c in sug.close:
            return f"; is it {sug.close[c]!r}? If so add \"{c}\": \"{sug.close[c]}\" to the columns file"
        return ""
    for c in REQUIRED + ("Ks_error",):
        if c not in tab.colnames:
            rep.errors.append(f"required column {c} ({COLUMN_INFO[c].unit or 'no unit'}) is missing{hint(c)}")
    if not ({"ra", "dec"} <= set(tab.colnames) or {"L", "alpha", "beta"} <= set(tab.colnames)):
        miss = [c for c in ("ra", "dec") if c not in tab.colnames]
        rep.errors.append(f"no position: need ra and dec (deg), or per-star L, alpha, beta; missing {', '.join(miss)}" + "".join(hint(c) for c in miss))
    if rep.errors:
        if not columns and sug:
            rep.errors.insert(0, f"no columns file given, and {len(sug)} of the table's columns have other names than DECIMA's: run `decima columns {label} -o cols.json`, check it, and pass --columns cols.json")
        return rep

    def vals(k):
        a = _f(tab, k)
        return a[np.isfinite(a)]

    teff = vals("teff")
    if len(teff) and np.median(teff) < 100:
        rep.warnings.append(f"teff: median {np.median(teff):g}. DECIMA wants Kelvin (e.g. 5800), not log10(Teff) or kK")
    if (teff <= 0).any():
        rep.warnings.append(f"teff: {(teff <= 0).sum()} value(s) <= 0 (not rejected: they give NaN). Set them to NaN to reject those stars")
    plx = np.abs(vals("parallax"))
    if len(plx) and np.percentile(plx, 90) < 0.02:
        rep.warnings.append(f"parallax: 90% of |values| are below 0.02. DECIMA wants milliarcseconds; is this arcseconds?")
    for k in ("ra", "dec"):
        if k in tab.colnames and len(vals(k)) and (np.abs(vals(k)) > (360 if k == "ra" else 90)).any():
            rep.warnings.append(f"{k}: values outside the range of degrees")
    if {"ra", "dec"} <= set(tab.colnames) and len(vals("ra")) and np.abs(vals("ra")).max() <= 2 * np.pi and np.abs(vals("dec")).max() <= np.pi / 2:
        rep.warnings.append("ra, dec: all values are small enough to be radians. DECIMA wants degrees (ICRS)")
    for k in ("teff", "feh", "alpha_fe", "logg"):
        a = vals(k)
        if (a <= -90).any():
            rep.warnings.append(f"{k}: {(a <= -90).sum()} value(s) <= -90 look like a missing-value flag. Use NaN for missing values")
    for b in BANDS:
        a = vals(b)
        if (a > 30).any():
            rep.warnings.append(f"{b}: {(a > 30).sum()} value(s) > 30 mag; a flag such as 99 for 'not measured'? or a flux, not a magnitude? Use NaN for missing values")
    for k in ("teff", "feh", "alpha_fe", "logg", *BANDS):
        a = vals(f"{k}_error")
        if len(a) and (a <= 0).any():
            rep.warnings.append(f"{k}_error: {(a <= 0).sum()} value(s) <= 0")
    if "id" in tab.colnames:
        ids = np.asarray(tab["id"])
        if len(np.unique(ids)) < len(ids):
            rep.warnings.append(f"id: not unique ({len(ids) - len(np.unique(ids))} repeats); rows in the summary cannot be matched to your table by id")
    else:
        rep.notes.append("no id column: the summary's `id` is the 0-based input row (rejected stars are dropped from the summary, so rows shift)")
    for b in cfg.bands:
        if b not in tab.colnames:
            rep.notes.append(f"band {b} is in the settings but not in the table: it is skipped for every star")
    if cfg.pseudo_mag_inflate and "BP" not in tab.colnames:
        rep.notes.append("no BP: the pseudo-magnitude error is not inflated for extinction (pseudo_mag_inflate)")

    try:
        _, rejected = build_inputs(tab, cfg)
    except Exception as e:
        rep.errors.append(f"dry run failed: {type(e).__name__}: {e}")
        return rep
    rep.rejected = dict(Counter(map(str, rejected["reason"])))
    rep.n_run = rep.n_rows - len(rejected)
    if rep.n_rows and rep.n_run == 0:
        rep.errors.append("no star can be run (see the rejection reasons above)" if rep.rejected else "the table is empty")
        rep.notes.append("rejections: " + ", ".join(f"{r}: {n}" for r, n in rep.rejected.items()))
    return rep
