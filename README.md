# DECIMA

Stellar ages, [Fe/H], [α/Fe], evolutionary stage, distances and extinctions, for stars with a spectroscopic T_eff and
[Fe/H], a Gaia parallax and Gaia/2MASS photometry.

DECIMA uses neural posterior estimators trained on MIST isochrones, in two stages:
1. The NPEs give fast posterior draws from the spectroscopy and a parallax-based pseudo-magnitude, marginalised over the
   noise on those inputs.
2. Importance sampling then folds in every photometric band, the exact parallax likelihood, a Galactic distance prior and
   the extinction.

It runs at about 5 s per star per CPU core, and in parallel over a big catalogue.
How it works: [docs/method.md](docs/method.md).

## Install

```bash
git clone <repo-url> decima && cd decima
uv sync                 # or: pip install -e .
```

Python ≥ 3.10.

The trained models (≈0.25 GB) are not downloadable yet. Point `DECIMA_DATA_DIR` at a directory holding `npe.pt`,
`npe_spec.pt`, `prior_spec.pt` and `grid.npz`. On linnunrata:

```bash
export DECIMA_DATA_DIR=/export/linnunrata/tsu/decima/models
```

## Quick start

```python
import decima

summary, samples, rejected = decima.infer("stars.fits", columns={"teff": "TEFF", "feh": "FE_H", ...})
summary["id", "age_p16", "age_p50", "age_p84", "ess_frac"]
```

For a big catalogue, use the chunked, checkpointed and resumable runner. Re-running the same command resumes it.

```bash
decima run catalogue.fits --out run/ --columns cols.json --workers 16
```

or, from Python:

```python
decima.run("catalogue.fits", out="run/", columns=cols, workers=16)
```

A worked example on five stars: [examples/quickstart.ipynb](examples/quickstart.ipynb).

## Input

Any table astropy reads (FITS, CSV, ECSV, VOTable, HDF5, Parquet), or an astropy `Table` / pandas `DataFrame`.
DECIMA looks for the canonical column names below. Map your own names with `columns={"teff": "TEFF", ...}` in Python, or
a JSON file of the same mapping with `--columns` on the shell.

| | Columns |
|---|---|
| **required** | `parallax`, `parallax_error` [mas]; `teff`, `teff_error` [K]; `feh`, `feh_error` [dex]; `Ks`, `Ks_error` [mag] |
| **position** | `ra`, `dec` [deg, ICRS] for the distance prior, unless per-star `L`, `alpha`, `beta` are given |
| optional | `id`; `alpha_fe`, `alpha_fe_error`; `logg`, `logg_error`; `G`, `BP`, `RP` (Gaia EDR3/DR3), `J`, `H` (2MASS), each with `_error` |

- **Parallax:** used as given. Apply the Gaia zero-point correction yourself first.
- **Missing values:** NaN or masked.
  - A missing magnitude means that band is skipped for that star.
  - A magnitude with no error gets a 0.5 mag error.
  - A missing [α/Fe] is filled with 0 ± 0.15.
- **Rejected stars:** stars without a required value, or with `parallax_error ≤ 0`, are not run. They are listed in
  `rejected` with the reason.
- **Spectroscopic errors** are used exactly as given. If your pipeline's errors are internal precisions, inflate them first.

## Settings

All settings that change results are in `decima.Config`. On the shell, `decima default-config > cfg.json` writes them out
to edit and pass back with `--config cfg.json`. Full details: [docs/settings.md](docs/settings.md).

| Setting | Default | |
|---|---|---|
| `n_latent_x`, `n_theta` | 800, 10 | Stage-1 draws per star: latent observables × θ per observable. Chosen by a convergence test. |
| `n_keep` | 400 | Posterior samples saved per star |
| `bands` | G BP RP J H Ks | Bands used in stage 2. Any subset that includes Ks. |
| `extinction` | `"F19"` | Extinction law. Colour-dependent by default. **[docs/extinction.md](docs/extinction.md)** |
| `ak_inflate` | `True` | Widen the parallax pseudo-magnitude error for reddened stars. **[docs/ak-inflate.md](docs/ak-inflate.md)** |
| `use_logg` | `False` | Also use the spectroscopic log g in stage 2 |
| `device` | `"cpu"` | or `"cuda"` |
| `stars_per_batch` | 100 | Stars per flow call (memory vs speed) |
| `seed` | 22 | Random seed |
| `advanced` | `Advanced()` | Method settings: defensive-mixture fraction, IMF prior, photometric error floors, … Their defaults are the validated ones. |

## Output

- **`summary`:** one row per star.
  - Percentiles (5, 16, 50, 84, 95) of `feh`, `alpha_fe`, `log_age`, `age` [Gyr], `eep`, `distance` [pc] and the
    extinction (`A0` for the F19 laws, `ebv` for fixed laws).
  - Quality diagnostics, of which **`ess_frac`** is the main one.
  - Per-band flags.
- **`samples`:** (n_stars, n_keep, 7) unweighted posterior draws, in the order `decima.quantity_names(cfg)`.
- **`rejected`:** input rows that could not be run, and why.

`decima run` writes `summary.fits`, `samples.h5` and `rejected.ecsv` into the run directory.

**What each column means and when to distrust a star: [docs/outputs.md](docs/outputs.md).** In short, an `ess_frac`
below ≈0.0125 (fewer than ≈100 effective draws) means the photometry and parallax disagree with the spectroscopy. Look
at those stars before using them.

## Validated domain and known limitations

DECIMA can be run on any star with the required inputs. It has been checked against real data on only one kind of star.

**Trained on** a uniform prior over the MIST v1.2 grid:

| Parameter | Range |
|---|---|
| [Fe/H] | −4 to +0.5 |
| [α/Fe] | −0.2 to +0.6 |
| age | 10⁵ yr to 20 Gyr (uniform in linear age) |
| EEP | ZAMS (202) to end of core-He burning (706) |

So the training covers the main sequence, turn-off, subgiant branch, red giant branch, red clump and horizontal branch.

**Validated on:**
- **Mock stars** from that prior (no extinction), against nested sampling. Posterior coverage is close to nominal:
  5.6–7.7% of true values fall outside the 95% interval, against 5% nominal.
- **Real stars:** the Xiang+24 LAMOST catalogue of turn-off and subgiant stars (≈250,000 single stars). These have
  T_eff ≈ 5000–6650 K, log g ≈ 3.35–4.3 and [Fe/H] ≈ −1.7 to +0.3. Against the catalogue ages, the residuals have mean
  +0.2 Gyr and RMS 1.05 Gyr.

**Not yet checked on real data:**
- red giants and the red clump
- cool K/M dwarfs and stars hotter than ≈6800 K
- very metal-poor stars
- highly reddened fields
- GALAH and APOGEE stars

**Known weaknesses:**
- **Red clump / horizontal branch:** these overlap with the red giant branch in the observables, giving occasional large
  outliers in EEP and age.
- **Age at the grid edges:** on mocks, ages below ≈10 Gyr come out up to 0.8 Gyr high, and ages above ≈14 Gyr come out
  2–3 Gyr low, pulled away from the 20 Gyr edge of the grid. Nested sampling shows the same bias, so it comes from the
  prior edges, not the NPE.
- **Binaries:** unresolved binaries are not modelled. They show up as low `ess_frac` or as ages that are too young.
- **[Fe/H] scale:** [Fe/H] is read on the MIST solar scale (Grevesse & Sauval 1998 solar mixture). A pipeline offset
  propagates into the age.

## GALAH and APOGEE

Column maps are in [examples/columns/](examples/columns). Each assumes the preparation below.

**APOGEE DR17** (`allStar-dr17-synspec_rev1.fits`):

```python
import json
import numpy as np
from astropy.table import Table
import decima

t = Table.read("allStar-dr17-synspec_rev1.fits", hdu=1)
t = t[(t["EXTRATARG"] == 0) & ((t["ASPCAPFLAG"] & 2**23) == 0)         # main sample, not STAR_BAD
      & (t["TEFF"] > 0) & (t["FE_H"] > -6)]
for b in ("J", "H", "K"):
    t[b][t[b] > 50] = np.nan                                            # 99 = not measured
t["ALPHA_FE"] = t["ALPHA_M"] + t["M_H"] - t["FE_H"]                     # APOGEE gives [α/M], not [α/Fe]
t["PARALLAX_ZPC"] = t["GAIAEDR3_PARALLAX"] + 0.017                      # crude global zero-point; better: Lindegren+21

cfg = decima.Config(advanced=decima.Advanced(missing_err=0))            # no Gaia mag errors: use the 0.02 mag floor
summary, samples, rejected = decima.infer(t, columns=json.load(open("examples/columns/apogee_dr17.json")), config=cfg)
```

**GALAH DR4:**
- **BP/RP:** join `galah_dr4_allstar_240705` with `galah_dr4_vac_wise_tmass_gaiadr3_240705` on `sobject_id`, for BP, RP
  and the Gaia magnitude errors.
- **Quality cuts:** `flag_sp == 0` and `snr_px_ccd3 > 30`.
- **[α/Fe]:** DR4 has no combined [α/Fe] column, so without one DR4 stars get the default 0 ± 0.15. You can build one
  from the individual elements.
- **Parallax:** the DR4 parallax may already be zero-point corrected. Check before correcting it again.

**GALAH DR3:**
- **Gaia data:** join `GALAH_DR3_main_allstar_v2` with `GALAH_DR3_VAC_GaiaEDR3` on `sobject_id`. The main table has
  only Gaia DR2 astrometry.
- **Parallax:** use the VAC's `parallax_corr` (Lindegren zero-point applied).
- **Quality cuts:** `flag_sp == 0`, `flag_fe_h == 0`, `flag_alpha_fe == 0` and `snr_c3_iraf > 30`.
- **Gaia magnitude errors:** there are none, so use `Advanced(missing_err=0)` as for APOGEE.

**For both surveys:**
- **Spectroscopic errors:** the quoted T_eff and [Fe/H] errors are internal precisions. Add a systematic term in
  quadrature.
- **Giants:** most of these stars are giants, which is outside the real-data validation above.

## Testing

<!-- TODO(Teresa): in your own words. What you'd like them to try, and what to send back. -->

When something looks wrong, send:
- the command or `decima.infer` call;
- `config.json` (or `cfg.to_json()`);
- the input rows and summary rows of the stars in question;
- `progress.log`, for `decima run`.

## Development

```bash
uv sync
uv run pytest                                            # fast tests (the model-dependent ones are skipped)
DECIMA_DATA_DIR=/path/to/models uv run pytest            # plus the regression against the reference outputs (~1 min)
```

## Licence

BSD-3-Clause.
