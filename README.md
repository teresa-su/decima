# DECIMA

Decima is the Roman Fate who measures the thread of life. DECIMA is also Distance, Extinction and Chronometry via
Isochrone Modelling, Amortised.

DECIMA infers a star's age, distance and extinction, together with its initial [Fe/H], [α/Fe] and evolutionary stage
(parametrised by the equivalent evolutionary point, EEP). It needs spectroscopic T_eff and [Fe/H] (and [α/Fe], which is
otherwise assumed solar), a Gaia parallax, and Gaia and 2MASS photometry.

DECIMA works in two stages:
1. Neural posterior estimation of the distance-independent parameters from the spectroscopy and a parallax-based
   pseudo-magnitude. Measurement noise is handled at inference time.
2. Importance sampling that folds in every available photometric band, the exact parallax likelihood, the priors and,
   optionally, the spectroscopic log g.

It is fast: about 0.04 s per star per GPU core.
How it works: [docs/method.md](docs/method.md).

## Install

```bash
git clone <repo-url> decima && cd decima
uv sync                 # or: pip install -e .
```

Python ≥ 3.10.

The trained models (≈0.25 GB) are not downloadable yet. Point `DECIMA_DATA_DIR` at a directory holding `npe.pt`,
`npe_spec.pt`, `prior_spec.pt` and `grid.npz`.

On linnunrata (internal only for now):

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
| **position** | `ra`, `dec` [deg, ICRS] for the distance prior, unless per-star `L`, `alpha`, `beta` are given; stars with neither are rejected |
| optional | `id`; `alpha_fe`, `alpha_fe_error`; `logg`, `logg_error`; `G`, `BP`, `RP` (Gaia EDR3/DR3), `J`, `H` (2MASS), each with `_error` |
| zero-point | `nu_eff_used_in_astrometry`, `pseudocolour`, `ecl_lat`, `astrometric_params_solved` (Gaia archive names), only with `parallax_zero_point=True` |

- **Parallax:** used as given, or with `Config(parallax_zero_point=True)` DECIMA subtracts the Lindegren+21 zero-point
  itself ([details](docs/settings.md#the-parallax-zero-point)).
- **Missing values:** NaN or masked.
  - A missing magnitude means that band is skipped for that star.
  - A magnitude with no error gets a 0.5 mag error.
  - A missing [α/Fe] is filled with 0 ± 0.2 by default.
- **Rejected stars:** stars without a required value or a position, or with `parallax_error ≤ 0`, are not run. They are listed in
  `rejected` with the reason.

## Settings

All settings that change results are in `decima.Config`. On the shell, `decima default-config > cfg.json` writes them out
to edit and pass back with `--config cfg.json`. Full details: [docs/settings.md](docs/settings.md).

| Setting | Default | |
|---|---|---|
| `n_latent_x`, `n_theta` | 800, 10 | Stage-1 draws per star: latent observables (the Monte Carlo sample over the noise) × θ per observable |
| `n_keep` | 400 | Posterior samples saved per star |
| `seed` | 22 | Random seed for every draw: the same inputs and seed give the same results |
| `bands` | G BP RP J H Ks | Bands used in stage 2. Any subset that includes Ks. |
| `extinction` | `"F19"` | Extinction law. Colour-dependent by default. **[docs/extinction.md](docs/extinction.md)** |
| `pseudo_mag_inflate` | `True` | Widen the pseudo absolute magnitude error for reddened stars to aid stage-1 sampling. **[docs/pseudo-mag-inflate.md](docs/pseudo-mag-inflate.md)** |
| `parallax_zero_point` | `False` | Subtract the Lindegren+21 Gaia parallax zero-point |
| `use_logg` | `False` | Include the spectroscopic log g in stage 2 |
| `device` | `"cpu"` | or `"cuda"` |
| `stars_per_batch` | 100 | Stars per flow call (memory vs speed) |

Plus the method settings (`mixture_alpha`, `npe_ratio_ceiling`, `imf_prior`, `grey`, `parallax_snr_min`) and the input
error floors and fills: all in [docs/settings.md](docs/settings.md).

## Output

- **`summary`:** one row per star.
  - Percentiles (5, 16, 50, 84, 95) of `feh_init`, `alpha_fe_init` (the model's initial abundances), `log_age`,
    `age` [Gyr], `eep`, `distance` [pc] and the extinction (`A0` for the F19 laws, `ebv` for fixed laws).
  - Quality flags and diagnostics: **`failed`** and **`ess_frac`** are the main ones.
  - Per-band flags.
- **`samples`:** (n_stars, n_keep, 7) unweighted posterior draws, in the order `decima.quantity_names(cfg)`.
- **`rejected`:** input rows that could not be run, and why.

`decima run` writes `summary.fits`, `samples.h5` and `rejected.ecsv` into the run directory.

**Failed stars** have `failed = True`: either no stage-1 draw was usable (percentiles NaN), or the age posterior has no
width on one side of the median (its 16th or 84th percentile equals the median). Don't use their results. What every
column means, and other warning signs: [docs/outputs.md](docs/outputs.md).

## Validated domain and known limitations

DECIMA can be run on any star with the required inputs.

**Trained on** a uniform prior over the MIST v2.5 grid:

| Parameter | Range |
|---|---|
| [Fe/H] | −4 to +0.5 |
| [α/Fe] | −0.2 to +0.6 |
| age | 10⁵ yr to 20 Gyr (uniform in linear age) |
| EEP | ZAMS (202) to end of core-He burning (707) |

So the training covers the main sequence, turn-off, subgiant branch, red giant branch, red clump and horizontal branch.

**Validated on:**
- **Mock stars** from that prior (no extinction), against nested sampling. Posterior coverage is close to nominal:
  5.6–7.7% of true values fall outside the 95% interval, against 5% nominal.
- **Real stars:** the Xiang+24 LAMOST catalogue of turn-off and subgiant stars (≈250,000 single stars). These have
  T_eff ≈ 5000–6650 K, log g ≈ 3.35–4.3 and [Fe/H] ≈ −1.7 to +0.3. Against the catalogue ages, the residuals have mean
  +0.2 Gyr and RMS 1.05 Gyr.

**Known weaknesses:**
- **Binaries:** unresolved binaries are not modelled.
- **Failures:** on the Xiang+24 catalogue, 0.1% of stars fail (see Output above).

## Development

```bash
uv sync
uv run pytest # fast tests (the model-dependent ones are skipped)
DECIMA_DATA_DIR=/path/to/models uv run pytest # plus the regression against the reference outputs (~1 min)
```

## Licence

BSD-3-Clause.
