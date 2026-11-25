# DECIMA

Stellar ages, [Fe/H], [α/Fe], EEP, distances and extinctions from spectroscopic [Fe/H], [α/Fe] and T_eff, a Gaia
parallax and Gaia/2MASS photometry. DECIMA uses neural posterior estimation (NPE) trained on MIST isochrones, in two
stages:

1. **Stage 1** marginalises over the noise on the observables. For each star it draws `n_latent_x` latent (noise-free)
   observables x = ([Fe/H], [α/Fe], log T_eff, parallax·10^(0.2 Ks − 2)) around the measured ones, and `n_theta`
   isochrone parameters θ for each from the NPE. By default (`mixture_alpha = 0.1`) these come from a defensive
   mixture of q(θ | x) and the spectroscopy-only q(θ | s).
2. **Stage 2** importance-resamples those draws with the photometry and parallax. It fits distance and extinction
   per draw, applies the Bailer-Jones+21 distance prior and the IMF, and swaps the proposal for the
   spectroscopy-only NPE, so the parallax is not counted twice.

## Install

```bash
pip install decima-stars        # import decima
```

The trained models (~0.23 GB, mostly the isochrone grid) are downloaded on first use. To use a local copy instead,
point `DECIMA_DATA_DIR` at a directory holding `npe.pt`, `npe_spec.pt`, `prior_spec.pt` and `grid.npz`.

## Input

Any table astropy reads (FITS, CSV, ECSV, VOTable, HDF5, Parquet), or an astropy `Table` / pandas `DataFrame`.
Columns use these canonical names; map your own with `columns={"teff": "TEFF", ...}`.

| | columns |
|---|---|
| required | `parallax`, `parallax_error` [mas]; `teff`, `teff_error` [K]; `feh`, `feh_error`; `Ks`, `Ks_error` |
| position | `ra`, `dec` [deg, ICRS], for the Bailer-Jones+21 prior, unless per-star `L`, `alpha`, `beta` are given |
| optional | `id`; `alpha_fe`, `alpha_fe_error`; `logg`, `logg_error`; `G`, `BP`, `RP` (Gaia EDR3), `J`, `H` (2MASS), each with `_error` |

- **Parallax:** used as given, so apply the Gaia zero-point correction yourself first.
- **Missing values:** a missing magnitude means that band is skipped. A missing [α/Fe] is filled with 0 ± 0.15.
- **Rejected stars:** stars without a required value are not run, and are returned with the reason.

## Use

```python
import decima

cfg = decima.Config(
    n_latent_x=100,          # latent observables per star
    n_theta=60,              # θ draws per latent observable
    n_keep=100,              # posterior samples kept per star
    bands=("G", "BP", "RP", "J", "H", "Ks"),   # any subset that includes Ks
    extinction="F19",        # or "F19_TEFF", "SS2018", "WC2019", "none", or a law object (below)
    ak_inflate=False,        # inflate the pseudo-magnitude error by a Casagrande+21 A_K estimate
)

# a few thousand stars, in memory
summary, samples, rejected = decima.infer("stars.csv", columns={"teff": "TEFF"}, config=cfg)

# a big catalog: chunked, checkpointed, resumable (re-run the same call), in parallel workers
decima.run("catalog.fits", out="run/", config=cfg, workers=8)
```

The same from the shell:

```bash
decima default-config > cfg.json      # edit
decima run catalog.fits --out run/ --config cfg.json --columns cols.json --workers 8
```

Outputs:
- `summary`: one row per star with percentiles of `feh`, `alpha_fe`, `log_age`, `age` [Gyr], `eep`, `distance` [pc]
  and the fitted extinction (`A0` for the F19 laws, `ebv` for fixed-R laws). It also holds diagnostics (importance
  sampling ESS, valid-draw fractions) and per-band flags.
- `samples`: the (n_stars, n_keep, 7) posterior draws, in the order `decima.quantity_names(cfg)`.

`decima.run` writes `summary.fits` (or `--format ecsv`, `csv`, ...), `samples.h5` and `rejected.ecsv`.

Method settings (the mixture fraction, IMF prior, error floors, ...) are in `decima.Advanced` (`Config(advanced=...)`).
Their defaults are the validated ones.

## Extinction laws

```python
decima.FixedLaw({"G": 2.29, "BP": 3.05, "RP": 1.74, "J": 0.63, "H": 0.32, "Ks": 0.16})   # constant R_b = A_b / E(B-V)
decima.PolynomialLaw(coeffs, x="BP-RP", x_range=(-0.06, 2.5))   # Fitzpatrick+19 form: k_b(X, A0), 10 coefficients per band
class MyLaw(decima.ExtinctionLaw): ...                           # anything else (see decima.extinction)
```

Fixed and polynomial laws are saved in `config.json`. A custom subclass needs a `to_dict` method before a run that
uses it can be rebuilt from `config.json`.

## Development

```bash
uv sync
uv run pytest                                      # fast tests
DECIMA_DATA_DIR=/path/to/models uv run pytest      # plus the bit-for-bit regression against the reference outputs
```

## Licence

BSD-3-Clause.
