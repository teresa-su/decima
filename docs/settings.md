# Settings

Every setting that can change a result lives in `decima.Config`.

**From Python:**

```python
cfg = decima.Config(bands=("BP", "RP", "J", "H", "Ks"), gaia_err_floor=0.03)
```

**From the shell:** write the defaults to a file, edit it, and pass it in.

```bash
decima default-config > cfg.json
decima run stars.fits --out run/ --config cfg.json
```

A run directory keeps its `config.json`. Re-running into the same directory with a different config is refused, so the outputs always match the settings stored next to them. The summary file also records the config, the DECIMA version and the model release in its header (FITS) or meta (ECSV).

## Sampling

| Setting | Default | What it does |
|---|---|---|
| `n_latent_x` | 800 | Latent (noise-free) observables drawn per star in stage 1. This is the Monte Carlo sample over the measurement noise, so if you change the sample size, change this first. |
| `n_theta` | 10 | θ draws per latent observable |
| `n_keep` | 400 | Posterior samples saved per star. The useful number depends on the effective sample size (`ess_frac` × `n_latent_x` × `n_theta`); beyond it, extra samples are repeats. |
| `percentiles` | (5, 16, 50, 84, 95) | Percentiles written to the summary table |
| `seed` | 22 | Random seed for every draw (latent observables, θ, distance and extinction). The same inputs and seed give the same results. A chunk of stars starting at row i is seeded `seed + i`. |

## Proposal and weights

| Setting | Default | What it does |
|---|---|---|
| `mixture_alpha` | 0.1 | Fraction of each latent observable's θ draws from q(θ \| s, M_pseudo,K); the rest come from q(θ \| s). `mixture_alpha × n_theta` must be a whole number. |
| `npe_ratio_ceiling` | 1.0 | Clip the NPE ratio q(θ \| s) / q(θ \| s, M_pseudo,K) at this per-star quantile (1.0 = no clip). With the mixture the ratio is already bounded by 1 / (1 − mixture_alpha). |
| `imf_prior` | `True` | Apply the IMF prior (Kroupa 2001). Off means a prior uniform in EEP. |
| `use_logg` | `False` | Add the spectroscopic log g likelihood in stage 2 |

## Photometry, parallax and extinction

| Setting | Default | What it does |
|---|---|---|
| `bands` | G, BP, RP, J, H, Ks | Photometric bands used in stage 2: any subset that includes Ks. A band missing for one star is skipped for that star. |
| `extinction` | `"F19"` | Extinction law: `"F19"`, `"F19_TEFF"`, `"SS2018"`, `"WC2019"`, `"none"`, or a law object. See [extinction.md](extinction.md). |
| `grey` | 0.0 | Grey extinction component added to every R_b of a built-in law |
| `pseudo_mag_inflate` | `True` | Widen the pseudo absolute magnitude error by a Casagrande+21 A_K estimate. See [pseudo-mag-inflate.md](pseudo-mag-inflate.md). |
| `parallax_zero_point` | `False` | Subtract the Lindegren+21 Gaia parallax zero-point (below) |
| `parallax_snr_min` | 3.0 | Parallaxes with parallax / error above this also enter the stage-2 distance–extinction fit directly. The exact parallax likelihood is used for every star. |

## Input errors and fills

| Setting | Default | What it does |
|---|---|---|
| `gaia_err_floor` | 0.02 mag | Floor on the G, BP, RP errors |
| `twomass_err_floor` | 0.03 mag | Floor on the J, H, Ks errors |
| `missing_err` | 0.5 mag | Error given to a magnitude that has no error. 0 uses the floor instead. |
| `alpha_fe_fill` | 0.0 | [α/Fe] used where the input has none … |
| `alpha_fe_err_fill` | 0.2 | … with this error |

Spectroscopic errors (T_eff, [Fe/H], [α/Fe]) are used exactly as given.

## Compute

| Setting | Default | What it does |
|---|---|---|
| `device` | `"cpu"` | `"cpu"` or `"cuda"` |
| `stars_per_batch` | 100 | Stars per stage-1 flow call: lower it if memory runs out, raise it on a GPU |

## The parallax zero-point

With `parallax_zero_point=True`, DECIMA subtracts the Lindegren+21 zero-point from each parallax before using it. This needs:
- the optional package: `pip install 'decima-stars[zeropoint]'`;
- the columns `G`, `nu_eff_used_in_astrometry`, `pseudocolour`, `ecl_lat` and `astrometric_params_solved`, with these Gaia archive names.

A star outside the recipe's validity keeps its parallax as given: G outside 6–21, a solution that is not 5- or 6-parameter, or a missing input. The summary column `zero_point_applied` records which stars were corrected.

## Not in the config: how to run, not what to compute

| Option | Python | Shell | Default |
|---|---|---|---|
| Worker processes | `run(..., workers=)` | `--workers` | 1 |
| Stars per checkpoint | `run(..., chunk=)`, `infer(..., chunk=)` | `--chunk` | 250 |
| Summary file format | `run(..., summary_format=)` | `--format` | `fits` |
| Column mapping | `columns={...}` | `--columns cols.json` | canonical names |

Results do not depend on the number of workers. They *do* depend on `chunk`, because chunks are seeded by their first row, so a resumed run must use the same chunk size.
