# Settings

Every setting that can change a result lives in `decima.Config`. The method settings sit in its `advanced` field
(`decima.Advanced`).

**From Python:**

```python
cfg = decima.Config(bands=("BP", "RP", "J", "H", "Ks"), advanced=decima.Advanced(gaia_err_floor=0.03))
```

**From the shell:** write the defaults to a file, edit it, and pass it in.

```bash
decima default-config > cfg.json
decima run stars.fits --out run/ --config cfg.json
```

A run directory keeps its `config.json`. Re-running into the same directory with a different config is refused, so the
outputs always match the settings stored next to them.

## Config

| Setting | Default | What it does | Change it when |
|---|---|---|---|
| `n_latent_x` | 800 | Latent (noise-free) observables drawn per star in stage 1 | Rarely. 800 × 10 was chosen by a convergence test on 1000 Xiang+24 stars: per-star medians moved by ≈0.06 posterior half-widths against 1000 × 10, the same as re-running with another seed. Fewer is faster but noisier per star. |
| `n_theta` | 10 | θ draws per latent observable | Rarely; see `n_latent_x`. Keep `mixture_alpha × n_theta` a whole number (below). |
| `n_keep` | 400 | Posterior samples saved per star | You need more or fewer samples per star. Stage-2 ESS is typically ≈ n_latent_x / 2 ≈ 400, so more samples mostly add repeats. `samples.h5` is ≈ 11 kB per star at 400. |
| `percentiles` | (5, 16, 50, 84, 95) | Percentiles written to the summary table | You want other quantiles. |
| `bands` | G, BP, RP, J, H, Ks | Photometric bands used in stage 2. Any subset that includes Ks (Ks is in the stage-1 pseudo-magnitude). | A band is unreliable for your sample, e.g. drop BP/RP for faint or crowded stars. A band missing for a single star is skipped automatically. |
| `extinction` | `"F19"` | Extinction law: `"F19"`, `"F19_TEFF"`, `"SS2018"`, `"WC2019"`, `"none"`, or a law object | See [extinction.md](extinction.md). |
| `ak_inflate` | `True` | Widen the stage-1 pseudo-magnitude error by a Casagrande+21 A_K estimate | Turn it off for mock stars with no extinction, or if your T_eff scale is far from Casagrande+21's. See [ak-inflate.md](ak-inflate.md). |
| `use_logg` | `False` | Add a spectroscopic log g likelihood in stage 2 | You trust your pipeline's log g and its errors. log g is *not* used by default: the NPEs were trained without it, and parallax plus photometry usually constrain it better. |
| `device` | `"cpu"` | `"cpu"` or `"cuda"` | You have a GPU. Draws on a GPU differ from CPU draws (another random stream). |
| `stars_per_batch` | 100 | Stars per stage-1 flow call | Lower it if you run out of memory; raise it (≈100–1000) on a GPU. With 1 each star is seeded on its own, so its draws don't depend on its neighbours, at ≈2.5× the CPU time. |
| `seed` | 22 | Base random seed. A chunk of stars starting at row i is seeded `seed + i`. | You want an independent re-run. |

## Advanced

These are the method's own settings. Their defaults are the validated ones; change them to study the method, not to
tune results.

| Setting | Default | What it does |
|---|---|---|
| `mixture_alpha` | 0.1 | Fraction of each latent observable's θ draws taken from q(θ \| x); the rest come from q(θ \| s). 1 = q(θ \| x) only, 0 = q(θ \| s) only. The draws are split as `round(mixture_alpha × n_theta)`, so with n_theta = 10 the possible values are 0, 0.1, … 1. |
| `npe_ratio_ceiling` | 1.0 | Clip the NPE ratio q(θ \| s) / q(θ \| x) at this per-star quantile (1.0 = no clip). With the mixture the ratio is already bounded by 1 / (1 − mixture_alpha). |
| `imf_prior` | `True` | Include the IMF in the stage-2 prior (the grid's `ln_dm_dEEP_prior`). Off means a prior uniform in EEP. |
| `grey` | 0.0 | Added to every R_b of a *built-in* law, a grey extinction term for sensitivity tests. |
| `truncate` | `False` | Truncated importance sampling (Ionides 2008) of the stage-2 weights. |
| `gaia_err_floor` | 0.02 mag | Floor on the G, BP, RP errors (Gaia's quoted errors ignore calibration systematics). |
| `twomass_err_floor` | 0.03 mag | Floor on the J, H, Ks errors. |
| `missing_err` | 0.5 mag | Error given to a magnitude that has no error. 0 uses the floor instead. |
| `alpha_fe_fill` | 0.0 | [α/Fe] used where the input has none … |
| `alpha_fe_err_fill` | 0.15 | … with this error. |

There are no floors on the *spectroscopic* errors: T_eff, [Fe/H] and [α/Fe] errors are used exactly as given. Several
pipelines quote internal precisions that are too small to be realistic accuracies. Inflate them yourself before running
(see the GALAH/APOGEE notes in the [README](../README.md#galah-and-apogee)).

## Not in the config: how to run, not what to compute

Options that change speed or file layout, not results:

| Option | Python | Shell | Default |
|---|---|---|---|
| Worker processes | `run(..., workers=)` | `--workers` | 1 |
| Stars per checkpoint | `run(..., chunk=)`, `infer(..., chunk=)` | `--chunk` | 250 |
| Summary file format | `run(..., summary_format=)` | `--format` | `fits` |
| Column mapping | `columns={...}` | `--columns cols.json` | canonical names |

Results do not depend on the number of workers. They *do* depend on `chunk`, because chunks are seeded by their first
row. A resumed run must use the same chunk size.
