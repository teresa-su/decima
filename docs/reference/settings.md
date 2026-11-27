# Settings

Every setting that can change a result lives in `decima.Config`. In a run they come from the **settings file**, which is a different file from the [columns file](../how-to/config-files.ipynb).

=== "Python"

    ```python
    cfg = decima.Config(bands=("BP", "RP", "J", "H", "Ks"), gaia_err_floor=0.03)
    ```

=== "Shell"

    ```bash
    decima default-config > settings.json     # write the defaults, edit what you want to change
    decima run stars.fits --out run/ --config settings.json
    ```

A settings file needs only the settings you change; the rest keep their defaults. A misspelt name or an impossible value is reported before anything runs (`decima check` does the same without starting a run).

A run directory keeps its `config.json`. Running into the same directory with different settings is refused, so the outputs always match the settings stored next to them. The summary file also records the settings, the DECIMA version and the model release in its header (FITS) or meta (ECSV).

Defaults are shown as they appear in the settings file (JSON: `true`, not `True`).

<!-- BEGIN GENERATED: settings -->
<!-- written by docs/gen_reference.py: edit the code, not this table -->

### Sampling

| Setting | Default | What it does |
|---|---|---|
| `n_latent_x` | `800` | Latent (noise-free) observables drawn per star in stage 1: the Monte Carlo sample over the observational noise |
| `n_theta` | `10` | Theta draws per latent observable |
| `n_keep` | `400` | Posterior samples kept per star (of n_latent_x * n_theta) |
| `percentiles` | `[5, 16, 50, 84, 95]` | Percentiles written to the summary table |
| `seed` | `22` | Random seed for every draw (latent observables, theta, distance and extinction): the same inputs and seed give the same results |

### Proposal and weights

| Setting | Default | What it does |
|---|---|---|
| `mixture_alpha` | `0.1` | Defensive mixture: fraction of each latent observable's theta draws from q(theta \| x_i), the rest from q(theta \| s_i); mixture_alpha * n_theta must be a whole number |
| `npe_ratio_ceiling` | `1.0` | Clip the NPE ratio q(theta \| s) / q(theta \| x) at this per-star quantile (1.0 = no clip) |
| `imf_prior` | `true` | Apply the IMF prior (Kroupa 2001) |
| `use_logg` | `false` | Add the spectroscopic log g likelihood in stage 2 |

### Photometry, parallax and extinction

| Setting | Default | What it does |
|---|---|---|
| `bands` | `["G", "BP", "RP", "J", "H", "Ks"]` | Any subset of G, BP, RP, J, H, Ks that includes Ks |
| `extinction` | `"F19"` | A name from extinction.EXTINCTION_LAWS, or a law object |
| `grey` | `0.0` | Grey component added to every R_b of a built-in extinction law |
| `pseudo_mag_inflate` | `true` | Inflate the pseudo absolute magnitude error by a Casagrande+21 A_K estimate (decima.ak) |
| `parallax_zero_point` | `false` | Subtract the Lindegren+21 Gaia parallax zero-point (needs gaiadr3-zeropoint and the astrometry columns; see decima.io) |
| `parallax_snr_min` | `3.0` | Parallaxes with parallax / error above this also enter the stage-2 (distance, extinction) fit directly |

### Input errors and fills

| Setting | Default | What it does |
|---|---|---|
| `gaia_err_floor` | `0.02` | [mag] floor on G, BP, RP errors |
| `twomass_err_floor` | `0.03` | [mag] floor on J, H, Ks errors |
| `missing_err` | `0.5` | [mag] error of a magnitude given without one (0: use the floor) |
| `alpha_fe_fill` | `0.0` | [alpha/Fe] where the input has none |
| `alpha_fe_err_fill` | `0.2` | Error in [alpha/Fe] when the input has none |

### Compute

| Setting | Default | What it does |
|---|---|---|
| `device` | `"cpu"` | "cpu" or "cuda" |
| `stars_per_batch` | `100` | Stars per stage-1 flow call |

<!-- END GENERATED: settings -->

## Notes on some settings

- **`n_latent_x`, `n_theta`, `n_keep`.** `n_latent_x` is the Monte Carlo sample over the measurement noise, so change it first if you want more draws. The useful number of samples is the effective sample size, `ess_frac` × `n_latent_x` × `n_theta`; beyond it the `n_keep` samples are repeats.
- **`seed`.** The same inputs, settings, `chunk` and seed give the same results. Chunk `c` of `chunk` stars is seeded `seed + c * chunk`, where stars are counted after rejected rows are dropped. CPU and CUDA draws differ.
- **`mixture_alpha`.** `mixture_alpha × n_theta` must be a whole number (with `n_theta = 10`: 0, 0.1, 0.2, …). A bad value is refused with the allowed values.
- **`extinction`.** One of `"F19"`, `"F19_TEFF"`, `"SS2018"`, `"WC2019"`, `"none"`, or a law object (Python only). See [Extinction laws](../explanation/extinction.md).
- **`pseudo_mag_inflate`.** See [Pseudo-magnitude inflation](../explanation/pseudo-mag-inflate.md).
- **`parallax_snr_min`.** The exact parallax likelihood is used for every star; this only decides whether the parallax also enters the distance-extinction fit directly.
- **Errors.** Spectroscopic errors (T_eff, [Fe/H], [α/Fe]) are used exactly as given: add a systematic term yourself if your survey needs one.
- **`device`.** `"cpu"`, `"cuda"` or `"cuda:N"`. See [Large catalogues](../how-to/large-runs.md).

## The parallax zero-point

With `parallax_zero_point=True`, DECIMA subtracts the Lindegren+21 zero-point from each parallax before using it. This needs:

- the optional package: `pip install 'decima-stars[zeropoint]'`;
- the columns `G`, `nu_eff_used_in_astrometry`, `pseudocolour`, `ecl_lat` and `astrometric_params_solved`, with these Gaia archive names.

A star outside the recipe's validity keeps its parallax as given: G outside 6–21, a solution that is not 5- or 6-parameter, or a missing input. The summary column `zero_point_applied` records which stars were corrected.

## Not in the settings: how to run, not what to compute

| Option | Python | Shell | Default |
|---|---|---|---|
| Worker processes | `run(..., workers=)` | `--workers` | 1 |
| Stars per checkpoint | `run(..., chunk=)`, `infer(..., chunk=)` | `--chunk` | 250 |
| Summary file format | `run(..., summary_format=)` | `--format` | `fits` |
| Column mapping | `columns={...}` | `--columns cols.json` | DECIMA's own names |

Results do not depend on the number of workers. They *do* depend on `chunk`, because chunks are seeded by their first star, so a resumed run must use the same chunk size.
