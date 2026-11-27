# Outputs, and how to tell a good result from a bad one

## What you get

**`decima.infer`** returns three things:
- `summary`: an astropy Table, one row per star that was run, in input order;
- `samples`: a numpy array of shape (n_stars, n_keep, 7);
- `rejected`: a Table of the input rows that could not be run.

**`decima.run`** / **`decima run`** writes a directory:

| File | Contents |
|---|---|
| `summary.<format>` | The summary table (default FITS) |
| `samples.h5` | `samples` (n_stars, n_keep, 7) float32, `id` (n_stars,), `quantity_names` (7,) |
| `rejected.ecsv` | Input rows that could not be run: `row`, `id` (if given), `reason` |
| `config.json`, `job.json` | The settings and chunk size of the run |
| `progress.log` | One line per finished or failed chunk, with timings |
| `inputs.pt`, `chunk_XXXXXX.npz` | Intermediate files, kept so the run can resume. Delete them once the run has finished. |

## The quantities

The seven columns of `samples`, in order (`decima.quantity_names(cfg)`):

| Name | Unit | |
|---|---|---|
| `feh_init` | dex | **initial** [Fe/H] of the model star. The present-day surface value is what you input. |
| `alpha_fe_init` | dex | initial [α/Fe] |
| `log_age` | log₁₀(yr) | |
| `eep` | — | MIST equivalent evolutionary point: 202 ZAMS, 454 TAMS, 605 RGB tip, 631 ZACHeB, 707 end of core-He burning |
| `age` | Gyr | 10^(log_age − 9) |
| `distance` | pc | |
| `A0` or `ebv` | mag | A0 for the colour-dependent laws (F19, F19_TEFF); E(B−V) for fixed laws. See [Extinction laws](../explanation/extinction.md). |

The samples are **unweighted** draws from the posterior. They were resampled from the `n_latent_x` × `n_theta`
importance-weighted draws, so some appear more than once. They are already in random order, so the first k of them are
a valid smaller sample.

## The summary table

`id` comes from the input `id` column, or is the 0-based input row. Then, for each quantity above and each of `Config.percentiles` (5, 16, 50, 84, 95), a column `<quantity>_p<P>`, such as `age_p50` or `distance_p16`. Then the diagnostics and flags:

<!-- BEGIN GENERATED: diagnostics -->
<!-- written by docs/gen_reference.py: edit the code, not this table -->

| Column | Meaning |
|---|---|
| `failed` | The star failed: no usable stage-1 draw, or an age posterior with no width on one side of the median. Do not use its results. |
| `uniform_fallback` | Every importance weight vanished, so the 'posterior' is just the stage-1 proposal. Treat as failed. |
| `ess_frac` | Effective sample size of the stage-2 importance weights as a fraction of the stage-1 draws. The main quality number. |
| `ess_norm_stage1` | Effective sample size of the stage-1 q(s) weights as a fraction of the draws. Low when the spectroscopy sits at the edge of the training prior. |
| `frac_valid_samples_pre` | Fraction of stage-1 draws usable in stage 2: on the grid, with enough bands, and a finite weight. |
| `frac_colour_clipped` | Fraction of draws whose intrinsic colour was outside the extinction law's calibration (colour-dependent laws only). |
| `n_negative_extinction` | Number of draws whose sampled extinction was negative or NaN, and which were dropped. |
| `par_valid` | The parallax was precise enough (`parallax_snr_min`) to enter the stage-2 distance-extinction fit directly. The exact parallax likelihood is used either way. |
| `pseudo_mag_inflated` | The pseudo absolute magnitude error was widened by an A_K estimate (`pseudo_mag_inflate`). |
| `zero_point_applied` | The Lindegren+21 parallax zero-point was subtracted from this star's parallax (`parallax_zero_point`). |
| `{band}_used` | The band had a magnitude and an error for this star, so it entered stage 2. |
| `{band}_err_missing` | The band had a magnitude but no error, so `missing_err` was used. |

<!-- END GENERATED: diagnostics -->

The table's header (FITS) or meta (ECSV) records the DECIMA version (`DECIMA`), the model release (`MODELS`) and the
full config (`CONFIG`, as JSON).

## When to distrust a star

**Failed stars** (`failed = True`):
- either no stage-1 draw was usable at all, and its percentiles and samples are NaN;
- or its age posterior has no width on one side of the median: the 16th or the 84th percentile equals the median.
  This happens when a few draws carry all the weight.

On the Xiang+24 catalogue, 0.1% of stars failed.

**Low effective sample size.** ESS = `ess_frac` × `n_latent_x` × `n_theta` is roughly the number of independent
posterior draws behind a star's result. A low ESS means the photometry and parallax disagree with what the spectroscopy
predicts, so look at those stars before using them. Typical causes:
- a binary;
- a bad magnitude;
- a T_eff or [Fe/H] that is wrong, or on a different scale from the models;
- high extinction.

## Resuming and re-running

Re-running the same `decima run` command into the same directory skips finished chunks and retries failed ones. A
different config or chunk size is refused. Use a new directory instead.
Results do not depend on the number of workers, and `decima.infer` with the same `chunk` gives the same draws as `decima.run`.
