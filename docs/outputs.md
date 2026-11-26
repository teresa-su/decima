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
| `feh` | dex | **initial** [Fe/H] of the isochrone (not the present surface value) |
| `alpha_fe` | dex | initial [α/Fe] |
| `log_age` | log₁₀(yr) | |
| `eep` | — | MIST equivalent evolutionary point: 202 ZAMS, 454 TAMS, 605 RGB tip, 631 ZACHeB, 707 end of core-He burning |
| `age` | Gyr | 10^(log_age − 9) |
| `distance` | pc | |
| `A0` or `ebv` | mag | A0 for the colour-dependent laws (F19, F19_TEFF); E(B−V) for fixed laws. See [extinction.md](extinction.md). |

The samples are **unweighted** draws from the posterior. They were resampled from 8000 importance-weighted draws, so
some appear more than once. They are already in random order, so the first k of them are a valid smaller sample.

## The summary table

| Column | Meaning |
|---|---|
| `id` | From the input `id` column, or the input row number |
| `<q>_p<P>` | Percentile P of quantity q (e.g. `age_p50`, `distance_p16`), for each quantity above and each `Config.percentiles` |
| `ess_frac` | Effective sample size of the stage-2 importance weights, as a fraction of the 8000 stage-1 draws. **The main quality number.** |
| `ess_frac_raw` | The same before truncation (equal to `ess_frac` unless `Advanced.truncate`) |
| `ess_norm_stage1` | ESS of the stage-1 q(s) weights / draws. Low when the spectroscopy sits at the edge of the training prior. |
| `frac_valid_samples_pre` | Fraction of stage-1 draws usable in stage 2: on the grid, with enough bands, and a finite weight |
| `frac_colour_clipped` | Fraction of draws whose intrinsic colour was outside the extinction law's calibration (colour-dependent laws only) |
| `n_negative_extinction` | Draws whose sampled extinction was negative or NaN, and which were dropped |
| `par_valid` | The parallax had S/N > 3 and entered the stage-2 fit directly. The exact parallax likelihood is used either way. |
| `ak_inflated` | The pseudo-magnitude error was widened by an A_K estimate ([ak-inflate.md](ak-inflate.md)) |
| `<band>_used` | The band had a magnitude and an error for this star |
| `<band>_err_missing` | The band had a magnitude but no error, so `Advanced.missing_err` (0.5 mag) was used |

## When to distrust a star

ESS = `ess_frac` × 8000 is roughly the number of independent posterior draws behind a star's result.

| Check | Rule of thumb | What it usually means |
|---|---|---|
| ESS < 100 (`ess_frac` < 0.0125) | treat with care | The photometry and parallax disagree with what the spectroscopy predicts. Typical causes: a binary, a bad magnitude, a T_eff or [Fe/H] off-scale, high extinction, or a star outside the trained range. On Xiang+24 turn-off stars this was about 7% of the sample. |
| ESS < 20 | unreliable | The percentiles rest on a handful of draws. Rerunning with more draws rarely helps: the problem is the mismatch, not the sample size. |
| `frac_valid_samples_pre` near 0 | broken | Almost no stage-1 draw was usable. If it is exactly 0, the star's samples are meaningless (see below). |
| `frac_colour_clipped` near 1 | caution | The star is redder or bluer than the extinction law's calibration. The law was evaluated at its edge. |
| `ess_norm_stage1` very low | caution | The spectroscopic values are at the edge of, or outside, the training prior. |
| `<band>_err_missing` | check | That band was effectively down-weighted. |

**A star whose weights all vanish is not flagged yet.**
- If it still has usable draws, it falls back to uniform weights. Its "posterior" is then just the stage-1 proposal, and
  `ess_frac` can look *high*.
- If it has none, every percentile collapses to a single value.

Check `frac_valid_samples_pre` and look for posteriors that are suspiciously narrow or that look identical to the
spectroscopy-only answer.

## Where the method is known to be weak

See [Validated domain and known limitations](../README.md#validated-domain-and-known-limitations) in the README.
In short:
- red clump and horizontal-branch stars;
- ages at the edges of the grid;
- stars unlike the turn-off/subgiant validation sample;
- unresolved binaries.

## Resuming and re-running

Re-running the same `decima run` command into the same directory skips finished chunks and retries failed ones. A
different config or chunk size is refused. Use a new directory instead.
Results do not depend on the number of workers, and `decima.infer` with the same `chunk` gives the same draws as `decima.run`.
