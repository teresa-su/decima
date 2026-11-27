# Decide which results to trust

DECIMA returns a posterior for every star it can run, including stars where something is wrong. Four things to look at before using a result.

## 1. Drop failed stars

`failed = True` means the star's result is not usable: either no stage-1 draw was usable (its percentiles are NaN), or its age posterior has no width on one side of the median, because a few draws carry all the weight. `uniform_fallback = True` means every weight vanished and the "posterior" is just the proposal: treat it as failed too.

<!-- notest -->
```python
good = ~summary["failed"] & ~summary["uniform_fallback"]
```

On the Xiang+24 catalogue (≈250,000 stars) 0.1% of stars failed.

## 2. Look at the effective sample size

`ess_frac` is the effective sample size of the final importance weights as a fraction of the `n_latent_x × n_theta` draws. The effective sample size is roughly the number of independent posterior draws behind the percentiles:

```{.python notest}
cfg = decima.Config()
summary["ess"] = summary["ess_frac"] * cfg.n_latent_x * cfg.n_theta
```

A low value means the photometry and parallax disagree with what the spectroscopy predicts. There is no calibrated cut-off. In the tests behind [`pseudo_mag_inflate`](../explanation/pseudo-mag-inflate.md) `ess_frac < 0.01` was the level counted as low, so a sensible first step is to look at those stars one by one rather than cutting blindly. Typical causes:

- an unresolved binary (not modelled by DECIMA);
- a bad magnitude, or a bad parallax;
- a T_eff or [Fe/H] that is wrong or on a different scale from the models;
- high extinction.

The [tutorial](../tutorial.ipynb) shows one such star.

## 3. Check what went into each star

| Columns | Look for |
|---|---|
| `G_used`, `BP_used`, … | Bands that were skipped for the star (no magnitude, or no error). Fewer bands means a weaker constraint on distance and extinction. |
| `G_err_missing`, … | The magnitude had no error, so `missing_err` (0.5 mag by default) was used. |
| `par_valid` | The parallax was precise enough to enter the distance-extinction fit directly. |
| `pseudo_mag_inflated` | The pseudo-magnitude error was widened for possible extinction. |
| `zero_point_applied` | The parallax zero-point was subtracted. |
| `ess_norm_stage1` | Low when the spectroscopy sits at the edge of the training prior. |

## 4. Check the star is inside what DECIMA was validated on

DECIMA runs on any star from the main sequence to the end of the horizontal branch, but it was tested on a limited domain:

| | |
|---|---|
| Trained on | [Fe/H] −4 to +0.5; [α/Fe] −0.2 to +0.6; age up to 20 Gyr; from the ZAMS to the end of core-He burning (main sequence to horizontal branch) |
| Tested on real stars | Xiang+24 turn-off and subgiant stars (T_eff ≈ 5000–6650 K, log g ≈ 3.35–4.3, [Fe/H] ≈ −1.7 to +0.3): residuals against catalogue ages have mean +0.2 Gyr and RMS 1.05 Gyr |
| Not modelled | Unresolved binaries. These will have underestimated ages. |
