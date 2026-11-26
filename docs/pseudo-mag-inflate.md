# `pseudo_mag_inflate`: widening the pseudo absolute magnitude error for reddened stars

**On by default.** `Config(pseudo_mag_inflate=False)` turns it off.

## The problem

Stage 1 conditions the main NPE on the pseudo absolute magnitude

    M_pseudo,K = ϖ · 10^(0.2 Ks − 2) = 10^(−0.2 (M_Ks − A_Ks))

The NPE was trained without extinction, so it reads M_pseudo,K as if A_Ks = 0. For a reddened star, M_pseudo,K is then
10^(0.2 A_Ks) times too large, and the star looks more luminous than it is.

Stage 2 fits the extinction and would correct this, but only by reweighting the stage-1 draws. If stage 1's proposal
is centred on the wrong luminosity *and* is narrow, few draws land where stage 2 wants them. Stage 2 then rests on a
handful of draws: low `ess_frac`, and noisy, sometimes biased results.

## What `pseudo_mag_inflate` does

1. **Estimate A_K cheaply from data already in the input.**
   - The Casagrande+21 colour–T_eff relation for Gaia EDR3 BP and 2MASS Ks (their Table A1, (BP−Ks)₀) gives
     T_eff = f((BP−Ks)₀, log g, [Fe/H]).
   - Inverting it at the star's spectroscopic T_eff, log g and [Fe/H] gives the intrinsic colour (BP−Ks)₀, so
     E(BP−Ks) = (BP−Ks)_obs − (BP−Ks)₀.
   - Then A_K = 0.124 · E(BP−Ks), clipped at 0.
   - This needs no dust map and no external E(B−V).
2. **Widen, don't correct.** The M_pseudo,K *error* is increased in quadrature by the fraction of M_pseudo,K the extinction could
   account for:

       σ² → σ² + (0.2 ln 10 · M_pseudo,K · A_K)²

   The value of M_pseudo,K is left alone. Stage 2 still does the actual extinction fit, with all bands. The inflation only
   makes the stage-1 proposal wide enough for stage 2 to work with.

The estimate is noisy: σ(A_K) ≈ 0.006 mag from star to star on Xiang+24. That is small next to the A_K of the stars
where the inflation matters, but not next to the median (≈0.017). This noise is why the estimate is used only to widen
the error, never to correct M_pseudo,K.

## When it applies

A star's error is inflated (`pseudo_mag_inflated = True` in the summary) only when all of these hold:
- BP and Ks are both present.
- BP > 5 and G > 6 (brighter stars are saturated in Gaia).
- The relation has a solution with (BP−Ks)₀ in [0.30, 4.20] that reproduces the star's T_eff to within 1 K. In practice
  this means 4000 K ≲ T_eff ≲ 8000 K, the calibrated range.

Otherwise the error is left as it was, and the star runs normally.

## Caveats

- **Giants and log g.** The Casagrande+21 relation covers giants as well as dwarfs: log g is one of its terms, so a giant
  with a measured log g is handled correctly. Only a *missing* log g is filled with the dwarf/subgiant value 4.0, so a
  giant without log g gets the wrong intrinsic colour, and so the wrong A_K. Pipelines that give log g (GALAH, APOGEE)
  are not affected.
- **T_eff scale.** A_K inherits any offset between your pipeline's T_eff scale and Casagrande+21's (an IRFM scale): about
  0.003 mag in A_K per 30 K. A pipeline whose T_eff is 100 K too cool gives A_K about 0.01 mag too high for every star. That only
  slightly over-widens the error, but turn the option off if you know your scale is far off.
- **Only widens.** If A_K is underestimated the inflation is too small, and the star behaves as with `pseudo_mag_inflate=False`.

## Evidence

An A/B test on 5000 single Xiang+24 stars, with and without the inflation:

| Stars | Effect of the inflation |
|---|---|
| A_K < 0.08 (84% of stars) | No change |
| A_K ≥ 0.08 | Ages 1% younger, distances +0.5%, posterior widths 7–12% larger |
| All stars | Stars with stage-2 `ess_frac` < 1% fell from **5.1% to 1.9%** |

Agreement with the catalogue ages was essentially unchanged.

So the gain is robustness: fewer stars where stage 2 has too few useful draws. It is not a shift in typical results.

This test used an earlier proposal: q(θ | s, M_pseudo,K) alone, without the defensive mixture. The mixture is itself
more robust to this kind of mismatch, so the gain under the current default may be smaller.

## Reference

Casagrande, L. et al. 2021, MNRAS 507, 2684: *The GALAH survey: effective temperature calibration from the InfraRed Flux
Method in the Gaia system.*
