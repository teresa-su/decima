"""Crude per-star A_K from the mismatch between a spectroscopic Teff and the observed Gaia BP - 2MASS Ks colour.

The Casagrande+21 colour-Teff relation (MNRAS 507, 2684, Gaia EDR3 + 2MASS, Table A1, (BP-Ks)_0) gives
Teff = f(colour_0, log g, [Fe/H]). With the survey's own Teff, log g and [Fe/H] we solve f(X0) = Teff for the intrinsic
colour X0, so E(BP-Ks) = (BP - Ks)_obs - X0 and A_K = A_K_PER_E_BPKS * E(BP-Ks). It only needs Teff, log g, [Fe/H], Gaia BP
and 2MASS Ks, so it works for any spectroscopic survey, with no dust map and no catalog E(B-V).

It is meant for inflating the stage-1 pseudo absolute magnitude error (`inflate_pseudo_mag_err`), not for correcting
the magnitude: on the Xiang+24 catalog the star-to-star noise is sigma(A_K) ~ 0.006, which is small against the A_K of the
stars where it matters but not against the median (~0.017).
"""
import numpy as np

# Casagrande+21 Table A1 (Gaia EDR3), (BP-Ks)_0 row, a0..a14 of
#   Teff = a0 + a1 X + a2 X^2 + a3 X^3 + a4 X^5
#          + log g (a5 + a6 X + a7 X^2 + a8 X^3 + a9 X^5)
#          + [Fe/H] (a10 + a11 X + a12 X^2 + a13 X^3 + a14 log g X)
_A = (8266.0, -2124.5574, 355.5051, -23.1719, 0.0,
      209.9927, -161.4505, 50.5904, -6.3337, 0.0,
      -27.2653, -160.3595, 67.9016, -6.5232, 16.5137)

COLOUR_RANGE = (0.30, 4.20) # recommended range of (BP-Ks)_0 for dwarfs (Casagrande+21 Table 2); giants go to 5.30
A_K_PER_E_BPKS = 0.124 # A_K / E(BP-Ks) = R_Ks / (R_BP - R_Ks), taken as a constant (it varies by ~0.01 over BP-RP_0 = 0.5-1.5)
LOGG_FILL = 4.0 # log g where it is missing: the dwarf value of Casagrande+21 Sec. 4


def teff_from_colour(x, logg, feh):
    """Casagrande+21 Teff [K] for intrinsic (BP-Ks)_0 `x`, log g and [Fe/H] (broadcast against each other)."""
    a = _A
    x, logg, feh = np.broadcast_arrays(np.asarray(x, float), np.asarray(logg, float), np.asarray(feh, float))
    return (a[0] + a[1] * x + a[2] * x ** 2 + a[3] * x ** 3 + a[4] * x ** 5
            + logg * (a[5] + a[6] * x + a[7] * x ** 2 + a[8] * x ** 3 + a[9] * x ** 5)
            + feh * (a[10] + a[11] * x + a[12] * x ** 2 + a[13] * x ** 3 + a[14] * logg * x))


def intrinsic_colour(teff, logg, feh, n_iter=50, tol=1.0):
    """(BP-Ks)_0 at which `teff_from_colour` returns `teff`, by bisection on COLOUR_RANGE (Teff falls with colour there).

    NaN where an input is not finite or no colour in the range gives `teff` to within `tol` K (a star hotter than the relation at the blue end or cooler than it at the red end, or Teff outside the calibrated 4000-8000 K)."""
    teff, logg, feh = np.broadcast_arrays(np.asarray(teff, float), np.asarray(logg, float), np.asarray(feh, float))
    lo, hi = np.full(teff.shape, COLOUR_RANGE[0]), np.full(teff.shape, COLOUR_RANGE[1])
    with np.errstate(invalid="ignore"):
        for _ in range(n_iter):
            mid = 0.5 * (lo + hi)
            too_hot = teff_from_colour(mid, logg, feh) > teff # model hotter than the star: the intrinsic colour is redder
            lo, hi = np.where(too_hot, mid, lo), np.where(too_hot, hi, mid)
        x0 = 0.5 * (lo + hi)
        ok = np.isfinite(teff + logg + feh) & (np.abs(teff_from_colour(x0, logg, feh) - teff) < tol)
    return np.where(ok, x0, np.nan)


def estimate_a_k(bp, ks, teff, logg, feh, g=None, logg_fill=LOGG_FILL):
    """A_K [mag] and E(BP-Ks) per star from Gaia EDR3 BP, 2MASS Ks and the spectroscopic Teff [K], log g, [Fe/H].

    A_K is clipped at 0 (the colour excess is noisy, ~0.045 mag, so about 16% of the catalog comes out below zero).
    NaN where BP or Ks is missing, the star is outside the calibration (BP <= 5, or G <= 6 when `g` is given: Gaia saturation), or `intrinsic_colour` finds no solution. A missing log g takes `logg_fill`.
    Returns a dict with a_k, e_bpks (unclipped) and x0 (intrinsic colour), each (n,).
    """
    bp, ks, teff, feh = (np.asarray(v, float) for v in (bp, ks, teff, feh))
    logg = np.where(np.isfinite(logg), logg, logg_fill)
    x0 = intrinsic_colour(teff, logg, feh)
    e = (bp - ks) - x0
    bright = (bp <= 5) if g is None else (bp <= 5) | (np.asarray(g, float) <= 6)
    e = np.where(bright, np.nan, e)
    return {"a_k": np.where(np.isfinite(e), np.fmax(A_K_PER_E_BPKS * e, 0.0), np.nan), "e_bpks": e, "x0": x0}


def inflate_pseudo_mag_err(pseudo_mag, pseudo_mag_err, a_k):
    """Add the unmodelled extinction to the error of the pseudo absolute magnitude, in quadrature. This is only to add error to pseudo absolute magnitude for stage-1 inference.
    """
    extra = 0.2 * np.log(10) * pseudo_mag * np.where(np.isfinite(a_k), a_k, 0.0)
    return np.sqrt(pseudo_mag_err ** 2 + extra ** 2)
