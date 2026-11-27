# Extinction laws

For every stage-1 draw, stage 2 fits a distance modulus μ and one extinction value E to the apparent magnitudes:

    m_b − M_b(θ) = μ + R_b E

The extinction law supplies R_b, the extinction in band b per unit E.

| Kind of law | E is | Output column |
|---|---|---|
| Fixed laws | E(B−V) | `ebv` |
| Colour-dependent laws | A0, the monochromatic extinction at 550 nm | `A0` |

Choose a law with `Config(extinction=...)`.

## Built-in laws

| Name | Kind | E | Source |
|---|---|---|---|
| `F19` (default) | colour-dependent, X = (BP−RP)₀ | A0 | Gaia EDR3 extinction coefficients for the Fitzpatrick+19 curve ("top of the HRD" fit) |
| `F19_TEFF` | colour-dependent, X = T_eff / 5040 K | A0 | the same, as a function of T_eff |
| `SS2018` | fixed | E(B−V) | Schlafly & Finkbeiner (2018)-type R_b |
| `WC2019` | fixed | E(B−V) | Wang & Chen (2019) |
| `none` | R_b = 0 | — | No extinction: stage 2 fits the distance alone, and gives no extinction estimate. For mock stars. |

## Colour-dependent laws

A fixed law uses one R_b per band for every star. A colour-dependent law gives each band a coefficient

    k_b = A_b / A0 = c₀ + c₁X + c₂X² + c₃X³ + c₄A + c₅A² + c₆A³ + c₇XA + c₈AX² + c₉XA²

where A = A0, and X is the *intrinsic* colour (or T_eff) of the draw being weighted, taken from the noise-free isochrone.
Since k_b depends on A0, the value being fitted, stage 2 fits once with k_b at A0 = 0, then refits once with k_b at
the fitted A0.

Two clamps keep the polynomial inside its calibration, so it is never extrapolated:
- X is clamped to the fit's range: (BP−RP)₀ ∈ [−0.06, 2.5] for F19, and T_eff ∈ [3500, 10000] K for F19_TEFF.
- A0 is clamped to [0, 20].

The summary column `frac_colour_clipped` is the fraction of a star's draws whose X was clamped. A value near 1 means the
star is outside the law's calibrated colours.

To convert, E(B−V) = A0 / 3.1.

## Grey component

`Config(grey=0.1)` adds a grey extinction component of 0.1 to every R_b of a *built-in* law.

## Your own law

**Constant R_b per band:**

```python
import decima

law = decima.FixedLaw({"G": 2.29, "BP": 3.05, "RP": 1.74, "J": 0.63, "H": 0.32, "Ks": 0.16})
cfg = decima.Config(extinction=law)
```

**A Fitzpatrick+19-form polynomial with your own coefficients:** ten per band, in the order
`decima.extinction.POLY_TERMS`, with X either `"BP-RP"` or `"teff"`:

```python notest
law = decima.PolynomialLaw(coeffs, x="BP-RP", x_range=(-0.06, 2.5), a0_max=20.0, r_v=3.1)
```

`FixedLaw` and `PolynomialLaw` are saved in `config.json`, so a run that uses them can be resumed and reproduced.

**Anything else:** subclass `decima.ExtinctionLaw`. Set `bands`, `colour_dependent`, `extinction_name` and `columns`
(the isochrone columns you need), and implement `__call__`. Its contract is in the docstring of
[`decima/extinction.py`](https://github.com/teresa-su/decima/blob/main/src/decima/extinction.py). Define `to_dict` as well if runs using the law should be reproducible
from `config.json`.

A law must have coefficients for every band in `Config.bands`.
