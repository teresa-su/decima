"""Extinction laws for stage 2: R_b = A_b / E, with E the extinction stage 2 fits (A0 or E(B-V), depending on the law).

Pick a built-in law by name, or build one:

    get_law("F19")  # default: Fitzpatrick+19 EDR3, colour-dependent, E = A0
    FixedLaw({"G": 2.29, "BP": 3.05, "RP": 1.74, "J": 0.63, "H": 0.32, "Ks": 0.16})  # constant R_b, E = E(B-V)
    PolynomialLaw(coeffs, x="BP-RP", x_range=(-0.06, 2.5)) # Fitzpatrick+19-form polynomial
    class MyLaw(ExtinctionLaw): ...   # anything else

Built-in laws:
    SS2018, WC2019  fixed R_b per band (Schlafly & Finkbeiner 2018; Wang & Chen 2019); E is E(B-V).
    F19             Fitzpatrick+19 EDR3 ("HRDTop" fit): k_b = A_b / A0 is a polynomial in the draw's intrinsic
                    (BP-RP)_0 and in A0. E is A0 (at 550 nm); `to_ebv` divides by R_V = 3.1.
    F19_TEFF        the same law with Teff / 5040 K of the draw in place of (BP-RP)_0.
    none            no extinction: R_b = 0, so stage 2 fits the distance modulus alone (E = 0): a distance but no extinction estimate, e.g. for mock stars.

A law is called as `law(mag_fields, iso_cols, (n_stars, n_draws), device, A0=None)` and returns (R_b, colour_clipped):
R_b is (n_bands, 1, 1) for a fixed law and (n_bands, n_stars, n_draws) for a colour-dependent one; colour_clipped
(n_stars, n_draws) flags draws whose colour is outside the law's calibrated range.
"""
import csv
from pathlib import Path

import torch

BANDS = ("G", "BP", "RP", "J", "H", "Ks")

# R_b = A_b / E(B-V)
SS2018 = {"G": 2.294, "BP": 3.046, "RP": 1.737, "J": 0.632, "H": 0.318, "Ks": 0.158}
WC2019 = {"G": 1.890, "BP": 2.429, "RP": 1.429, "J": 0.582, "H": 0.313, "Ks": 0.186}

F19_BPRP_RANGE = (-0.06, 2.5) # validity range of the (BP-RP)_0 fit
F19_TEFF_NORM = 5040.0
F19_TEFF_RANGE = (3500.0 / F19_TEFF_NORM, 10000.0 / F19_TEFF_NORM) # validity range of the TeffNorm fit
POLY_TERMS = ("Intercept", "X", "X2", "X3", "A", "A2", "A3", "XA", "AX2", "XA2")
_F19_KNAME = {"G": "kG", "BP": "kBP", "RP": "kRP", "J": "kJ", "H": "kH", "Ks": "kK"}

# coefficients per independent variable ("BPRP", "TeffNorm"), then per band; the file has other fits too
_F19_COEFFS = {}
with open(Path(__file__).parent / "data" / "Fitz19_EDR3_HRDTop.csv") as _f:
    for row in csv.DictReader(_f):
        _F19_COEFFS.setdefault(row["Xname"], {})[row["Kname"]] = [float(row[t]) for t in POLY_TERMS]
F19_BPRP = {b: _F19_COEFFS["BPRP"][k] for b, k in _F19_KNAME.items()}
F19_TEFF = {b: _F19_COEFFS["TeffNorm"][k] for b, k in _F19_KNAME.items()}

EXTINCTION_LAWS = ("SS2018", "WC2019", "F19", "F19_TEFF", "none")


class ExtinctionLaw:
    """Base class. A subclass sets `bands` (band -> anything; the bands it has coefficients for) and implements
    `__call__` (see the module docstring), and `to_dict` if runs using it should be reproducible from config.json."""
    colour_dependent = False # True: R_b depends on the fitted E, so stage 2 refits once at the fitted value
    columns = () # isochrone columns read by __call__ besides the fitted bands
    bands = {}
    extinction_name = "ebv" # what E is; names the output column

    def __init__(self, grey: float = 0.0):
        self.grey = grey # added to every R_b

    def check(self, mag_fields):
        missing = [b for b in mag_fields if b not in self.bands]
        if missing:
            raise ValueError(f"No R_b extinction coefficient for bands: {missing}")

    def to_ebv(self, e):
        """The fitted E as E(B-V)."""
        return e

    def to_dict(self) -> dict:
        raise NotImplementedError(f"{type(self).__name__} cannot be saved to config.json; define to_dict()")


class FixedLaw(ExtinctionLaw):
    """Constant R_b = A_b / E(B-V) per band, e.g. FixedLaw({"G": 2.294, ..., "Ks": 0.158})."""

    def __init__(self, R: dict, grey: float = 0.0):
        super().__init__(grey)
        self.R = self.bands = dict(R)

    def __call__(self, mag_fields, iso_cols, shape, device, A0=None):
        R_b = torch.tensor([self.R[b] for b in mag_fields], dtype=torch.float32, device=device)
        return R_b.view(-1, 1, 1) + self.grey, torch.zeros(shape, dtype=torch.bool, device=device)

    def to_dict(self):
        return {"type": "fixed", "R": self.R, "grey": self.grey}


class PolynomialLaw(ExtinctionLaw):
    """Fitzpatrick+19-form law: k_b = A_b / A0 = sum_t c_bt basis_t, with basis
    (1, X, X^2, X^3, A, A^2, A^3, XA, AX^2, XA^2) (`POLY_TERMS`), A = A0 and X the draw's intrinsic colour.

    coeffs : {band: 10 coefficients}
    x : "BP-RP" (X = (BP-RP)_0 of the isochrone) or "teff" (X = Teff / `teff_norm`)
    x_range : X is clamped to this range (the fit's validity), so the polynomial is never extrapolated
    a0_max : A0 is clamped to [0, a0_max]
    r_v : E(B-V) = A0 / r_v (`to_ebv`)
    """
    colour_dependent = True
    extinction_name = "A0"

    def __init__(self, coeffs: dict, x: str = "BP-RP", x_range=F19_BPRP_RANGE, a0_max: float = 20.0,
                 r_v: float = 3.1, teff_norm: float = F19_TEFF_NORM, grey: float = 0.0):
        super().__init__(grey)
        if x not in ("BP-RP", "teff"):
            raise ValueError(f"x must be 'BP-RP' or 'teff', got {x!r}")
        if any(len(c) != len(POLY_TERMS) for c in coeffs.values()):
            raise ValueError(f"each band needs {len(POLY_TERMS)} coefficients {POLY_TERMS}")
        self.coeffs = self.bands = {b: [float(v) for v in c] for b, c in coeffs.items()}
        self.x, self.x_range, self.a0_max, self.r_v, self.teff_norm = x, tuple(x_range), a0_max, r_v, teff_norm
        # a draw's intrinsic colour comes from the (extinction-free) isochrone, even if BP / RP are not fitted
        self.columns = ("log_Teff",) if x == "teff" else ("BP", "RP")

    def k(self, mag_fields, X, A0):
        """k_b = A_b / A0 for each band, shape (n_bands, *X.shape); X and A0 clamped as described above."""
        X = X.clamp(*self.x_range)
        A = A0.clamp(0.0, self.a0_max)
        basis = torch.stack([torch.ones_like(X), X, X**2, X**3, A, A**2, A**3, X * A, A * X**2, X * A**2])
        a = torch.tensor([self.coeffs[b] for b in mag_fields], dtype=X.dtype, device=X.device) # (n_bands, 10)
        return torch.einsum("bt,t...->b...", a, basis)

    def __call__(self, mag_fields, iso_cols, shape, device, A0=None):
        """R_b at extinction A0 (n_stars, n_draws); A0 = 0 if None."""
        if self.x == "teff":
            x = (10 ** iso_cols["log_Teff"] / self.teff_norm).view(shape).float()
        else:
            x = (iso_cols["BP"] - iso_cols["RP"]).view(shape).float()
        clipped = (x < self.x_range[0]) | (x > self.x_range[1]) # off-grid draws (NaN) count as not clipped
        A0 = torch.zeros_like(x) if A0 is None else A0
        x = x.to(A0.dtype) # a float64 fit evaluates the polynomial in float64
        return self.k(mag_fields, x, A0) + self.grey, clipped

    def to_ebv(self, e):
        return e / self.r_v

    def to_dict(self):
        return {"type": "polynomial", "coeffs": self.coeffs, "x": self.x, "x_range": list(self.x_range),
                "a0_max": self.a0_max, "r_v": self.r_v, "teff_norm": self.teff_norm, "grey": self.grey}


def get_law(name: str, grey: float = 0.0) -> ExtinctionLaw:
    """The built-in law called `name` (one of EXTINCTION_LAWS), with `grey` added to every R_b."""
    laws = {
        "SS2018": lambda: FixedLaw(SS2018, grey),
        "WC2019": lambda: FixedLaw(WC2019, grey),
        "F19": lambda: PolynomialLaw(F19_BPRP, x="BP-RP", x_range=F19_BPRP_RANGE, grey=grey),
        "F19_TEFF": lambda: PolynomialLaw(F19_TEFF, x="teff", x_range=F19_TEFF_RANGE, grey=grey),
        "none": lambda: FixedLaw({b: 0.0 for b in BANDS}),
    }
    if name not in laws:
        raise ValueError(f"unknown extinction law {name!r}; choose from {EXTINCTION_LAWS}")
    if name == "none" and grey != 0.0:
        raise ValueError("extinction law 'none' takes no grey offset (it would make every R_b non-zero)")
    return laws[name]()


def from_dict(d: dict) -> ExtinctionLaw:
    """Inverse of `FixedLaw.to_dict` / `PolynomialLaw.to_dict`."""
    d = dict(d)
    kind = d.pop("type")
    if kind == "fixed":
        return FixedLaw(**d)
    if kind == "polynomial":
        return PolynomialLaw(**d)
    raise ValueError(f"unknown extinction law type {kind!r}")
