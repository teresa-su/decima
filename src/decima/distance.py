"""Bailer-Jones+21 generalised-gamma distance prior, with per-HEALPix parameters (`bj_prior_params`)."""
from functools import lru_cache
from pathlib import Path

import healpy as hp
import numpy as np
import torch
from torch.distributions import Gamma, TransformedDistribution
from torch.distributions.transforms import PowerTransform, AffineTransform
from torch.distributions.utils import broadcast_all

# GGD columns (healpix, GGDrlen, GGDalpha, GGDbeta) of Bailer-Jones+21 (EDR3) prior_summary.csv: 12288 level-5
# HEALPix, nested, *equatorial*. https://keeper.mpdl.mpg.de/d/0e9919daa8fa4e119ebe/
BJ21_PRIOR_TABLE = Path(__file__).parent / "data" / "bj21_ggd.csv"

# Fallback for stars without a sky position: the s^2 exp(-s/1000) EDSD prior (GGD with alpha=1, beta=2)
BJ_DEFAULT_PARAMS = (1000.0, 1.0, 2.0)

class GeneralizedGamma(TransformedDistribution):
    """
    Bailer-Jones+21 generalised gamma distribution (GGD) over distance r (pc):
        p(r | L, alpha, beta) = alpha / (Gamma((beta+1)/alpha) L^(beta+1)) * r^beta * exp(-(r/L)^alpha)
    i.e. r = L * y^(1/alpha) with y ~ Gamma((beta+1)/alpha, 1). Parameters broadcast, so
    shape (n_stars, 1) gives one prior per star.
    """
    arg_constraints = {}

    def __init__(self, alpha, beta, L, validate_args=None):
        self.alpha, self.beta, self.L = broadcast_all(alpha, beta, L)
        base = Gamma(concentration=(self.beta + 1) / self.alpha, rate=torch.ones_like(self.alpha))
        transforms = [PowerTransform(1 / self.alpha), AffineTransform(loc=0.0, scale=self.L)]  # y -> L y^(1/alpha)
        super().__init__(base, transforms, validate_args=validate_args)


@lru_cache(maxsize=1)
def _bj_ggd_table():
    t = np.loadtxt(BJ21_PRIOR_TABLE, delimiter=",", skiprows=1)
    t = t[np.argsort(t[:, 0])]
    assert len(t) == 12288 and (t[:, 0] == np.arange(12288)).all()
    return t[:, 1:]  # (12288, 3): L, alpha, beta


def bj_prior_params(ra, dec, default=BJ_DEFAULT_PARAMS):
    """
    Per-star GGD prior parameters (L [pc], alpha, beta) from the Bailer-Jones+21 per-HEALPix prior.

    Stars with non-finite ra or dec get `default`.

    Parameters
    ----------
    ra, dec : array-like
        ICRS right ascension and declination in degrees.

    Returns
    -------
    L, alpha, beta : torch.Tensor
        Each of shape (n_stars,); pass to `bailer_jones`.
    """
    ra, dec = np.atleast_1d(np.asarray(ra, float)), np.atleast_1d(np.asarray(dec, float))
    ok = np.isfinite(ra) & np.isfinite(dec)
    params = np.tile(np.asarray(default, float), (len(ra), 1))
    params[ok] = _bj_ggd_table()[hp.ang2pix(32, ra[ok], dec[ok], nest=True, lonlat=True)]
    L, alpha, beta = (torch.tensor(params[:, i], dtype=torch.float32) for i in range(3))
    return L, alpha, beta


def bailer_jones(L=1000, alpha=1.0, beta=2.0):
    """
    Bailer-Jones+21 generalised-gamma distance prior.

    The defaults (alpha=1, beta=2) are the exponentially-decreasing space-density prior
    s^2 exp(-s/L); per-HEALPix (L, alpha, beta) come from `bj_prior_params`.

    Parameters
    ----------
    L, alpha, beta : float or torch.Tensor
        GGD length scale (pc) and shape parameters. Tensors broadcast against the
        distances, e.g. shape (n_stars, 1) for per-star parameters.

    Returns
    -------
    GeneralizedGamma
        Distance prior over distance s in parsecs.
    """
    return GeneralizedGamma(torch.as_tensor(alpha, dtype=torch.float32),
                            torch.as_tensor(beta, dtype=torch.float32),
                            torch.as_tensor(L, dtype=torch.float32))
