"""The trained flows and the isochrone grid: where they live and how they are rebuilt.

Files (model release v1):
    npe.pt          q(theta | x), x = ([Fe/H], [alpha/Fe], log Teff, M_pseudo,K)   sbi NSF posterior
    npe_spec.pt     q(theta | s), s = the first three of x                                sbi NSF posterior
    prior_spec.pt   q(s), the prior-predictive density of s (weights the latent observables)  sbi/zuko NSF
    grid.npz        the MIST grid (`decima.grid.IsochroneGrid`)

The two NPEs are trained on, sample and score the transformed theta_tilde (logit of theta rescaled to the grid range, `decima.theta.ThetaTransformer`), not physical theta.

Each .pt is a plain dict {"arch": builder kwargs, "state_dict": weights, "prior_loc"/"prior_scale" (posteriors)}, so it loads with `torch.load(weights_only=True)` and does not depend on pickled classes. The files are downloaded from Zenodo (doi:10.5281/zenodo.24584265) on first use, into $DECIMA_DATA_DIR if set, else the user cache directory.
"""
import os
from dataclasses import dataclass
from pathlib import Path

import pooch
import torch
from sbi.inference.posteriors import DirectPosterior
from sbi.neural_nets import marginal_nn, posterior_nn
from sbi.neural_nets.factory import ZukoFlowType

from decima.grid import IsochroneGrid
from decima.theta import ThetaTransformer

MODEL_VERSION = "v1"
DOI = "10.5281/zenodo.24584265" # pinned to the v1 record: a new model release is a new record, DOI and MODEL_VERSION
REGISTRY = { # file -> md5, as listed on the Zenodo record
    "npe.pt": "124af31916f1fd9887d892b28f532451",
    "npe_spec.pt": "9a2d2884a9b135a83ec77e3638027d93",
    "prior_spec.pt": "945c3df2be09868cf0a8516bafbbd645",
    "grid.npz": "c260ac033485594a48ce0cb766d8334d",
}


def data_dir() -> Path | None:
    """$DECIMA_DATA_DIR if set, else None (the user cache directory is used)."""
    d = os.environ.get("DECIMA_DATA_DIR")
    return Path(d) if d else None


def fetch(name: str) -> Path:
    """Local path of model file `name`, downloaded from Zenodo and checked if missing."""
    d = data_dir()
    if d is not None and (d / name).exists():
        return d / name # used as is: pooch would overwrite a copy whose hash differs
    return Path(pooch.retrieve(f"doi:{DOI}/{name}", known_hash=f"md5:{REGISTRY[name]}", fname=name,
                               path=d or pooch.os_cache("decima") / MODEL_VERSION))


class _MovableIndependent(torch.distributions.Independent):
    """Independent(Normal) with the in-place `.to(device)` that sbi's `DirectPosterior.to` requires of its prior."""
    def to(self, device):
        b = self.base_dist
        self.base_dist = torch.distributions.Normal(b.loc.to(device), b.scale.to(device), validate_args=False)
        return self


def build_posterior(d: dict) -> DirectPosterior:
    """An sbi NSF posterior from a saved {"arch", "state_dict", "prior_loc", "prior_scale"} dict."""
    arch = dict(d["arch"])
    theta_dim, x_dim = arch.pop("theta_dim"), arch.pop("x_dim")
    # the batches only size the net and seed the z-score buffers, which the state dict then overwrites
    estimator = posterior_nn(**arch)(torch.randn(10, theta_dim), torch.randn(10, x_dim))
    estimator.load_state_dict(d["state_dict"])
    estimator.eval()
    prior = _MovableIndependent(torch.distributions.Normal(d["prior_loc"], d["prior_scale"], validate_args=False), 1,
                                validate_args=False)
    return DirectPosterior(posterior_estimator=estimator, prior=prior)


def build_marginal(d: dict):
    """An sbi/zuko unconditional flow from a saved {"arch", "state_dict"} dict."""
    arch = dict(d["arch"])
    dim = arch.pop("dim")
    flow = marginal_nn(model=ZukoFlowType[arch.pop("model")], **arch)(torch.randn(10, dim))
    flow.load_state_dict(d["state_dict"])
    flow.eval()
    return flow


@dataclass
class Models:
    iso_grid: IsochroneGrid
    theta_mapper: ThetaTransformer
    npe: DirectPosterior
    npe_spec: DirectPosterior
    prior_spec: object # q(s), the density of the 3 spectroscopic observables under the uniform training prior
    device: str = "cpu" # where stage 1 and 2 run; the outputs come back to the CPU


def load_models(device: str = "cpu") -> Models:
    """Load the flows and the grid (from data_dir(); see `fetch`). With device="cuda" both stages run on the GPU;
    draws then differ from a CPU run (other RNG stream)."""
    load = lambda name: torch.load(fetch(name), weights_only=True)
    npe, npe_spec = build_posterior(load("npe.pt")), build_posterior(load("npe_spec.pt"))
    prior_spec = build_marginal(load("prior_spec.pt"))
    iso_grid = IsochroneGrid.load(fetch("grid.npz"), device=device)
    if device != "cpu":
        npe.to(device)
        npe_spec.to(device)
        prior_spec.to(device)
    return Models(iso_grid=iso_grid, theta_mapper=ThetaTransformer(iso_grid, device=device), npe=npe,
                  npe_spec=npe_spec, prior_spec=prior_spec, device=device)
