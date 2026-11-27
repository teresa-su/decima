"""The two-stage pipeline around `decima.inference`: stage 1 (noise-marginalised NPE draws), then stage 2 (distance /
extinction importance resampling), on `StarInputs` built by `decima.io`.

summary, samples, rejected = decima.infer("stars.fits", columns={"teff": "TEFF", ...}, config=Config())
"""
from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass, fields
from typing import Optional, Union

import numpy as np
import torch
from astropy.table import Table

from decima.extinction import BANDS, EXTINCTION_LAWS, ExtinctionLaw, from_dict, get_law
from decima.inference import inference_stage1_chunks, inference_stage2_chunks
from decima.models import MODEL_VERSION, Models, load_models

QUANTITY_THETA = ("feh_init", "alpha_fe_init", "log_age", "eep") # the theta columns of the output: [Fe/H] and [alpha/Fe] are the model's initial abundances

@dataclass
class Config:
    """Every setting that changes the result. Saved as config.json next to a run's output."""
    # sampling
    n_latent_x: int = 800 # latent (noise-free) observables drawn per star in stage 1: the Monte Carlo sample over the observational noise
    n_theta: int = 10 # theta draws per latent observable
    n_keep: int = 400 # posterior samples kept per star (of n_latent_x * n_theta)
    percentiles: tuple = (5, 16, 50, 84, 95) # percentiles written to the summary table
    seed: int = 22 # random seed for every draw (latent observables, theta, distance and extinction): the same inputs and seed give the same results
    # proposal and weights
    mixture_alpha: float = 0.1 # defensive mixture: fraction of each latent observable's theta draws from q(theta | x_i), the rest from q(theta | s_i); mixture_alpha * n_theta must be a whole number
    npe_ratio_ceiling: float = 1.0 # clip the NPE ratio q(theta | s) / q(theta | x) at this per-star quantile (1.0 = no clip)
    imf_prior: bool = True # apply the IMF prior (Kroupa 2001)
    use_logg: bool = False # add the spectroscopic log g likelihood in stage 2
    # photometry, parallax and extinction
    bands: tuple = BANDS # any subset of G, BP, RP, J, H, Ks that includes Ks
    extinction: Union[str, ExtinctionLaw] = "F19" # a name from extinction.EXTINCTION_LAWS, or a law object
    grey: float = 0.0 # grey component added to every R_b of a built-in extinction law
    pseudo_mag_inflate: bool = True # inflate the pseudo absolute magnitude error by a Casagrande+21 A_K estimate (decima.ak)
    parallax_zero_point: bool = False # subtract the Lindegren+21 Gaia parallax zero-point (needs gaiadr3-zeropoint and the astrometry columns; see decima.io)
    parallax_snr_min: float = 3.0 # parallaxes with parallax / error above this also enter the stage-2 (distance, extinction) fit directly
    # input errors and fills
    gaia_err_floor: float = 0.02 # [mag] floor on G, BP, RP errors
    twomass_err_floor: float = 0.03 # [mag] floor on J, H, Ks errors
    missing_err: float = 0.5 # [mag] error of a magnitude given without one (0: use the floor)
    alpha_fe_fill: float = 0.0 # [alpha/Fe] where the input has none
    alpha_fe_err_fill: float = 0.2 # error in [alpha/Fe] when the input has none
    # compute
    device: str = "cpu" # "cpu" or "cuda"
    stars_per_batch: int = 100 # stars per stage-1 flow call

    def __post_init__(self):
        self.bands, self.percentiles = tuple(self.bands), tuple(self.percentiles)
        if "Ks" not in self.bands or not set(self.bands) <= set(BANDS):
            raise ValueError(f"bands must be a subset of {BANDS} that includes Ks, got {self.bands}")
        m_x = self.mixture_alpha * self.n_theta
        if not 0 <= self.mixture_alpha <= 1 or abs(m_x - round(m_x)) > 1e-9:
            allowed = ", ".join(f"{k / self.n_theta:g}" for k in range(self.n_theta + 1)) if self.n_theta <= 20 else "k / n_theta for whole k"
            raise ValueError(f"mixture_alpha * n_theta must be a whole number in [0, n_theta], got {self.mixture_alpha} * {self.n_theta} = {m_x:g}. "
                             f"With n_theta={self.n_theta}, mixture_alpha can be {allowed}; or change n_theta.")
        if isinstance(self.extinction, dict):
            self.extinction = from_dict(self.extinction)
        if isinstance(self.extinction, str) and self.extinction not in EXTINCTION_LAWS:
            raise ValueError(f"unknown extinction law {self.extinction!r}{_did_you_mean(self.extinction, EXTINCTION_LAWS)}; choose from {EXTINCTION_LAWS}")
        if not isinstance(self.device, str) or not re.fullmatch(r"cpu|cuda(:\d+)?", self.device):
            hint = " (for a GPU use 'cuda')" if str(self.device).lower() in ("gpu", "cuda0", "gpu0") else ""
            raise ValueError(f"device must be 'cpu', 'cuda' or 'cuda:N', got {self.device!r}{hint}")

    def law(self) -> ExtinctionLaw:
        if isinstance(self.extinction, ExtinctionLaw):
            return self.extinction
        return get_law(self.extinction, grey=self.grey)

    def to_json(self) -> str:
        d = {f.name: getattr(self, f.name) for f in fields(self)}
        if isinstance(self.extinction, ExtinctionLaw):
            d["extinction"] = self.extinction.to_dict()
        return json.dumps(d, indent=2)

    @classmethod
    def from_json(cls, text: str) -> "Config":
        """Settings from the text of a settings file (`decima default-config > settings.json`). Missing settings keep their defaults."""
        d = json.loads(text)
        valid = [f.name for f in fields(cls)]
        for k in d:
            if k not in valid:
                raise ValueError(f"unknown setting {k!r} in the settings file{_did_you_mean(k, valid)}. Valid settings: {', '.join(valid)}")
        return cls(**d)


def _did_you_mean(name, options) -> str:
    close = difflib.get_close_matches(str(name), [str(o) for o in options], n=1)
    return f" (did you mean {close[0]!r}?)" if close else ""


@dataclass
class StarInputs:
    """Per-star inputs, all with leading dimension n_stars. `ids` is carried through to the output.

    x_obs, x_obs_err   (n, 4)  [Fe/H], [alpha/Fe], log Teff, M_pseudo,K (pseudo absolute magnitude) and their errors (stage 1)
    m_obs, m_stds      (n, n_bands) apparent magnitudes and errors, in the order of Config.bands (NaN where missing)
    par_obs, par_error (n,) parallax and error [mas]
    log_g, log_g_err   (n,) spectroscopic log g and error (used only with Config.use_logg)
    err_missing        (n, n_bands) bool, magnitude present but its error missing (Config.missing_err used)
    L, alpha, beta     (n,) per-star distance-prior (generalised gamma) parameters
    pseudo_mag_inflated (n,) bool, the pseudo absolute magnitude error was inflated by an A_K estimate
    zero_point_applied (n,) bool, the Lindegren+21 parallax zero-point was subtracted
    """
    ids: np.ndarray
    x_obs: torch.Tensor
    x_obs_err: torch.Tensor
    m_obs: torch.Tensor
    m_stds: torch.Tensor
    par_obs: torch.Tensor
    par_error: torch.Tensor
    log_g: torch.Tensor
    log_g_err: torch.Tensor
    err_missing: torch.Tensor
    L: torch.Tensor
    alpha: torch.Tensor
    beta: torch.Tensor
    pseudo_mag_inflated: torch.Tensor
    zero_point_applied: torch.Tensor

    _TENSORS = ("x_obs", "x_obs_err", "m_obs", "m_stds", "par_obs", "par_error", "log_g", "log_g_err",
                "err_missing", "L", "alpha", "beta", "pseudo_mag_inflated", "zero_point_applied")

    def __len__(self):
        return len(self.ids)

    def slice(self, lo: int, hi: int) -> "StarInputs":
        return StarInputs(ids=self.ids[lo:hi], **{k: getattr(self, k)[lo:hi] for k in self._TENSORS})

    def save(self, path):
        torch.save({"ids": np.asarray(self.ids), **{k: getattr(self, k) for k in self._TENSORS}}, path)

    @classmethod
    def load(cls, path) -> "StarInputs":
        d = torch.load(path, weights_only=False)
        return cls(ids=d.pop("ids"), **d)


def _to_cpu(x):
    """Tensors (also inside dicts / tuples, e.g. the stage-2 diagnostics) moved to the CPU."""
    if torch.is_tensor(x):
        return x.cpu()
    if isinstance(x, dict):
        return {k: _to_cpu(v) for k, v in x.items()}
    if isinstance(x, tuple):
        return tuple(_to_cpu(v) for v in x)
    return x


def stage1(inputs: StarInputs, models: Models, cfg: Config, seed: int) -> dict:
    """Stage 1 on the stars of `inputs`: the `inference_stage1` outputs, on the CPU."""
    out = inference_stage1_chunks(
        npe=models.npe,
        npe_spec=models.npe_spec,
        x_obs_err=inputs.x_obs_err.to(models.device),
        x_obs=inputs.x_obs.to(models.device),
        iso_grid=models.iso_grid,
        theta_mapper=models.theta_mapper,
        chunk_size=cfg.stars_per_batch * cfg.n_latent_x * cfg.n_theta,
        n_latent_x=cfg.n_latent_x,
        n_theta=cfg.n_theta,
        seed=seed,
        prior_spec=models.prior_spec,
        mixture_alpha=cfg.mixture_alpha,
    )
    return _to_cpu(out)


def stage2(s1: dict, inputs: StarInputs, models: Models, cfg: Config, torch_seed: Optional[int] = None):
    """Stage 2 on a `stage1` result. Returns, on the CPU, (outputs, theta_hat, diagnostics) with
    outputs = (distance [pc], E (as fitted by the law), theta, weights)."""
    if torch_seed is not None:
        torch.manual_seed(torch_seed)
    dev = lambda x: torch.as_tensor(x).to(models.device)
    outputs, theta_hat = inference_stage2_chunks(
        {k: dev(s1[k]) for k in ("theta", "log_q_theta_x", "log_q_theta_s", "log_q_s")},
        m_obs=dev(inputs.m_obs),
        m_err=dev(inputs.m_stds),
        par_obs=dev(inputs.par_obs),
        par_err=dev(inputs.par_error),
        log_g=dev(inputs.log_g) if cfg.use_logg else None, # None skips the log g term
        log_g_err=dev(inputs.log_g_err) if cfg.use_logg else None,
        L=dev(inputs.L), alpha=dev(inputs.alpha), beta=dev(inputs.beta),
        iso_grid=models.iso_grid,
        mag_fields=list(cfg.bands),
        return_diagnostics=True,
        extinction_law=cfg.law(),
        npe_ratio_ceiling=cfg.npe_ratio_ceiling,
        imf_prior=cfg.imf_prior,
        parallax_snr_min=cfg.parallax_snr_min,
    )
    d, e, theta, w, diag = outputs
    return _to_cpu(((d, e, theta, w), theta_hat, diag))


def quantity_names(cfg: Config) -> tuple:
    """Columns of the saved samples: theta, age [Gyr], distance [pc] and the fitted extinction (A0 or ebv)."""
    return QUANTITY_THETA + ("age", "distance", cfg.law().extinction_name)


def failed_stars(age: np.ndarray, no_valid_draw: np.ndarray) -> np.ndarray:
    """Failed stars: no usable stage-1 draw at all, or an age posterior (n_stars, n_draws) with no width on one side of
    the median (16th or 84th percentile equal to the median)."""
    age_lo, age_med, age_hi = np.percentile(age, (16, 50, 84), axis=1)
    return np.asarray(no_valid_draw) | (age_lo == age_med) | (age_hi == age_med)


def run_chunk(inputs: StarInputs, models: Models, cfg: Config, seed: int) -> dict:
    """Stage 1 then stage 2 on one chunk of stars. Returns numpy arrays: ids, percentiles (n, n_pct, n_q),
    samples (n, n_keep, n_q) and per-star diagnostics."""
    torch.manual_seed(seed)
    s1 = stage1(inputs, models, cfg, seed)
    (d, e, theta, _), _, diag = stage2(s1, inputs, models, cfg, torch_seed=seed)
    # (n_stars, n_draws, n_q); stage 2 resamples, so draws are unweighted
    theta = theta.double()
    age = 10 ** (theta[..., 2:3] - 9)
    q = torch.cat([theta, age, d.double().unsqueeze(-1), e.double().unsqueeze(-1)], dim=-1).numpy()
    pct = np.percentile(q, cfg.percentiles, axis=1).transpose(1, 0, 2) # numpy, as torch.quantile caps at 16M elements
    samples = q[:, :cfg.n_keep] # draws are iid, so the first n_keep
    no_valid_draw = diag["no_valid_draw"].numpy()
    failed = failed_stars(q[..., 4], no_valid_draw)
    pct[no_valid_draw], samples[no_valid_draw] = np.nan, np.nan
    return {
        "ids": np.asarray(inputs.ids),
        "percentiles": pct.astype(np.float32),
        "samples": samples.astype(np.float32),
        "failed": failed,
        "uniform_fallback": diag["uniform_fallback"].numpy(),
        **{k: diag[k].float().numpy() for k in ("frac_valid_samples_pre", "ess_frac", "frac_colour_clipped")},
        "n_negative_extinction": diag["n_negative_ebv_per_star"].numpy(),
        "par_valid": diag["par_valid"].numpy(),
        "ess_norm_stage1": s1["ess_norm"].float().numpy(),
        "band_used": (torch.isfinite(inputs.m_obs) & torch.isfinite(inputs.m_stds)).numpy(),
        "band_err_missing": inputs.err_missing.numpy(),
        "pseudo_mag_inflated": inputs.pseudo_mag_inflated.numpy(),
        "zero_point_applied": inputs.zero_point_applied.numpy(),
    }


# the diagnostic and flag columns of the summary table (after the percentiles), with what they mean; docs/reference/outputs.md is generated from this
DIAGNOSTICS = {
    "failed": "The star failed: no usable stage-1 draw, or an age posterior with no width on one side of the median. Do not use its results.",
    "uniform_fallback": "Every importance weight vanished, so the 'posterior' is just the stage-1 proposal. Treat as failed.",
    "ess_frac": "Effective sample size of the stage-2 importance weights as a fraction of the stage-1 draws. The main quality number.",
    "ess_norm_stage1": "Effective sample size of the stage-1 q(s) weights as a fraction of the draws. Low when the spectroscopy sits at the edge of the training prior.",
    "frac_valid_samples_pre": "Fraction of stage-1 draws usable in stage 2: on the grid, with enough bands, and a finite weight.",
    "frac_colour_clipped": "Fraction of draws whose intrinsic colour was outside the extinction law's calibration (colour-dependent laws only).",
    "n_negative_extinction": "Number of draws whose sampled extinction was negative or NaN, and which were dropped.",
    "par_valid": "The parallax was precise enough (`parallax_snr_min`) to enter the stage-2 distance-extinction fit directly. The exact parallax likelihood is used either way.",
    "pseudo_mag_inflated": "The pseudo absolute magnitude error was widened by an A_K estimate (`pseudo_mag_inflate`).",
    "zero_point_applied": "The Lindegren+21 parallax zero-point was subtracted from this star's parallax (`parallax_zero_point`).",
}
BAND_DIAGNOSTICS = {
    "{band}_used": "The band had a magnitude and an error for this star, so it entered stage 2.",
    "{band}_err_missing": "The band had a magnitude but no error, so `missing_err` was used.",
}


def summary_table(res: dict, cfg: Config) -> Table:
    """One row per star: id, percentiles of every quantity, diagnostics and flags. The table's meta records the DECIMA
    version, the model release and the config (written to the FITS header / ECSV meta)."""
    from decima import __version__
    t = Table({"id": res["ids"]})
    for j, q in enumerate(quantity_names(cfg)):
        for i, p in enumerate(cfg.percentiles):
            t[f"{q}_p{p:g}"] = res["percentiles"][:, i, j]
    for k in DIAGNOSTICS:
        t[k] = res[k]
    for i, b in enumerate(cfg.bands):
        t[f"{b}_used"] = res["band_used"][:, i]
        t[f"{b}_err_missing"] = res["band_err_missing"][:, i]
    # one-line JSON: FITS header values cannot hold newlines
    t.meta.update({"DECIMA": __version__, "MODELS": MODEL_VERSION, "CONFIG": json.dumps(json.loads(cfg.to_json()))})
    return t


def concat(parts: list) -> dict:
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def infer(source, columns: Optional[dict] = None, config: Optional[Config] = None, models: Optional[Models] = None,
          chunk: int = 250):
    """Run DECIMA in memory on a table (path, astropy Table or pandas DataFrame); for big catalogs use `decima.run`.

    Returns (summary, samples, rejected): the per-star astropy Table, the (n_stars, n_keep, n_q) posterior samples
    (columns `quantity_names(config)`) and a Table of the input rows that could not be run, with the reason.
    Chunk c of `chunk` stars is seeded config.seed + c * chunk, as in `decima.run`, so both give the same draws.
    """
    from decima.io import build_inputs, read_table
    cfg = config or Config()
    models = models or load_models(cfg.device)
    inputs, rejected = build_inputs(read_table(source, columns), cfg)
    res = concat([run_chunk(inputs.slice(lo, lo + chunk), models, cfg, seed=cfg.seed + lo)
                  for lo in range(0, len(inputs), chunk)])
    return summary_table(res, cfg), res["samples"], rejected
