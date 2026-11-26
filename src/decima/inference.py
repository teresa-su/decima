"""Two-stage inference of theta = ([Fe/H], [alpha/Fe], log age, EEP) for each star; see docs/method.md and the paper.

Stage 1 (`inference_stage1`) draws latent (noise-free) observables x_i = (s_i, M_pseudo,K) around the measured ones and
theta from the NPEs q(theta | x_i) and q(theta | s_i) (a defensive mixture), with s = ([Fe/H], [alpha/Fe], log Teff) and
M_pseudo,K = parallax * 10^(0.2 Ks - 2) the pseudo absolute magnitude. The flows sample the transformed theta_tilde
(`decima.theta`). Stage 2 (`inference_stage2`) fits distance and extinction E to every draw, then importance-resamples
the draws with the photometry, the exact parallax likelihood, the distance prior and the IMF.

Precision: the flows run in float32, as trained, and a batch whose spline inverse fails is redrawn in float64. The
(mu, E) fit and its sampler run in float64.
"""

import logging
import math
import warnings
from contextlib import ExitStack, contextmanager
from typing import Any, Optional

import torch
from sbi.inference.posteriors import DirectPosterior
from sbi.neural_nets.estimators.zuko_flow import ZukoUnconditionalFlow
from nflows.transforms.splines import rational_quadratic as _nflows_rq
from sbi.utils.sbiutils import within_support
from torch.distributions import Normal

from decima.distance import bailer_jones
from decima.extinction import ExtinctionLaw
from decima.grid import IsochroneGrid
from decima.theta import ThetaTransformer

_MAX_DRAWS_PER_FLOW_PASS = 100_000 # bounds memory
_N_S = 3 # q(theta | s) conditions on the first _N_S columns of x = (s, M_pseudo,K)
log = logging.getLogger("decima")


class _TorchWithFastCumsum:
    """`torch` as nflows' rational-quadratic spline sees it, with a faster CUDA cumsum.

    The spline cumsums (n_draws, num_bins) tensors along the bins. On CUDA, torch's scan over a short innermost dim is
    ~20x slower than over the outer dim, and was ~40% of the GPU time of stage 1. The outer-dim scan gives the
    sequential sum (as the CPU does). CPU tensors use torch.cumsum unchanged.
    """
    def __getattr__(self, name):
        return getattr(torch, name)

    @staticmethod
    def cumsum(x, dim):
        if not x.is_cuda:
            return torch.cumsum(x, dim)
        return x.transpose(dim, 0).contiguous().cumsum(0).transpose(dim, 0)


_nflows_rq.torch = _TorchWithFastCumsum()


# ========================================== #
# Flow helpers                               #
# ========================================== #

def _estimator(model: Any) -> torch.nn.Module:
    """The torch module holding the weights of an sbi posterior (a bare flow is its own module)."""
    return model.posterior_estimator if hasattr(model, "posterior_estimator") else model


def _model_dtype(model: Any) -> torch.dtype:
    return next(_estimator(model).parameters()).dtype


@contextmanager
def _module_dtype(module: torch.nn.Module, dtype: torch.dtype):
    """Cast `module` to `dtype` in place for the duration of the block, then back (exact for float32 <-> float64)."""
    original = next(module.parameters()).dtype
    module.to(dtype)
    try:
        yield module
    finally:
        module.to(original)


def _sample_and_log_prob_batched(posterior: DirectPosterior, n_samples: int, x: torch.Tensor,
                                 max_draws_per_pass: int = _MAX_DRAWS_PER_FLOW_PASS):
    """`n_samples` draws of the NPEs (transformed) theta for every row of `x`, with their log q(theta | x).

    Returns samples (n_samples, n_obs, theta_dim) and log_probs (n_samples, n_obs). `x` must have the flow's dtype.
    """
    estimator = posterior.posterior_estimator
    estimator.eval()
    obs_per_pass = max(1, max_draws_per_pass // n_samples)
    net = getattr(estimator, "net", None)
    fused = net is not None and hasattr(net, "sample_and_log_prob") # nflows: one pass for both

    samples, log_probs = [], []
    with torch.no_grad():
        for lo in range(0, x.shape[0], obs_per_pass):
            cond = x[lo:lo + obs_per_pass]
            if fused: # nflows layout (n_obs, n_samples, ...) -> sbi layout (n_samples, n_obs, ...)
                s, lp = net.sample_and_log_prob(n_samples, context=cond)
                s, lp = s.transpose(0, 1), lp.transpose(0, 1)
            else:
                s = estimator.sample((n_samples,), condition=cond)
                lp = estimator.log_prob(s, condition=cond)
            samples.append(s)
            log_probs.append(lp)
    samples = torch.cat(samples, dim=1)
    # nflows' base density is float32 whatever the flow's dtype, so a float64 flow gives float32 log q
    log_probs = torch.cat(log_probs, dim=1).to(samples.dtype)

    if not within_support(posterior.prior, samples).all():
        raise ValueError("Some NPE draws fall outside the prior support; use `posterior.sample_batched` instead.")
    return samples, log_probs


def _draw_theta_tilde(npe: DirectPosterior, n_samples: int, x: torch.Tensor, seed: int):
    """`_sample_and_log_prob_batched` in the dtype of `x`, redrawn in float64 if the float32 spline inverse fails.

    nflows' rational-quadratic spline asserts that a discriminant is non-negative, and float32 rounding can break that for an outlying context. The whole batch is then redrawn with the flow cast to float64, and cast back.
    """
    try:
        return _sample_and_log_prob_batched(npe, n_samples, x)
    except AssertionError:
        if x.dtype == torch.float64:
            raise
        warnings.warn(f"Flow sampling failed in {x.dtype} ({len(x)} contexts, seed {seed}); redrawing in float64.")
    with _module_dtype(_estimator(npe), torch.float64):
        samples, log_probs = _sample_and_log_prob_batched(npe, n_samples, x.to(torch.float64))
    return samples.to(x.dtype), log_probs.to(x.dtype)


def _log_q_theta_s(npe: DirectPosterior, theta_tilde: torch.Tensor, context: torch.Tensor, n_theta: int):
    """log q(theta | context_i) of every draw, evaluated at the latent observable it was drawn for.

    theta_tilde (n_stars, n_draws, theta_dim): draw k belongs to latent observable k // `n_theta`.
    context (n_stars, n_latent_x, n_context): s_i for q(theta | s), or x_i for q(theta | x).
    Returns (n_stars, n_draws) in the dtype of `theta_tilde`.
    """
    n_stars, n_draws, theta_dim = theta_tilde.shape
    theta = theta_tilde.reshape(-1, theta_dim)
    ctx = context.repeat_interleave(n_theta, dim=1).reshape(-1, context.shape[-1])
    log_q = torch.cat([
        npe.log_prob_batched(theta[lo:lo + _MAX_DRAWS_PER_FLOW_PASS].reshape(1, -1, theta_dim),
                             x=ctx[lo:lo + _MAX_DRAWS_PER_FLOW_PASS], norm_posterior=False).reshape(-1)
        for lo in range(0, len(theta), _MAX_DRAWS_PER_FLOW_PASS)
    ])
    return log_q.view(n_stars, n_draws)


def _log_q(npe: DirectPosterior, theta_tilde: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
    """log q(theta | context) in the sampling layout: theta_tilde (n, n_ctx, theta_dim), draw [j, c] at context[c].

    Returns (n, n_ctx) in the dtype of `theta_tilde` (which must be that of `npe`).
    """
    n, n_ctx, theta_dim = theta_tilde.shape
    theta = theta_tilde.reshape(-1, theta_dim) # row j * n_ctx + c
    ctx = context.repeat(n, 1) # row j * n_ctx + c -> context[c]
    log_q = torch.cat([
        npe.log_prob_batched(theta[lo:lo + _MAX_DRAWS_PER_FLOW_PASS].reshape(1, -1, theta_dim),
                             x=ctx[lo:lo + _MAX_DRAWS_PER_FLOW_PASS], norm_posterior=False).reshape(-1)
        for lo in range(0, len(theta), _MAX_DRAWS_PER_FLOW_PASS)
    ])
    return log_q.view(n, n_ctx)


def _star_major(t: torch.Tensor, n_stars: int, n_latent_x: int) -> torch.Tensor:
    """(n_theta, n_stars * n_latent_x, ...) -> (n_stars, n_latent_x * n_theta, ...), draw k in latent observable k // n_theta."""
    n_theta, rest = t.shape[0], t.shape[2:]
    return t.view(n_theta, n_stars, n_latent_x, *rest).movedim(0, 2).reshape(n_stars, n_latent_x * n_theta, *rest)


def _cat(parts: list):
    """Concatenate results along the star axis (dicts key by key)."""
    if isinstance(parts[0], dict):
        return {k: _cat([p[k] for p in parts]) for k in parts[0]}
    return torch.cat(parts)


# ========================================== #
# Stage 1: noise-marginalised NPE draws      #
# ========================================== #

def inference_stage1(
        npe: DirectPosterior,
        npe_spec: DirectPosterior,
        x_obs: torch.Tensor,
        x_obs_err: torch.Tensor,
        iso_grid: IsochroneGrid,
        theta_mapper: ThetaTransformer,
        n_latent_x: int = 800,
        n_theta: int = 10,
        seed: int = 22,
        prior_spec: Optional[ZukoUnconditionalFlow] = None,
        ess_warning_threshold: float = 0.1,
        dtype: Optional[torch.dtype] = None,
        mixture_alpha: float = 0.1,
    ) -> dict[str, torch.Tensor]:
    """
    Stage 1: weighted draws of theta for each star, marginalised over the noise on the context.

    Parameters
    ----------
    npe, npe_spec : DirectPosterior
        q(theta | x), conditioned on x = (s, M_pseudo,K), and q(theta | s), conditioned on s.
    x_obs, x_obs_err : torch.Tensor, (n_stars, 4) or (4,)
        Observed context and its 1-sigma errors; a single row of errors is used for every star.
    iso_grid : IsochroneGrid
        Flags off-grid draws (a TorchIsochroneGrid keeps the lookup on the GPU).
    theta_mapper : ThetaTransformer
        Maps the NPEs unbounded theta_tilde to physical theta.
    n_latent_x, n_theta : int
        Latent observables per star, theta draws per latent observable.
    seed : int
        Seed of the torch RNG.
    prior_spec : ZukoUnconditionalFlow, optional
        q(s), the weight of each latent observable. None weights them equally (with a warning).
    ess_warning_threshold : float
        Stars whose normalised ESS falls below this are listed.
    dtype : torch.dtype, optional
        Run in this precision (the flows are cast for the call and restored after). None = the flows' own dtype.
    mixture_alpha : float
        Fraction of each latent observable's draws from q(theta | x_i); the rest come from q(theta | s_i).

    Returns
    -------
    dict of torch.Tensor, with P = n_latent_x * n_theta draws per star (draw k from latent observable k // n_theta):
        ess_norm (n_stars,)            effective sample size of the q(s_i) weights / P
        theta, theta_tilde (n_stars, P, 4)   draws in physical units and in the NPEs space
        log_q_theta_x (n_stars, P)     log density of the proposal: q(theta | x_i), or the mixture density
        log_q_theta_s (n_stars, P)     log q(theta | s_i)
        log_q_s (n_stars, P)           log q(s_i) of the draw's latent observable, -inf off the grid (float64)
        all_invalid (n_stars,)         bool, no on-grid draw at all (stage 2 cannot use the star)
        x_i (n_stars, n_latent_x, 4)   the latent observables
    """
    if dtype is not None:
        with ExitStack() as stack: # cast the models in place for this call, then restore them
            for model in (npe, npe_spec, prior_spec):
                if model is not None:
                    stack.enter_context(_module_dtype(_estimator(model), dtype))
            return inference_stage1(npe, npe_spec, x_obs, x_obs_err, iso_grid, theta_mapper, n_latent_x, n_theta,
                                    seed, prior_spec, ess_warning_threshold, mixture_alpha=mixture_alpha)

    torch.manual_seed(seed)
    dtype = _model_dtype(npe)
    x_obs, x_obs_err = torch.atleast_2d(x_obs.to(dtype)), torch.atleast_2d(x_obs_err.to(dtype))
    n_stars, n_features = x_obs.shape
    n_draws = n_latent_x * n_theta
    device = x_obs_err.device

    # Latent observables x_i ~ N(x_obs, sigma_x), each weighted by q(s_i)
    x_i = x_obs.unsqueeze(1) + torch.randn(n_stars, n_latent_x, n_features, device=device, dtype=dtype) \
        * x_obs_err.unsqueeze(1)
    s_i = x_i[..., :_N_S]
    if prior_spec is None:
        warnings.warn("No prior_spec: the latent observables are weighted flat, i.e. q(s) is taken to be uniform.")
        log_q_s_i = torch.zeros(n_stars, n_latent_x, device=device, dtype=torch.float64)
    else:
        with torch.no_grad():
            log_q_s_i = prior_spec.log_prob(s_i.reshape(-1, _N_S)).view(n_stars, n_latent_x).to(torch.float64)

    # Per latent observable: m_x draws from q(theta | x_i) first, then n_theta - m_x from q(theta | s_i). Each flow's
    # log q of its own draws comes with them; it is evaluated only on the other flow's draws.
    m_x = int(round(mixture_alpha * n_theta))
    x_flat, s_flat = x_i.reshape(-1, n_features), s_i.reshape(-1, _N_S).contiguous()
    thetas, log_q_x, log_q_sp = [], [], [] # sampling layout (draw, n_stars * n_latent_x, ...)
    if m_x > 0:
        theta_x, lq = _draw_theta_tilde(npe, m_x, x_flat, seed)
        thetas.append(theta_x)
        log_q_x.append(lq)
        log_q_sp.append(_log_q(npe_spec, theta_x, s_flat))
    if m_x < n_theta:
        theta_s, lq = _draw_theta_tilde(npe_spec, n_theta - m_x, s_flat, seed)
        thetas.append(theta_s)
        log_q_sp.append(lq)
        if m_x > 0:
            log_q_x.append(_log_q(npe, theta_s, x_flat))
    theta_tilde = _star_major(torch.cat(thetas), n_stars, n_latent_x)

    log_q_theta_s = _star_major(torch.cat(log_q_sp), n_stars, n_latent_x)
    if m_x == 0: # pure q(theta | s_i): the NPE ratio is exactly 1
        log_q_theta_x = log_q_theta_s
    else:
        log_q_theta_x = _star_major(torch.cat(log_q_x), n_stars, n_latent_x)
        if m_x < n_theta: # mixture density a q(theta | x_i) + (1 - a) q(theta | s_i)
            a = m_x / n_theta
            log_q_theta_x = torch.logaddexp(math.log(a) + log_q_theta_x, math.log1p(-a) + log_q_theta_s)

    # A draw off the isochrone grid is outside the prior support: weight 0
    theta = theta_mapper.inverse_transform(theta_tilde.view(-1, 4)).view(n_stars, n_draws, 4)
    on_grid = torch.isfinite(torch.as_tensor(iso_grid.lookup(theta.view(-1, 4), ["G"])[0], device=device))
    on_grid = on_grid.view(n_stars, n_latent_x, n_theta)
    log_q_s = torch.where(on_grid, log_q_s_i.unsqueeze(2), float("-inf")).reshape(n_stars, n_draws)

    log_q_max = log_q_s.max(dim=1, keepdim=True).values
    all_invalid = ~torch.isfinite(log_q_max.squeeze(1))
    w = torch.exp(log_q_s - torch.where(torch.isfinite(log_q_max), log_q_max, 0.0))
    ess = w.sum(dim=1) ** 2 / (w ** 2).sum(dim=1).clamp(min=1e-12)
    ess_norm = ess / n_draws

    n_off = (~on_grid).sum().item()
    log.info(f"Invalid posterior samples (off-grid): {n_off} / {n_stars * n_draws} ({100 * n_off / (n_stars * n_draws):.1f}%)")
    log.info(f"Per-star ESS: min={ess.min().item():.1f}, median={ess.median().item():.1f}, max={ess.max().item():.1f} "
          f"of {n_draws}; normalised: min={ess_norm.min().item():.3f}, median={ess_norm.median().item():.3f}, "
          f"max={ess_norm.max().item():.3f}")
    low = torch.nonzero(ess_norm < ess_warning_threshold).flatten().tolist()
    if low:
        log.info(f"Stars below normalised ESS {ess_warning_threshold:.0%}: {len(low)} / {n_stars}: "
              + ", ".join(f"{i} ({ess_norm[i].item():.3f})" for i in low))
    if all_invalid.any():
        warnings.warn(f"{all_invalid.sum().item()} star(s) have no on-grid theta draw.")

    return {
        "ess_norm": ess_norm,
        "theta": theta,
        "theta_tilde": theta_tilde,
        "log_q_theta_x": log_q_theta_x,
        "log_q_theta_s": log_q_theta_s,
        "log_q_s": log_q_s,
        "all_invalid": all_invalid,
        "x_i": x_i,
    }


def inference_stage1_chunks(npe, npe_spec, x_obs, x_obs_err, iso_grid, theta_mapper, chunk_size: int = 250000,
                            n_latent_x: int = 800, n_theta: int = 10, seed: int = 22, prior_spec=None,
                            dtype: Optional[torch.dtype] = None, mixture_alpha: float = 0.1) -> dict[str, torch.Tensor]:
    """`inference_stage1` in chunks of at most `chunk_size` draws, to bound memory.

    The chunk starting at star i is seeded `seed + i`. Other parameters and the output are those of `inference_stage1`.
    """
    x_obs, x_obs_err = torch.atleast_2d(x_obs), torch.atleast_2d(x_obs_err)
    n_stars = x_obs.shape[0]
    step = max(1, chunk_size // (n_latent_x * n_theta))
    log.info(f"Processing {step} stars per chunk to stay under {chunk_size} samples.")
    return _cat([
        inference_stage1(npe, npe_spec, x_obs[i:i + step], x_obs_err[i:i + step] if len(x_obs_err) == n_stars else x_obs_err,
                         iso_grid, theta_mapper, n_latent_x, n_theta, seed + i, prior_spec, dtype=dtype,
                         mixture_alpha=mixture_alpha)
        for i in range(0, n_stars, step)
    ])


# ========================================== #
# Stage 2: distance / extinction resampling  #
# ========================================== #

def sir_weights(log_w: torch.Tensor):
    """Self-normalised importance weights from `log_w` (n_stars, S); -inf draws get weight 0."""
    return torch.nan_to_num((log_w - torch.logsumexp(log_w, dim=1, keepdim=True)).exp(), nan=0.0).clamp(min=0.0)


def _bivariate_gls(m_obs, m_err, M_iso, R_b, band_valid):
    """GLS fit of beta = (mu, E) for every (star, draw) from m_b - M_b = mu + R_b E.

    m_obs, m_err (n_bands, n_stars, 1); M_iso, band_valid (n_bands, n_stars, n_draws); R_b (n_bands, 1, 1) or
    (n_bands, n_stars, n_draws). The 2x2 normal equations are summed and solved in float64: the 1e-6 ridge on draws with < 2 bands is below float32 precision.

    Returns theta_hat (n_stars, n_draws, 2) and its covariance theta_cov (..., 2, 2), both float64, then chi2 at the fit, the number of valid bands and log det(theta_cov), in the dtype of `M_iso`.
    """
    f64 = torch.float64
    # zero missing bands before multiplying (0 * NaN = NaN); a per-draw R_b is NaN where the isochrone colour is
    y = torch.where(band_valid, m_obs - M_iso, 0.0)
    inv_var = torch.where(band_valid, m_err.pow(-2), 0.0)
    r = torch.where(band_valid, R_b, 0.0)
    inv_var_r = inv_var * r

    n_valid_bands = band_valid.sum(dim=0)
    ridge = (n_valid_bands < 2).to(f64) * 1e-6
    A11 = inv_var.sum(dim=0, dtype=f64) + ridge
    A12 = inv_var_r.sum(dim=0, dtype=f64)
    A22 = (inv_var_r * r).sum(dim=0, dtype=f64) + ridge
    b1 = (inv_var * y).sum(dim=0, dtype=f64)
    b2 = (inv_var_r * y).sum(dim=0, dtype=f64)

    det = A11 * A22 - A12 * A12
    C11, C12, C22 = A22 / det, -A12 / det, A11 / det
    theta_hat = torch.stack([C11 * b1 + C12 * b2, C12 * b1 + C22 * b2], dim=-1)
    theta_cov = torch.stack([torch.stack([C11, C12], dim=-1), torch.stack([C12, C22], dim=-1)], dim=-2)

    mu_hat, e_hat = (theta_hat[..., i].to(M_iso.dtype).unsqueeze(0) for i in (0, 1))
    resid = y - (mu_hat + r * e_hat)
    chi2 = torch.where(band_valid, inv_var * resid.pow(2), 0.0).sum(dim=0)
    log_det_cov = (-torch.log(det)).to(M_iso.dtype) # det(C) = 1 / det(A)
    return theta_hat, theta_cov, chi2, n_valid_bands, log_det_cov


def _univariate_gls(m_obs, m_err, M_iso, band_valid):
    """Inverse-variance fit of mu alone, for all R_b = 0 (no extinction: E is unconstrained, the 2x2 fit singular).

    Arguments as `_bivariate_gls` without R_b. Returns mu_hat, sigma_mu, chi2, n_valid_bands, each (n_stars, n_draws).
    """
    y = torch.where(band_valid, m_obs - M_iso, 0.0)
    inv_var = torch.where(band_valid, m_err.pow(-2), 0.0)
    norm = inv_var.sum(dim=0).clamp(min=1e-30)
    n_valid_bands = band_valid.sum(dim=0)
    good = n_valid_bands >= 1
    mu_hat = torch.where(good, (inv_var * y).sum(dim=0) / norm, 0.0)
    sigma_mu = torch.where(good, norm.pow(-0.5), 1.0)
    chi2 = (inv_var * (y - mu_hat.unsqueeze(0)).pow(2)).sum(dim=0)
    return mu_hat, sigma_mu, chi2, n_valid_bands


def _sample_truncated_bivariate_normal(theta_hat, theta_cov, eps: float = 1e-12, out_dtype=torch.float32):
    """Exact draw of (mu, E) from N(theta_hat, theta_cov) truncated to E >= 0.

    E is drawn by inverse CDF in float64 (the tail is sensitive), then mu from its conditional normal. Returns mu, E and log Z (log of the probability mass kept by the truncation), each (n_stars, n_draws) in `out_dtype`.
    """
    mu_hat, e_hat = theta_hat[..., 0], theta_hat[..., 1]
    var_mu, cov_mue = theta_cov[..., 0, 0], theta_cov[..., 0, 1]
    var_e = theta_cov[..., 1, 1].clamp(min=eps)
    sigma_e = var_e.sqrt()

    a = -e_hat / sigma_e # E = 0 in standard units
    log_Z = torch.special.log_ndtr(-a)
    u = torch.rand_like(e_hat).clamp(min=eps, max=1 - eps)
    p = torch.logaddexp(torch.special.log_ndtr(a), log_Z + u.log()).exp().clamp(max=1 - 1e-15)
    z = torch.erfinv(2 * p - 1) * math.sqrt(2) # standard-normal quantile of p
    e_i = (e_hat + sigma_e * z).clamp(min=0.0)

    cond_mean = mu_hat + (cov_mue / var_e) * (e_i - e_hat)
    cond_std = (var_mu - cov_mue.pow(2) / var_e).clamp(min=eps).sqrt()
    mu_i = torch.normal(cond_mean, cond_std)
    return mu_i.to(out_dtype), e_i.to(out_dtype), log_Z.to(out_dtype)


def inference_stage2(
    stage1: dict[str, torch.Tensor],
    m_obs: torch.Tensor,
    m_err: torch.Tensor,
    par_obs: torch.Tensor,
    par_err: torch.Tensor,
    L: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    iso_grid: IsochroneGrid,
    mag_fields: list[str],
    extinction_law: ExtinctionLaw,
    log_g: Optional[torch.Tensor] = None,
    log_g_err: Optional[torch.Tensor] = None,
    npe_ratio_ceiling: float = 1.0,
    imf_prior: bool = True,
    parallax_snr_min: float = 3.0,
    return_diagnostics: bool = False,
    return_terms: bool = False,
    dtype: Optional[torch.dtype] = None,
):
    """
    Stage 2: importance resampling of the stage-1 draws with the photometry and parallax (see the module docstring).

    All stars are processed at once; use `inference_stage2_chunks` for many.

    Parameters
    ----------
    stage1 : dict
        Output of `inference_stage1` (uses theta, log_q_theta_x, log_q_theta_s, log_q_s).
    m_obs, m_err : torch.Tensor, (n_stars, n_bands)
        Apparent magnitudes in `mag_fields` order (NaN where missing) and their errors.
    par_obs, par_err : torch.Tensor, (n_stars,)
        Parallax and its error [mas].
    L, alpha, beta : torch.Tensor, (n_stars,)
        Bailer-Jones+21 distance prior parameters (`decima.distance.bj_prior_params`).
    iso_grid : IsochroneGrid
        Isochrone lookup (a TorchIsochroneGrid keeps it on the GPU).
    mag_fields : list[str]
        Isochrone names of the bands, matching the columns of `m_obs`.
    extinction_law : ExtinctionLaw
        Supplies R_b (`decima.extinction`).
    log_g, log_g_err : torch.Tensor, (n_stars,), optional
        Spectroscopic log g and error. None skips the term; a star with NaN gets no term.
    npe_ratio_ceiling : float
        Per-star quantile at which q(theta | s_i) / q(theta | x_i) is clipped (1.0 = no clip).
    imf_prior : bool
        Include the IMF prior on theta (the grid's ln_dm_dEEP_prior column).
    parallax_snr_min : float
        A parallax with parallax / error above this also enters the (mu, E) fit as a linearised band.
    return_diagnostics, return_terms : bool
        Append a diagnostics dict to the outputs; with `return_terms` it holds every per-draw log-weight term.
    dtype : torch.dtype, optional
        Run the stage in this precision. None = float32, with the (mu, E) fit in float64.

    Returns
    -------
    outputs : tuple
        (d [pc], E, theta, weights) of the resampled draws, (n_stars, P) except theta (n_stars, P, 4). E is as fitted
        by the law (`extinction_law.to_ebv` converts it to E(B-V)); `weights` are the normalised importance weights of
        the pool. A star whose weights all vanish gets uniform weights over its valid draws.
    theta_hat : torch.Tensor, (n_stars, P, 2)
        Best-fit (mu, E) of every draw.
    """
    if not mag_fields:
        raise ValueError("stage 2 needs at least one band (with no band R_b = 0 and E is unconstrained)")
    extinction_law.check(mag_fields)
    theta, log_q_theta_x, log_q_theta_s, log_q_s = (stage1[k] for k in ("theta", "log_q_theta_x", "log_q_theta_s", "log_q_s"))
    if dtype is not None:
        cast = lambda t: None if t is None else t.to(dtype)
        theta, log_q_theta_x, log_q_theta_s, log_q_s = map(cast, (theta, log_q_theta_x, log_q_theta_s, log_q_s))
        m_obs, m_err, par_obs, par_err, log_g, log_g_err, L, alpha, beta = map(
            cast, (m_obs, m_err, par_obs, par_err, log_g, log_g_err, L, alpha, beta))
    dist_prior = bailer_jones(L[:, None], alpha[:, None], beta[:, None])

    n_stars, n_draws, _ = theta.shape
    shape = (n_stars, n_draws)
    device = theta.device
    fit_dtype = torch.float32 if dtype is None else dtype
    use_logg = log_g is not None and log_g_err is not None

    # One isochrone lookup (float64) for the bands, log g, the IMF term and whatever the extinction law reads
    cols = [*mag_fields, *(["log_g"] if use_logg else []), *(["ln_dm_dEEP_prior"] if imf_prior else []),
            *extinction_law.columns]
    iso = {c: torch.as_tensor(v, device=device)
           for c, v in iso_grid.get_columns(theta.flatten(0, 1), list(dict.fromkeys(cols))).items()}

    m_err, par_err = m_err.clamp(min=1e-6), par_err.clamp(min=1e-6)
    par_valid = (par_obs > 0) & (par_obs / par_err > parallax_snr_min)
    mu_par = 10.0 - 5.0 * torch.log10(par_obs.clamp(min=1e-6)) # distance modulus of the parallax, and its error
    sigma_mu_par = (5.0 / math.log(10.0)) * (par_err / par_obs.clamp(min=1e-6))

    # Fit arrays, (n_bands + 1, n_stars, ...): the bands, then the parallax as a pseudo-band with M = 0 and R_b = 0
    n_bands = len(mag_fields)
    M_iso = torch.zeros(n_bands + 1, *shape, dtype=fit_dtype, device=device)
    for i, band in enumerate(mag_fields):
        M_iso[i] = iso[band].view(shape)
    m_obs_b, m_err_b = (t.t().unsqueeze(-1).to(fit_dtype) for t in (m_obs, m_err))
    m_fit = torch.cat([m_obs_b, mu_par.view(1, n_stars, 1).to(fit_dtype)])
    err_fit = torch.cat([m_err_b, sigma_mu_par.clamp(min=1e-6).view(1, n_stars, 1).to(fit_dtype)])
    band_valid = torch.empty(n_bands + 1, *shape, dtype=torch.bool, device=device)
    band_valid[:-1] = ~torch.isnan(M_iso[:-1]) & ~(torch.isnan(m_obs_b) | torch.isnan(m_err_b))
    band_valid[-1] = par_valid.unsqueeze(-1)

    def R_b_fit(A0=None):
        """R_b of the bands at extinction A0, plus the parallax row of zeros."""
        R_b, clipped = extinction_law(mag_fields, iso, shape, device, A0=A0)
        R_b = R_b.to(fit_dtype)
        return torch.cat([R_b, R_b.new_zeros(1, *R_b.shape[1:])]), clipped

    R_b, colour_clipped = R_b_fit()
    if bool((R_b == 0).all()): # no extinction: fit mu alone (E = 0, no truncation, log Z = 0); one band suffices
        mu_hat, sigma_mu, chi2, n_valid_bands = _univariate_gls(m_fit, err_fit, M_iso, band_valid)
        theta_hat = torch.stack([mu_hat, torch.zeros_like(mu_hat)], dim=-1)
        mu_i = torch.normal(mu_hat, sigma_mu)
        e_i, log_Z, log_det_cov = (torch.zeros_like(mu_i) for _ in range(3)) # sigma_mu is constant per star
        min_bands = 1
    else:
        fit = _bivariate_gls(m_fit, err_fit, M_iso, R_b, band_valid)
        if extinction_law.colour_dependent: # R_b depends on the A0 being fitted: refit once at the fitted A0
            fit = _bivariate_gls(m_fit, err_fit, M_iso, R_b_fit(fit[0][..., 1].to(fit_dtype))[0], band_valid)
        theta_hat, theta_cov, chi2, n_valid_bands, log_det_cov = fit
        mu_i, e_i, log_Z = _sample_truncated_bivariate_normal(theta_hat, theta_cov, out_dtype=fit_dtype)
        min_bands = 2
    d_i = 10 ** ((mu_i + 5) / 5)

    # Parallax: the exact likelihood, divided by the linearised one wherever that was in the fit (inside chi2)
    log_L_par = Normal(1000 / d_i, par_err.unsqueeze(-1)).log_prob(par_obs.unsqueeze(-1))
    log_L_par_linear = Normal(mu_par.unsqueeze(-1), sigma_mu_par.unsqueeze(-1)).log_prob(mu_i)
    log_L_par = torch.where(par_valid.unsqueeze(-1), log_L_par - log_L_par_linear, log_L_par)
    has_par = torch.isfinite(par_obs) & (par_err > 0)
    log_L_par = torch.where(has_par.unsqueeze(-1), log_L_par, 0.0)

    log_prior_d = dist_prior.log_prob(d_i)
    log_jacobian = torch.log(d_i)

    log_L_logg = 0.0
    if use_logg:
        logg_ok = torch.isfinite(log_g) & torch.isfinite(log_g_err) & (log_g_err > 0)
        log_L_logg = Normal(iso["log_g"].view(shape).to(fit_dtype), log_g_err.unsqueeze(-1).clamp(min=1e-6)
                            ).log_prob(log_g.unsqueeze(-1))
        log_L_logg = torch.where(logg_ok.unsqueeze(-1), log_L_logg, 0.0)

    # NaN off the grid -> -inf weight below
    log_imf = iso["ln_dm_dEEP_prior"].view(shape).to(fit_dtype) if imf_prior else 0.0

    npe_ratio = log_q_theta_s - log_q_theta_x
    npe_ratio_clipped = torch.minimum(npe_ratio, torch.quantile(npe_ratio, npe_ratio_ceiling, dim=1, keepdim=True))

    log_w = (log_L_par + log_prior_d + log_jacobian - 0.5 * chi2 + log_Z + 0.5 * log_det_cov + log_L_logg
             + npe_ratio_clipped + log_imf)
    if log_q_s is not None: # stage-1 weights q(s_i) of the draws' latent observables
        log_w = log_w + log_q_s
    is_valid = (n_valid_bands >= min_bands) & (e_i >= 0) & torch.isfinite(log_w)
    log_w = torch.where(is_valid, log_w, float("-inf"))

    weights = sir_weights(log_w)
    row_sums = weights.sum(dim=1, keepdim=True)
    degenerate = (row_sums == 0).squeeze(1)
    weights = weights / row_sums.clamp(min=1e-30)
    # A star whose weights all vanished: uniform over its valid draws, or all on draw 0 if it has none
    valid_w = is_valid.to(weights.dtype)
    n_valid = valid_w.sum(dim=1, keepdim=True)
    fallback = degenerate & (n_valid.squeeze(1) > 0)
    weights[fallback] = valid_w[fallback] / n_valid[fallback]
    broken = degenerate & (n_valid.squeeze(1) == 0)
    weights[broken] = 0.0
    weights[broken, 0] = 1.0

    idx = torch.multinomial(weights, n_draws, replacement=True)
    star = torch.arange(n_stars, device=idx.device).unsqueeze(1)
    outputs = (d_i[star, idx], e_i[star, idx], theta[star, idx], weights)

    if return_diagnostics:
        diag = {
            "par_valid": par_valid,
            "n_negative_ebv_per_star": (~(e_i >= 0)).sum(dim=1), # counts NaN E
            "frac_valid_samples_pre": is_valid.float().mean(dim=1),
            "ess_frac": 1.0 / weights.pow(2).sum(dim=1).clamp(min=1e-30) / n_draws,
            "uniform_fallback": fallback, # every weight vanished: uniform over the valid draws (the proposal)
            "no_valid_draw": broken, # no usable draw at all
            "frac_colour_clipped": colour_clipped.float().mean(dim=1), # draws outside the law's calibrated colours
        }
        if return_terms:
            diag["log_w_terms"] = {
                "log_likelihood_par": log_L_par, "log_prior_d": log_prior_d, "log_jacobian": log_jacobian,
                "neg_half_chi2": -0.5 * chi2, "log_Z": log_Z, "half_log_det_cov": 0.5 * log_det_cov,
                **({"log_likelihood_logg": log_L_logg} if use_logg else {}),
                "npe_ratio": npe_ratio,
                **({"log_imf": log_imf} if imf_prior else {}),
                **({"log_q_s": log_q_s} if log_q_s is not None else {}),
                "log_w_total": log_w,
            }
        outputs = outputs + (diag,)
    return outputs, theta_hat.to(fit_dtype)


def inference_stage2_chunks(stage1, m_obs, m_err, par_obs, par_err, L, alpha, beta, iso_grid, mag_fields,
                            extinction_law, chunk_size: int = 6_000_000, log_g=None, log_g_err=None, **kwargs):
    """`inference_stage2` in chunks of at most `chunk_size` draws (about 1000 stars at 6000 draws), to bound memory.

    `kwargs` go to `inference_stage2`; per-star arguments are sliced with the stars. Same output as `inference_stage2`.
    """
    n_stars, n_draws = stage1["theta"].shape[:2]
    step = max(1, chunk_size // n_draws)
    per_star = lambda t, sl: None if t is None else t[sl]
    results = []
    for lo in range(0, n_stars, step):
        sl = slice(lo, lo + step)
        results.append(inference_stage2(
            {k: stage1[k][sl] for k in ("theta", "log_q_theta_x", "log_q_theta_s", "log_q_s")},
            m_obs[sl], m_err[sl], par_obs[sl], par_err[sl], L[sl], alpha[sl], beta[sl], iso_grid, mag_fields,
            extinction_law, log_g=per_star(log_g, sl), log_g_err=per_star(log_g_err, sl), **kwargs,
        ))
    outputs = tuple(_cat([r[0][i] for r in results]) for i in range(len(results[0][0])))
    return outputs, torch.cat([r[1] for r in results])
