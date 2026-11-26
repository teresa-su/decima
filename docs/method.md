# How DECIMA works

DECIMA infers θ = ([Fe/H], [α/Fe], log age, EEP) for each star, plus its distance and extinction.
θ lives on a grid of MIST v2.5 isochrones (v/v_crit = 0.4).
Two neural posterior estimators (NPEs) and one density estimator, all trained on simulations from that grid provide initial posteriors over θ. Importance sampling refines the the posterior with the photometry and parallax, and provide posteriors over distance and extinction.

## Inputs, in two groups

| Symbol | What | Used in |
|---|---|---|
| s = ([Fe/H], [α/Fe], log T_eff) | spectroscopic observables | stage 1 and 2 |
| ϖ_K = ϖ · 10^(0.2 Ks − 2) | the *pseudo absolute magnitude* (parallax ϖ in mas), equal to 10^(−0.2 M_Ks) for a star without extinction | stage 1 |
| m_b, b ∈ {G, BP, RP, J, H, Ks} | apparent magnitudes | stage 2 |
| ϖ | parallax | stage 2 |

x = (s, ϖ_K) is the context of the main NPE.
The pseudo-magnitude form lets the parallax enter the NPE without dividing by it, so negative and low-S/N parallaxes are fine.

## The trained models

| Model | What it is |
|---|---|
| q(θ \| x) | NPE conditioned on the spectroscopy *and* ϖ_K |
| q(θ \| s) | NPE conditioned on the spectroscopy only |
| q(s) | density of the spectroscopic observables under the training prior |

All three were trained on 70,000 simulations from a prior that is uniform over the grid:

| Parameter | Range |
|---|---|
| [Fe/H] | −4 to +0.5 |
| [α/Fe] | −0.2 to +0.6 |
| age | uniform in *linear* age, 10⁵ yr to 20 Gyr |
| EEP | 202 (ZAMS) to 706 (end of core-He burning) |

## Stage 1: marginalise over the noise on the observables

For each star:
1. Draw `n_latent_x` latent (noise-free) observables x_i ~ N(x_obs, σ_x).
2. Weight each by q(s_i), the prior density of its spectroscopic part.
3. For each x_i, draw `n_theta` values of θ.

The θ draws come from a *defensive mixture*: a fraction α (`Advanced.mixture_alpha`, default 0.1) from q(θ | x_i) and
the rest from q(θ | s_i). The mixture keeps the proposal wide enough that stage 2 can find the region the photometry
and parallax prefer, even where q(θ | x_i) is overconfident.

Draws that land off the isochrone grid get zero weight. With the defaults this gives 800 × 10 = 8000 draws per star.

## Stage 2: importance-resample with photometry and parallax

For each stage-1 draw, the isochrone gives absolute magnitudes M_b(θ).

**Fit (μ, E′).** A generalised least-squares fit of the distance modulus μ and the extinction E′ to

    m_b − M_b(θ) = μ + R_b E′

uses every available band. The parallax enters as one more, linearised "band".
R_b comes from the extinction law (see [extinction.md](extinction.md)). For a colour-dependent law it depends on the
draw's intrinsic colour and on E′ itself, so the fit is repeated once at the fitted E′.

**Sample (μ, E′).** (μ, E′) is drawn from the fitted Gaussian, truncated to E′ ≥ 0.

**Weight the draw.** The weight is target / proposal:

    log w = log q(s_i) + log q(θ | s_i) − log q(θ_mix | x_i)     NPE ratio
          − χ²_min / 2 + log Z + ½ log |C|                       photometry; correction for sampling (μ, E′)
          + log L(ϖ) − log L_linear(ϖ)                           exact parallax likelihood replaces the linear one
          + log p_BJ21(d) + log d                                Bailer-Jones+21 distance prior; Jacobian μ → d
          + log IMF(θ)                                           (Advanced.imf_prior)
          + log L(log g)                                         (Config.use_logg)

The NPE ratio swaps the proposal q(θ | x) for q(θ | s). The parallax information that q(θ | x) carried through ϖ_K is
then counted only once, through the exact parallax likelihood.

**Resample.** The draws are resampled with replacement in proportion to w. The kept samples are therefore unweighted, but
contain repeats. The effective number of independent draws is the importance-sampling ESS, reported as `ess_frac`.

## The distance prior

The Bailer-Jones+21 generalised-gamma prior, with parameters taken from the star's level-5 HEALPix pixel (from `ra`, `dec`).
Per-star `L`, `alpha`, `beta` columns override the map.
A star with no position and no override gets an exponentially decreasing space-density prior with L = 1 kpc.

## The A_K inflation (optional, on by default)

ϖ_K assumes no extinction, so a reddened star's ϖ_K is too large by 10^(0.2 A_K). Stage 1 cannot know this.
With `ak_inflate`, a cheap A_K estimate widens the error on ϖ_K, so the stage-1 proposal is broad enough for stage 2 to fix.
See [ak-inflate.md](ak-inflate.md).

## Parallax quality

- The parallax enters the stage-2 *fit* only when ϖ > 0 and ϖ/σ_ϖ > 3.
- The *exact* parallax likelihood is applied to every star with a finite parallax and a positive error.

The `par_valid` column records which case applied.

## Further reading

- [settings.md](settings.md): every setting, its default and when to change it.
- [extinction.md](extinction.md): the extinction laws.
- [outputs.md](outputs.md): what comes out, and how to tell a good result from a bad one.
