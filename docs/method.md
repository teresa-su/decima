# How DECIMA works

DECIMA infers θ = ([Fe/H]_init, [α/Fe]_init, log age, EEP) for each star, plus its distance and extinction.
θ lives on a grid of MIST v2.5 isochrones (v/v_crit = 0.4).
Two neural posterior estimators (NPEs) and one density estimator, all trained on simulations from that grid, give a first posterior over θ.
Importance sampling then refines it with the photometry and parallax, and adds the posteriors over distance and extinction.

The details, derivations and validation are in the DECIMA paper (in preparation). This page is the short version.

## Inputs

| Symbol | What |
|---|---|
| s = ([Fe/H], [α/Fe], log T_eff) | spectroscopic surface observables |
| M_pseudo,K = ϖ · 10^(0.2 Ks − 2) | the pseudo absolute magnitude (parallax ϖ in mas), equal to 10^(−0.2 M_Ks) for a star without extinction |
| m_b, b ∈ {G, BP, RP, J, H, Ks} | apparent magnitudes |
| ϖ | parallax |
| log g | optional (`use_logg`) |

The spectroscopy and M_pseudo,K are distance-independent; the magnitudes and parallax are not.

## The trained models

| Model | What it is |
|---|---|
| q(θ \| s, M_pseudo,K) | NPE conditioned on the spectroscopy and the pseudo absolute magnitude |
| q(θ \| s) | NPE conditioned on the spectroscopy only |
| q(s) | density of the spectroscopic observables under the training prior |

All three were trained with `sbi` on 70,000 noise-free simulations drawn uniformly over the grid. The NPEs work in a transformed θ, rescaled to the grid range and logit-transformed (`decima.theta`). DECIMA converts their draws back to physical θ.

## Stage 1: spectroscopic posterior

Measurement noise is handled at inference time, not in the networks:
1. For each star, DECIMA draws `n_latent_x` noise-free versions of the observables around the measured ones, and weights each by q(s).
2. For each of these, it draws `n_theta` values of θ from a defensive mixture: a fraction `mixture_alpha` (0.1) from the luminosity-informed q(θ | s, M_pseudo,K), the rest from q(θ | s).

The mixture concentrates draws where the parallax will put the star, while still covering the tails of the posterior.

## Stage 2: photometry and parallax

For every stage-1 draw, the isochrone gives the absolute magnitudes:
1. DECIMA fits a distance modulus and an extinction E to the observed magnitudes, and draws (d, E) around the fit, with E ≥ 0.
2. Each draw is weighted by the photometric likelihood, the parallax likelihood, the distance prior, the IMF prior and, optionally, the log g likelihood.
3. The draws are resampled in proportion to their weights.

The weights also swap the proposal for q(θ | s), so the parallax is counted once.

## Priors

| On | Prior |
|---|---|
| θ | uniform over the grid in training; the IMF (Kroupa 2001) is applied in stage 2 (`imf_prior`) |
| distance | Bailer-Jones+21 generalised gamma, with parameters from the star's HEALPix pixel (per-star `L`, `alpha`, `beta` override the map) |
| extinction | flat, E ≥ 0 |

## Further reading

- [settings.md](settings.md): every setting.
- [extinction.md](extinction.md): the extinction laws.
- [pseudo-mag-inflate.md](pseudo-mag-inflate.md): widening the pseudo-magnitude error for reddened stars.
- [outputs.md](outputs.md): what comes out, and how to spot a failed star.
