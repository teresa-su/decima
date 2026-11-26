# Changelog

## 0.1.0 (unreleased)

First version for testing outside the group.

- Two-stage NPE inference of [Fe/H], [α/Fe], age, EEP, distance and extinction (extracted from isosbi).
- Production sample sizes: 800 latent observables × 10 θ draws per star, 400 posterior samples kept.
- Colour-dependent Fitzpatrick+19 extinction law (`F19`) by default; `F19_TEFF`, `SS2018`, `WC2019`, `none`, or a
  custom law.
- `ak_inflate` on by default: the pseudo absolute magnitude error is inflated by a Casagrande+21 A_K estimate.
- `decima.infer` (in memory) and `decima.run` / `decima run` (chunked, resumable, parallel).
