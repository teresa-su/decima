# Changelog

## 0.1.0 (unreleased)

First version for testing outside the group.

- Two-stage NPE inference of [Fe/H], [α/Fe], age, EEP, distance and extinction (extracted from isosbi).
- Production sample sizes: 800 latent observables × 10 θ draws per star, 400 posterior samples kept.
- Colour-dependent Fitzpatrick+19 extinction law (`F19`) by default; `F19_TEFF`, `SS2018`, `WC2019`, `none`, or a
  custom law.
- `pseudo_mag_inflate` on by default: the pseudo absolute magnitude error is inflated by a Casagrande+21 A_K estimate.
- `decima.infer` (in memory) and `decima.run` / `decima run` (chunked, resumable, parallel).
- Summary flags failed stars (`failed`, `uniform_fallback`) and records the version, model release and config.
- Optional Lindegren+21 parallax zero-point (`parallax_zero_point`, extra `decima-stars[zeropoint]`).
- Stars with no position (ra/dec or per-star distance prior) are rejected.
- One flat `Config` (no `Advanced`); output abundances are `feh_init` / `alpha_fe_init`.
- `decima columns` / `decima check` (and `decima.suggest_columns`, `decima.check_inputs`): write the columns file and check table, columns and settings before a run, without the models.
- Settings errors say how to fix them and are raised on load (unknown setting, `extinction`, `device`, `mixture_alpha`).
- Fixed: `decima run` crashed after the compute when ids were strings; a run interrupted during preparation could not resume.
- Trained models (release v1) are downloaded from Zenodo (doi:10.5281/zenodo.24584265) on first use, into the user cache or `$DECIMA_DATA_DIR`; `decima download-models` fetches them ahead of a run.
- Docs reorganised (tutorial, how-to, reference, explanation) with a mkdocs site; notebooks and code blocks are tested in CI.
