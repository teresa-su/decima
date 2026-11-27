# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.7
#   kernelspec:
#     display_name: decima-stars (3.13.12)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Tutorial: from a catalogue to ages
#
# This is a short tutorial to show how to infer ages, distances and extinctions for a handful of stars. It also shows how to identify when the pipeline has failed/produced unreliable results.
#
#
# | Step | |
# |---|---|
# | 1 | Look at your table |
# | 2 | Make the **columns file**: your column names → DECIMA's |
# | 3 | Check everything before running |
# | 4 | Choose the **settings** |
# | 5 | Run |
# | 6 | Read the summary, the samples and the rejected stars |
# | 7 | Spot a star you should not trust |
#
# ## Install
#
# ```bash
# git clone https://github.com/teresa-su/decima && cd decima
# uv sync --group dev        # or: pip install -e .
# ```
#
# `uv sync` creates the environment in `decima/.venv`. To run this notebook in it, register it as a Jupyter kernel and select the `decima` kernel (in VS Code, choose `.venv/bin/python` as the interpreter instead):
#
# ```bash
# source .venv/bin/activate
# python -m ipykernel install --user --name decima
# ```
#
# The first time DECIMA runs inference it downloads the trained models ([doi:10.5281/zenodo.24584265](https://doi.org/10.5281/zenodo.24584265), CC-BY-4.0, 245 MB) from Zenodo and caches them (`~/.cache/decima/v1` on Linux). If your compute nodes have no internet, run `decima download-models` on the login node first. To keep the models elsewhere, or use a copy you already have, set `DECIMA_DATA_DIR` to that directory.

# %%
import warnings

warnings.filterwarnings("ignore", message=".*triangular_solve.*")  # harmless warnings from dependencies
warnings.filterwarnings("ignore", message=".*IProgress.*")

import matplotlib.pyplot as plt
from astropy.table import Table

import decima

# %% [markdown]
# ## 1. Look at your table
#
# `data/example_stars.ecsv` has seven stars with spectroscopy (LAMOST-style names), Gaia and 2MASS photometry. Five are real stars from the Xiang+24 catalogue. Two are copies altered for this tutorial: one has no T_eff, and one has a parallax that does not fit its spectroscopy. DECIMA reads any table astropy reads (FITS, CSV, ECSV, ...) and also pandas data frames.

# %%
stars = Table.read("data/example_stars.ecsv")
print(stars.colnames)
stars["GAIAEDR3_SOURCE_ID_1", "TEFF", "FEH", "PARALLAX", "E_PARALLAX", "TWOMASS_K"]

# %% [markdown]
# ## 2. Make the columns file
#
# DECIMA has its own names for the columns it needs (`teff`, `feh`, `parallax`, `Ks`, ... see the [input reference](reference/inputs.md)). If your table calls them something else, the **columns file** will provide the mapping: *DECIMA's name → table name*.
#
# You do not have to write it by hand. `suggest_columns` (on the shell: `decima columns stars.ecsv -o cols.json`) matches names for you:

# %%
suggestion = decima.suggest_columns(stars)
print(suggestion)

# %% [markdown]
# `ok` rows were matched. If a required column is `NOT FOUND`, or two columns could both be the answer (`??`), DECIMA says so instead of guessing: [Make your columns file](how-to/config-files.ipynb) shows how to fix those. Here everything was found, so the columns file is just the suggestion:

# %%
columns = dict(suggestion)
columns

# %% [markdown]
# ## 3. Check before you run
#
# `check_inputs` (shell: `decima check stars.ecsv --columns cols.json`) reads your table and the columns file and reports what would go wrong, including values that look like the wrong unit (log T_eff instead of K, parallax in arcsec, `99` as a "missing" magnitude).

# %%
print(decima.check_inputs(stars, columns))

# %% [markdown]
# Read the end of the report: six of the seven stars can be run, and one is rejected for having no T_eff.
#
# ## 4. Choose the settings
#
# Everything that changes the result is in `decima.Config`; the defaults are the ones DECIMA was validated with. The **settings file** is a different file from the columns file: it holds these settings, not column names. Change what you need and leave the rest:

# %%
cfg = decima.Config(n_keep=1000)  # keep 1000 posterior samples per star instead of 400
print(cfg.to_json()[:300], "...")

# %% [markdown]
# On the shell: `decima default-config > settings.json`, edit it, and pass it with `--config settings.json`. A misspelt setting or an impossible value is refused before any computation, and the message says how to fix it. See the [settings reference](reference/settings.md) for all of them.
#
# ## 5. Run

# %%
summary, samples, rejected = decima.infer(stars, columns, cfg)
len(summary), samples.shape

# %% [markdown]
# `infer` is for notebooks and up to a few thousand stars. For a big catalogue, the same inputs go to `decima run` on the shell, which works in chunks, uses several processes and resumes if interrupted: see [Run a large catalogue](how-to/large-runs.md).
#
# ## 6. Read the results
#
# **`summary`** has one row per star that was run. For every quantity it has percentiles (5, 16, 50, 84, 95): `age_p16`, `age_p50`, `age_p84` give the median age and a 68% interval, in Gyr.

# %%
summary["id", "age_p16", "age_p50", "age_p84", "distance_p50", "A0_p50", "ess_frac", "failed"]

# %% [markdown]
# **`samples`** has the posterior draws themselves, shape `(stars, n_keep, 7)`. The seven quantities, in order, are:

# %%
decima.quantity_names(cfg)

# %% [markdown]
# **`rejected`** lists the rows that could not be run, and why. They are not in the summary, which is why `id` matters: map an `id` column to match results to your table.

# %%
rejected

# %% [markdown]
# ## 7. Spot a star you should not trust
#
# `failed` flags stars with no usable result. None failed here. But a star can come out without being flagged and still be doubtful. `ess_frac` is the effective sample size of the final weights as a fraction of the draws DECIMA made. A small value means the photometry and parallax disagree with what the spectroscopy predicts.

# %%
summary["ess"] = summary["ess_frac"] * cfg.n_latent_x * cfg.n_theta
summary["id", "ess_frac", "ess", "age_p50"]

# %% [markdown]
# The star with `id = 2` has the lowest effective sample size (about 13 of 8000 draws): its parallax was changed to a quarter of the real one. It is a copy of the star with `id = 680764209182123904`, so the spectroscopy is the same, yet the ages disagree (0.8 Gyr against 7.8 Gyr). Compare the two posteriors:

# %%
ids = list(summary["id"])
real, altered = ids.index(680764209182123904), ids.index(2)
fig, ax = plt.subplots(figsize=(6, 3))
age = decima.quantity_names(cfg).index("age")
for k, label in ((real, "real parallax"), (altered, "parallax / 4 (id 2)")):
    ax.hist(samples[k, :, age], bins=40, histtype="step", density=True, label=label)
ax.set_xlabel("age [Gyr]")
ax.legend();

# %% [markdown]
# Use `failed` and `ess_frac` as a first filter, then look at the stars it leaves you with. [Decide which results to trust](how-to/trust-results.md) says what else to look at and which stars DECIMA was validated on.
#
# ## Where next
#
# - Your own catalogue: [Make your columns file](how-to/config-files.ipynb), then [Run a large catalogue](how-to/large-runs.md).
# - What DECIMA does with your numbers: [How DECIMA works](explanation/method.md).
