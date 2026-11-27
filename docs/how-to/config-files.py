# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Make your columns file and settings file
#
# A DECIMA run is steered by two small JSON files. They are different things, and mixing them up is the commonest early mistake:
#
# | | **Columns file** | **Settings file** |
# |---|---|---|
# | Answers | *Which of my columns is which?* | *How should DECIMA compute?* |
# | Content | DECIMA's column name → your column name | Every setting in `decima.Config` |
# | Example | `{"teff": "TEFF", "Ks": "ks_m"}` | `{"n_keep": 1000, "extinction": "F19"}` |
# | Make it | `decima columns stars.fits -o cols.json` | `decima default-config > settings.json` |
# | Pass it | `--columns cols.json` or `columns={...}` | `--config settings.json` or `config=Config(...)` |
# | Needed? | Unless your table already uses DECIMA's names | No: the defaults are the validated ones |
#
# This page needs no models. It uses the example table from the [tutorial](../tutorial.ipynb).

# %%
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*IProgress.*")  # a harmless warning from a dependency

import decima
from astropy.table import Table

stars = Table.read("../data/example_stars.ecsv")
work = Path("../work")  # scratch folder for the files this page writes
work.mkdir(exist_ok=True)

# %% [markdown]
# ## 1. Let DECIMA suggest the columns file
#
# `suggest_columns` compares your column names with the usual names for each input: Gaia archive names, `E_X` / `X_err` / `X_error` error columns, `*_msigcom` for 2MASS, and so on. Case and punctuation are ignored.

# %%
suggestion = decima.suggest_columns(stars)
print(suggestion)

# %% [markdown]
# `suggestion` is a dict, which is the columns file. Save it, and read it back as the shell does:

# %%
(work / "cols.json").write_text(json.dumps(dict(suggestion), indent=2))
columns = json.loads((work / "cols.json").read_text())
list(columns.items())[:4]

# %% [markdown]
# ## 2. When the suggestion is not enough
#
# DECIMA never guesses between two candidates. Here a table has two T_eff columns, and no column that looks like [Fe/H]:

# %%
messy = Table({"TEFF": [5800.0], "teff_spec": [5750.0], "FEH_x": [0.0], "parallax": [2.0]})
s = decima.suggest_columns(messy)
print(s)
print("ambiguous:", s.ambiguous)
print("missing:  ", s.missing)

# %% [markdown]
# Decide yourself, and edit the dict. `ambiguous` lists the candidates, and for a missing column `close` names the closest column that was not used:

# %%
columns = dict(s)
columns["teff"] = "TEFF"        # chose one of the two candidates
columns["feh"] = s.close["feh"]  # 'FEH_x' really is [Fe/H]
columns

# %% [markdown]
# ## 3. Check the table, the columns file and the settings together
#
# `check_inputs` (shell: `decima check stars.fits --columns cols.json --config settings.json`) is the step that saves hours. It reads your data and tells you what would go wrong. Here is a table with four planted mistakes (and one renamed column, which we map by hand):

# %%
import numpy as np

bad = Table.read("../data/example_stars.ecsv")
bad["TEFF"] = np.log10(bad["TEFF"])                       # log T_eff instead of K
bad["PARALLAX"] = bad["PARALLAX"] / 1000                  # arcseconds instead of mas
bad["TWOMASS_K"][1] = 99.0                                # 99 as "not measured"
bad["E_FEH"][2] = 0.0                                     # a zero error
bad.rename_column("FEH", "FEH_x")                         # a column DECIMA cannot find on its own
cols = dict(decima.suggest_columns(bad))
cols["feh"] = "FEH_x"
print(decima.check_inputs(bad, cols))

# %% [markdown]
# - **Errors** mean a run would stop (a required column is missing or misnamed, the settings are invalid).
# - **Warnings** mean the run will go ahead on suspicious numbers. Fix them in your table; DECIMA does not change your data.
# - The last lines say how many stars would be rejected, and why.
#
# A mistake in the columns file itself is named the way you wrote it, with a suggestion:

# %%
print(decima.check_inputs(stars, {"teff": "TEF", "ks": "TWOMASS_K"}))

# %% [markdown]
# ## 4. The settings file
#
# `decima default-config` prints the defaults. Save them, change what you want, keep the rest:

# %%
(work / "settings.json").write_text(decima.Config().to_json())
settings = json.loads((work / "settings.json").read_text())
settings["n_keep"] = 1000
settings["extinction"] = "F19_TEFF"
(work / "settings.json").write_text(json.dumps(settings, indent=2))

cfg = decima.Config.from_json((work / "settings.json").read_text())
cfg.n_keep, cfg.extinction

# %% [markdown]
# A settings file may hold only the settings you change; the others keep their defaults. Mistakes are caught when the file is read, before any computation, and say how to fix them:

# %%
for text in ('{"n_thetas": 5}', '{"extinction": "F99"}', '{"device": "gpu"}', '{"mixture_alpha": 0.05}'):
    try:
        decima.Config.from_json(text)
    except ValueError as e:
        print(f"{text}\n    -> {str(e)[:170]}\n")

# %% [markdown]
# ## 5. Use them
#
# ```bash
# decima check stars.fits --columns cols.json --config settings.json
# decima run   stars.fits --columns cols.json --config settings.json --out run/
# ```
#
# or in Python, `decima.infer("stars.fits", columns, cfg)`. The [tutorial](../tutorial.ipynb) does this end to end, and [Run a large catalogue](large-runs.md) covers `decima run`.
#
# ## Common problems
#
# | You see | Cause and fix |
# |---|---|
# | `required column teff is missing; 'TEFF' looks right` | The table's name differs from DECIMA's: add `"teff": "TEFF"` to the columns file, or run `decima columns`. |
# | `unknown canonical column ... did you mean 'teff'` | The columns file has the keys the wrong way round or misspelt: the key is **DECIMA's** name, the value is yours. |
# | `column(s) not in the table: teff -> 'TEF'` | The value is not a column of your table. Names are case-sensitive. |
# | `teff: median 3.77 ... Kelvin` | The column holds log₁₀ T_eff. Convert it to K in your table. |
# | `Ks: 1 value(s) > 30 mag` | A flag such as 99 for "not measured". Replace it with NaN. |
# | `parallax: 90% of |values| are below 0.02` | Parallax is in arcseconds. DECIMA wants mas. |
# | `unknown setting 'n_thetas'` | A misspelt setting in the settings file; the message lists the valid ones. |
