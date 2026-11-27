"""DECIMA: stellar ages, [Fe/H], [alpha/Fe], distances and extinctions with neural posterior estimation on MIST
isochrones, from spectroscopic [Fe/H], [alpha/Fe], Teff, a Gaia parallax and Gaia / 2MASS photometry."""
from decima.extinction import EXTINCTION_LAWS, ExtinctionLaw, FixedLaw, PolynomialLaw, get_law
from decima.io import COLUMN_INFO, build_inputs, check_inputs, read_table, suggest_columns
from decima.models import load_models
from decima.pipeline import Config, infer, quantity_names
from decima.runner import run

__version__ = "0.1.0"
