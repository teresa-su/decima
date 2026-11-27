"""The command line: columns, check, default-config, run's help. These need no models."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from astropy.table import Table

STARS = Path(__file__).parent / "data" / "reference_stars.ecsv"


class Result:
    def __init__(self, returncode, stdout, stderr):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


def decima(*args, capfd=None):
    """Run the command line in this process: (return code, stdout, stderr)."""
    import contextlib
    import io
    from decima.cli import main
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = main([str(a) for a in args])
        except SystemExit as e: # argparse --version / --help
            code = e.code
    return Result(code or 0, out.getvalue(), err.getvalue())


def test_installed_entry_point_runs():
    r = subprocess.run([sys.executable, "-m", "decima.cli", "default-config"], capture_output=True, text=True)
    assert r.returncode == 0 and json.loads(r.stdout)["seed"] == 22


def test_columns_then_check_then_default_config(tmp_path):
    out = tmp_path / "cols.json"
    r = decima("columns", STARS, "-o", out)
    assert r.returncode == 0, r.stderr
    cols = json.loads(out.read_text())
    assert cols["teff"] == "TEFF" and cols["Ks_error"] == "ks_msigcom"

    r = decima("check", STARS, "--columns", out)
    assert r.returncode == 0 and "Result: OK" in r.stdout and "5 of 5 stars can be run" in r.stdout

    r = decima("default-config")
    cfg = json.loads(r.stdout)
    assert cfg["n_theta"] == 10 and cfg["device"] == "cpu"
    (tmp_path / "settings.json").write_text(r.stdout)
    assert decima("check", STARS, "--columns", out, "--config", tmp_path / "settings.json").returncode == 0


def test_columns_to_stdout_is_pure_json():
    r = decima("columns", STARS)
    assert json.loads(r.stdout)["teff"] == "TEFF" # the readable summary goes to stderr


def test_check_on_a_broken_table_fails_and_says_why(tmp_path):
    t = Table.read(STARS)
    t["TEFF"] = np.log10(t["TEFF"])
    t["PARALLAX"] = t["PARALLAX"] / 1000
    t["TWOMASS_K"][1] = 99.0
    t.rename_column("FEH", "FEH_x")
    t.write(tmp_path / "broken.ecsv")

    r = decima("check", tmp_path / "broken.ecsv")
    assert r.returncode == 1 and "decima columns" in r.stdout

    r = decima("columns", tmp_path / "broken.ecsv", "-o", tmp_path / "c.json")
    assert r.returncode == 1 and "FEH_x" in r.stderr  # feh not found; the closest column is named

    cols = json.loads((tmp_path / "c.json").read_text())
    cols["feh"] = "FEH_x"
    (tmp_path / "c.json").write_text(json.dumps(cols))
    r = decima("check", tmp_path / "broken.ecsv", "--columns", tmp_path / "c.json")
    assert r.returncode == 0
    for word in ("Kelvin", "arcseconds", "99"):
        assert word in r.stdout


def test_version_and_help_explain_the_two_files():
    assert decima("--version").stdout.startswith("decima ")
    h = decima("run", "--help").stdout
    assert "columns file" in h and "settings file" in h
