"""The docs cannot drift from the code: generated tables are current, everything is documented, links resolve."""
import importlib.util
import json
import re
from dataclasses import fields
from pathlib import Path

from decima.io import COLUMNS
from decima.pipeline import Config

ROOT = Path(__file__).parent.parent
PAGES = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]


def test_generated_regions_are_current():
    spec = importlib.util.spec_from_file_location("gen_reference", ROOT / "docs" / "gen_reference.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    assert gen.main(["--check"]) == 0, "run `python docs/gen_reference.py`"


def test_every_setting_column_and_quantity_is_documented():
    ref = {p.name: p.read_text() for p in (ROOT / "docs" / "reference").glob("*.md")}
    for f in fields(Config):
        assert f"`{f.name}`" in ref["settings.md"], f.name
    for c in COLUMNS:
        assert f"`{c}`" in ref["inputs.md"], c
    for q in ("feh_init", "alpha_fe_init", "log_age", "eep", "age", "distance", "A0", "ebv"):
        assert f"`{q}`" in ref["outputs.md"], q


def test_relative_links_resolve():
    for page in PAGES:
        for target in re.findall(r"\]\(([^)\s]+)\)", page.read_text()):
            target = target.split("#")[0]
            if target and not target.startswith(("http", "mailto")):
                assert (page.parent / target).exists(), f"{page.relative_to(ROOT)}: {target}"


def test_example_columns_files_use_decima_names():
    for f in (ROOT / "examples" / "columns").glob("*.json"):
        assert set(json.loads(f.read_text())) <= set(COLUMNS), f.name
