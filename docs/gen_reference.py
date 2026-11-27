"""Write the generated parts of the reference pages from the code, so they cannot drift from it.

    python docs/gen_reference.py           rewrite the generated regions
    python docs/gen_reference.py --check   exit 1 if any region is stale (run in the tests)

Regions sit between `<!-- BEGIN GENERATED: name -->` and `<!-- END GENERATED: name -->` in the pages:
    reference/settings.md   settings    every `decima.Config` field with its default and the comment next to it in pipeline.py
    reference/inputs.md     inputs      every column in `decima.io.COLUMN_INFO`
    reference/outputs.md    diagnostics the summary table's diagnostic columns (`decima.pipeline.DIAGNOSTICS`)
"""
import inspect
import json
import re
import sys
from pathlib import Path

from decima.io import COLUMN_INFO
from decima.pipeline import BAND_DIAGNOSTICS, DIAGNOSTICS, Config

DOCS = Path(__file__).parent


def _cell(text):
    return text.replace("|", "\\|")


def settings():
    """One table per group of Config settings, in the order and with the words of the dataclass."""
    cfg = Config()
    out, group, rows = [], None, []

    def flush():
        if rows:
            out.append(f"### {group.capitalize()}\n\n| Setting | Default | What it does |\n|---|---|---|\n" + "\n".join(rows) + "\n")
            rows.clear()

    for line in inspect.getsource(Config).splitlines():
        if m := re.fullmatch(r"    # ([a-z ,]+)", line):
            flush()
            group = m.group(1)
        elif m := re.fullmatch(r"    (\w+): [^=]+ = .+? # (.+)", line):
            name, comment = m.groups()
            default = json.dumps(getattr(cfg, name))
            rows.append(f"| `{name}` | `{default}` | {_cell(comment[0].upper() + comment[1:])} |")
    flush()
    assert group is not None and len(re.findall(r"^\| `", "\n".join(out), re.M)) == len(Config.__dataclass_fields__), \
        "a Config field has no `name: type = default # comment` line, or no group comment above it"
    return "\n".join(out)


def inputs():
    """One table per group of canonical input columns."""
    titles = {"required": "Required", "position": "Position (ra and dec, or L, alpha and beta)", "optional": "Optional",
              "zero-point": "Parallax zero-point (only with `parallax_zero_point=True`)"}
    out = []
    for group, title in titles.items():
        rows = [f"| `{c}` | {i.unit or ''} | {_cell(i.description)} |" for c, i in COLUMN_INFO.items() if i.group == group]
        out.append(f"### {title}\n\n| Column | Unit | Meaning |\n|---|---|---|\n" + "\n".join(rows) + "\n")
    return "\n".join(out)


def diagnostics():
    rows = [f"| `{k}` | {_cell(v)} |" for k, v in {**DIAGNOSTICS, **BAND_DIAGNOSTICS}.items()]
    return "| Column | Meaning |\n|---|---|\n" + "\n".join(rows) + "\n"


REGIONS = {"reference/settings.md": ("settings", settings), "reference/inputs.md": ("inputs", inputs),
           "reference/outputs.md": ("diagnostics", diagnostics)}


def render(text, name, body):
    pat = re.compile(rf"(<!-- BEGIN GENERATED: {name} -->\n).*?(<!-- END GENERATED: {name} -->)", re.S)
    if not pat.search(text):
        raise SystemExit(f"no `BEGIN GENERATED: {name}` region")
    return pat.sub(lambda m: m.group(1) + "<!-- written by docs/gen_reference.py: edit the code, not this table -->\n\n" + body + "\n" + m.group(2), text)


def main(argv):
    stale = []
    for page, (name, fn) in REGIONS.items():
        path = DOCS / page
        old = path.read_text()
        new = render(old, name, fn())
        if new != old:
            stale.append(page)
            if "--check" not in argv:
                path.write_text(new)
    if "--check" in argv and stale:
        print("stale generated regions (run `python docs/gen_reference.py`): " + ", ".join(stale))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
