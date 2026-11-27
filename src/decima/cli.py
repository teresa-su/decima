"""Command line. Two small JSON files steer a run, and they are different things:

    columns file   your column names -> DECIMA's, e.g. {"teff": "TEFF", "Ks": "ks_m"}   `decima columns stars.fits -o cols.json`
    settings file  every setting that changes the result (a `decima.Config`)            `decima default-config > settings.json`

    decima columns stars.fits -o cols.json                      suggest the columns file
    decima check stars.fits --columns cols.json                 check table, columns and settings before running
    decima run stars.fits --out run/ --columns cols.json        run (resumable: re-run the same command)
    decima download-models                                      fetch the models now (e.g. on a login node)
"""
import argparse
import json
import sys
from pathlib import Path

import decima
from decima.io import check_inputs, suggest_columns
from decima.models import REGISTRY, fetch
from decima.pipeline import Config
from decima.runner import run


def main(argv=None):
    ap = argparse.ArgumentParser(prog="decima", description="Stellar ages, distances and extinctions with DECIMA. "
                                 "A run takes a table of stars, a columns file (your column names -> DECIMA's) and optionally a settings file.")
    ap.add_argument("--version", action="version", version=f"decima {decima.__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("columns", help="suggest the columns file for a table of stars",
                       description="Match your table's column names to DECIMA's and print the columns file (your names for DECIMA's). Check the result with `decima check`.")
    c.add_argument("input", help="table of stars (any format astropy reads)")
    c.add_argument("-o", "--output", help="write the columns file here (default: print it to stdout)")

    k = sub.add_parser("check", help="check a table, columns file and settings before running (needs no models)")
    k.add_argument("input", help="table of stars (any format astropy reads)")
    k.add_argument("--columns", help="columns file: JSON mapping DECIMA's column names to your table's")
    k.add_argument("--config", help="settings file: JSON written by `decima default-config` (default: the default settings)")

    sub.add_parser("download-models", help="download the trained models (about 0.25 GB) now and print where they are",
                   description="Download the models before a run, e.g. on a login node when the compute nodes have no "
                               "internet. They go to $DECIMA_DATA_DIR if set, else the user cache directory.")

    sub.add_parser("default-config", help="print the default settings file, to save, edit and pass to --config")

    r = sub.add_parser("run", help="run a catalogue (resumable: re-run the same command)")
    r.add_argument("input", help="table of stars (any format astropy reads)")
    r.add_argument("--out", required=True, help="run directory")
    r.add_argument("--config", help="settings file: JSON written by `decima default-config` (default: the default settings)")
    r.add_argument("--columns", help="columns file: JSON mapping DECIMA's column names to your table's (see `decima columns`)")
    r.add_argument("--workers", type=int, default=1, help="worker processes (default 1)")
    r.add_argument("--chunk", type=int, default=250, help="stars per checkpoint (default 250); changing it changes the random draws")
    r.add_argument("--format", default="fits", help="summary table format (file extension), e.g. fits, ecsv, csv")
    args = ap.parse_args(argv)

    if args.cmd == "download-models":
        for name in REGISTRY:
            print(fetch(name))
        return 0
    if args.cmd == "default-config":
        print(Config().to_json())
        return 0
    if args.cmd == "columns":
        sug = suggest_columns(args.input)
        print(sug, file=sys.stderr)
        for c in sug.ambiguous:
            print(f"?? {c}: choose one of {sug.ambiguous[c]} and add it to the columns file by hand", file=sys.stderr)
        if sug.unused:
            print(f"not used: {sug.unused}", file=sys.stderr)
        text = json.dumps(dict(sug), indent=2)
        if args.output:
            Path(args.output).write_text(text + "\n")
            print(f"wrote {args.output}. Next: decima check {args.input} --columns {args.output}", file=sys.stderr)
        else:
            print(text)
        return 1 if sug.missing else 0
    if args.cmd == "check":
        columns = json.loads(Path(args.columns).read_text()) if args.columns else None
        rep = check_inputs(args.input, columns, args.config)
        print(rep)
        return 0 if rep.ok else 1

    cfg = Config.from_json(Path(args.config).read_text()) if args.config else Config()
    columns = json.loads(Path(args.columns).read_text()) if args.columns else None
    print(run(args.input, args.out, columns=columns, config=cfg, workers=args.workers, chunk=args.chunk,
              summary_format=args.format))
    return 0


if __name__ == "__main__":
    sys.exit(main())
