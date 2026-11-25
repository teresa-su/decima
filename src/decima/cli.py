"""Command line: `decima run stars.fits --out run/ [--config cfg.json] [--columns cols.json] [--workers 4]`.

`decima default-config > cfg.json` prints the default settings to edit. cols.json maps canonical column names to yours,
e.g. {"teff": "TEFF", "Ks": "ks_m"} (see `decima.io`).
"""
import argparse
import json
from pathlib import Path

from decima.pipeline import Config
from decima.runner import run


def main(argv=None):
    ap = argparse.ArgumentParser(prog="decima")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("default-config", help="print the default config.json")
    r = sub.add_parser("run", help="run a catalog (resumable: re-run the same command)")
    r.add_argument("input", help="table of stars (any format astropy reads)")
    r.add_argument("--out", required=True, help="run directory")
    r.add_argument("--config", help="config.json (default: decima default-config)")
    r.add_argument("--columns", help="JSON file mapping canonical column names to the table's")
    r.add_argument("--workers", type=int, default=1)
    r.add_argument("--chunk", type=int, default=250, help="stars per checkpoint")
    r.add_argument("--format", default="fits", help="summary table format (file extension), e.g. fits, ecsv, csv")
    args = ap.parse_args(argv)
    if args.cmd == "default-config":
        print(Config().to_json())
        return
    cfg = Config.from_json(Path(args.config).read_text()) if args.config else Config()
    columns = json.loads(Path(args.columns).read_text()) if args.columns else None
    print(run(args.input, args.out, columns=columns, config=cfg, workers=args.workers, chunk=args.chunk,
              summary_format=args.format))


if __name__ == "__main__":
    main()
