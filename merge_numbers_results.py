#!/usr/bin/env python3
"""One-off: fold the old per-range results_numbers_<LO>-<HI>.json files into results_numbers.json.

eval.py used to write a separate results file for each non-default --range; it now
writes every range to results_numbers.json (rows differ by eval_config). Rows
already in results_numbers.json win over duplicates from the old files. Merged
files are renamed to *.json.merged (not deleted); every row gets a ``range``
field and the distribution plots are regenerated.

    python merge_numbers_results.py [--results-dir results] [--dry-run]
"""

import argparse
import glob
import json
import os

from tasks.numbers import plot_distributions, row_range
from utils import load_json


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--dry-run", action="store_true", help="report what would change, write nothing")
    args = ap.parse_args()

    out = os.path.join(args.results_dir, "results_numbers.json")
    old_files = sorted(glob.glob(os.path.join(args.results_dir, "results_numbers_*.json")))
    if not old_files:
        print("No per-range results_numbers_*.json files found; nothing to merge.")

    by_key = {r["eval_config"]: r for r in load_json(out, [])}
    print(f"{out}: {len(by_key)} row(s)")
    for path in old_files:
        rows = load_json(path, [])
        new = [r for r in rows if r["eval_config"] not in by_key]
        print(f"{path}: {len(rows)} row(s), {len(new)} new, {len(rows) - len(new)} already present")
        for r in new:
            by_key[r["eval_config"]] = r

    for r in by_key.values():
        r["range"] = row_range(r)

    if args.dry_run:
        print(f"Dry run: would write {len(by_key)} row(s) to {out}.")
        return

    with open(out, "w") as f:
        json.dump(list(by_key.values()), f, indent=2)
    for path in old_files:
        os.rename(path, path + ".merged")
    print(f"Wrote {len(by_key)} row(s) to {out}; renamed {len(old_files)} old file(s) to *.json.merged.")

    plot_dir = os.path.join(args.results_dir, "plots", "numbers")
    paths = plot_distributions(list(by_key.values()), plot_dir)
    print(f"{len(paths)} number distribution plot(s) written to {plot_dir}/<range>/")


if __name__ == "__main__":
    main()
