#!/usr/bin/env python3
"""
make_summary.py -- assemble results/summary.md for one variant from the JSON the
pipeline produced (eval_base.json, forced/aux_metrics.json). Pure stdlib. A
missing input is printed as "(missing)", never as a plausible number.

Usage:
  python3 make_summary.py --results results --out results/summary.md \
          --split navtest --variant R3gfull
"""
from __future__ import annotations

import argparse
import glob
import json
import os


def _load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        return {"_error": str(e)}


def _fmt(v, nd=3):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="navtest")
    ap.add_argument("--variant", default="?")
    args = ap.parse_args()

    rd = args.results
    parts = [f"# HeavyBridge -- variant {args.variant} on {args.split}\n",
             "Real navsim PDM scorer. The plan is the argmax-logit anchor; "
             "forced-anchor selections are in forced/ and are scored separately "
             "(see the navtest_*.sh drivers).\n",
             "| selection | PDMS | NC | DAC | TTC | EP | C | n |",
             "|---|---|---|---|---|---|---|---|"]
    p = os.path.join(rd, "eval_base.json")
    if os.path.exists(p):
        d = _load_json(p)
        parts.append("| argmax | {} | {} | {} | {} | {} | {} | {} |".format(
            _fmt(d.get("PDMS"), 2), _fmt(d.get("NC")), _fmt(d.get("DAC")),
            _fmt(d.get("TTC")), _fmt(d.get("EP")), _fmt(d.get("C")),
            d.get("n_scenes", "?")))
    else:
        parts.append("| argmax | (missing: no eval_base.json) | | | | | | |")

    aux = os.path.join(rd, "forced", "aux_metrics.json")
    sels = sorted(glob.glob(os.path.join(rd, "forced", "selection_k*.json")))
    parts.append("\n## Forced-anchor selections\n")
    parts.append(", ".join(os.path.basename(s) for s in sels) or "_(none)_")
    if os.path.exists(aux):
        parts.append("\n```json\n" + json.dumps(_load_json(aux), indent=2) + "\n```\n")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(parts) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
