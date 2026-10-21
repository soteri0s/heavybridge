#!/usr/bin/env python3
"""
select_base.py -- CPU-only: turn the per-scene dumps into the selection map of
the DEPLOYED plan { token: (T,3) trajectory of the argmax-logit anchor },
written as JSON for ReplayAgent to score with the navsim PDM scorer.

Usage:
  python3 select_base.py --dump-dir dumps --out selections [--name base]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXP = os.path.dirname(_HERE)
if _EXP not in sys.path:
    sys.path.insert(0, _EXP)

import utils                       # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument("--out", required=True, help="output dir for the selection JSON")
    ap.add_argument("--name", default="base")
    args = ap.parse_args()

    paths = utils.list_dumps(args.dump_dir)
    if not paths:
        raise FileNotFoundError(f"no .npz dumps in {args.dump_dir}; run run_dump.py first.")
    sel = {}
    for p in paths:
        d = utils.load_dump(p)
        k = int(np.argmax(np.asarray(d["logits"], dtype=float)))
        token = str(d.get("scene_token", os.path.splitext(os.path.basename(p))[0]))
        sel[token] = np.asarray(d["modes"][k], dtype=float).tolist()

    os.makedirs(args.out, exist_ok=True)
    dst = os.path.join(args.out, f"{args.name}.json")
    with open(dst, "w") as f:
        json.dump(sel, f)
    print(f"[select] {args.name}: {len(sel)} tokens -> {dst}")


if __name__ == "__main__":
    main()
