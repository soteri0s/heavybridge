#!/usr/bin/env python3
"""
collect_pdm.py -- read the navsim PDM-scorer output CSV for one selection and
emit the aggregated PDMS + sub-scores as JSON (mirroring evaluate.py's schema:
PDMS, NC, DAC, TTC, EP, C). The navsim eval script writes a per-token CSV under
$NAVSIM_EXP_ROOT; we average the rows over the split.

This is the REAL-scorer counterpart to evaluate.py's _aggregate(): no proxy, the
numbers come straight from the devkit's PDMScorer.

Usage:
  python3 collect_pdm.py --csv <navsim_score.csv> --method base \
          --out results/eval_base.json
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os

# === VERIFY against installed navsim API ===
# Column names the navsim PDM scorer writes. The devkit's run_*_pdm_score
# evaluation produces a CSV with one row per token plus an aggregate row; these
# are the documented metric columns. If the installed version renames them, map
# them here (one place).
COLS = {
    "PDMS": ["pdm_score", "score"],
    "NC":   ["no_at_fault_collisions", "no_at_fault_collision"],
    "DAC":  ["drivable_area_compliance"],
    "TTC":  ["time_to_collision_within_bound", "ttc"],
    "EP":   ["ego_progress"],
    "C":    ["comfort"],
}


def _pick(fieldnames, candidates):
    for c in candidates:
        if c in fieldnames:
            return c
    return None


def _resolve_csv(path: str) -> str:
    """Accept a file or a directory/glob; pick the newest matching CSV."""
    if os.path.isfile(path):
        return path
    cands = sorted(glob.glob(path) + glob.glob(os.path.join(path, "**", "*.csv"),
                                               recursive=True),
                   key=lambda p: os.path.getmtime(p) if os.path.exists(p) else 0)
    if not cands:
        raise FileNotFoundError(
            f"no navsim PDM-score CSV found at {path!r}. The eval script did not "
            f"produce output -- check the scoring step log. FAILING LOUDLY."
        )
    return cands[-1]


def aggregate(csv_path: str) -> dict:
    csv_path = _resolve_csv(csv_path)
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    if not rows:
        raise RuntimeError(f"PDM CSV {csv_path} has no rows; scoring produced nothing.")

    colmap = {k: _pick(fieldnames, v) for k, v in COLS.items()}
    missing = [k for k, v in colmap.items() if v is None]
    if "PDMS" in missing:
        raise KeyError(
            f"could not find a PDM score column in {csv_path} (looked for "
            f"{COLS['PDMS']}; saw {fieldnames}). Fix collect_pdm.COLS to match "
            f"the installed navsim CSV header."
        )

    # drop a trailing aggregate/"average" row if the devkit appends one (we
    # average ourselves to be version-agnostic).
    def _is_token_row(r):
        tok = r.get("token") or r.get("scene_token") or ""
        return tok.lower() not in ("", "average", "mean", "aggregate", "total")

    data_rows = [r for r in rows if _is_token_row(r)] or rows

    out = {}
    for k, col in colmap.items():
        if col is None:
            continue
        vals = []
        for r in data_rows:
            try:
                vals.append(float(r[col]))
            except (TypeError, ValueError):
                pass
        out[k] = float(sum(vals) / len(vals)) if vals else None
    out["n_scenes"] = len(data_rows)
    out["_csv"] = csv_path
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True, help="navsim PDM score CSV (file/dir/glob)")
    ap.add_argument("--method", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    agg = aggregate(args.csv)
    agg["method"] = args.method
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(agg, f, indent=2)
    print(json.dumps({k: v for k, v in agg.items() if not k.startswith("_")}, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
