"""Forced-anchor selections from the per-scene dumps (the anchor stress test).

Reads the per-scene npz dumps (keys: modes (K,8,3), logits (K,), ...) and
writes one selection JSON per rank k in --ks: for every scene, the refined
trajectory of the k-th ranked anchor (k=1 is the deployed argmax plan). The
navsim PDM scorer then scores each JSON on CPU through ReplayAgent.

Also writes aux_metrics.json with the kinematic-feasibility violation rate
over all K modes (|accel| > 8 m/s^2 or |jerk| > 15 m/s^3, dt = 0.5 s), and,
with --feasibility-filter, ranks only feasible modes (falling back to the
unfiltered ranking when none passes).
"""
import argparse, glob, itertools, json, os
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", required=True, help="dir of per-scene .npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ks", default="1 2 3 5")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--feasibility-filter", action="store_true",
                    help="rank only kinematically feasible modes (accel/jerk); "
                         "falls back to the unfiltered ranking if none pass.")
    args = ap.parse_args()
    ks = [int(k) for k in args.ks.split()]
    os.makedirs(args.out, exist_ok=True)

    sel = {k: {} for k in ks}
    div_vals, feas_bad, feas_tot = [], 0, 0
    for f in sorted(glob.glob(os.path.join(args.dumps, "*.npz"))):
        d = np.load(f, allow_pickle=True)
        token = str(d["scene_token"])
        order = np.argsort(-d["logits"])                      # rank anchors
        if args.feasibility_filter:
            def _feas(tr):
                v = np.diff(tr[:, :2], axis=0) / 0.5
                a = np.diff(v, axis=0) / 0.5
                j = np.diff(a, axis=0) / 0.5
                return (np.linalg.norm(a, axis=-1) <= 8.0).all() and \
                       (np.linalg.norm(j, axis=-1) <= 15.0).all()
            forder = [i for i in order if _feas(d["modes"][i])]
            if forder:
                order = np.asarray(forder)
        for k in ks:
            # ReplayAgent-compatible selection: the full k-th-ranked trajectory.
            idx = int(order[min(k - 1, len(order) - 1)])
            sel[k][token] = np.asarray(d["modes"][idx], dtype=float).tolist()
        # optional diversity metric: pairwise ADE over repeated samples of the
        # top mode, only if a dump carries them (key modes_mc)
        if "modes_mc" in d.files:
            mc = d["modes_mc"][order[0]]                      # (S,8,3)
            div_vals.append(float(np.mean([
                np.mean(np.linalg.norm(a[:, :2] - b[:, :2], axis=-1))
                for a, b in itertools.combinations(mc, 2)])))
        # feasibility on all modes (accel/jerk, dt=0.5)
        for tr in d["modes"]:
            v = np.diff(tr[:, :2], axis=0) / 0.5
            a = np.diff(v, axis=0) / 0.5
            j = np.diff(a, axis=0) / 0.5
            feas_tot += 1
            feas_bad += int((np.linalg.norm(a, axis=-1) > 8.0).any()
                            or (np.linalg.norm(j, axis=-1) > 15.0).any())

    for k in ks:
        with open(os.path.join(args.out, f"selection_k{k}.json"), "w") as fh:
            json.dump(sel[k], fh)
    with open(os.path.join(args.out, "aux_metrics.json"), "w") as fh:
        json.dump({"run_id": args.run_id,
                   "within_anchor_diversity": float(np.mean(div_vals)) if div_vals else None,
                   "feasibility_violation_rate": feas_bad / max(feas_tot, 1)}, fh)
    print(f"[forced-anchor] wrote selections for k={ks}; "
          f"diversity={(np.mean(div_vals) if div_vals else float('nan')):.3f}, "
          f"feas_viol={feas_bad/max(feas_tot,1):.4f}")


if __name__ == "__main__":
    main()
