#!/usr/bin/env python3
"""Monte-Carlo check of the jump-energy calibration (paper: "Choosing alpha and
gamma_J"). For each calibration (alpha, target varpi, kappa), draws tempered
per-step profiles lambda_{0:N-1} with the released sampler and compares the
realized jump-energy factor  varpi_MC = sum_i E[lambda_i] u_i / sum_i u_i  with
its target. Prints |diff| next to the Monte-Carlo standard error.

    python3 calibration_mc_check.py            # paper sample sizes
    python3 calibration_mc_check.py --fast     # 10x fewer draws
"""
import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, os.pardir, "experiments_navsim", "runpod"))
from heavybridge_mixing import precompute, sample_profile  # noqa: E402

# name: (alpha, target varpi, kappa)
CALIBS = {
    "R3t  (aggressive)": (1.7, 5 / 3, 0.0285),
    "R3g  (deployed)":   (1.8, 5 / 4, 0.956338),
    "R3g' (deeper)":     (1.8, 10 / 9, 3.105537),
    "alpha=1.7 @ 5/4":   (1.7, 5 / 4, 0.193775),
    "alpha=1.6 @ 5/4":   (1.6, 5 / 4, 0.085732),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2_000_000)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    n_total = args.n // 10 if args.fast else args.n
    rng = np.random.default_rng(args.seed)
    ts = np.linspace(0.0, 1.0, 21)
    for name, (alpha, target, kappa) in CALIBS.items():
        tab = precompute(ts, alpha=alpha)
        tot = np.zeros(len(tab.u)); sq = np.zeros(len(tab.u)); n = 0
        while n < n_total:
            b = min(100_000, n_total - n)
            lam = sample_profile(tab, b, rng, kappa=kappa, max_tries=2048)
            tot += lam.sum(axis=0); sq += (lam ** 2).sum(axis=0); n += b
        mean = tot / n; var = sq / n - mean ** 2
        g = float(np.sum(mean * tab.u) / np.sum(tab.u))
        se = float(np.sqrt(np.sum(var * tab.u ** 2)) / np.sum(tab.u) / np.sqrt(n))
        print(f"{name:18s} alpha={alpha} kappa={kappa:<9g} n={n:.1e}  "
              f"varpi_MC={g:.5f}  target={target:.5f}  |diff|={abs(g - target):.1e}  (SE {se:.1e})")


if __name__ == "__main__":
    main()
