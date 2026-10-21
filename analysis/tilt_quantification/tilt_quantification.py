#!/usr/bin/env python3
"""Quantify the prior-profile approximation to the endpoint-tilted posterior
(appendix subsection sec:Stiltquant of the paper).

Prop. 3(ii): the exact bridge draws the mixing profile lambda_{0:N-1} from the
tilted posterior  pi_N(lambda | x0, y)  \propto  prior(lambda) * w(v(lambda)),
    w(v) = (2*pi*v)^{-D/2} * exp(-r^2 / (2 v)),      v(lambda) = sum_i rho_i lambda_i,
    r^2  = ||y - m_T x0||^2   (normalized bridge coordinates),
while training/planning draw from the PRIOR. This script measures, per scene of
the real corpus, how far the tilted posterior is from the prior:

  * ESS/M      : effective sample size of self-normalized weights over M prior
                 draws (the SNIS diagnostic of App. D.3);
  * TV         : total-variation distance between tilted posterior and prior
                 (= 0.5 * E_prior |w/Z - 1|; exact identity because the weight
                 depends on the profile only through the scalar v);
  * E[v] shift : E_tilt[v] / E_prior[v];
  * max E[lambda_i] shift : worst per-step first-moment ratio.

Inputs (all offline; same code path as the experiments):
  * heavybridge_mixing.py  (precompute + sample_profile: Kanter + tilt rejection)
  * kmeans_navsim_traj_20.npy       -- the public DiffusionDrive anchor vocabulary
  * gt_trajs_navmini.npz            -- expert trajectories extracted from the
                                       phase-1 dumps (396 navmini scenes)

Constants below mirror heavybridge_head.py exactly:
  D = 16 (8 waypoints x 2 normalized coords; the bridge noises [..., :2] only),
  norm_xy: x -> 2(x+1.2)/56.9 - 1,  y -> 2(y+20)/46 - 1,
  VP schedule beta_d=2.0, beta_min=0.1, N=20  =>  m_T = 0.5769, sigma_T^2 = 0.6671.
"""
import numpy as np
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, os.pardir, os.pardir, "experiments_navsim", "runpod"))
os.chdir(_HERE)  # inputs and outputs are next to this script
from heavybridge_mixing import precompute, sample_profile

M = 500_000          # prior draws (shared across scenes)
SEED = 0
CALIBS = [("gentle (R3g, deployed)", 1.8, 0.956338),
          ("aggressive (R3t)",       1.7, 0.0285)]

def norm_xy(p):
    x = 2 * (p[..., 0:1] + 1.2) / 56.9 - 1
    y = 2 * (p[..., 1:2] + 20.0) / 46.0 - 1
    return np.concatenate([x, y], axis=-1)

def main(gt_path="gt_trajs_navmini.npz", anchor_path="../kmeans_navsim_traj_20.npy"):
    gt = np.load(gt_path)["gt"][..., :2]          # (S,8,2) metric ego frame
    anc = np.load(anchor_path)                    # (20,8,2) metric ego frame
    x0, y = norm_xy(gt), norm_xy(anc)
    S, D = len(x0), x0.shape[1] * x0.shape[2]

    rng = np.random.default_rng(SEED)
    report = {}
    for name, alpha, kappa in CALIBS:
        tab = precompute(np.linspace(0.0, 1.0, 21), alpha=alpha)
        mT, sigT2 = tab.m[-1], tab.sig2[-1]

        # residuals r^2 per scene: nearest anchor (the loss-matched mode) and
        # the worst case over the whole 20-anchor vocabulary (every anchor's
        # bridge is noised in training).
        diff = y[None] - mT * x0[:, None]                     # (S,20,8,2)
        r2_all = (diff ** 2).sum(axis=(2, 3))                 # (S,20)
        near = ((y[None] - x0[:, None]) ** 2).sum(axis=(2, 3)).argmin(axis=1)
        r2_near = r2_all[np.arange(S), near]

        lam = sample_profile(tab, M, rng, kappa=kappa)        # (M,N) prior draws
        v = lam @ tab.rho                                     # (M,)
        logv = np.log(v)
        Elam = lam.mean(axis=0)                               # prior per-step mean

        def stats(r2):
            logw = -0.5 * D * logv - 0.5 * r2 / v
            logw -= logw.max()
            w = np.exp(logw); w /= w.sum()
            ess = 1.0 / np.sum(w ** 2) / M
            tv = 0.5 * np.abs(w - 1.0 / M).sum()
            ev = np.sum(w * v) / v.mean()
            eshift = np.abs(np.sum(w[:, None] * lam, axis=0) / Elam - 1).max()
            return ess, tv, ev, eshift

        per_scene = np.array([stats(r) for r in r2_near])     # (S,4)
        worst_anchor = stats(r2_all.max())                    # single worst case
        report[name] = dict(
            mT=mT, sigT2=sigT2, D=D,
            r2_near=r2_near, r2_all=r2_all, per_scene=per_scene,
            worst=worst_anchor,
            v_relsd=v.std() / v.mean(),
        )

        u_near = r2_near / (D * sigT2)
        u_all = r2_all / (D * sigT2)
        ess, tv, ev, esh = per_scene.T
        print(f"\n=== {name}: alpha={alpha}, kappa={kappa} "
              f"(D={D}, m_T={mT:.4f}, sigma_T^2={sigT2:.4f}, M={M}) ===")
        print(f"  v(lambda) prior rel-sd: {v.std()/v.mean():.3f}")
        print(f"  r^2/(D sigma_T^2): nearest-anchor median {np.median(u_near):.3f} "
              f"(range {u_near.min():.3f}-{u_near.max():.3f}); "
              f"max over all 20 anchors {u_all.max():.3f}")
        print(f"  ESS/M      : median {np.median(ess):.4f}  min {ess.min():.4f}")
        print(f"  TV(tilt,prior): median {np.median(tv):.4f}  max {tv.max():.4f}")
        print(f"  E_tilt[v]/E_prior[v]: range {ev.min():.4f}-{ev.max():.4f}")
        print(f"  max |E_tilt[lam_i]/E_prior[lam_i]-1|: median {np.median(esh)*100:.2f}%  "
              f"max {esh.max()*100:.2f}%")
        we, wt, wv, ws = worst_anchor
        print(f"  worst anchor in vocabulary: ESS/M {we:.4f}, TV {wt:.4f}, "
              f"E[v] ratio {wv:.4f}, max step shift {ws*100:.2f}%")

        # Uniformity of the tilt across steps at the median residual, and the
        # equivalent jump-energy recalibration ("effective varpi") it implies.
        r2m = np.median(r2_near)
        logw = -0.5 * D * logv - 0.5 * r2m / v
        logw -= logw.max()
        w = np.exp(logw); w /= w.sum()
        ratio1 = np.sum(w[:, None] * lam, axis=0) / Elam
        ratio2 = np.sum(w[:, None] * lam ** 2, axis=0) / (lam ** 2).mean(axis=0)
        gJ = {1.8: 1.25, 1.7: 5.0 / 3.0}[alpha]
        ev_med = np.sum(w * v) / v.mean()
        print(f"  per-step E[lam_i] ratio (median scene): "
              f"{ratio1.min():.4f}-{ratio1.max():.4f} (mean {ratio1.mean():.4f}, "
              f"sd {ratio1.std():.4f}); E[lam_i^2] ratio {ratio2.min():.4f}-{ratio2.max():.4f}")
        print(f"  effective jump-energy factor: varpi {gJ:.4f} -> "
              f"~{gJ * ev_med:.4f} under the tilt")
    return report

if __name__ == "__main__":
    main()
