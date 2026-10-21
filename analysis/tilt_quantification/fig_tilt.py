#!/usr/bin/env python3
"""fig_tilt.pdf — appendix figure for the tilted-posterior quantification.
Left: ESS/M of prior-draw SNIS weights vs the dimensionless residual
u = r^2/(D sigma_T^2). Right: tilted-vs-prior jump-energy ratio E_tilt[v]/E_prior[v].
Shaded band: the empirical nearest-anchor residual range of the training corpus;
dotted marker: the largest residual over the whole 20-anchor vocabulary."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, os.pardir, os.pardir, "experiments_navsim", "runpod"))
os.chdir(_HERE)  # inputs and outputs are next to this script
from heavybridge_mixing import precompute, sample_profile

BLUE, VERM, GREY = "#0072B2", "#D55E00", "#999999"
D, M = 16, 400_000
rng = np.random.default_rng(0)

def curves(alpha, kappa, us, sigT2):
    tab = precompute(np.linspace(0, 1, 21), alpha=alpha)
    lam = sample_profile(tab, M, rng, kappa=kappa)
    v = lam @ tab.rho
    logv = np.log(v)
    ess, ev = [], []
    for u in us:
        r2 = u * D * sigT2
        logw = -0.5 * D * logv - 0.5 * r2 / v
        logw -= logw.max()
        w = np.exp(logw); w /= w.sum()
        ess.append(1.0 / np.sum(w ** 2) / M)
        ev.append(np.sum(w * v) / v.mean())
    return np.array(ess), np.array(ev)

tab18 = precompute(np.linspace(0, 1, 21), alpha=1.8)
sigT2 = tab18.sig2[-1]
us = np.linspace(0.0, 1.2, 61)
ess_g, ev_g = curves(1.8, 0.956338, us, sigT2)
ess_a, ev_a = curves(1.7, 0.0285, us, sigT2)

# empirical corpus markers (computed by tilt_quantification.py)
U_NEAR_LO, U_NEAR_HI, U_VOCAB_MAX = 0.023, 0.126, 0.805

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.labelsize": 9,
    "axes.linewidth": 0.6, "xtick.direction": "out", "ytick.direction": "out",
    "pdf.fonttype": 42,
})
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.35), constrained_layout=True)

for ax, (yg, ya), ylab, ident in zip(
        axes, [(ess_g, ess_a), (ev_g, ev_a)],
        [r"ESS$/M$ of prior-draw weights",
         r"$\mathbb{E}_{\mathrm{tilt}}[v]\,/\,\mathbb{E}_{\mathrm{prior}}[v]$"],
        ["a", "b"]):
    ax.axvspan(U_NEAR_LO, U_NEAR_HI, color=GREY, alpha=0.25, lw=0, zorder=0)
    ax.axvline(U_VOCAB_MAX, color=GREY, lw=0.9, ls=":", zorder=0)
    ax.plot(us, yg, color=BLUE, lw=2.0, solid_capstyle="round", zorder=3)
    ax.plot(us, ya, color=VERM, lw=2.0, solid_capstyle="round", zorder=3)
    ax.set_xlabel(r"$u \;=\; r^2 / (D\,\sigma_T^2)$")
    ax.set_ylabel(ylab)
    ax.set_xlim(0, 1.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#dddddd", lw=0.5, zorder=-5)
    ax.set_axisbelow(True)

# direct labels
axes[0].set_ylim(0, 1.02)
axes[0].text(0.42, 0.86, r"gentle ($\alpha{=}1.8$, deployed)", color=BLUE, fontsize=8)
axes[0].text(0.42, 0.36, r"aggressive ($\alpha{=}1.7$)", color=VERM, fontsize=8)
axes[0].text(0.075, 0.06, "corpus,\nnearest\nanchor", color="#555555",
             fontsize=6.5, ha="center")
axes[0].text(U_VOCAB_MAX + 0.02, 0.06, "worst anchor\nin vocabulary",
             color="#555555", fontsize=6.5, ha="left")
axes[1].set_ylim(0.55, 1.45)
axes[1].axhline(1.0, color="#bbbbbb", lw=0.7)
axes[1].text(0.30, 1.04, r"gentle", color=BLUE, fontsize=8)
axes[1].text(0.30, 0.62, r"aggressive", color=VERM, fontsize=8)

fig.savefig("fig_tilt.pdf")
print("wrote fig_tilt.pdf")
