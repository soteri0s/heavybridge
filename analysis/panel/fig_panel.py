#!/usr/bin/env python3
"""fig_panel.pdf — worst-anchor refinements on navmini.
Left: ECDF of the distance (ADE, m) between each planner's refinement of its own
worst-ranked anchor and the expert plan, over the 396 navmini scenes, with the
raw anchor for reference. Right: three scenes, chosen by a declared rule on the
official per-scene worst-anchor PDMS (R2 < 0.5, i.e. a catastrophe as defined
in the paper, and R0, R3g >= 0.9; 106 of the 144 R2 catastrophes qualify) and
for maneuver diversity. Inputs: panel_{R0,R2,R3g}.npz (modes, logits, gt, tokens
from the phase-1 dumps), panel_scene.npz (drivable raster, agents),
kmeans_navsim_traj_20.npy (anchor vocabulary).
BEV raster geometry (matches the calibration): ego at the
top-centre pixel, 0.25 m/px, row = x/0.25 (forward), col = 128 + y/0.25."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # inputs next to this script

BLUE, VERM, GREY, INK = "#0072B2", "#D55E00", "#8a8a8a", "#222222"
PS = 0.25
SCENES = [302, 252, 49]           # slow left turn, right turn, slow straight
PD = np.load("panel_pdms.npz")   # official per-scene PDMS (perscene_matrix.csv.gz)

S = np.load("panel_scene.npz"); V = {v: np.load(f"panel_{v}.npz") for v in ["R0", "R2", "R3g"]}
anc = np.load("../kmeans_navsim_traj_20.npy"); gt = V["R0"]["gt"]; drv = S["drivable"]; agents = S["agents"]

def ade(a, b): return np.linalg.norm(a[..., :2] - b[..., :2], axis=-1).mean(-1)
order = {v: np.argsort(-V[v]["logits"], axis=1) for v in V}
worst = {v: V[v]["modes"][np.arange(396), order[v][:, 19]] for v in V}
d_ref = {v: ade(worst[v], gt) for v in V}
d_raw = ade(anc[order["R2"][:, 19]], gt)

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.linewidth": 0.6, "pdf.fonttype": 42})
fig = plt.figure(figsize=(6.8, 2.45))
gs = fig.add_gridspec(1, 4, width_ratios=[1.25, 1, 1, 1], wspace=0.12, left=0.09, right=0.99, top=0.97, bottom=0.25)

# ---- left: ECDF ---------------------------------------------------------------
ax = fig.add_subplot(gs[0])
def ecdf(x): x = np.sort(x); return x, np.arange(1, len(x) + 1) / len(x)
for x, col, lab, ls, lw in [(d_ref["R0"], GREY, "Gaussian bridge (R0)", "-", 2.8), (d_ref["R3g"], BLUE, "per-step Lévy (R3g)", "-", 1.2),
                            (d_ref["R2"], VERM, "amortized per-rollout (R2)", "-", 1.8), (d_raw, INK, "raw worst anchor", ":", 1.2)]:
    xs, ys = ecdf(x); ax.step(xs, ys, where="post", color=col, lw=lw, ls=ls, label=lab)
ax.set_xscale("log"); ax.set_xlim(0.05, 40); ax.set_ylim(0, 1.02)
ax.set_xlabel("distance of worst-anchor refinement\nto the expert plan (ADE, m)")
ax.set_ylabel("fraction of scenes")
ax.grid(axis="y", color="#dddddd", lw=0.5); ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=6.3, loc="lower right", handlelength=1.8)

# ---- right: scenes ------------------------------------------------------------
def draw_agents(axx, s):
    """agent boxes: metric (x fwd, y left, heading from x toward y) -> plot (y, x)"""
    for cx, cy, hd, L, W in agents[s]:
        if np.isnan(cx): continue
        c, si = np.cos(hd), np.sin(hd)
        corners = np.array([[dx, dy] for dx, dy in [(L/2, W/2), (L/2, -W/2), (-L/2, -W/2), (-L/2, W/2)]])
        xy = np.stack([cx + c*corners[:,0] - si*corners[:,1], cy + si*corners[:,0] + c*corners[:,1]], 1)
        axx.add_patch(Polygon(xy[:, ::-1], closed=True, facecolor="#cfcfcf", edgecolor="#8f8f8f", lw=0.4, zorder=2))

for j, s in enumerate(SCENES):
    axx = fig.add_subplot(gs[j + 1])
    # raster in metric coords: horizontal = y (left positive, so invert), vertical = x (forward)
    img = np.where(drv[s], 0.93, 1.0)
    axx.imshow(img.astype(float), cmap="gray", vmin=0, vmax=1, extent=[-32, 32, 0, 32], origin="lower", zorder=0,
               interpolation="nearest")
    # patches/curves live in (x fwd, y left) metric coordinates -> plot as (y, x)
    draw_agents(axx, s)
    axx.plot(anc[order["R2"][s, 19], :, 1], anc[order["R2"][s, 19], :, 0], color=INK, lw=0.9, ls=":", zorder=3)
    axx.plot(worst["R2"][s, :, 1], worst["R2"][s, :, 0], color=VERM, lw=1.8, marker="o", ms=2.2, zorder=5)
    axx.plot(worst["R0"][s, :, 1], worst["R0"][s, :, 0], color=GREY, lw=3.0, marker="o", ms=3.4, zorder=6)
    axx.plot(worst["R3g"][s, :, 1], worst["R3g"][s, :, 0], color=BLUE, lw=1.3, marker="o", ms=1.8, zorder=7)
    axx.plot(gt[s, :, 1], gt[s, :, 0], color=INK, lw=1.0, ls="--", zorder=8)
    axx.plot(0, 0, marker=(3, 0, 0), color=INK, ms=6, zorder=8)
    axx.set_xlim(20, -20); axx.set_ylim(-2, 30)         # +y (left) drawn to the left
    axx.set_aspect("equal"); axx.set_xticks([]); axx.set_yticks([])
    for sp in axx.spines.values(): sp.set_linewidth(0.5)
    axx.text(0.03, 0.97, f"scene {j + 1}", transform=axx.transAxes, fontsize=7, va="top", ha="left", color=INK)
    axx.set_xlabel(f"R2 {PD['R2_k20'][s]:.2f}   R0 {PD['R0_k20'][s]:.2f}   R3g {PD['R3g_k20'][s]:.2f}", fontsize=6.4, labelpad=2)

fig.savefig("fig_panel.pdf")
print("wrote fig_panel.pdf")
