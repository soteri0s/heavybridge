# Analysis scripts

All scripts run on CPU from the released files; none needs NAVSIM.

| script | what it computes | paper |
|---|---|---|
| `seeds_analysis.py` | per-seed PDMS, seed spread, per-seed and seed-pooled paired bootstrap (scene resampling, 20,000 draws) of R3g-full minus R0-full at k = 1, 2, 20, from `results/navtest/` | full-schedule confirmation table and its bootstrap table |
| `calibration_mc_check.py` | Monte-Carlo check that each tempered variant realizes its target jump-energy factor | "Choosing alpha and gamma_J", calibration grid |
| `tilt_quantification/tilt_quantification.py` | distance between the endpoint-tilted posterior of the mixing profile and the prior used in training (ESS/M, TV, moment shifts) on the 396 navmini expert trajectories | appendix "tilt quantification" |
| `tilt_quantification/fig_tilt.py` | the corresponding appendix figure | Fig. tilt |
| `panel/fig_panel.py` | worst-anchor refinements on navmini (ECDF of ADE to the expert plan, three example scenes) | Fig. panel |

`kmeans_navsim_traj_20.npy` is DiffusionDrive's public K = 20 anchor vocabulary.
`tilt_quantification/gt_trajs_navmini.npz` and `panel/*.npz` are extracts of the
navmini dumps (expert trajectories, refined modes, logits, drivable raster and
agents).

Expected outputs, from a run on the released files:

- `seeds_analysis.py`: pooled k = 20 difference +0.157 PDMS, 95% CI [+0.030, +0.286], one-sided p = 0.0075.
- `tilt_quantification.py`, deployed calibration (alpha = 1.8, kappa = 0.956338): ESS/M median 0.774 (min 0.761), TV median 0.213, maximum per-step first-moment shift 6.5%, effective jump-energy factor 1.25 -> 1.20.
