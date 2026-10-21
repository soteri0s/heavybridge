# HeavyBridge: Tempered Lévy Bridges for Anchored Diffusion Driving Policies

Code, per-scene results and pre-registration documents for

> Sotirios Chatzis and Loukas Papadoulas. **HeavyBridge: Tempered Lévy Bridges for Anchored Diffusion Driving Policies.** *Transactions on Machine Learning Research*, 2026. [OpenReview](https://openreview.net/forum?id=5p1S532fXM)

Anchored diffusion planners first pick one of K anchor trajectories and then refine it with a Gaussian denoising bridge. HeavyBridge replaces the Gaussian bridge noise with a per-step tempered α-stable scale mixture. The forward chain is exactly an α-stable Lévy bridge on any time grid, and all standard bridge machinery (training, fast samplers, warm starts) carries over. On NAVSIM the per-step Lévy planner scores the same whichever anchor it is handed, at no cost at the deployed argmax. A per-rollout heavy tail, by contrast, is provably Gaussian in its deterministic dynamics and, when its scale is learned per scene, collapses under the forced-anchor stress test.

Checkpoints are on Hugging Face: **[huggingface.co/soteri0s/heavybridge](https://huggingface.co/soteri0s/heavybridge)**. The raw selection files and training logs are available from the authors on request.

## Repository layout

```
experiments_navsim/
  runpod/heavybridge_mixing.py   the method: schedule table, closed-form bridge coefficients,
                                 Inverse-Gamma / alpha-stable / tempered-stable samplers (numpy)
  runpod/heavybridge_head.py     drop-in replacement for DiffusionDrive's TrajectoryHead (torch):
                                 bridge forward process, mixing law, noise-level FiLM, DBIM sampler
  runpod/apply_patch.py          swaps the head into DiffusionDrive (selected by HB_MIXING)
  runpod/train_stage.sh          training; the VARIANT -> mixing-law table is defined here
  runpod/remote_run.sh           full pipeline on one GPU pod: setup, data, train, dump, scoring
  runpod/run_dump.py             one inference pass per scene -> per-scene .npz dumps
  runpod/select_base.py          deployed plan (argmax anchor) from the dumps
  runpod/select_forced_anchor.py forced-anchor stress test: the k-th ranked anchor, k = 1..20
  runpod/navsim_agent.py         DumpAgent / ReplayAgent (CPU-only PDM scoring of selections)
  runpod/navtest_*.sh, phase*.sh, controls_m3J8.sh, ablation_alpha.sh
                                 the drivers that produced the paper's numbers (see below)
  runpod/runpod_session/         parallel NAVSIM data download helpers
verification/                    numerical checks of the closed-form claims (appendix "Numerical verification")
analysis/                        seed bootstrap, calibration check, tilt quantification, figure scripts
results/                         per-scene PDMS for every reported evaluation (see results/README.md)
docs/                            pre-registered decision rules and predictions
```

## Variants

The code uses the variant ids of the paper.

| id | mixing law | role in the paper |
|---|---|---|
| R0 | none (Gaussian bridge) | baseline |
| R1a / R1b | per-rollout Inverse-Gamma scale, ν = 5 / 10 | fixed-scale per-rollout tail |
| R2 | per-rollout scale from a scene-amortized head | the collapse (Findings 2, 2b) |
| R2b | R2 with the scale bounded, m(z) ∈ [1/4, 4] | control (Finding 2b) |
| R2c0 | the R2 checkpoint evaluated with churn off (η = 0) | control (Finding 4) |
| R3t | per-step tempered stable, (α, ϖ) = (1.7, 5/3) | aggressive calibration |
| **R3g** | **per-step tempered stable, (α, ϖ) = (1.8, 5/4), κ = 0.956338** | **the proposed planner** |
| R3k, R23 | R3 with a step ramp / an amortized jump intensity | screening |
| R3g2, R3gk, R23g | R3g′, R3g*k*, R23′ in the paper | navtest sweep around R3g |
| R3ga7, R3ga6 | α = 1.7 / 1.6 at ϖ = 5/4 | α ablation |
| R0full, R3gfull (+ `_s2`) | R0-full, R3g-full, seeds 1 and 2 | confirmation, 100-epoch schedule |

## Quick checks (CPU, no data)

```bash
pip install numpy scipy matplotlib
python verification/verify_all.py          # the closed-form claims of the appendix
python analysis/seeds_analysis.py          # seed-pooled paired bootstrap on navtest
python analysis/calibration_mc_check.py    # jump-energy calibration of every tempered variant
python analysis/tilt_quantification/tilt_quantification.py
```

`seeds_analysis.py` reproduces the headline statistic from the released per-scene scores: pooled over the two seeds, R3g-full minus R0-full at the worst anchor (k = 20) is +0.157 PDMS, 95% CI [+0.030, +0.286].

## Reproducing the NAVSIM experiments

Requirements: a CUDA GPU (we used RTX 4090 and L40S pods on RunPod), Python 3.9, the [DiffusionDrive](https://github.com/hustvl/DiffusionDrive) fork of the [NAVSIM](https://github.com/autonomousvision/navsim) devkit, and access to the gated OpenScene/nuPlan data on Hugging Face. `remote_run.sh` installs the devkit and its dependencies, downloads maps and the split, and fetches the public DiffusionDrive checkpoint that warm-starts every variant.

```bash
cd experiments_navsim/runpod
cp config.env.example config.env      # set HF_TOKEN, VARIANT, storage paths
bash launch.sh --host <pod-ip> --port <ssh-port>           # one variant: train -> dump -> score
```

The drivers that produced the paper's tables run on the pod and call `remote_run.sh` per variant:

| driver | what it runs |
|---|---|
| `phase1.sh`, `phase2_forced.sh`, `phase3.sh` | navmini screening (15-epoch recipe) and forced-anchor sweeps |
| `navtest_confirm.sh`, `navtest_train3.sh`, `navtest_filt.sh` | held-out navtest replication, sweep variants, feasibility filter |
| `navtest_full.sh`, `navtest_seeds.sh` | confirmation phase: 100-epoch R0-full / R3g-full, two seeds |
| `controls_m3J8.sh`, `navtest_controls.sh` | collapse-mechanism controls R2b and R2c0 |
| `ablation_alpha.sh` | α ablation at fixed jump energy |

Every reported score is a single evaluation pass: one checkpoint, one plan per scene, scored once by the official NAVSIM PDM scorer. The dump stores all K = 20 refined anchors per scene, so the argmax plan and every forced-anchor rank are scored from the same inference pass on CPU. The full study (fifteen trainings, about 250 scoring passes) cost under $300 of commodity-GPU time.

**About this release.** These are the scripts that produced the paper's numbers. For the release we translated comments, removed code paths that no reported number uses, and replaced pod-specific settings with `config.env.example`. The CPU-side code (`verification/`, `analysis/`, the selection and summary scripts) was re-run after this cleaning; the GPU pipeline was not re-run end to end. Please open an issue if something does not run.

## Citation

```bibtex
@article{chatzis2026heavybridge,
  title   = {HeavyBridge: Tempered L{\'e}vy Bridges for Anchored Diffusion Driving Policies},
  author  = {Chatzis, Sotirios and Papadoulas, Loukas},
  journal = {Transactions on Machine Learning Research},
  year    = {2026},
  url     = {https://openreview.net/forum?id=5p1S532fXM}
}
```

## License and acknowledgments

Code: MIT (see `LICENSE`). HeavyBridge builds on [DiffusionDrive](https://github.com/hustvl/DiffusionDrive) (MIT) and the [NAVSIM](https://github.com/autonomousvision/navsim) devkit (Apache-2.0); `analysis/kmeans_navsim_traj_20.npy` is DiffusionDrive's public anchor vocabulary. The per-scene results and the trajectory extracts in `analysis/` are derived from NAVSIM/OpenScene/nuPlan data and are subject to those datasets' licenses.

This work was funded by the European Union under the HIDDEN project, grant agreement No 101202228. Views and opinions expressed are, however, those of the authors only and do not necessarily reflect those of the European Union. Neither the European Union nor the granting authority can be held responsible for them.
