# Headline-Decision Experiment (NAVSIM, RunPod)

**Goal.** Decide early which algorithm is the paper's headline — Alg. 1 (fixed-ν
Student-t), Alg. 2 (amortized/learnable ν), or Alg. 3 (per-step Lévy, tempered) —
so the prose is built around the winner. Bench2Drive full protocol runs only for
the chosen headline + baseline afterwards.

**Why NAVSIM.** Open-loop, pipeline already proven on RunPod, cheap. Raw PDMS on
navtest is saturated (~88) and will NOT discriminate; the *decision axes* below will.

---

## Runs (Phase 1) — 2–4 pods in parallel, identical protocol

| id  | mixing          | flags                          | pod |
|-----|-----------------|--------------------------------|-----|
| R0  | none (baseline) | `--mixing none`                | A   |
| R1a | Alg.1           | `--mixing ig --nu 5`           | A   |
| R1b | Alg.1           | `--mixing ig --nu 10`          | B   |
| R2  | Alg.2           | `--mixing ig-learned`          | B   |
| R3t | Alg.3 tempered  | `--mixing tempered --alpha 1.7 --kappa <fit>` | C |
| R3s | Alg.3 stable    | `--mixing stable --alpha 1.9`  | C (optional) |

Phase 0 gate: R0 must reproduce ≈87.5–88 PDMS before anything else runs.
κ for R3t: fit so that Var of the mixed increment matches R1a's ν=5 variance
(`kappa_from_nu()` helper in eval_decision.py) — keeps R1a/R3t comparable.

## Decision metrics (Phase 2, inference-only, per checkpoint)

1. **Forced-anchor PDMS** at k ∈ {1,2,3,5}: replace argmax anchor with k-th
   best. Primary discriminator (open-loop analogue of H1).
2. **Feasibility rate**: fraction of samples violating kinematic/comfort
   limits (catches untempered-A3 pathology).
3. **Within-anchor diversity**: mean pairwise ADE over 20 samples.
4. **Churn-off contrast** (same checkpoint, ς=0): must collapse R1/R2 to R0
   (Prop. 3 negative control); if it doesn't, the implementation is wrong.

## Pre-registered decision rule

- **Headline = Alg. 3** iff R3t beats best-of(R1a,R1b) on forced-anchor
  PDMS at k≥2 by >0.5 PDMS AND feasibility within 1% of R0.
- else **Headline = Alg. 2** iff R2 beats best-of-R1 on (1) or on navtest
  PDMS by >0.3, or its learned ν(z) shows clear scene dependence
  (std(ν)/mean(ν) > 0.2 across scenes).
- else **Headline = Alg. 1** (simplicity story; 2 & 3 become ablations).
- Any branch: if churn-off ≠ baseline within noise → STOP, fix bug first.

## Budget

Per training: ~same as our previous NAVSIM runs (you know the wall-clock).
6 trainings / 3 pods ≈ 2 rounds each. Phase 2 is hours, single pod.

## Integration points with the old NAVSIM pipeline  (TODO when repo is shared)

- [ ] hook `heavybridge_mixing.precompute` at trainer init (grid = diffusion steps)
- [ ] hook noise draw in the trunc/bridge noising step → `sample_ig` / `sample_profile`
- [ ] add FiLM conditioning input (`torch_noise_level_embedding`)
- [ ] Alg.2: attach IG head (2 outputs, softplus) + ELBO weight A/B + KL(IG‖IG)
- [ ] sampler: replace update with `dbim_step` (ς schedule: ς_s = η·c_{s|s'} , η=0.5 default)
- [ ] eval: anchor-forcing switch in the anchor classifier output
