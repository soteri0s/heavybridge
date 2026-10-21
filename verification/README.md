# Numerical verification

This folder holds the scripts behind the appendix section "Numerical verification" of
*HeavyBridge: Tempered Lévy Bridges for Anchored Diffusion Driving Policies* (TMLR).
`verify_all.py` has one function per item of that list, numbered as in the paper (items 1-8).

## How to run

```bash
pip install numpy scipy               # the only dependencies
python verify_all.py                  # full suite, paper sample sizes (about 3 min, under 1 GB RAM)
python verify_all.py --fast           # samples divided by 20 (about 6 s)
python verify_all.py --items 2 4 8    # selected items only
```

The script imports the released mixing module unchanged from
`../experiments_navsim/runpod/heavybridge_mixing.py`. Each quantity that module computes is taken
from it: `precompute`, `ScheduleTable.abc`, `profile_sig2`, `profile_abc`, `sample_ig`,
`sample_kanter`, `sample_profile` and `_s_alpha_only`. Closed forms that appear only in the paper
(the Student-t density, score and posterior moments, the composition identities, and the
tempered-stable moments and transforms) are written out from the paper's equations. Every random
stream is seeded (seed = 1000 + item number). The exit code is 0 only if every item passes.
`RESULTS.txt` holds the output of a full default run.

## Settings

- VP-Karras schedule with β_d = 2.0 and β_min = 0.1. The horizon is T = 1 and the grid is uniform
  with N = 20. The paper does not print the grid, but T = 1 matches its stated m_T = 0.577 and
  σ_T² = 0.667.
- Tolerances for deterministic items are the ones the paper states. Monte-Carlo items require
  |estimate − closed form| ≤ 4 standard errors, with the SE estimated from the same samples. Each
  line prints the error both as a number and in SE units.

## What each item checks

| Item | Paper claim | Check |
|---|---|---|
| 1 | Lemma 2: a_t and b_t are unchanged and (c_t^λ)² = λc_t², at t ∈ {0.2, 0.5, 0.8} and λ ∈ {1, 3.7}, to machine precision | `ScheduleTable.abc` with σ² → λσ², including the restricted coefficients a_{s\|t} and b_{s\|t}. `profile_sig2` with a constant profile λ, followed by `profile_abc`. Both must agree within 1e-14 relative. As a cross-check, σ_λ² is also computed by quadrature of m_t²∫λg²/m² (tolerance 1e-12, because the quadrature limits the accuracy). |
| 2 | Prop. 1(i): the Student-t density (Eq. Stdensity) against a 2e6-sample Inverse-Gamma mixture, D = 4, ν = 5 | The formula is compared with `scipy.stats.multivariate_t`. The mixture average of N(x; μ, λc²I) with λ drawn by `sample_ig` (i.i.d.) is evaluated at r_t ∈ {0, 0.5, 1, 2, 3, 5} and must fall within 4 SE. The stratified default of `sample_ig` is also reported. |
| 3 | Prop. 1(iii): the score agrees with central differences to 1e-6 | Central differences with h = 1e-5 of the log-density at 26 points. The maximum absolute error must be at most 1e-6. |
| 4 | Prop. 1(ii): E[λ⁻¹ \| ·] and E[λ \| ·] against SNIS | Self-normalised importance sampling with 2e6 prior draws (the paper gives no size). The delta-method SE is used, with a 4 SE tolerance, at the same 6 residuals. |
| 5 | Lemma S-composition at (s, t) = (0.3, 0.7), to 1e-15 | All three identities through `ScheduleTable.abc`, as absolute errors. As an extra check outside the paper, the same identities are tested for the profile coefficients. |
| 6 | Prop. 3(i), α = 1.5, 4e6 samples | λ_i comes from `sample_profile` (Kanter's transform with c_i^{2/α}). The characteristic function of √(λ_i u_i) ε_i (D = 4) is compared with exp(−s_i^α‖ξ‖^α) for all 20 steps and ‖ξ‖s_i ∈ {0.5, 1, 2}, real and imaginary parts. Two-step closure, after propagating the first step through the drift, is compared with exp(−(s₁'^α + s₂^α)‖ξ‖^α) for steps (0,1), (9,10) and (18,19). |
| 7 | Corollary 1, α = 1.5, κ = 1.3, 1.5e6 samples | `sample_profile(kappa=1.3)` (tilt rejection). For all 20 steps the script checks the Laplace transform at θ ∈ {0.5, 2}, the mean (α/2)cκ^{α/2−1}, the variance (α/2)(1−α/2)cκ^{α/2−2}, the mixed-increment characteristic function (Eq. Stempcf) and the increment variance u E[λ]. The scaling identity (Eq. Stsscaling) is checked by quantile match at h ∈ {0.5, 2}: the CDF of an independent TS(ch^α, κ/h²) sample is evaluated at the quantiles of h²λ. |
| 8 | sec:Spractical | (a) The closed forms for σ_t², g², m_t and u_i against quadrature, below 1e-10. (b) Kanter's X and λ_i = c_i^{2/α}X_i against their Laplace transforms, with 2e6 samples. (c) The maximum relative error of Gauss–Legendre s_i^α at K = 2, 4, 8 must reproduce the stated 2.6e-5 and 6.2e-9 to two significant digits, and must reach round-off level (≤ 5e-15) at K = 8. (d) s_i² = u_i at α = 2, below 1e-13. (e) Σ_i(m_T/m_{t_{i+1}})^α s_i^α must agree across N ∈ {20, 40, 80} to better than 1e-15; the coarse grid N = 5 is checked against the stated 2e-11. |

## Status of the current release (see RESULTS.txt)

All eight items pass.

On the coarse grid N = 5, the 8-node Gauss–Legendre rule for s_i^α leaves a relative spread of
1.9e-11 against N = 20, as the paper states; with 12 or 16 nodes it falls to 4e-15 or 0. For the
same reason `precompute` raises its built-in `AssertionError("propagated sum must be grid-invariant")`
on that grid (internal tolerance 1e-12), and the script then uses `_s_alpha_only`, which applies the
same 8-node rule. All experiments use N = 20.

Runtime of a full run on 2 CPU cores: about 3 minutes (item 7 takes about 2 minutes, item 6 about 50 s).
