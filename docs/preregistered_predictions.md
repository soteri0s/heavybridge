# Pre-registered decision rules and predictions

The paper states that the screening decision rules, and the predictions of the
two collapse-mechanism controls and of the alpha ablation, were written down
before the corresponding runs. This file collects them.

1. **Screening decision rules** (written 2026-07-13, before the first training
   run): [`preregistration_PLAN.md`](preregistration_PLAN.md), reproduced
   verbatim. The variant ids there are the ones used throughout the code and
   the paper (R0, R1a, R1b, R2, R3t, ...). The verdict was reached under these
   rules on navmini; the paper reports how the study then continued.

2. **Collapse-mechanism controls** (written 2026-09-13, before the pod was
   started), translated from the run book:
   - *Control B (R2c0)*: the existing R2 checkpoint evaluated with the churn
     switched off (eta = 0), forced sweep k = 1, 2, 3, 5, 10, 20 on navmini, with
     the per-scene scale logged. **Prediction: the collapse persists (about 58
     PDMS at k = 20).**
   - *Control A (R2b)*: bounded amortized per-rollout scale,
     lambda = m(z) * lambda_prior, lambda_prior ~ IG(5, 5) (nu = 10),
     m(z) in [1/4, 4], zero-initialized, regularizer (log m)^2; 15-epoch screening
     recipe. **Prediction: no collapse; argmax close to R0/R1b.**
   - Whatever the outcome, it goes into the paper.

   Outcome (`results/controls/navmini_forced_summary.csv`): both held. R2c0
   scores 92.98 at the argmax and 58.31 at k = 20; R2b scores 92.74 at the
   argmax and 93.13 at k = 20.

3. **Alpha ablation at fixed jump energy gamma_J = 5/4** (written 2026-09-15,
   before the pod was started), translated from the run book. Variants R3ga7
   (alpha = 1.7, kappa = 0.193775) and R3ga6 (alpha = 1.6, kappa = 0.085732),
   compared with R3g (alpha = 1.8, kappa = 0.956338).
   1. No collapse: k = 20 within 1.0 PDMS of the argmax for both.
   2. navmini argmax: both within +/-0.7 of R3g (93.5); a monotone trend in
      alpha will not be distinguishable from the noise of one screening seed.
   3. navtest: argmax 87.2-87.5, k = 20 86.6-87.0, catastrophes 780-860.
   4. Feasibility violations increase as alpha decreases (alpha = 1.7: about
      14-17%, alpha = 1.6: about 16-20%).

   *Revised prediction, added offline before any PDMS result* (2026-09-15), after
   running `analysis/tilt_quantification/tilt_quantification.py` at the new
   calibrations: at fixed gamma_J, a heavier index widens the profile and
   strengthens the tilt (ESS/M 0.77 / 0.40 / 0.19 for alpha = 1.8 / 1.7 / 1.6).
   If there is a difference, it will be monotone in favour of alpha = 1.8, with
   alpha = 1.6 losing most.

   Outcome (numbers in `results/ablation_alpha/`): prediction 1 held (no
   collapse at any alpha). Prediction 2 held for alpha = 1.7 (navmini argmax
   92.6) and failed for alpha = 1.6 (89.2). Prediction 3 failed: navtest argmax
   86.9 and 85.6, catastrophes 855 and 1,050. Prediction 4 failed (7.6% at 1.7,
   13.3% at 1.8, 20.0% at 1.6). The revised prediction held: the argmax falls in
   the order of the offline tilt, with alpha = 1.6 losing most.
