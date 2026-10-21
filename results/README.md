# Results

Per-scene and aggregate PDMS for every evaluation reported in the paper, scored
by the official NAVSIM PDM scorer. Every score is a single evaluation pass: one
trained checkpoint, one plan per scene, scored once.

## Files

`navtest/<variant>/` (12,146 scenes each)

| file | content |
|---|---|
| `perscene_base.csv.gz` | per-scene PDMS and sub-scores of the deployed plan (argmax anchor, k = 1) |
| `perscene_k2.csv.gz`, `perscene_k20.csv.gz` | forced-anchor stress test: the 2nd-ranked and the worst-ranked (20th) anchor |
| `perscene_filt_k1.csv.gz`, `perscene_filt_k2.csv.gz` | ranking restricted to kinematically feasible modes (R0, R2, R3g) |
| `eval_base.json` | aggregate of `perscene_base` |
| `aux_metrics.json` | within-anchor sample diversity and kinematic-feasibility violation rate |

The per-scene CSVs are the scorer's own output: columns `token`, `valid`,
`no_at_fault_collisions` (NC), `drivable_area_compliance` (DAC), `ego_progress`
(EP), `time_to_collision_within_bound` (TTC), `comfort` (C),
`driving_direction_compliance` (DDC), `score` (PDMS). The last row, token
`average`, is the split average.

`navtest/summary.csv`, `navtest/summary_filt.csv`: one row per (variant, k).

`navmini/` (396 scenes, screening checkpoints R0, R1a, R1b, R2, R3t, R3k, R3g, R23)

| file | content |
|---|---|
| `perscene_matrix.csv.gz` | per-scene PDMS for every variant and every forced rank k = 1..20 |
| `forced_summary.csv` | aggregate PDMS and sub-scores at k = 2, 3, 5, 10, 20, and with the feasibility filter (k = 1, 2, 5) |
| `oracle_table.csv` | per-scene best over the 20 ranked anchors (the oracle study) |

`controls/`: the collapse-mechanism controls R2b (bounded amortized scale) and
R2c0 (R2 with the churn off): navmini forced sweep, navtest k = 1/2/20, and the
realized per-scene scale (`lambda_log.csv.gz`).

`ablation_alpha/`: alpha = 1.7 (R3ga7) and alpha = 1.6 (R3ga6) at jump energy
5/4: navmini forced sweep and navtest k = 1/2/20.

Variant names are explained in the top-level README. The two seeds of the
confirmation phase are `R0full`, `R3gfull` (seed 1) and `R0full_s2`,
`R3gfull_s2` (seed 2); `analysis/seeds_analysis.py` computes the paired
bootstrap statistics of the paper from these files.

The checkpoints are on Hugging Face (huggingface.co/soteri0s/heavybridge). The
selection files (the chosen trajectory of every scene at every rank) and the
training logs are available from the authors on request.

These results are derived from NAVSIM (OpenScene / nuPlan) and inherit the
licenses of those datasets.
