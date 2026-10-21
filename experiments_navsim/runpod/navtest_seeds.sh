#!/bin/bash
# =============================================================================
# navtest_seeds.sh — seed replication for the full-schedule finalists (2026-08-04).
# Trains R3gfull/R0full at extra seeds SERIALLY on ONE pod, so the navtrain
# download (~450 GB) is paid once. Each variant-seed reuses navtest_full.sh
# end-to-end (100-epoch train -> navtest dump -> base/k2/k20 scoring) and is
# idempotent: finished variants are skipped via their DONE sentinels, so the
# driver can be re-launched after any failure and resumes where it stopped.
#
# Order matters: one COMPLETE seed pair first (R3gfull_s2, R0full_s2), so if
# the budget dies mid-way we still have a usable paired replicate.
#
# Before running:
#   1. Put a valid HF token (gated NAVSIM data) in the pod's config.env.
#      NEVER commit a config.env that contains secrets.
#   2. rsync the whole runpod/ to the pod and adjust the paths in config.env.
#
# Start:
#   setsid nohup bash navtest_seeds.sh > /scratch/mc/seeds_driver.log 2>&1 &
# Progress:
#   grep FULL /scratch/mc/navtest.log ; tail -f /scratch/mc/seeds_driver.log
#   table: /workspace/mc/navtest/summary.csv (new rows: R3gfull_s2, ...)
# Afterwards, download BEFORE closing the pod:
#   /workspace/mc/navtest/<variant>/  (results, perscene_*.csv, train.log)
#   /workspace/mc/ckpt_backup/<variant>_final.pth
# =============================================================================
set -u
cd "$(cd "$(dirname "$0")" && pwd)"

SEEDS_LOG=/scratch/mc/seeds_driver.log
VARIANTS=(R3gfull_s2 R0full_s2 R3gfull_s3 R0full_s3)

for V in "${VARIANTS[@]}"; do
  echo "SEEDS ==== starting $V $(date -u) ====" | tee -a "$SEEDS_LOG"
  if bash navtest_full.sh "$V"; then
    echo "SEEDS ==== $V DONE $(date -u) ====" | tee -a "$SEEDS_LOG"
  else
    echo "SEEDS ==== $V FAILED $(date -u) — driver stops here; fix and relaunch (it will resume) ====" | tee -a "$SEEDS_LOG"
    exit 1
  fi
done

echo "SEEDS ==== all variants complete $(date -u) ====" | tee -a "$SEEDS_LOG"
cat /workspace/mc/navtest/summary.csv | tee -a "$SEEDS_LOG"
