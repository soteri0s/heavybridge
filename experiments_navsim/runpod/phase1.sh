#!/bin/bash
# =============================================================================
# phase1.sh — Phase-1 decision-experiment driver (2026-07-21).
# Sequential warm-start fine-tunes on ONE pod: R1a, R1b, R2, R3t
# (+ churn-off Prop.-3 controls for R1a & R2, per docs/preregistration_PLAN.md), reusing the shared
# dataset+cache. Archives per-variant under /scratch/phase1/<V>/ and pushes a
# DURABLE checkpoint copy to /workspace/mc/ckpt_backup/ after every training
# (the container disk is ephemeral — we lost a checkpoint to a crash once).
# Run detached:  setsid nohup bash phase1.sh > /scratch/mc/phase1_driver.log 2>&1 &
# Progress:      /scratch/mc/phase1.log
# =============================================================================
set -u
cd /scratch/mc/experiments_navsim/runpod
CFG=config.env
LOG=/scratch/mc/phase1.log
mkdir -p /workspace/mc/ckpt_backup /scratch/phase1

setk() {  # key value — replace or append in config.env
  if grep -q "^$1=" "$CFG"; then sed -i "s|^$1=.*|$1=\"$2\"|" "$CFG"; else echo "$1=\"$2\"" >> "$CFG"; fi
}

run_once() {  # label -> 0 on DONE, 1 on FAILED
  rm -f DONE FAILED run.log
  rm -rf results selections dumps
  mkdir -p results selections dumps
  bash remote_run.sh > run.log 2>&1
  local dest="/scratch/phase1/$1"
  mkdir -p "$dest"
  cp -r results "$dest/" 2>/dev/null || true
  cp run.log "$dest/" 2>/dev/null || true
  # keep the dump too: selections are index-only, forced-anchor (Phase-2)
  # scoring replays trajectories FROM the dump (~140MB/variant)
  cp -r dumps "$dest/dumps" 2>/dev/null || true
  if [[ -f DONE ]]; then
    echo "PHASE1 $1 DONE $(date -u)" >> "$LOG"; return 0
  else
    echo "PHASE1 $1 FAILED $(date -u)" >> "$LOG"; cp FAILED "$dest/" 2>/dev/null || true; return 1
  fi
}

# archive the completed R0 run before touching anything
if [[ ! -d /scratch/phase1/R0 ]]; then
  mkdir -p /scratch/phase1/R0
  cp -r results /scratch/phase1/R0/ 2>/dev/null || true
  cp run.log /scratch/phase1/R0/ 2>/dev/null || true
  echo "PHASE1 R0 archived $(date -u)" >> "$LOG"
fi

for V in R1a R1b R2 R3t; do
  case $V in
    R1a) M="ig";         setk HB_NU 5 ;;
    R1b) M="ig";         setk HB_NU 10 ;;
    R2)  M="ig-learned" ;;
    R3t) M="tempered";   setk HB_ALPHA 1.7 ;;   # HB_KAPPA=0.028469 already in config
  esac
  setk VARIANT "$V"
  setk HB_MIXING "$M"
  setk HB_CHURN_ETA "0.5"
  echo "PHASE1 $V starting (mixing=$M) $(date -u)" >> "$LOG"
  run_once "$V" || { echo "PHASE1 STOPPED at $V $(date -u)" >> "$LOG"; exit 1; }
  # durable checkpoint backup — IMMEDIATELY after each training
  if [[ -f "/scratch/data/checkpoints/$V/${V}_final.pth" ]]; then
    cp "/scratch/data/checkpoints/$V/${V}_final.pth" /workspace/mc/ckpt_backup/
    echo "PHASE1 $V ckpt -> /workspace/mc/ckpt_backup/ $(date -u)" >> "$LOG"
  fi
  # Prop.-3 negative control: churn-off eval on the SAME checkpoint (R1a, R2)
  if [[ "$V" == "R1a" || "$V" == "R2" ]]; then
    setk HB_CHURN_ETA "0"
    echo "PHASE1 ${V}_churn0 starting $(date -u)" >> "$LOG"
    run_once "${V}_churn0" || { echo "PHASE1 STOPPED at ${V}_churn0 $(date -u)" >> "$LOG"; exit 1; }
    setk HB_CHURN_ETA "0.5"
  fi
done
echo "PHASE1 ALL DONE $(date -u)" >> "$LOG"
