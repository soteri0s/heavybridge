#!/bin/bash
# =============================================================================
# phase3.sh — "R3 redemption" driver (2026-07-22). Runs the five alternatives:
#   #1 R3k : step-dependent tempering (HB_KAPPA_END_MULT=20)      [train]
#   #4 R3g : gentle mixing (alpha=1.8, kappa matched to nu=10)     [train]
#   #5 R23 : amortized per-scene jump intensity m(z) (Alg.2∘Alg.3) [train]
#   #2     : feasibility-filtered selection, ALL variants          [eval]
#   #3     : deep forced-anchor k in {10,20}, ALL variants         [eval]
# Prereqs on a fresh pod: harness rsynced to /scratch/mc/experiments_navsim,
# and the ORIGINAL Phase-1 dumps uploaded to /scratch/phase1_dumps/<V>/
# (R0 R1a R1b R2 R3t — from the laptop bundle; their checkpoints are gone).
# Run: setsid nohup bash phase3.sh > /scratch/mc/phase3_driver.log 2>&1 &
# Progress: grep PHASE3 /scratch/mc/phase1.log ; table: /scratch/phase1/forced_summary_phase3.csv
# =============================================================================
set -u
RUNPOD=/scratch/mc/experiments_navsim/runpod
cd "$RUNPOD"
CFG=config.env
LOG=/scratch/mc/phase1.log
SUM=/scratch/phase1/forced_summary_phase3.csv
mkdir -p /workspace/mc/ckpt_backup /scratch/phase1
log() { echo "PHASE3 $* $(date -u)" >> "$LOG"; }
setk() { if grep -q "^$1=" "$CFG"; then sed -i "s|^$1=.*|$1=\"$2\"|" "$CFG"; else echo "$1=\"$2\"" >> "$CFG"; fi; }

# ---- part B first: train the three new variants (bootstraps data on run #1) --
run_variant() {  # V mixing extra-setk-pairs...
  local V="$1"; shift
  setk VARIANT "$V"; setk HB_CHURN_ETA "0.5"
  while [[ $# -gt 0 ]]; do setk "$1" "$2"; shift 2; done
  rm -f DONE FAILED run.log; rm -rf results selections dumps; mkdir -p results selections dumps
  log "$V run starting"
  bash remote_run.sh > run.log 2>&1
  local dest="/scratch/phase1/$V"; mkdir -p "$dest"
  cp -r results run.log "$dest/" 2>/dev/null || true
  cp -r dumps "$dest/dumps" 2>/dev/null || true
  [[ -f "/scratch/data/checkpoints/$V/${V}_final.pth" ]] && \
    cp "/scratch/data/checkpoints/$V/${V}_final.pth" /workspace/mc/ckpt_backup/
  if [[ -f DONE ]]; then log "$V DONE (ckpt backed up)"; else log "$V FAILED"; return 1; fi
}
run_variant R3k HB_KAPPA_END_MULT 20                 || exit 1
run_variant R3g HB_ALPHA 1.8 HB_KAPPA 0.956338       || exit 1
run_variant R23 HB_M_REG 1.0                         || exit 1

# ---- part A: selections (deep + filtered) & scoring for ALL variants ---------
source /scratch/mc/miniconda/etc/profile.d/conda.sh
conda activate /scratch/mc/miniconda/envs/navsim 2>/dev/null || conda activate navsim
set -a; source "$CFG"; set +a
export NUPLAN_MAPS_ROOT=/scratch/data/navsim/maps OPENSCENE_DATA_ROOT=/scratch/data/navsim
export NAVSIM_DEVKIT_ROOT=/scratch/mc/src/DiffusionDrive NAVSIM_EXP_ROOT=/scratch/mc/exp
export NAVSIM_LOG_PATH=/scratch/data/navsim/mini_navsim_logs/mini
export NAVSIM_SENSOR_ROOT=/scratch/data/navsim/mini_sensor_blobs/mini
export PYTHONPATH="$RUNPOD:${PYTHONPATH:-}"
export MC_PLAN_ANCHOR=/scratch/mc/kmeans_navsim_traj_20.npy MC_CONFIG="$RUNPOD/../config.yaml"
echo "variant,k,filtered,PDMS,NC,DAC,EP,TTC,C,DDC" > "$SUM"

# stage the uploaded original dumps
for V in R0 R1a R1b R2 R3t; do
  if [[ -d "/scratch/phase1_dumps/$V" && ! -d "/scratch/phase1/$V/dumps" ]]; then
    mkdir -p "/scratch/phase1/$V"; cp -r "/scratch/phase1_dumps/$V" "/scratch/phase1/$V/dumps"
  fi
done

score_pass() {  # V k selfile label worker
  local V="$1" K="$2" SEL="$3" L="$4" W="$5"
  local outdir="/scratch/mc/exp/p3_${V}_${L}_k${K}_${W}"
  [[ -s "$SEL" ]] || { log "$V $L k$K missing selection, skip"; return 2; }
  MC_SELECTION="$SEL" MC_DUMP_DIR="/scratch/phase1/$V/dumps" timeout 5400 python \
    "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py" \
    agent=replay_agent train_test_split=navmini \
    metric_cache_path="$NAVSIM_EXP_ROOT/metric_cache" \
    navsim_log_path="$NAVSIM_LOG_PATH" sensor_blobs_path="$NAVSIM_SENSOR_ROOT" \
    output_dir="$outdir" worker="$W" >> /scratch/mc/phase3_scoring.log 2>&1 || return 1
  local csv=$(ls -t "$outdir"/*.csv 2>/dev/null | head -1); [[ -n "$csv" ]] || return 1
  python - "$csv" "$V" "$K" "$L" "$SUM" <<'PY'
import sys, csv as c
rows = list(c.reader(open(sys.argv[1])))
avg = [r for r in rows if len(r) > 1 and r[1] == "average"][-1]
v = avg[3:]  # NC DAC EP TTC C DDC PDMS(last)
open(sys.argv[5], "a").write(",".join([sys.argv[2], sys.argv[3], sys.argv[4], v[-1], v[0], v[1], v[2], v[3], v[4], v[5]]) + "\n")
PY
}

ALL="R0 R1a R1b R2 R3t R3k R3g R23"
for V in $ALL; do
  D="/scratch/phase1/$V/dumps"
  [[ -d "$D" ]] || { log "$V no dumps, skipping selections"; continue; }
  mkdir -p "/scratch/phase1/$V/results/forced" "/scratch/phase1/$V/results/forced_filt"
  python select_forced_anchor.py --dumps "$D" --out "/scratch/phase1/$V/results/forced" \
      --ks "1 2 3 5 10 20" --run-id "$V" >> /scratch/mc/phase3_scoring.log 2>&1
  python select_forced_anchor.py --dumps "$D" --out "/scratch/phase1/$V/results/forced_filt" \
      --ks "1 2 3 5 10 20" --run-id "${V}_filt" --feasibility-filter >> /scratch/mc/phase3_scoring.log 2>&1
done
log "selections built for: $ALL"

pass_list() {  # which (V,k,label) combos to score
  for V in R0 R1a R1b R2 R3t; do
    for K in 10 20;      do echo "$V $K plain"; done
    for K in 1 2 5;      do echo "$V $K filt";  done
  done
  for V in R3k R3g R23; do
    for K in 2 3 5 10 20; do echo "$V $K plain"; done
    for K in 1 2 5;       do echo "$V $K filt";  done
  done
}
pass_list | while read -r V K L; do
  SEL="/scratch/phase1/$V/results/forced/selection_k${K}.json"
  [[ "$L" == "filt" ]] && SEL="/scratch/phase1/$V/results/forced_filt/selection_k${K}.json"
  log "$V $L k$K scoring (ray)"
  if ! score_pass "$V" "$K" "$SEL" "$L" ray_distributed; then
    log "$V $L k$K ray failed -> sequential"
    score_pass "$V" "$K" "$SEL" "$L" sequential || log "$V $L k$K FAILED (both)"
  fi
done
log "PHASE3 ALL DONE"
cat "$SUM" >> "$LOG"
