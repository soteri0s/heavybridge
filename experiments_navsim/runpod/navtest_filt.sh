#!/bin/bash
# navtest_filt.sh -- feasibility-filtered selection on the existing navtest
# dumps (R0/R2/R3g), k in {1,2}. No GPU, no training. Rows: variant,k,filt in
# /workspace/mc/navtest/summary_filt.csv.
set -u
RUNPOD=/scratch/mc/experiments_navsim/runpod
LOG=/scratch/mc/navtest.log
SUM=/workspace/mc/navtest/summary_filt.csv
cd "$RUNPOD"
log() { echo "NAVTEST-FILT $* $(date -u)" >> "$LOG"; }
source /scratch/mc/miniconda/etc/profile.d/conda.sh
conda activate /scratch/mc/miniconda/envs/navsim 2>/dev/null || conda activate navsim
set -a; source config.env; set +a
export NUPLAN_MAPS_ROOT=/scratch/data/navsim/maps OPENSCENE_DATA_ROOT=/scratch/data/navsim
export NAVSIM_DEVKIT_ROOT=/scratch/mc/src/DiffusionDrive NAVSIM_EXP_ROOT=/scratch/mc/exp
_p="$(find /scratch/data/navsim -maxdepth 3 -type d -name 'CAM_F0' -path '*test*' -print -quit 2>/dev/null)"
export NAVSIM_SENSOR_ROOT="$(dirname "$(dirname "$_p")")"
_l="$(find /scratch/data/navsim -maxdepth 4 -name '*.pkl' -path '*test*' -print -quit 2>/dev/null)"
export NAVSIM_LOG_PATH="$(dirname "$_l")"
export PYTHONPATH="$RUNPOD:${PYTHONPATH:-}" MC_PLAN_ANCHOR=/scratch/mc/kmeans_navsim_traj_20.npy MC_CONFIG="$RUNPOD/../config.yaml"
[ -s "$SUM" ] || echo "variant,k,PDMS,NC,DAC,EP,TTC,C,DDC" > "$SUM"

for V in R0 R2 R3g; do
  D="/scratch/navtest/$V/dumps"
  O="/workspace/mc/navtest/$V/results/forced_filt"
  mkdir -p "$O"
  log "$V filt selections"
  python select_forced_anchor.py --dumps "$D" --out "$O" --ks "1 2" \
    --run-id "${V}_filt" --feasibility-filter >> /scratch/mc/navtest_scoring.log 2>&1 \
    || { log "$V filt selection FAILED"; continue; }
  for K in 1 2; do
    sel="$O/selection_k${K}.json"
    out="/scratch/mc/exp/nt_${V}_filt_k${K}"
    log "$V filt k$K scoring"
    MC_SELECTION="$sel" MC_DUMP_DIR="$D" timeout 7200 python \
      "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py" \
      agent=replay_agent train_test_split=navtest \
      metric_cache_path="$NAVSIM_EXP_ROOT/metric_cache" \
      navsim_log_path="$NAVSIM_LOG_PATH" sensor_blobs_path="$NAVSIM_SENSOR_ROOT" \
      output_dir="$out" worker=ray_distributed "worker.threads_per_node=64" \
      >> /scratch/mc/navtest_scoring.log 2>&1 || { log "$V filt k$K FAILED"; continue; }
    csv=$(ls -t "$out"/*.csv 2>/dev/null | head -1)
    python - "$csv" "${V}" "$K" "$SUM" <<'PY'
import sys, csv as c
rows = list(c.reader(open(sys.argv[1])))
avg = [r for r in rows if len(r) > 1 and r[1] == "average"][-1]
v = avg[3:]
open(sys.argv[4], "a").write(",".join([sys.argv[2], sys.argv[3], v[-1], v[0], v[1], v[2], v[3], v[4], v[5]]) + "\n")
PY
    cp "$csv" "/workspace/mc/navtest/$V/perscene_filt_k${K}.csv" 2>/dev/null || true
  done
done
log "FILT ALL DONE"
cat "$SUM" >> "$LOG"
