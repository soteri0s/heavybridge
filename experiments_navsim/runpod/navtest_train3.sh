#!/bin/bash
# =============================================================================
# navtest_train3.sh -- three additional "gentler Levy" screening variants:
#   R3g2: alpha=1.8, kappa=3.105537 (<-> nu=20), next step of the gentle trend
#   R3gk: alpha=1.8, kappa=0.956338, ramp END_MULT=2 (early jumps, mild late)
#   R23g: tempered-learned m(z) in [1/4,4] on top of alpha=1.8/kappa=0.956338
# Full pipeline per variant: 15-epoch warm-start fine-tune (downloads navtrain
# once) -> navtest dump -> argmax + k=2/k=20 scoring. Rows go to the shared
# /workspace/mc/navtest/summary.csv. Checkpoints -> /workspace/mc/ckpt_backup immediately.
# Start: setsid nohup bash navtest_train3.sh > /scratch/mc/train3_driver.log 2>&1 &
# =============================================================================
set -u
RUNPOD=/scratch/mc/experiments_navsim/runpod
LOG=/scratch/mc/navtest.log
SUM=/workspace/mc/navtest/summary.csv
cd "$RUNPOD"
CFG=config.env
mkdir -p /workspace/mc/navtest /workspace/mc/ckpt_backup
log() { echo "TRAIN3 $* $(date -u)" >> "$LOG"; }
setk() { if grep -q "^$1=" "$CFG"; then sed -i "s|^$1=.*|$1=\"$2\"|" "$CFG"; else echo "$1=\"$2\"" >> "$CFG"; fi; }

setk EVAL_SPLIT navtest
setk PDM_WORKER ray_distributed
setk HB_CHURN_ETA "0.5"
setk HB_FINETUNE 1
setk HB_FT_EPOCHS 15
setk HB_FT_LR 1e-4
setk HB_BEV_WEIGHT 14.0
setk HB_ZERO_INIT_LAM 1

mix_env() { case "$1" in
  R3g2) echo "HB_MIXING tempered HB_ALPHA 1.8 HB_KAPPA 3.105537 HB_KAPPA_END_MULT 1 HB_M_REG 0" ;;
  R3gk) echo "HB_MIXING tempered HB_ALPHA 1.8 HB_KAPPA 0.956338 HB_KAPPA_END_MULT 2 HB_M_REG 0" ;;
  R23g) echo "HB_MIXING tempered-learned HB_ALPHA 1.8 HB_KAPPA 0.956338 HB_KAPPA_END_MULT 1 HB_M_REG 1.0" ;;
esac; }

score_k() {  # V K
  local V="$1" K="$2"
  local sel="/workspace/mc/navtest/$V/results/forced/selection_k${K}.json"
  local dumps="/scratch/navtest/$V/dumps"
  local out="/scratch/mc/exp/nt_${V}_k${K}"
  [ -s "$sel" ] || { log "$V k$K missing selection"; return 1; }
  MC_SELECTION="$sel" MC_DUMP_DIR="$dumps" timeout 14400 python \
    "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py" \
    agent=replay_agent train_test_split=navtest \
    metric_cache_path="$NAVSIM_EXP_ROOT/metric_cache" \
    navsim_log_path="$NAVSIM_LOG_PATH" sensor_blobs_path="$NAVSIM_SENSOR_ROOT" \
    output_dir="$out" worker=ray_distributed "worker.threads_per_node=64" \
    >> /scratch/mc/navtest_scoring.log 2>&1 || return 1
  local csv=$(ls -t "$out"/*.csv 2>/dev/null | head -1); [ -n "$csv" ] || return 1
  python - "$csv" "$V" "$K" "$SUM" <<'PY'
import sys, csv as c
rows = list(c.reader(open(sys.argv[1])))
avg = [r for r in rows if len(r) > 1 and r[1] == "average"][-1]
v = avg[3:]
open(sys.argv[4], "a").write(",".join([sys.argv[2], sys.argv[3], v[-1], v[0], v[1], v[2], v[3], v[4], v[5]]) + "\n")
PY
}

for V in R3g2 R3gk R23g; do
  D="/workspace/mc/navtest/$V"
  if [ -f "$D/DONE" ]; then log "$V already done — skipping"; continue; fi
  setk VARIANT "$V"
  set -- $(mix_env "$V"); while [ $# -gt 0 ]; do setk "$1" "$2"; shift 2; done
  rm -f DONE FAILED run.log
  LASTV=$(cat /scratch/mc/.last_variant 2>/dev/null || echo none)
  if [ "$LASTV" != "$V" ]; then rm -rf results selections dumps; fi
  echo "$V" > /scratch/mc/.last_variant
  if [ -d "/scratch/navtest/$V/dumps" ] && [ ! -d dumps ]; then
    mv "/scratch/navtest/$V/dumps" dumps
  fi
  mkdir -p results selections dumps
  log "$V pipeline starting (TRAIN 15ep + dump + score)"
  bash remote_run.sh > run.log 2>&1
  # checkpoint to the durable /workspace BEFORE anything else
  [ -f "/scratch/data/checkpoints/$V/${V}_final.pth" ] && \
    cp -f "/scratch/data/checkpoints/$V/${V}_final.pth" /workspace/mc/ckpt_backup/ && \
    log "$V ckpt backed up"
  mkdir -p "$D" "/scratch/navtest/$V"
  cp -r results "$D/" 2>/dev/null || true
  cp run.log "$D/" 2>/dev/null || true
  cp "/scratch/data/checkpoints/$V/train.log" "$D/" 2>/dev/null || true
  rm -rf "/scratch/navtest/$V/dumps"; mv dumps "/scratch/navtest/$V/dumps" 2>/dev/null || true
  if [ -f DONE ]; then log "$V remote_run DONE"; else log "$V FAILED"; exit 1; fi
  _b=$(ls -t /scratch/mc/exp/base/*.csv 2>/dev/null | head -1)
  [ -n "$_b" ] && cp "$_b" "$D/perscene_base.csv"
  python - "$D/results/eval_base.json" "$V" "$SUM" <<'PY' || true
import sys, json
d = json.load(open(sys.argv[1]))
open(sys.argv[3], "a").write(f"{sys.argv[2]},1,{d['PDMS']},{d['NC']},{d['DAC']},{d['EP']},{d['TTC']},{d['C']},-\n")
PY
  source /scratch/mc/miniconda/etc/profile.d/conda.sh
  conda activate /scratch/mc/miniconda/envs/navsim 2>/dev/null || conda activate navsim
  set -a; source "$CFG"; set +a
  export NUPLAN_MAPS_ROOT=/scratch/data/navsim/maps OPENSCENE_DATA_ROOT=/scratch/data/navsim
  export NAVSIM_DEVKIT_ROOT=/scratch/mc/src/DiffusionDrive NAVSIM_EXP_ROOT=/scratch/mc/exp
  _p="$(find /scratch/data/navsim -maxdepth 3 -type d -name 'CAM_F0' -path '*test*' -print -quit 2>/dev/null)"
  export NAVSIM_SENSOR_ROOT="$(dirname "$(dirname "$_p")")"
  _l="$(find /scratch/data/navsim -maxdepth 4 -name '*.pkl' -path '*test*' -print -quit 2>/dev/null)"
  export NAVSIM_LOG_PATH="$(dirname "$_l")"
  export PYTHONPATH="$RUNPOD:${PYTHONPATH:-}" MC_PLAN_ANCHOR=/scratch/mc/kmeans_navsim_traj_20.npy MC_CONFIG="$RUNPOD/../config.yaml"
  for K in 2 20; do
    log "$V k$K stress scoring"
    score_k "$V" "$K" || log "$V k$K FAILED"
    _c=$(ls -t "/scratch/mc/exp/nt_${V}_k${K}"/*.csv 2>/dev/null | head -1)
    [ -n "$_c" ] && cp "$_c" "$D/perscene_k${K}.csv"
  done
  touch "$D/DONE"; log "$V ALL DONE"
done
log "TRAIN3 ALL DONE"
cat "$SUM" >> "$LOG"
