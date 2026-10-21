#!/bin/bash
# =============================================================================
# navtest_controls.sh — navtest k=1/2/20 for the two reviewer-m3J8 controls
# (2026-09-14), mirroring navtest_confirm.sh: NO training, checkpoints given.
#   R2c0 : the R2 checkpoint with HB_CHURN_ETA=0, HB_LOG_LAMBDA -> lam,A,B per scene
#   R2b  : the bounded amortized checkpoint (HB_MIXING=ig-bounded), HB_LOG_LAMBDA -> lam,m,lam_prior
# Per variant: remote_run.sh (navtest dump, metric cache once, base scoring, forced
# selections) -> k=2 & k=20 stress scoring (ray). Results archived in /workspace/mc/navtest/.
# Prereqs: harness at /scratch/mc/experiments_navsim (patched head + train_stage R2c0 case),
#   /workspace/mc/ckpt_backup/{R2_final.pth,R2b_final.pth}; navtest data present or fetched by stage 4.
# Run:  setsid nohup bash navtest_controls.sh > /scratch/mc/navtest_driver.log 2>&1 &
# =============================================================================
set -u
RUNPOD=/scratch/mc/experiments_navsim/runpod
LOG=/scratch/mc/navtest.log
SUM=/workspace/mc/navtest/summary.csv
cd "$RUNPOD"
CFG=config.env
mkdir -p /workspace/mc/navtest
log() { echo "NAVTEST $* $(date -u)" >> "$LOG"; }
setk() { if grep -q "^$1=" "$CFG"; then sed -i "s|^$1=.*|$1=\"$2\"|" "$CFG"; else echo "$1=\"$2\"" >> "$CFG"; fi; }

setk EVAL_SPLIT navtest
setk PDM_WORKER ray_distributed
setk DATA_ROOT /scratch/data
setk DATA_VOLUME /scratch/data
setk REMOTE_WORKDIR /scratch/mc
setk HB_NUM_WORKERS 12

ckpt_of() { case "$1" in R2c0) echo R2_final.pth;; *) echo "$1_final.pth";; esac; }
mix_env() { case "$1" in
  R2c0) echo "HB_MIXING ig-learned HB_CHURN_ETA 0" ;;
  R2b)  echo "HB_MIXING ig-bounded HB_NU 10 HB_M_REG 1.0 HB_CHURN_ETA 0.5" ;;
esac; }

[ -s "$SUM" ] || echo "variant,k,PDMS,NC,DAC,EP,TTC,C,DDC" > "$SUM"

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
    output_dir="$out" worker=ray_distributed "worker.threads_per_node=${HB_RAY_THREADS:-64}" \
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

for V in R2c0 R2b; do
  D="/workspace/mc/navtest/$V"
  if [ -f "$D/DONE" ]; then log "$V already done — skipping"; continue; fi
  mkdir -p "/scratch/data/checkpoints/$V" "$D"
  CK="/workspace/mc/ckpt_backup/$(ckpt_of "$V")"
  [ -s "$CK" ] || { log "$V ckpt missing ($CK) — aborting"; exit 1; }
  cp -n "$CK" "/scratch/data/checkpoints/$V/${V}_final.pth" || true
  touch "/scratch/data/checkpoints/$V/TRAIN_DONE"
  setk VARIANT "$V"
  set -- $(mix_env "$V"); while [ $# -gt 0 ]; do setk "$1" "$2"; shift 2; done
  rm -f "$D/lambda_log.csv"; setk HB_LOG_LAMBDA "$D/lambda_log.csv"
  rm -f DONE FAILED run.log
  LASTV=$(cat /scratch/mc/.last_variant 2>/dev/null || echo none)
  if [ "$LASTV" != "$V" ]; then rm -rf results selections dumps; fi
  echo "$V" > /scratch/mc/.last_variant
  if [ -d "/scratch/navtest/$V/dumps" ] && [ ! -d dumps ]; then mv "/scratch/navtest/$V/dumps" dumps; fi
  mkdir -p results selections dumps
  log "$V pipeline starting (dump+score, no training)"
  bash remote_run.sh > run.log 2>&1
  mkdir -p "/scratch/navtest/$V"
  cp -r results "$D/" 2>/dev/null || true
  cp run.log "$D/" 2>/dev/null || true
  rm -rf "/scratch/navtest/$V/dumps"; mv dumps "/scratch/navtest/$V/dumps" 2>/dev/null || true
  if [ -f DONE ]; then log "$V remote_run DONE (lambda log lines: $(wc -l < "$D/lambda_log.csv" 2>/dev/null))"; else log "$V FAILED"; exit 1; fi
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
  export NAVSIM_LOG_PATH=/scratch/data/navsim/test_navsim_logs/test
  export NAVSIM_SENSOR_ROOT=/scratch/data/navsim/test_sensor_blobs/test
  _p="$(find /scratch/data/navsim -maxdepth 3 -type d -name 'CAM_F0' -path '*test*' -print -quit 2>/dev/null)"
  [ -n "$_p" ] && export NAVSIM_SENSOR_ROOT="$(dirname "$(dirname "$_p")")"
  _l="$(find /scratch/data/navsim -maxdepth 4 -name '*.pkl' -path '*test*' -print -quit 2>/dev/null)"
  [ -n "$_l" ] && export NAVSIM_LOG_PATH="$(dirname "$_l")"
  export PYTHONPATH="$RUNPOD:${PYTHONPATH:-}" MC_PLAN_ANCHOR=/scratch/mc/kmeans_navsim_traj_20.npy MC_CONFIG="$RUNPOD/../config.yaml"
  for K in 2 20; do
    log "$V k$K stress scoring"
    score_k "$V" "$K" || log "$V k$K FAILED"
    _c=$(ls -t "/scratch/mc/exp/nt_${V}_k${K}"/*.csv 2>/dev/null | head -1)
    [ -n "$_c" ] && cp "$_c" "$D/perscene_k${K}.csv"
  done
  touch "$D/DONE"; log "$V ALL DONE"
done
log "ALL DONE"
cat "$SUM" >> "$LOG"
