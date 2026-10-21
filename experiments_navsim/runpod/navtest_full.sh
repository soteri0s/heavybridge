#!/bin/bash
# =============================================================================
# navtest_full.sh <R0full|R3gfull|R2bfull>[_sN] -- full-schedule fine-tune.
# 100 epochs warm-started from the public DiffusionDrive checkpoint, so that
# DiffusionDrive's hard-coded WarmupCosLR (warmup 3, cosine -> 1e-6 over 100
# epochs) COMPLETES; this fixes the early stop that keeps all 15-epoch
# screening runs below 88.1. Then: navtest dump + argmax/k2/k20 scoring.
# ONE variant per pod (2x RTX 4090). No checkpoint upload needed: the warm
# start is the public checkpoint (downloaded in stage 5 of remote_run.sh).
# Includes a background backup of the lightning checkpoints to /workspace
# every 2 h.
# Start:     setsid nohup bash navtest_full.sh R3gfull > /scratch/mc/full_driver.log 2>&1 &
# Progress:  grep FULL /scratch/mc/navtest.log ; table: /workspace/mc/navtest/summary.csv
# =============================================================================
set -u
V="${1:?usage: navtest_full.sh R0full|R3gfull|R0full_sN|R3gfull_sN}"
case "$V" in R0full|R3gfull|R2bfull|R0full_s[0-9]*|R3gfull_s[0-9]*|R2bfull_s[0-9]*) ;; *) echo "unknown variant $V"; exit 1;; esac
# seed replicates (2026-08-04): suffix _sN pins the training seed to N via
# train_stage.sh's HB_SEED; no suffix = stock behavior (the original runs).
SEED=""
[[ "$V" == *_s* ]] && SEED="${V##*_s}" 
RUNPOD=/scratch/mc/experiments_navsim/runpod
LOG=/scratch/mc/navtest.log
SUM=/workspace/mc/navtest/summary.csv
cd "$RUNPOD"
CFG=config.env
mkdir -p /workspace/mc/navtest /workspace/mc/ckpt_backup /workspace/mc/lightning_backup
log() { echo "FULL $* $(date -u)" >> "$LOG"; }
setk() { if grep -q "^$1=" "$CFG"; then sed -i "s|^$1=.*|$1=\"$2\"|" "$CFG"; else echo "$1=\"$2\"" >> "$CFG"; fi; }

setk EVAL_SPLIT navtest
setk PDM_WORKER ray_distributed
setk REMOTE_WORKDIR /scratch/mc
setk DATA_ROOT /scratch/data
setk DATA_VOLUME /scratch/data
setk HB_CHURN_ETA "0.5"
setk HB_FINETUNE 1
setk HB_FT_EPOCHS 100
setk HB_FT_LR 1e-4
setk HB_BEV_WEIGHT 14.0
setk HB_ZERO_INIT_LAM 1
setk HB_NUM_WORKERS 12
_np=$(nproc); [ "$_np" -gt 64 ] && _np=64
setk HB_RAY_THREADS "$_np"
setk VARIANT "$V"
case "$V" in
  R0full*)   setk HB_MIXING none ;;
  R3gfull*)  setk HB_MIXING tempered; setk HB_ALPHA 1.8; setk HB_KAPPA 0.956338; setk HB_KAPPA_END_MULT 1; setk HB_M_REG 0 ;;
  R2bfull*)  setk HB_MIXING ig-bounded; setk HB_NU 10; setk HB_M_REG 1.0 ;;   # bounded amortized per-rollout (R2b), full schedule (2026-09-14, STANDBY)
esac
# R2bfull*: log the realized per-rollout scale (lam,m,lam_prior per scene) during the navtest dump;
# empty HB_LOG_LAMBDA disables the log for every other variant (the dump stage sources config.env).
mkdir -p "/workspace/mc/navtest/$V"
case "$V" in R2bfull*) rm -f "/workspace/mc/navtest/$V/lambda_log.csv"; setk HB_LOG_LAMBDA "/workspace/mc/navtest/$V/lambda_log.csv" ;; *) setk HB_LOG_LAMBDA "" ;; esac
if [ -n "$SEED" ]; then setk HB_SEED "$SEED"; else setk HB_SEED ""; fi

[ -s "$SUM" ] || echo "variant,k,PDMS,NC,DAC,EP,TTC,C,DDC" > "$SUM"

# ---- background: periodic backup of the newest lightning checkpoint (every 2 h) -----
( while true; do
    sleep 7200
    ck=$(ls -t "/scratch/data/checkpoints/$V"/lightning_logs/version_*/checkpoints/*.ckpt 2>/dev/null | head -1)
    [ -n "$ck" ] && cp -f "$ck" "/workspace/mc/lightning_backup/${V}_latest.ckpt" && \
      echo "FULL $V lightning backup $(date -u)" >> "$LOG"
  done ) &
BK_PID=$!

score_k() {  # K
  local K="$1"
  local sel="/workspace/mc/navtest/$V/results/forced/selection_k${K}.json"
  local dumps="/scratch/navtest/$V/dumps"
  local out="/scratch/mc/exp/nt_${V}_k${K}"
  [ -s "$sel" ] || { log "$V k$K missing selection"; return 1; }
  MC_SELECTION="$sel" MC_DUMP_DIR="$dumps" timeout 14400 python \
    "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py" \
    agent=replay_agent train_test_split=navtest \
    metric_cache_path="$NAVSIM_EXP_ROOT/metric_cache" \
    navsim_log_path="$NAVSIM_LOG_PATH" sensor_blobs_path="$NAVSIM_SENSOR_ROOT" \
    output_dir="$out" worker=ray_distributed "worker.threads_per_node=${HB_RAY_THREADS:-32}" \
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

D="/workspace/mc/navtest/$V"
if [ -f "$D/DONE" ]; then log "$V already done"; kill $BK_PID 2>/dev/null; exit 0; fi
rm -f DONE FAILED run.log
LASTV=$(cat /scratch/mc/.last_variant 2>/dev/null || echo none)
if [ "$LASTV" != "$V" ]; then rm -rf results selections dumps; fi
echo "$V" > /scratch/mc/.last_variant
if [ -d "/scratch/navtest/$V/dumps" ] && [ ! -d dumps ]; then mv "/scratch/navtest/$V/dumps" dumps; fi
mkdir -p results selections dumps
log "$V pipeline starting (TRAIN 100ep full-cosine + dump + score)"
bash remote_run.sh > run.log 2>&1
[ -f "/scratch/data/checkpoints/$V/${V}_final.pth" ] && \
  cp -f "/scratch/data/checkpoints/$V/${V}_final.pth" /workspace/mc/ckpt_backup/ && log "$V ckpt backed up"
mkdir -p "$D" "/scratch/navtest/$V"
cp -r results "$D/" 2>/dev/null || true
cp run.log "$D/" 2>/dev/null || true
cp "/scratch/data/checkpoints/$V/train.log" "$D/" 2>/dev/null || true
rm -rf "/scratch/navtest/$V/dumps"; mv dumps "/scratch/navtest/$V/dumps" 2>/dev/null || true
if [ -f DONE ]; then log "$V remote_run DONE"; else log "$V FAILED"; kill $BK_PID 2>/dev/null; exit 1; fi
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
  score_k "$K" || log "$V k$K FAILED"
  _c=$(ls -t "/scratch/mc/exp/nt_${V}_k${K}"/*.csv 2>/dev/null | head -1)
  [ -n "$_c" ] && cp "$_c" "$D/perscene_k${K}.csv"
done
touch "$D/DONE"; log "$V ALL DONE"
kill $BK_PID 2>/dev/null
cat "$SUM" >> "$LOG"
