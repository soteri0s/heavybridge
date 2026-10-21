#!/bin/bash
# =============================================================================
# navtest_confirm.sh -- navtest confirmation of the screening finalists.
# R0 / R2 / R3g ONLY, WITHOUT training: the fine-tuned checkpoints already sit
# in /workspace/mc/ckpt_backup/. Per variant: dump (12,146 scenes, GPU ~1-2 h)
# -> argmax PDM scoring (ray) -> forced selections -> k=2 and k=20 stress
# scoring (ray). Results/selections are archived IMMEDIATELY under
# /workspace/mc/navtest/ (survives restarts); the dumps (~15 GB/variant) stay
# in /scratch (reproducible from the checkpoints).
# Pod prerequisites: L40S (or 4090), container disk >= 800 GB, no volume.
# Before running: experiments_navsim/ rsynced to /scratch/mc/experiments_navsim
# and the 3 checkpoints in /workspace/mc/ckpt_backup/.
# Start:     setsid nohup bash navtest_confirm.sh > /scratch/mc/navtest_driver.log 2>&1 &
# Progress:  grep NAVTEST /scratch/mc/navtest.log ; table: /workspace/mc/navtest/summary.csv
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

# ---- global switches: navtest split, ray scorer (sequential scoring takes ~25 h per pass) --
setk EVAL_SPLIT navtest
setk PDM_WORKER ray_distributed
setk HB_CHURN_ETA "0.5"
setk DATA_ROOT /scratch/data
setk DATA_VOLUME /scratch/data
setk REMOTE_WORKDIR /scratch/mc

ckpt_of() { case "$1" in R0) echo R0_finetune_ep15.pth;; *) echo "$1_final.pth";; esac; }
mix_env() { case "$1" in
  R0)  echo "HB_MIXING none" ;;
  R2)  echo "HB_MIXING ig-learned" ;;
  R3g) echo "HB_MIXING tempered HB_ALPHA 1.8 HB_KAPPA 0.956338" ;;
esac; }

[ -s "$SUM" ] || echo "variant,k,PDMS,NC,DAC,EP,TTC,C,DDC" > "$SUM"

score_k() {  #  V K  (runs AFTER remote_run of V, with the conda env active)
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

for V in R0 R2 R3g; do
  D="/workspace/mc/navtest/$V"
  if [ -f "$D/DONE" ]; then log "$V already done — skipping"; continue; fi
  # checkpoint in place + sentinel, so that train_stage skips training.
  # If the upload is still running, wait (size >700MB and stable for 60 s).
  mkdir -p "/scratch/data/checkpoints/$V"
  CK="/workspace/mc/ckpt_backup/$(ckpt_of "$V")"
  for _i in $(seq 1 240); do
    s1=$(stat -c%s "$CK" 2>/dev/null || echo 0)
    [ "$s1" -gt 700000000 ] && { sleep 60; s2=$(stat -c%s "$CK" 2>/dev/null || echo 0); [ "$s1" = "$s2" ] && break; }
    [ "$_i" = 1 ] && log "$V waiting for ckpt upload ($CK)"
    sleep 60
  done
  s1=$(stat -c%s "$CK" 2>/dev/null || echo 0)
  [ "$s1" -gt 700000000 ] || { log "$V ckpt still missing after 4h — aborting"; exit 1; }
  cp -n "$CK" "/scratch/data/checkpoints/$V/${V}_final.pth" || true
  touch "/scratch/data/checkpoints/$V/TRAIN_DONE"
  setk VARIANT "$V"
  set -- $(mix_env "$V"); while [ $# -gt 0 ]; do setk "$1" "$2"; shift 2; done
  rm -f DONE FAILED run.log
  # resume-aware cleanup: for a NEW variant wipe everything; for a retry of the SAME
  # variant keep results/selections (score_selection skips finished ones).
  LASTV=$(cat /scratch/mc/.last_variant 2>/dev/null || echo none)
  if [ "$LASTV" != "$V" ]; then rm -rf results selections dumps; fi
  echo "$V" > /scratch/mc/.last_variant
  # dumps: if a saved copy exists from a previous (failed) attempt,
  # restore it; stage 7 is resumable and will pass over it in minutes.
  if [ -d "/scratch/navtest/$V/dumps" ] && [ ! -d dumps ]; then
    mv "/scratch/navtest/$V/dumps" dumps
  fi
  mkdir -p results selections dumps
  log "$V pipeline starting (dump+score, no training)"
  bash remote_run.sh > run.log 2>&1
  mkdir -p "$D" "/scratch/navtest/$V"
  cp -r results "$D/" 2>/dev/null || true          # small: json/csv/selections
  cp run.log "$D/" 2>/dev/null || true
  rm -rf "/scratch/navtest/$V/dumps"; mv dumps "/scratch/navtest/$V/dumps" 2>/dev/null || true
  if [ -f DONE ]; then log "$V remote_run DONE"; else log "$V FAILED"; exit 1; fi
  _b=$(ls -t /scratch/mc/exp/base/*.csv 2>/dev/null | head -1)
  [ -n "$_b" ] && cp "$_b" "$D/perscene_base.csv"   # per-scene PDMS (difficulty analysis)
  # base PDMS -> summary
  python - "$D/results/eval_base.json" "$V" "$SUM" <<'PY' || true
import sys, json
d = json.load(open(sys.argv[1]))
open(sys.argv[3], "a").write(f"{sys.argv[2]},1,{d['PDMS']},{d['NC']},{d['DAC']},{d['EP']},{d['TTC']},{d['C']},-\n")
PY
  # env for the k-stress passes (same environment as remote_run)
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
