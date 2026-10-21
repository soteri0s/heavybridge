#!/bin/bash
# =============================================================================
# ablation_alpha.sh — alpha ablation at fixed jump energy gJ=5/4 (reviewer 3,
# TMLR #11166, 2026-09-15). ONE pod, sequential. Mirrors controls_m3J8.sh
# (train + navmini forced sweep, via remote_run.sh) and navtest_controls.sh
# (navtest dump + k=1/2/20) code paths verbatim.
#
#   R3ga7 : tempered per-step, alpha=1.7, gJ=5/4 (kappa=0.193775)   [train+eval]
#   R3ga6 : tempered per-step, alpha=1.6, gJ=5/4 (kappa=0.085732)   [train+eval]
#   (R3g, the headline, is alpha=1.8 at the same gJ, kappa=0.956338)
#
# Predictions written before the run: docs/preregistered_predictions.md.
#
# Data: waits for the markers written by runpod_session/bootstrap_data.sh
#   /scratch/data/navsim/MINI_COMPLETE, NAVTRAIN_COMPLETE (phase mini),
#   /scratch/data/navsim/TEST_COMPLETE (phase test).
# Run:      setsid nohup bash ablation_alpha.sh > /scratch/mc/ablation_driver.log 2>&1 &
# Progress: grep ABL /scratch/mc/phase1.log ; tables: /scratch/phase1/forced_summary_alpha.csv,
#           /workspace/mc/navtest/summary.csv
# Download afterwards (before terminating!):
#   /scratch/phase1/forced_summary_alpha.csv
#   /scratch/phase1/{R3ga7,R3ga6}/results  (+ run.log; dumps optional, ~140MB each)
#   /workspace/mc/navtest/{summary.csv,R3ga7/,R3ga6/}
#   /workspace/mc/ckpt_backup/{R3ga7,R3ga6}_final.pth
# =============================================================================
set -u
RUNPOD=/scratch/mc/experiments_navsim/runpod
cd "$RUNPOD"
CFG=config.env
LOG=/scratch/mc/phase1.log
SUM=/scratch/phase1/forced_summary_alpha.csv
NTSUM=/workspace/mc/navtest/summary.csv
KS="1 2 3 5 10 20"
VARIANTS="R3ga7 R3ga6"
mkdir -p /workspace/mc/ckpt_backup /scratch/phase1 /workspace/mc/navtest
log() { echo "ABL $* $(date -u)" >> "$LOG"; }
setk() { if grep -q "^$1=" "$CFG"; then sed -i "s|^$1=.*|$1=\"$2\"|" "$CFG"; else echo "$1=\"$2\"" >> "$CFG"; fi; }
mix_env() { case "$1" in
  R3ga7) echo "HB_MIXING tempered HB_ALPHA 1.7 HB_KAPPA 0.193775 HB_CHURN_ETA 0.5" ;;
  R3ga6) echo "HB_MIXING tempered HB_ALPHA 1.6 HB_KAPPA 0.085732 HB_CHURN_ETA 0.5" ;;
esac; }

grep -q 'R3ga7)' train_stage.sh || { log "train_stage.sh is NOT patched (R3ga7 case missing) — abort"; exit 1; }

# ---------------- common config (paths) ---------------------------------------
setk REMOTE_WORKDIR /scratch/mc
setk DATA_ROOT /scratch/data
setk DATA_VOLUME /scratch/data
setk HB_NUM_WORKERS 12
_np=$(nproc); [ "$_np" -gt 64 ] && _np=64
setk HB_RAY_THREADS "$_np"
setk HB_LOG_LAMBDA ""

# ============================ PHASE MINI ======================================
setk EVAL_SPLIT navmini
setk PDM_WORKER ray_distributed
log "waiting for navmini + navtrain markers"
while [[ ! -f /scratch/data/navsim/MINI_COMPLETE || ! -f /scratch/data/navsim/NAVTRAIN_COMPLETE ]]; do sleep 60; done
log "data markers present"

run_variant() {  # V extra-setk-pairs...   (phase3.sh / controls_m3J8.sh, verbatim)
  local V="$1"; shift
  setk VARIANT "$V"
  while [[ $# -gt 0 ]]; do setk "$1" "$2"; shift 2; done
  rm -f DONE FAILED run.log; rm -rf results selections dumps; mkdir -p results selections dumps
  log "$V run starting"
  bash remote_run.sh > run.log 2>&1
  local dest="/scratch/phase1/$V"; mkdir -p "$dest"
  cp -r results run.log "$dest/" 2>/dev/null || true
  rm -rf "$dest/dumps"; cp -r dumps "$dest/dumps" 2>/dev/null || true
  [[ -f "/scratch/data/checkpoints/$V/${V}_final.pth" ]] && \
    cp "/scratch/data/checkpoints/$V/${V}_final.pth" /workspace/mc/ckpt_backup/
  if [[ -f DONE ]]; then log "$V DONE (ckpt backed up)"; else log "$V FAILED"; return 1; fi
}
for V in $VARIANTS; do
  if [[ $(ls /scratch/phase1/$V/dumps/*.npz 2>/dev/null | wc -l) -lt 300 ]]; then
    set -- $(mix_env "$V"); run_variant "$V" "$@" || exit 1
  else
    log "$V already trained/dumped — skipping training"
  fi
done

# ---------------- eval env (phase2_forced.sh, verbatim) -----------------------
source /scratch/mc/miniconda/etc/profile.d/conda.sh
conda activate /scratch/mc/miniconda/envs/navsim 2>/dev/null || conda activate navsim
set -a; source "$CFG"; set +a
export NUPLAN_MAPS_ROOT=/scratch/data/navsim/maps OPENSCENE_DATA_ROOT=/scratch/data/navsim
export NAVSIM_DEVKIT_ROOT=/scratch/mc/src/DiffusionDrive NAVSIM_EXP_ROOT=/scratch/mc/exp
export NAVSIM_LOG_PATH=/scratch/data/navsim/mini_navsim_logs/mini
export NAVSIM_SENSOR_ROOT=/scratch/data/navsim/mini_sensor_blobs/mini
export PYTHONPATH="$RUNPOD:${PYTHONPATH:-}"
export MC_PLAN_ANCHOR=/scratch/mc/kmeans_navsim_traj_20.npy MC_CONFIG="$RUNPOD/../config.yaml"
[ -s "$SUM" ] || echo "variant,k,PDMS,NC,DAC,EP,TTC,C" > "$SUM"

for V in $VARIANTS; do
  mkdir -p "/scratch/phase1/$V/results/forced"
  python select_forced_anchor.py --dumps "/scratch/phase1/$V/dumps" \
      --out "/scratch/phase1/$V/results/forced" --ks "$KS" --run-id "$V" \
      >> /scratch/mc/ablation_dump.log 2>&1 || { log "$V selections FAILED"; exit 1; }
  log "$V selections built (k=$KS)"
done

score_pass() {  # variant k worker -> 0 ok   (controls_m3J8.sh, verbatim)
  local V="$1" K="$2" W="$3"
  local sel="/scratch/phase1/$V/results/forced/selection_k${K}.json"
  local dumps="/scratch/phase1/$V/dumps"
  local outdir="/scratch/mc/exp/abl_${V}_k${K}_${W}"
  local WARGS=""; [[ "$W" == "ray_distributed" ]] && WARGS="worker.threads_per_node=${HB_RAY_THREADS:-64}"
  [[ -s "$sel" && -d "$dumps" ]] || { log "$V k$K missing inputs, skip"; return 2; }
  MC_SELECTION="$sel" MC_DUMP_DIR="$dumps" timeout 5400 python \
    "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py" \
    agent=replay_agent train_test_split=navmini \
    metric_cache_path="$NAVSIM_EXP_ROOT/metric_cache" \
    navsim_log_path="$NAVSIM_LOG_PATH" sensor_blobs_path="$NAVSIM_SENSOR_ROOT" \
    output_dir="$outdir" worker="$W" $WARGS >> /scratch/mc/ablation_scoring.log 2>&1 || return 1
  local csv=$(ls -t "$outdir"/*.csv 2>/dev/null | head -1)
  [[ -n "$csv" ]] || return 1
  cp "$csv" "/scratch/phase1/$V/results/perscene_k${K}.csv" 2>/dev/null || true
  python - "$csv" "$V" "$K" "$SUM" <<'PY'
import sys, csv as c
rows = list(c.reader(open(sys.argv[1])))
hdr, avg = rows[0], [r for r in rows if len(r) > 1 and r[1] == "average"][-1]
vals = avg[3:]
open(sys.argv[4], "a").write(f"{sys.argv[2]},{sys.argv[3]}," + ",".join([vals[-1]] + vals[:-1]) + "\n")
PY
  return 0
}
for V in $VARIANTS; do
  for K in $KS; do
    grep -q "^$V,$K," "$SUM" && continue
    log "$V k$K scoring (ray)"
    if ! score_pass "$V" "$K" ray_distributed; then
      log "$V k$K ray failed -> sequential"
      score_pass "$V" "$K" sequential || log "$V k$K FAILED (both workers)"
    fi
  done
done
log "PHASE MINI DONE"; cat "$SUM" >> "$LOG"

# ============================ PHASE TEST ======================================
log "waiting for navtest marker"
while [[ ! -f /scratch/data/navsim/TEST_COMPLETE ]]; do sleep 60; done
log "navtest present"
setk EVAL_SPLIT navtest
setk PDM_WORKER ray_distributed
[ -s "$NTSUM" ] || echo "variant,k,PDMS,NC,DAC,EP,TTC,C,DDC" > "$NTSUM"

score_k() {  # V K   (navtest_controls.sh, verbatim)
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
  python - "$csv" "$V" "$K" "$NTSUM" <<'PY'
import sys, csv as c
rows = list(c.reader(open(sys.argv[1])))
avg = [r for r in rows if len(r) > 1 and r[1] == "average"][-1]
v = avg[3:]
open(sys.argv[4], "a").write(",".join([sys.argv[2], sys.argv[3], v[-1], v[0], v[1], v[2], v[3], v[4], v[5]]) + "\n")
PY
}

for V in $VARIANTS; do
  D="/workspace/mc/navtest/$V"
  if [ -f "$D/DONE" ]; then log "$V navtest already done — skipping"; continue; fi
  mkdir -p "$D"
  [ -s "/scratch/data/checkpoints/$V/${V}_final.pth" ] || { log "$V ckpt missing — aborting"; exit 1; }
  touch "/scratch/data/checkpoints/$V/TRAIN_DONE"
  setk VARIANT "$V"
  set -- $(mix_env "$V"); while [ $# -gt 0 ]; do setk "$1" "$2"; shift 2; done
  rm -f DONE FAILED run.log
  rm -rf results selections dumps; mkdir -p results selections dumps
  echo "$V" > /scratch/mc/.last_variant
  log "$V navtest pipeline starting (dump+score, no training)"
  bash remote_run.sh > run.log 2>&1
  mkdir -p "/scratch/navtest/$V"
  cp -r results "$D/" 2>/dev/null || true
  cp run.log "$D/" 2>/dev/null || true
  rm -rf "/scratch/navtest/$V/dumps"; mv dumps "/scratch/navtest/$V/dumps" 2>/dev/null || true
  if [ -f DONE ]; then log "$V navtest remote_run DONE"; else log "$V navtest FAILED"; exit 1; fi
  _b=$(ls -t /scratch/mc/exp/base/*.csv 2>/dev/null | head -1)
  [ -n "$_b" ] && cp "$_b" "$D/perscene_base.csv"
  python - "$D/results/eval_base.json" "$V" "$NTSUM" <<'PY' || true
import sys, json
d = json.load(open(sys.argv[1]))
open(sys.argv[3], "a").write(f"{sys.argv[2]},1,{d['PDMS']},{d['NC']},{d['DAC']},{d['EP']},{d['TTC']},{d['C']},-\n")
PY
  set -a; source "$CFG"; set +a
  export NAVSIM_LOG_PATH=/scratch/data/navsim/test_navsim_logs/test
  export NAVSIM_SENSOR_ROOT=/scratch/data/navsim/test_sensor_blobs/test
  _p="$(find /scratch/data/navsim -maxdepth 3 -type d -name 'CAM_F0' -path '*test*' -print -quit 2>/dev/null)"
  [ -n "$_p" ] && export NAVSIM_SENSOR_ROOT="$(dirname "$(dirname "$_p")")"
  _l="$(find /scratch/data/navsim -maxdepth 4 -name '*.pkl' -path '*test*' -print -quit 2>/dev/null)"
  [ -n "$_l" ] && export NAVSIM_LOG_PATH="$(dirname "$_l")"
  for K in 2 20; do
    log "$V k$K stress scoring"
    score_k "$V" "$K" || log "$V k$K FAILED"
    _c=$(ls -t "/scratch/mc/exp/nt_${V}_k${K}"/*.csv 2>/dev/null | head -1)
    [ -n "$_c" ] && cp "$_c" "$D/perscene_k${K}.csv"
  done
  touch "$D/DONE"; log "$V navtest ALL DONE"
done
log "ALL DONE"
cat "$NTSUM" >> "$LOG"
