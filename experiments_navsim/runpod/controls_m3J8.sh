#!/bin/bash
# =============================================================================
# controls_m3J8.sh — the two collapse-mechanism controls requested by reviewer
# m3J8 (TMLR #11166, 2026-09). ONE pod, sequential, navmini. Mirrors phase3.sh
# (training/eval via remote_run.sh) and phase2_forced.sh (redump + forced
# scoring) code paths verbatim.
#
#   Control B  R2c0 : the EXISTING amortized R2 checkpoint, churn OFF (eta=0),
#                     forced-anchor sweep k in {1,2,3,5,10,20}   [eval only]
#                     + log of the realized per-rollout lambda and the head's
#                       IG parameters (A,B) per scene  (HB_LOG_LAMBDA)
#   Control A  R2b  : bounded amortized per-rollout, lam = m(z)*IG(nu/2,nu/2),
#                     m(z) in [1/4,4], nu=10 (patched heavybridge_head.py,
#                     mixing=ig-bounded), 15-epoch screening recipe, then the
#                     same forced sweep                           [train+eval]
#
# Pre-registered predictions (docs/preregistered_predictions.md):
#   B: collapse persists at k=20 (~58) — churn-off adds no escape channel.
#   A: no collapse (k=20 within ~1 of argmax), argmax at R0/R1b level.
#
# Prereqs on the pod (same as phase3.sh):
#   - harness rsynced to /scratch/mc/experiments_navsim  WITH the patched
#     runpod/heavybridge_head.py and runpod/train_stage.sh
#   - config.env paths sed'ed as usual (REMOTE_WORKDIR, DATA_ROOT ...); the
#     15-epoch screening recipe is the config default (as in phases 1-3)
#   - /workspace/mc/ckpt_backup/R2_final.pth uploaded from checkpoints/
#   - navmini data present or bootstrapped by remote_run.sh on first run
# Run:      setsid nohup bash controls_m3J8.sh > /scratch/mc/controls_driver.log 2>&1 &
# Progress: grep CTRL /scratch/mc/phase1.log ; table: /scratch/phase1/forced_summary_m3J8.csv
# Download afterwards (before terminating!):
#   /scratch/phase1/forced_summary_m3J8.csv
#   /scratch/phase1/R2c0/{results,lambda_log.csv}   /scratch/phase1/R2b/results
#   /workspace/mc/ckpt_backup/R2b_final.pth
#   dumps (optional, ~140MB each): /scratch/phase1/R2c0/dumps /scratch/phase1/R2b/dumps
# =============================================================================
set -u
RUNPOD=/scratch/mc/experiments_navsim/runpod
cd "$RUNPOD"
CFG=config.env
LOG=/scratch/mc/phase1.log
SUM=/scratch/phase1/forced_summary_m3J8.csv
KS="1 2 3 5 10 20"
mkdir -p /workspace/mc/ckpt_backup /scratch/phase1
log() { echo "CTRL $* $(date -u)" >> "$LOG"; }
setk() { if grep -q "^$1=" "$CFG"; then sed -i "s|^$1=.*|$1=\"$2\"|" "$CFG"; else echo "$1=\"$2\"" >> "$CFG"; fi; }

# sanity: the patch must be in place
grep -q '"ig-bounded"' heavybridge_head.py || { log "heavybridge_head.py is NOT patched (ig-bounded missing) — abort"; exit 1; }
grep -q 'R2b)' train_stage.sh || { log "train_stage.sh is NOT patched (R2b case missing) — abort"; exit 1; }

# ---------------- Control A: train R2b (phase3.sh run_variant, verbatim) -------
run_variant() {  # V mixing extra-setk-pairs...
  local V="$1"; shift
  setk VARIANT "$V"; setk HB_CHURN_ETA "0.5"
  while [[ $# -gt 0 ]]; do setk "$1" "$2"; shift 2; done
  rm -f DONE FAILED run.log; rm -rf results selections dumps; mkdir -p results selections dumps
  log "$V run starting"
  bash remote_run.sh > run.log 2>&1
  local dest="/scratch/phase1/$V"; mkdir -p "$dest"
  cp -r results run.log "$dest/" 2>/dev/null || true
  rm -rf "$dest/dumps"; cp -r dumps "$dest/dumps" 2>/dev/null || true   # never nest into a stale dest/dumps (2026-09-14)
  [[ -f "/scratch/data/checkpoints/$V/${V}_final.pth" ]] && \
    cp "/scratch/data/checkpoints/$V/${V}_final.pth" /workspace/mc/ckpt_backup/
  if [[ -f DONE ]]; then log "$V DONE (ckpt backed up)"; else log "$V FAILED"; return 1; fi
}
if [[ $(ls /scratch/phase1/R2b/dumps/*.npz 2>/dev/null | wc -l) -lt 300 ]]; then   # a failed attempt leaves an EMPTY dumps dir: re-run
  run_variant R2b HB_MIXING ig-bounded HB_NU 10 HB_M_REG 1.0 || exit 1
else
  log "R2b already trained/dumped — skipping training"
fi

# ---------------- common eval env (phase2_forced.sh, verbatim) ----------------
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

# ---------------- Control B: R2 churn-off redump + lambda log -----------------
if [[ ! -d /scratch/phase1/R2c0/dumps || $(ls /scratch/phase1/R2c0/dumps 2>/dev/null | wc -l) -lt 300 ]]; then
  log "R2c0 dump starting (R2 checkpoint, HB_CHURN_ETA=0)"
  mkdir -p /scratch/phase1/R2c0/dumps
  rm -f /scratch/phase1/R2c0/lambda_log.csv
  env HB_MIXING=ig-learned HB_CHURN_ETA=0 HB_LOG_LAMBDA=/scratch/phase1/R2c0/lambda_log.csv \
      MC_CHECKPOINT=/workspace/mc/ckpt_backup/R2_final.pth CUDA_VISIBLE_DEVICES=0 \
      python hb_run.py run_dump.py --split navmini --out /scratch/phase1/R2c0/dumps --config ../config.yaml \
      >> /scratch/mc/controls_dump.log 2>&1 || { log "R2c0 dump FAILED"; exit 1; }
  log "R2c0 dump done; lambda log lines: $(wc -l < /scratch/phase1/R2c0/lambda_log.csv)"
fi

# ---------------- forced selections + scoring (phase2_forced.sh, verbatim) ----
for V in R2c0 R2b; do
  mkdir -p "/scratch/phase1/$V/results/forced"
  python select_forced_anchor.py --dumps "/scratch/phase1/$V/dumps" \
      --out "/scratch/phase1/$V/results/forced" --ks "$KS" --run-id "$V" \
      >> /scratch/mc/controls_dump.log 2>&1 || { log "$V selections FAILED"; exit 1; }
  log "$V selections built (k=$KS)"
done

score_pass() {  # variant k worker -> 0 ok
  local V="$1" K="$2" W="$3"
  local sel="/scratch/phase1/$V/results/forced/selection_k${K}.json"
  local dumps="/scratch/phase1/$V/dumps"
  local outdir="/scratch/mc/exp/ctrl_${V}_k${K}_${W}"
  # cap ray workers as remote_run.sh does (128-core host: uncapped ray hangs idle, 2026-09-14)
  local WARGS=""; [[ "$W" == "ray_distributed" ]] && WARGS="worker.threads_per_node=${HB_RAY_THREADS:-64}"
  [[ -s "$sel" && -d "$dumps" ]] || { log "$V k$K missing inputs, skip"; return 2; }
  MC_SELECTION="$sel" MC_DUMP_DIR="$dumps" timeout 5400 python \
    "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py" \
    agent=replay_agent train_test_split=navmini \
    metric_cache_path="$NAVSIM_EXP_ROOT/metric_cache" \
    navsim_log_path="$NAVSIM_LOG_PATH" sensor_blobs_path="$NAVSIM_SENSOR_ROOT" \
    output_dir="$outdir" worker="$W" $WARGS >> /scratch/mc/controls_scoring.log 2>&1 || return 1
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

for V in R2c0 R2b; do
  for K in $KS; do
    log "$V k$K scoring (ray)"
    if ! score_pass "$V" "$K" ray_distributed; then
      log "$V k$K ray failed -> sequential"
      score_pass "$V" "$K" sequential || log "$V k$K FAILED (both workers)"
    fi
  done
done
log "ALL DONE"
cat "$SUM" >> "$LOG"
