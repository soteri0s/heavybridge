#!/bin/bash
# =============================================================================
# phase2_forced.sh — Phase-2 decision metrics (2026-07-22): forced-anchor PDMS
# at k in {2,3,5} for R0,R1a,R1b,R2,R3t on the archived navmini dumps.
#   1. R0 lacks an archived dump (predates the archiver) -> GPU redump from the
#      durable ckpt + fresh forced selections.
#   2. Each (variant,k): ReplayAgent PDM scoring of selection_k<k>.json against
#      the variant dump. Tries ray_distributed (all vCPUs), falls back to
#      sequential per pass.
# Output: /scratch/phase1/forced_summary.csv  (variant,k,PDMS + subscores)
# Run detached: setsid nohup bash phase2_forced.sh > /scratch/mc/phase2.log 2>&1 &
# =============================================================================
set -u
source /scratch/mc/miniconda/etc/profile.d/conda.sh
conda activate /scratch/mc/miniconda/envs/navsim 2>/dev/null || conda activate navsim
RUNPOD=/scratch/mc/experiments_navsim/runpod
cd "$RUNPOD"
set -a; source config.env; set +a
export NUPLAN_MAPS_ROOT=/scratch/data/navsim/maps
export OPENSCENE_DATA_ROOT=/scratch/data/navsim
export NAVSIM_DEVKIT_ROOT=/scratch/mc/src/DiffusionDrive
export NAVSIM_EXP_ROOT=/scratch/mc/exp
export NAVSIM_LOG_PATH=/scratch/data/navsim/mini_navsim_logs/mini
export NAVSIM_SENSOR_ROOT=/scratch/data/navsim/mini_sensor_blobs/mini
export PYTHONPATH="$RUNPOD:${PYTHONPATH:-}"
export MC_PLAN_ANCHOR=/scratch/mc/kmeans_navsim_traj_20.npy
export MC_CONFIG="$RUNPOD/../config.yaml"
SUMMARY=/scratch/phase1/forced_summary.csv
echo "variant,k,PDMS,NC,DAC,EP,TTC,C" > "$SUMMARY"
log() { echo "PHASE2 $* $(date -u)" >> /scratch/mc/phase1.log; }

# --- 1. redump any variant whose archived dump is missing/empty ---------------
# (the live archiver lost the R1a/R1b/R2/R3t dumps to driver clean-up races;
# every dump is regenerable from the durable checkpoints)
ckpt_of() { case "$1" in
  R0)  echo R0_finetune_ep15.pth ;;
  *)   echo "${1}_final.pth" ;;
esac; }
env_of() { case "$1" in   # echoed as VAR=VAL pairs for the dump env
  R0)  echo "HB_MIXING=none" ;;
  R1a) echo "HB_MIXING=ig HB_NU=5" ;;
  R1b) echo "HB_MIXING=ig HB_NU=10" ;;
  R2)  echo "HB_MIXING=ig-learned" ;;
  R3t) echo "HB_MIXING=tempered HB_ALPHA=1.7" ;;
esac; }
ensure_dump() {  # variant
  local V="$1"
  if [[ -d "/scratch/phase1/$V/dumps" && $(ls "/scratch/phase1/$V/dumps" 2>/dev/null | wc -l) -ge 300 ]]; then
    return 0
  fi
  log "$V redump starting"
  mkdir -p "/scratch/phase1/$V/dumps"
  env $(env_of "$V") HB_CHURN_ETA=0.5 \
      MC_CHECKPOINT="/workspace/mc/ckpt_backup/$(ckpt_of "$V")" \
      CUDA_VISIBLE_DEVICES=0 python hb_run.py run_dump.py --split navmini \
      --out "/scratch/phase1/$V/dumps" --config ../config.yaml \
      >> /scratch/mc/phase2.log 2>&1 || { log "$V redump FAILED"; return 1; }
  mkdir -p "/scratch/phase1/$V/results/forced"
  python select_forced_anchor.py --dumps "/scratch/phase1/$V/dumps" \
      --out "/scratch/phase1/$V/results/forced" --ks "1 2 3 5" --run-id "$V" \
      >> /scratch/mc/phase2.log 2>&1 || { log "$V selections FAILED"; return 1; }
  log "$V redump done"
}
for V in R0 R1a R1b R2 R3t; do ensure_dump "$V" || exit 1; done

# always (re)build selections from the dumps — cheap, and guarantees the
# trajectory-based format the ReplayAgent needs
for V in R0 R1a R1b R2 R3t; do
  mkdir -p "/scratch/phase1/$V/results/forced"
  python select_forced_anchor.py --dumps "/scratch/phase1/$V/dumps" \
      --out "/scratch/phase1/$V/results/forced" --ks "1 2 3 5" --run-id "$V" \
      >> /scratch/mc/phase2.log 2>&1 || { log "$V selections FAILED"; exit 1; }
  log "$V selections rebuilt"
done

# --- 2. scoring passes --------------------------------------------------------
score_pass() {  # variant k worker -> 0 ok
  local V="$1" K="$2" W="$3"
  local sel="/scratch/phase1/$V/results/forced/selection_k${K}.json"
  local dumps="/scratch/phase1/$V/dumps"
  local outdir="/scratch/mc/exp/forced_${V}_k${K}_${W}"
  [[ -s "$sel" && -d "$dumps" ]] || { log "$V k$K missing inputs, skip"; return 2; }
  MC_SELECTION="$sel" MC_DUMP_DIR="$dumps" timeout 5400 python \
    "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py" \
    agent=replay_agent train_test_split=navmini \
    metric_cache_path="$NAVSIM_EXP_ROOT/metric_cache" \
    navsim_log_path="$NAVSIM_LOG_PATH" sensor_blobs_path="$NAVSIM_SENSOR_ROOT" \
    output_dir="$outdir" worker="$W" >> /scratch/mc/phase2.log 2>&1 || return 1
  local csv=$(ls -t "$outdir"/*.csv 2>/dev/null | head -1)
  [[ -n "$csv" ]] || return 1
  python - "$csv" "$V" "$K" "$SUMMARY" <<'PY'
import sys, csv as c
rows = list(c.reader(open(sys.argv[1])))
hdr, avg = rows[0], [r for r in rows if len(r) > 1 and r[1] == "average"][-1]
# columns: score fields; last col = PDMS-like aggregate (as in eval collection)
vals = avg[3:]
open(sys.argv[4], "a").write(f"{sys.argv[2]},{sys.argv[3]}," + ",".join([vals[-1]] + vals[:-1]) + "\n")
PY
  return 0
}

for V in R0 R1a R1b R2 R3t; do
  for K in 2 3 5; do
    log "$V k$K scoring (ray)"
    if ! score_pass "$V" "$K" ray_distributed; then
      log "$V k$K ray failed -> sequential"
      score_pass "$V" "$K" sequential || log "$V k$K FAILED (both workers)"
    fi
  done
done
log "PHASE2 ALL DONE"
cat "$SUMMARY" >> /scratch/mc/phase1.log
