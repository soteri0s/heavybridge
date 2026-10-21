#!/usr/bin/env bash
# =============================================================================
# train_stage.sh — TRAIN stage for the headline-decision experiment.
# Runs ON THE POD, after remote_run.sh's env/dataset setup and BEFORE the dump.
# Same conventions as remote_run.sh: set -euo pipefail, STAGE markers, idempotent
# (skips finished variants via sentinel files), fail-loud via the caller's trap.
#
# Inputs (from config.env additions): VARIANT, HB_* envs, NAVTRAIN under
# $DATA_ROOT, RCLONE_REMOTE for checkpoint push. One variant per pod.
# =============================================================================
set -euo pipefail
RUNPOD_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$RUNPOD_DIR/config.env"

STAGE="train:${VARIANT:?set VARIANT in config.env (R0|R1a|R1b|R2|R3t|R3s)}"
CKPT_DIR="${DATA_ROOT:?}/checkpoints/${VARIANT}"
SENTINEL="$CKPT_DIR/TRAIN_DONE"
mkdir -p "$CKPT_DIR"

# variant -> HB_* env mapping (single source of truth)
case "$VARIANT" in
  R0)   export HB_MIXING=none ;;
  R1a)  export HB_MIXING=ig HB_NU=5 ;;
  R1b)  export HB_MIXING=ig HB_NU=10 ;;
  R2)   export HB_MIXING=ig-learned ;;
  R2b)  export HB_MIXING=ig-bounded HB_NU=10 HB_M_REG=1.0 ;;
  R2c0) export HB_MIXING=ig-learned ;;   # reviewer-m3J8 control B: the R2 checkpoint evaluated with HB_CHURN_ETA=0 (eval-only; TRAIN_DONE sentinel)  # reviewer-m3J8 control A (2026-09): bounded amortized per-rollout
  R3t)  export HB_MIXING=tempered HB_ALPHA=1.7 HB_KAPPA="${HB_KAPPA:?run kappa_from_nu first, put value in config.env}" ;;
  R3s)  export HB_MIXING=stable HB_ALPHA=1.9 ;;
  # ---- phase-3 "R3 redemption" variants (2026-07-22) ----
  R3k)  export HB_MIXING=tempered HB_ALPHA=1.7 HB_KAPPA="${HB_KAPPA:?}" \
               HB_KAPPA_END_MULT=20 ;;   # step-dependent tempering: jumps early, tame late
  R3g)  export HB_MIXING=tempered HB_ALPHA=1.8 HB_KAPPA=0.956338 ;;  # gentle: kappa_from_nu(a=1.8, nu=10), MC-verified
  # ---- alpha ablation at fixed jump energy gJ=5/4 (reviewer 3, 2026-09-15): kappa from Eq. (kappacal), MC-checked to 1e-4 ----
  R3ga7) export HB_MIXING=tempered HB_ALPHA=1.7 HB_KAPPA=0.193775 ;;  # alpha=1.7, gJ=5/4 (tilt-rejection acceptance 0.75)
  R3ga6) export HB_MIXING=tempered HB_ALPHA=1.6 HB_KAPPA=0.085732 ;;  # alpha=1.6, gJ=5/4 (acceptance 0.88)
  R23)  export HB_MIXING=tempered-learned HB_ALPHA=1.7 HB_KAPPA="${HB_KAPPA:?}" ;;  # amortized jump intensity m(z)
  # ---- full-schedule (100-epoch, stock cosine completes) variants (2026-07-30) ----
  R0full*)  export HB_MIXING=none ;;              # R0full, R0full_s2, R0full_s3 (seed replicates)
  R3gfull*) export HB_MIXING=tempered HB_ALPHA=1.8 HB_KAPPA=0.956338 ;;  # + seed replicates
  R2bfull*) export HB_MIXING=ig-bounded HB_NU=10 HB_M_REG=1.0 ;;  # bounded amortized per-rollout, full schedule (STANDBY runbook SEEDS_RUNBOOK_R2b.md, 2026-09-14)
  # ---- navtest-era "gentler Lévy" variants (2026-07-28) ----
  R3g2) export HB_MIXING=tempered HB_ALPHA=1.8 HB_KAPPA=3.105537 ;;  # kappa_from_nu(1.8, nu=20), acc-checked
  R3gk) export HB_MIXING=tempered HB_ALPHA=1.8 HB_KAPPA=0.956338 \
               HB_KAPPA_END_MULT=2 ;;  # gentle base + mild early-jump ramp
  R23g) export HB_MIXING=tempered-learned HB_ALPHA=1.8 HB_KAPPA=0.956338 ;;  # bounded m(z) on gentle base
  UP)   export HB_MIXING=upstream ;;   # true upstream DiffusionDrive control
  *) echo "unknown VARIANT=$VARIANT"; exit 1 ;;
esac

if [[ -f "$SENTINEL" ]]; then
  echo "[train_stage] $VARIANT already trained (sentinel present) — skipping."
  exit 0
fi

# HB_CACHE_RESUME=1 (2026-09-13): a PARTIAL training_cache (interrupted caching) must
# not be used as cache-only; with both flags False navsim caches the missing tokens
# and trains on the full navtrain split.
# --- navtrain data (idempotent, big: ~450 GB — must be on DATA_VOLUME) -------
if [[ ! -d "$DATA_ROOT/navtrain_sensors" ]]; then
  echo "[train_stage] downloading navtrain split (this is the big one)..."
  # TODO-VERIFY-ON-POD: exact HF dataset paths as in remote_run.sh's navtest
  # block, switching the split to trainval/navtrain. Mirror that block here.
  bash "$RUNPOD_DIR/download_navtrain.sh" "$DATA_ROOT"
fi

# --- run training -------------------------------------------------------------
# TODO-VERIFY-ON-POD (dry run will surface the exact entrypoint): DiffusionDrive
# trains via navsim's run_training with its agent config, e.g.:
#   python $NAVSIM_DIR/navsim/planning/script/run_training.py \
#       agent=diffusiondrive_agent experiment_name=$VARIANT ...
# We wrap it so apply_patch.py is imported FIRST.
cat > /tmp/hb_train_entry.py <<'PY'
import apply_patch  # noqa: F401  (monkeypatch before anything navsim)
import runpy, sys
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
PY

# navsim's configs resolve data as $OPENSCENE_DATA_ROOT/navsim_logs/<split> and
# $OPENSCENE_DATA_ROOT/sensor_blobs/<split>; our download layout is
# trainval_navsim_logs/trainval etc. Bridge with symlinks (idempotent, zero-copy).
# (verified on pod 2026-07-14: default_evaluation.yaml oc.env OPENSCENE_DATA_ROOT)
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-$DATA_ROOT/navsim}"
mkdir -p "$OPENSCENE_DATA_ROOT/navsim_logs" "$OPENSCENE_DATA_ROOT/sensor_blobs"
ln -sfn "$OPENSCENE_DATA_ROOT/trainval_navsim_logs/trainval" "$OPENSCENE_DATA_ROOT/navsim_logs/trainval"
ln -sfn "$OPENSCENE_DATA_ROOT/trainval_sensor_blobs/trainval" "$OPENSCENE_DATA_ROOT/sensor_blobs/trainval"

# ResNet34 backbone weights + clustered anchors: transfuser_config.py hardcodes
# the author's paths (found on dry run 2026-07-14) -> download once, override via
# hydra. Anchors URL is the official DiffusionDrive release (docs/train_eval.md).
BKB="${REMOTE_WORKDIR:?}/ckpt/resnet34_a1_in1k.bin"
ANCHORS="${REMOTE_WORKDIR:?}/ckpt/kmeans_navsim_traj_20.npy"
mkdir -p "$(dirname "$BKB")"
if [[ ! -s "$BKB" ]]; then
  curl -fL -o "$BKB" "https://huggingface.co/timm/resnet34.a1_in1k/resolve/main/pytorch_model.bin"
fi
if [[ ! -s "$ANCHORS" ]]; then
  curl -fL -o "$ANCHORS" "https://github.com/hustvl/DiffusionDrive/releases/download/DiffusionDrive_88p1_PDMS_Eval_file/kmeans_navsim_traj_20.npy"
fi

# fine-tune mode (HB_FINETUNE=1): warm-start every variant from the public
# DiffusionDrive checkpoint (already downloaded by remote_run stage 5), few
# epochs, reduced lr. This is the cheap Phase-1 protocol (2026-07-18): all
# variants share the same strong init; from-scratch is the expensive fallback.
FT_ARGS=""
if [[ "${HB_FINETUNE:-0}" == "1" ]]; then
  export HB_WARMSTART_CKPT="${HB_WARMSTART_CKPT:-$REMOTE_WORKDIR/ckpt/diffusiondrive.pth}"
  [[ -s "$HB_WARMSTART_CKPT" ]] || { echo "[train_stage] HB_FINETUNE=1 but warm-start ckpt missing: $HB_WARMSTART_CKPT"; exit 1; }
  FT_ARGS="trainer.params.max_epochs=${HB_FT_EPOCHS:-15} agent.lr=${HB_FT_LR:-1e-4}"
  echo "[train_stage] FINE-TUNE mode: init=$HB_WARMSTART_CKPT epochs=${HB_FT_EPOCHS:-15} lr=${HB_FT_LR:-1e-4}"
fi

# resume from the newest lightning checkpoint if one exists (apply_patch.py
# injects it into Trainer.fit via HB_RESUME_CKPT; overrides the warm start)
RESUME_CKPT=$(ls -t "$CKPT_DIR"/lightning_logs/version_*/checkpoints/*.ckpt 2>/dev/null | head -1 || true)
if [[ -n "$RESUME_CKPT" ]]; then
  export HB_RESUME_CKPT="$RESUME_CKPT"
  echo "[train_stage] resuming from $RESUME_CKPT"
fi

# --- seed replicates (2026-08-04): HB_SEED sets the hydra seed override.
# navsim's training config exposes `seed`; if this navsim version names it
# differently, the first attempt fails fast with "Could not override 'seed'"
# and we retry once with `+seed=` (append). Empty HB_SEED = stock behavior,
# identical to the original R0full/R3gfull runs.
SEED_ARG=""
[[ -n "${HB_SEED:-}" ]] && SEED_ARG="seed=${HB_SEED}"

run_training() {  # $1 = seed override (may be empty)
  python /tmp/hb_train_entry.py \
    "${NAVSIM_DEVKIT_ROOT:?remote_run.sh exports this}/navsim/planning/script/run_training.py" \
    agent=diffusiondrive_agent \
    +agent.config.bkb_path="$BKB" \
    +agent.config.plan_anchor_path="$ANCHORS" \
    train_test_split=navtrain \
    experiment_name="hb_${VARIANT}" \
    output_dir="$CKPT_DIR" \
    use_cache_without_dataset=$([[ -d "$NAVSIM_EXP_ROOT/training_cache" && "${HB_CACHE_RESUME:-0}" != "1" ]] && echo True || echo False) \
    force_cache_computation=$([[ -d "$NAVSIM_EXP_ROOT/training_cache" ]] && echo False || echo True) \
    trainer.params.precision=bf16-mixed \
    trainer.params.gradient_clip_val=1.0 \
    dataloader.params.num_workers="${HB_NUM_WORKERS:-4}" \
    +agent.config.bev_semantic_weight="${HB_BEV_WEIGHT:-1.0}" \
    $FT_ARGS \
    $1 \
    ${TRAIN_EXTRA_ARGS:-} 2>&1 | tee "$CKPT_DIR/train.log"
}

export PYTHONPATH="$RUNPOD_DIR:$PYTHONPATH"
if ! run_training "$SEED_ARG"; then
  if [[ -n "$SEED_ARG" ]] && grep -q "Could not override .seed." "$CKPT_DIR/train.log"; then
    echo "[train_stage] hydra has no 'seed' key; retrying with '+seed=' append"
    run_training "+seed=${HB_SEED}"
  else
    exit 1
  fi
fi

# --- export final ckpt as .pth (remote_run's 6b check + agent loader expect it;
# lightning ckpt already has the {"state_dict": {"agent.*": ...}} layout) -------
FINAL_CKPT=$(ls -t "$CKPT_DIR"/lightning_logs/version_*/checkpoints/*.ckpt 2>/dev/null | head -1)
if [[ -n "$FINAL_CKPT" ]]; then
  cp -f "$FINAL_CKPT" "$CKPT_DIR/${VARIANT}_final.pth"
  echo "[train_stage] exported $FINAL_CKPT -> $CKPT_DIR/${VARIANT}_final.pth"
fi

# --- backup: NEVER volume-only again ------------------------------------------
if [[ -n "${RCLONE_REMOTE:-}" ]]; then
  rclone copy "$CKPT_DIR" "$RCLONE_REMOTE/heavybridge/$VARIANT" \
    --include "*.pth" --include "train.log" -P
fi
touch "$SENTINEL"
echo "[train_stage] $VARIANT done."
