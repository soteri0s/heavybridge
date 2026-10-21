#!/usr/bin/env bash
# =============================================================================
# remote_run.sh -- runs ON THE RUNPOD POD. Drives the whole pipeline:
#   setup -> dataset/checkpoint download -> (train the HeavyBridge VARIANT)
#   -> ONE inference dump over the split -> CPU-only PDM scoring of the
#   deployed argmax plan -> forced-anchor selections -> results/ + summary.md
#   -> touch DONE.
#
# It is idempotent and re-runnable (skips work already done) and FAILS LOUDLY:
# `set -euo pipefail` + an ERR trap that writes a FAILED sentinel with the line
# and reason, so launch.sh never mistakes a crash for a finished run and we
# never emit silent wrong numbers.
#
# Invoked by launch.sh as:
#     bash remote_run.sh   (cwd = $REMOTE_WORKDIR/experiments_navsim/runpod)
# Logs to run.log; sentinels DONE / FAILED are written in this same dir.
# =============================================================================
set -euo pipefail

RUNPOD_DIR="$(cd "$(dirname "$0")" && pwd)"
EXP_DIR="$(dirname "$RUNPOD_DIR")"          # experiments_navsim/
cd "$RUNPOD_DIR"

# remove any stale sentinels from a previous attempt
rm -f DONE FAILED

# --- fail-loud trap ----------------------------------------------------------
on_err() {
  local ec=$?
  local ln=${1:-?}
  {
    echo "STEP_FAILED at line ${ln} (exit ${ec})"
    echo "stage: ${STAGE:-<unset>}"
    echo "see run.log for the full output"
  } > FAILED
  echo "=============================================================="
  echo "[remote_run] FAILED in stage '${STAGE:-?}' at line ${ln} (exit ${ec})."
  echo "[remote_run] wrote FAILED sentinel; launch.sh will surface this."
  echo "=============================================================="
  exit "$ec"
}
trap 'on_err $LINENO' ERR

STAGE="init"
echo "=============================================================="
echo "[remote_run] start $(date -u +%FT%TZ)"
echo "=============================================================="

# --- config ------------------------------------------------------------------
if [[ ! -f config.env ]]; then
  echo "config.env not found in $RUNPOD_DIR; launch.sh should have shipped it." >&2
  exit 1
fi
# shellcheck disable=SC1091
set -a; source ./config.env; set +a   # export HB_* to python children (fix 2026-07-17)

# --- HeavyBridge: keep EVERYTHING big off the container disk ------------------
# caches (pip/HF/tmp) go to the network volume; the container disk stays small.
_CACHE_ROOT="${DATA_VOLUME:-${DATA_ROOT:-/workspace}}/cache"
mkdir -p "$_CACHE_ROOT/pip" "$_CACHE_ROOT/hf" "$_CACHE_ROOT/tmp"
export PIP_CACHE_DIR="$_CACHE_ROOT/pip"
export HF_HOME="$_CACHE_ROOT/hf"
export HUGGINGFACE_HUB_CACHE="$_CACHE_ROOT/hf"
export TMPDIR="$_CACHE_ROOT/tmp"
echo "    caches -> $_CACHE_ROOT ; disk state:"; df -h / "${DATA_VOLUME:-${DATA_ROOT:-/workspace}}" 2>/dev/null || true

# The code's ACTUAL location wins over config.env's REMOTE_WORKDIR, so that
# src/data/exp/ckpt land under the same --workdir launch.sh shipped the code to
# (e.g. /workspace/mc-dry), not config.env's default. Dataset sharing across
# runs is via DATA_VOLUME, not via this path.
REMOTE_WORKDIR="$(dirname "$EXP_DIR")"

# RunPod's system python is PEP-668 "externally-managed"; allow pip to install
# (the pod is disposable). Covers our installs AND navsim/DiffusionDrive's.
export PIP_BREAK_SYSTEM_PACKAGES=1

# The dataset download scripts untar OpenScene blobs; as root in a container,
# tar tries to restore the archive's original uid/gid and fails with
# "Cannot change ownership ... Operation not permitted". GNU tar reads
# TAR_OPTIONS and applies it to EVERY tar (incl. the ones inside navsim's
# download scripts), so this makes extraction ignore ownership cleanly.
export TAR_OPTIONS="--no-same-owner"

PY="${PYTHON:-python3}"
DUMP_DIR="$RUNPOD_DIR/dumps"
SEL_DIR="$RUNPOD_DIR/selections"
RESULTS="$RUNPOD_DIR/results"
mkdir -p "$DUMP_DIR" "$SEL_DIR" "$RESULTS"

# resolve the shared config.yaml (CONFIG_YAML is relative to experiments_navsim/)
CONFIG_ABS="$(cd "$EXP_DIR" && cd "$(dirname "$CONFIG_YAML")" && pwd)/$(basename "$CONFIG_YAML")"
[[ -f "$CONFIG_ABS" ]] || { echo "config.yaml not found at $CONFIG_ABS" >&2; exit 1; }

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MC_CONFIG="$CONFIG_ABS"
export MC_CUDA_DEVICE="${CUDA_DEVICE:-0}"
export MC_DUMP_DIR="$DUMP_DIR"

echo "[remote_run] split=$EVAL_SPLIT variant=${VARIANT:-UP}"
echo "[remote_run] workdir=$REMOTE_WORKDIR config=$CONFIG_ABS"

# =============================================================================
# (1) system deps + python env
# =============================================================================
STAGE="1-deps"
echo; echo "[1] system tools + Python 3.9 conda env (navsim/nuplan-devkit require py3.9) ..."
# IMPORTANT: navsim and nuplan-devkit require Python 3.9. RunPod's system python
# is usually 3.12, on which navsim's pinned deps (old numpy/setuptools) fail to
# build (pkgutil.ImpImporter was removed in 3.12). So we set up a conda py3.9 env
# and run EVERYTHING (navsim, DiffusionDrive, dump, PDM eval, analyses) inside it.
for tool in git rsync wget unzip; do   # unzip: needed by devkit download_maps.sh (missing on the 2026-09 image)
  command -v "$tool" >/dev/null || (apt-get update -y >/dev/null && apt-get install -y "$tool" >/dev/null) || true
done

# (a) ensure a conda is available (use a pre-installed one, else install miniconda)
CONDA_ROOT="${CONDA_ROOT:-$REMOTE_WORKDIR/miniconda}"
if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
elif [[ -x "$CONDA_ROOT/bin/conda" ]]; then
  CONDA_BASE="$CONDA_ROOT"
else
  echo "    installing miniconda -> $CONDA_ROOT"
  wget -qO /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    || { echo "miniconda download failed"; exit 1; }
  bash /tmp/miniconda.sh -b -p "$CONDA_ROOT" >/dev/null
  CONDA_BASE="$CONDA_ROOT"
fi

# (b) load conda's shell hook and create/activate the py3.9 env
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
set +u                                   # conda's activate scripts trip `set -u`
if ! conda env list | grep -qE '(^|/)navsim[[:space:]]'; then
  echo "    creating conda env 'navsim' (python=3.9, conda-forge only)"
  # Use conda-forge ONLY (--override-channels): avoids the Anaconda default
  # channels (pkgs/main, pkgs/r) and their Terms-of-Service gate, which newer
  # conda refuses non-interactively. conda-forge has no such gate.
  conda create -y -n navsim -c conda-forge --override-channels python=3.9 pip setuptools wheel >/dev/null
fi
# activate by PATH, not name: name-resolution can pick up a same-named env from
# another conda root via ~/.conda/environments.txt (bit us 2026-07-19 when the
# tree was migrated across disks and pip then wrote into the wrong, quota-bound env)
conda activate "$REMOTE_WORKDIR/miniconda/envs/navsim" 2>/dev/null || conda activate navsim
set -u

PY="python"                              # the py3.9 env python; used everywhere below
echo "    python: $($PY --version 2>&1) at $(command -v "$PY")"
# conda-forge's bare python may ship without pip, and an env created earlier
# (before this fix) lacks it -> install pip into the env if missing (idempotent).
$PY -m pip --version >/dev/null 2>&1 \
  || conda install -y -n navsim -c conda-forge --override-channels pip setuptools wheel >/dev/null
$PY -m pip install --quiet --upgrade pip wheel >/dev/null 2>&1 || true
# navsim/nuplan-devkit import pkg_resources at runtime; setuptools>=81 (2025+)
# removed it, so pin a version that still ships it (do NOT upgrade to latest).
$PY -m pip install --quiet "setuptools<80" >/dev/null 2>&1 || true
$PY -m pip install --quiet numpy pyyaml matplotlib >/dev/null

# =============================================================================
# (2) clone + install navsim devkit and DiffusionDrive (pinned if requested)
# =============================================================================
STAGE="2-repos"
echo; echo "[2] navsim devkit + DiffusionDrive ..."
SRC="$REMOTE_WORKDIR/src"; mkdir -p "$SRC"

clone_pin() {  # url dir commit
  local url="$1" dir="$2" commit="$3"
  if [[ ! -d "$dir/.git" ]]; then
    git clone "$url" "$dir"
  else
    echo "    $dir already cloned; reusing."
  fi
  if [[ -n "$commit" ]]; then
    ( cd "$dir" && git fetch --all --quiet && git checkout "$commit" )
  fi
}

# DiffusionDrive is a FULL fork of the navsim devkit: it contains the entire
# navsim tree PLUS navsim/agents/diffusiondrive/*. We use it as THE devkit;
# installing autonomousvision/navsim separately would shadow it and the
# diffusiondrive agent would be missing (ModuleNotFoundError). === VERIFY ===
clone_pin "$DIFFUSIONDRIVE_REPO"   "$SRC/DiffusionDrive"  "${DIFFUSIONDRIVE_COMMIT:-}"
export NAVSIM_DEVKIT_ROOT="$SRC/DiffusionDrive"

# DiffusionDrive runtime deps: diffusers (DDIMScheduler) + einops. The version
# must sit in a compatibility window with navsim's pinned torch + huggingface_hub:
#   - too new (>=~0.32) references torch.xpu, absent in torch<2.4;
#   - too old (<=0.27) imports `cached_download`, removed from huggingface_hub>=0.26.
# 0.30.0 satisfies both. Do this BEFORE the import check below (which imports
# diffusers transitively via the model).
$PY -c "from diffusers.schedulers import DDIMScheduler" 2>/dev/null \
  || $PY -m pip install --quiet "diffusers==0.30.0" einops >/dev/null 2>&1 || true

# NOTE: not --quiet on purpose -- this is the heavy step (navsim + nuplan-devkit
# + torch, ~15-25 min); streaming pip output to run.log is the only sign of life.
if ! $PY -c "import navsim.agents.diffusiondrive.transfuser_agent" 2>/dev/null; then
  echo "    installing DiffusionDrive (navsim fork + nuplan-devkit + torch; HEAVY ~15-25 min) ..."
  # drop any previously-installed 'navsim' (e.g. autonomousvision/navsim from an
  # earlier run) so the editable install cleanly repoints to the DiffusionDrive fork.
  $PY -m pip uninstall -y navsim >/dev/null 2>&1 || true
  ( cd "$SRC/DiffusionDrive" && $PY -m pip install --progress-bar off -e . ) \
    || { echo "DiffusionDrive/navsim install failed (see its docs/install.md)"; exit 1; }
  echo "    DiffusionDrive (navsim fork) installed."
fi

# torch >= 2.1 + cu121: navsim pins torch 2.0.1+cu117, which (a) has no sm_90
# kernels (H100 "no kernel image"/illegal-instruction, hit 2026-07-14) and
# (b) lacks bf16 CUDA kernels for upsample_bilinear2d (hit on L40S 2026-07-19).
# MUST run AFTER the navsim editable install (its deps pin/downgrade torch —
# that ordering bug cost a fresh-pod run on 2026-07-20). Idempotent.
if ! $PY -c "import torch; assert torch.__version__ >= \"2.1\"" 2>/dev/null; then
  echo "    upgrading torch -> 2.1.2+cu121 (bf16/H100 kernel coverage) ..."
  mkdir -p "${DATA_ROOT:-/tmp}/cache/tmp"
  TMPDIR="${DATA_ROOT:-/tmp}/cache/tmp" $PY -m pip install --no-cache-dir --quiet \
    torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
fi

# DiffusionDrive's TransfuserConfig loads the kmeans plan anchors at construction
# (np.load(plan_anchor_path)); its default is the authors' local path. Fetch the
# released anchors and expose them so navsim_agent.py points the config at them
# (the ResNet-34 backbone is fetched by timm online, so no bkb_path is needed).
export MC_PLAN_ANCHOR="$REMOTE_WORKDIR/kmeans_navsim_traj_20.npy"
if [[ ! -s "$MC_PLAN_ANCHOR" ]]; then
  echo "    downloading kmeans plan anchors ..."
  wget -qO "$MC_PLAN_ANCHOR" \
    "https://github.com/hustvl/DiffusionDrive/releases/download/DiffusionDrive_88p1_PDMS_Eval_file/kmeans_navsim_traj_20.npy" \
    || { echo "failed to download kmeans_navsim_traj_20.npy"; exit 1; }
fi

# Final guard: navsim/nuplan-devkit import pkg_resources at runtime; some 2025+
# setuptools releases removed it (and one may have been pulled in as a build dep
# above). Ensure it is importable in the env, downgrading setuptools if needed.
$PY -c "import pkg_resources" 2>/dev/null || {
  echo "    pkg_resources missing; pinning setuptools<80 (ships pkg_resources) ..."
  $PY -m pip install --quiet "setuptools<80"
}

# make our runpod/ + experiments_navsim/ importable by Hydra-instantiated agents
export PYTHONPATH="$RUNPOD_DIR:$EXP_DIR:${PYTHONPATH:-}"

# =============================================================================
# (3) NAVSIM environment variables
# =============================================================================
STAGE="3-env"
echo; echo "[3] NAVSIM env vars ..."
# pick the data root: the persistent volume if given, else pod-local.
if [[ -n "${DATA_VOLUME:-}" ]]; then
  DATA_ROOT="$DATA_VOLUME/navsim"
else
  DATA_ROOT="$REMOTE_WORKDIR/data"
fi
mkdir -p "$DATA_ROOT"

# The official navsim download scripts (run with cwd=$DATA_ROOT) produce, under
# $DATA_ROOT:  maps/ , ${prefix}_navsim_logs/ , ${prefix}_sensor_blobs/  where
# prefix is the OpenScene split name (mini for navmini, test for navtest). navsim
# itself expects OPENSCENE_DATA_ROOT = the dataset ROOT and finds the per-split
# dirs from it; run_dump.py reads the explicit per-split paths below.
case "$EVAL_SPLIT" in
  navmini|mini)      SPLIT_PREFIX="mini" ;;
  navtest|test)      SPLIT_PREFIX="test" ;;
  navtrain|trainval) SPLIT_PREFIX="trainval" ;;
  *)                 SPLIT_PREFIX="$EVAL_SPLIT" ;;
esac

export OPENSCENE_DATA_ROOT="$DATA_ROOT"
export NUPLAN_MAPS_ROOT="$DATA_ROOT/maps"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NAVSIM_EXP_ROOT="$REMOTE_WORKDIR/exp"
export NAVSIM_LOG_PATH="$DATA_ROOT/${SPLIT_PREFIX}_navsim_logs"
export NAVSIM_SENSOR_ROOT="$DATA_ROOT/${SPLIT_PREFIX}_sensor_blobs"
# Do NOT pre-create maps/log/sensor dirs: the download scripts populate them via
# `mv`, which would nest wrongly into a pre-existing empty directory.
mkdir -p "$NAVSIM_EXP_ROOT"
echo "    OPENSCENE_DATA_ROOT=$OPENSCENE_DATA_ROOT (dataset root)"
echo "    NAVSIM_LOG_PATH=$NAVSIM_LOG_PATH"
echo "    NAVSIM_SENSOR_ROOT=$NAVSIM_SENSOR_ROOT"
echo "    NUPLAN_MAPS_ROOT=$NUPLAN_MAPS_ROOT"
echo "    NAVSIM_DEVKIT_ROOT=$NAVSIM_DEVKIT_ROOT"
echo "    NAVSIM_EXP_ROOT=$NAVSIM_EXP_ROOT"

# =============================================================================
# (4) download maps + the split (idempotent; skip if already present)
# =============================================================================
STAGE="4-data"
echo; echo "[4] dataset + maps ($EVAL_SPLIT) ..."
export HF_TOKEN="${HF_TOKEN:-}"
huggingface_login() {
  [[ -n "$HF_TOKEN" ]] || { echo "HF_TOKEN is empty but the dataset is gated."; return 1; }
  $PY -m pip install --quiet "huggingface_hub" >/dev/null 2>&1 || true
  $PY - <<'PYHF' || true
import os
try:
    from huggingface_hub import login
    login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
    print("    huggingface login ok")
except Exception as e:
    print("    huggingface login skipped:", e)
PYHF
}

have_maps()  { [[ -n "$(find "$NUPLAN_MAPS_ROOT" -maxdepth 3 -name 'nuplan-maps-v1.0.json' -print -quit 2>/dev/null)" ]]; }
# present iff at least one per-log .pkl exists anywhere under the split log dir
# (OpenScene nests them under a split subdir, e.g. <prefix>_navsim_logs/<prefix>/).
have_split() { [[ -n "$(find "$NAVSIM_LOG_PATH" -maxdepth 4 -name '*.pkl' -print -quit 2>/dev/null)" ]]; }

# === VERIFY against the navsim download/ scripts ===
# The navsim repo ships download scripts under download/. Their exact names vary
# by release (e.g. download_maps.sh, download_navtest.sh / download_test.sh).
# We resolve them by glob and run the ones that match the requested split.
DL_DIR="$NAVSIM_DEVKIT_ROOT/download"

# The devkit's download/*.sh are often checked out with CRLF line endings, which
# append a literal %0D (\r) to the S3 URLs (-> HTTP 404) and break their rm/mv
# (e.g. "cannot remove 'nuplan-maps-v1.1.zip'$'\r'"). Normalize to LF first.
if [[ -d "$DL_DIR" ]]; then
  sed -i 's/\r$//' "$DL_DIR"/*.sh 2>/dev/null || true
fi

run_first_match() {  # human-name  glob...
  local label="$1"; shift
  local f
  for pat in "$@"; do
    for f in $DL_DIR/$pat; do
      if [[ -f "$f" ]]; then
        echo "    running $label: $(basename "$f")"
        ( cd "$DATA_ROOT" && bash "$f" ) && return 0
      fi
    done
  done
  echo "    [warn] no download script matched $label in $DL_DIR (patterns: $*)."
  echo "           If the data is already on \$DATA_VOLUME this is fine."
  return 1
}

if have_maps; then
  echo "    maps already present in $NUPLAN_MAPS_ROOT (skipping)."
else
  huggingface_login || true
  run_first_match "maps" "download_maps*.sh" "*map*.sh" || {
    have_maps || { echo "maps missing and no download script ran; cannot eval."; exit 1; }
  }
fi

# nuplan's GPKGMapsDB opens {NUPLAN_MAPS_ROOT}/nuplan-maps-v1.0.json. download_maps
# renames nuplan-maps-v1.0/ -> maps/, but a pre-existing maps/ dir (from an earlier
# run) nests it as maps/nuplan-maps-v1.0/. Point the root at whichever dir holds the
# json (its parent), so both the clean and nested layouts work.
_maps_json="$(find "$NUPLAN_MAPS_ROOT" -maxdepth 3 -name 'nuplan-maps-v1.0.json' -print -quit 2>/dev/null || true)"
if [[ -n "$_maps_json" ]]; then
  export NUPLAN_MAPS_ROOT="$(dirname "$_maps_json")"
fi
echo "    resolved NUPLAN_MAPS_ROOT=$NUPLAN_MAPS_ROOT"

if have_split; then
  echo "    split data already present in $OPENSCENE_DATA_ROOT (skipping)."
else
  huggingface_login || true
  if [[ "$EVAL_SPLIT" == "navtest" ]]; then
    run_first_match "navtest" "download_navtest*.sh" "download_test*.sh" "*test*.sh"
  else
    run_first_match "navmini" "download_navmini*.sh" "download_mini*.sh" "*mini*.sh"
  fi
  # metadata layout fix (2026-07-18): newer HF dumps extract to openscene-v1.1
  # (hyphen); the stock devkit scripts expect openscene_v1.1, fail the mv, and
  # their final `rm -r openscene-v1.1` then DELETES the metadata. Rescue it, or
  # re-fetch just the metadata tgz if it was already lost.
  _meta_prefix=$([[ "$EVAL_SPLIT" == "navtest" ]] && echo test || echo mini)
  if [[ -d "$DATA_ROOT/openscene-v1.1/meta_datas" ]]; then
    mv "$DATA_ROOT/openscene-v1.1/meta_datas" "$DATA_ROOT/${_meta_prefix}_navsim_logs"
    rm -rf "$DATA_ROOT/openscene-v1.1"
  fi
  if ! have_split; then
    echo "    metadata missing (stock-script layout bug) — re-fetching just the metadata tgz ..."
    ( cd "$DATA_ROOT" \
      && wget -c -nv "https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_metadata_${_meta_prefix}.tgz" \
      && tar -xzf "openscene_metadata_${_meta_prefix}.tgz" --no-same-owner \
      && rm -f "openscene_metadata_${_meta_prefix}.tgz" \
      && if [[ -d openscene-v1.1/meta_datas ]]; then mv openscene-v1.1/meta_datas "${_meta_prefix}_navsim_logs"; rm -rf openscene-v1.1; \
         elif [[ -d openscene_v1.1/meta_datas ]]; then mv openscene_v1.1/meta_datas "${_meta_prefix}_navsim_logs"; rm -rf openscene_v1.1; fi )
  fi
  have_split || { echo "split '$EVAL_SPLIT' data missing after download attempt."; exit 1; }
fi

# navsim's filter_scenes() does data_path.iterdir() and pickle.load()s EACH entry,
# so NAVSIM_LOG_PATH must be the directory that DIRECTLY contains the per-log .pkl
# files. OpenScene nests them (e.g. <prefix>_navsim_logs/<prefix>/*.pkl, and a
# repeated download can nest further under meta_datas/). Resolve to the parent of
# the first .pkl we find so the loader sees only .pkl files.
_first_pkl="$(find "$NAVSIM_LOG_PATH" -maxdepth 4 -name '*.pkl' -print -quit 2>/dev/null || true)"
if [[ -n "$_first_pkl" ]]; then
  export NAVSIM_LOG_PATH="$(dirname "$_first_pkl")"
fi
echo "    resolved NAVSIM_LOG_PATH=$NAVSIM_LOG_PATH"
# Sensor blobs: the frame dicts store paths as "<log>/CAM_*/..." (no split prefix),
# while the actual per-log sensor folders are nested (e.g.
# <prefix>_sensor_blobs/<prefix>/<log>/CAM_F0/...). Locate a CAM_F0 dir and set the
# sensor ROOT to the dir that holds the per-log folders (two levels up), so
# root + "<log>/CAM_*/..." resolves to a real file.
_cam="$(find "$NAVSIM_SENSOR_ROOT" -maxdepth 4 -type d -name 'CAM_F0' -print -quit 2>/dev/null || true)"
if [[ -n "$_cam" ]]; then
  export NAVSIM_SENSOR_ROOT="$(dirname "$(dirname "$_cam")")"
fi
echo "    resolved NAVSIM_SENSOR_ROOT=$NAVSIM_SENSOR_ROOT"
# =============================================================================
STAGE="5-ckpt"
echo; echo "[5] DiffusionDrive checkpoint ..."
CKPT_DIR="$REMOTE_WORKDIR/ckpt"; mkdir -p "$CKPT_DIR"
export MC_CHECKPOINT="$CKPT_DIR/diffusiondrive.pth"
if [[ -s "$MC_CHECKPOINT" ]]; then
  echo "    checkpoint already at $MC_CHECKPOINT (skipping download)."
else
  [[ -n "${CKPT_URL:-}" ]] || { echo "CKPT_URL is empty; set it in config.env."; exit 1; }
  echo "    downloading checkpoint from \$CKPT_URL ..."
  if [[ -n "${HF_TOKEN:-}" ]]; then
    curl -fL -H "Authorization: Bearer $HF_TOKEN" -o "$MC_CHECKPOINT" "$CKPT_URL"
  else
    curl -fL -o "$MC_CHECKPOINT" "$CKPT_URL"
  fi
  [[ -s "$MC_CHECKPOINT" ]] || { echo "checkpoint download produced an empty file."; exit 1; }
fi

# =============================================================================
# (6) register the agent Hydra configs
# =============================================================================
STAGE="6-register"
echo; echo "[6] registering agent configs ..."
# resolve via the IMPORTED package, not $NAVSIM_DEVKIT_ROOT: after a tree
# migration the editable install may point at a different clone than the env
# var (bit us 2026-07-20 — hydra then can't see replay_agent & co).
_navsim_pkg=$($PY -c "import navsim, os; print(os.path.dirname(navsim.__file__))" 2>/dev/null || true)
AGENT_CFG_DST="${_navsim_pkg:-$NAVSIM_DEVKIT_ROOT/navsim}/planning/script/config/common/agent"
if [[ -d "$AGENT_CFG_DST" ]]; then
  cp -f "$RUNPOD_DIR/agent_configs/"*.yaml "$AGENT_CFG_DST/"
  echo "    copied agent yamls into $AGENT_CFG_DST"
else
  echo "    [warn] $AGENT_CFG_DST not found; the installed navsim may keep agent"
  echo "           configs elsewhere. === VERIFY === locate config/common/agent/"
  echo "           and copy agent_configs/*.yaml there, or pass them via Hydra."
fi

# =============================================================================
# (6b) HeavyBridge: train the VARIANT (skipped for VARIANT=UP)
# =============================================================================
STAGE="6b-heavybridge-train"
echo; echo "[6b] HeavyBridge variant: ${VARIANT:-UP}"
if [[ "${VARIANT:-UP}" != "UP" ]]; then
  bash "$RUNPOD_DIR/train_stage.sh"
  # NOTE: DATA_ROOT is reassigned to $DATA_VOLUME/navsim earlier in this script,
  # while train_stage.sh (which sources config.env) writes checkpoints under the
  # config-level DATA_ROOT. Look in BOTH places (permanent fix 2026-07-19; was a
  # manual symlink on the first pod).
  _hb_ckpt=$(ls -t "${DATA_ROOT:-/runpod-volume}/checkpoints/${VARIANT}"/*.pth \
                   "${DATA_VOLUME:-/nonexistent}/checkpoints/${VARIANT}"/*.pth 2>/dev/null | head -1 || true)
  if [[ -n "$_hb_ckpt" ]]; then
    export MC_CHECKPOINT="$_hb_ckpt"
    echo "    dump will use HeavyBridge checkpoint: $MC_CHECKPOINT"
  else
    echo "    trained checkpoint not found for $VARIANT"; exit 1
  fi
fi

# =============================================================================
# (6c) HeavyBridge: GPU compile-smoke of the head (fast, no data)
# =============================================================================
STAGE="6c-heavybridge-smoke"
if [[ "${HB_SMOKE:-0}" == "1" ]]; then
  echo; echo "[6c] HeavyBridge head compile-smoke ..."
  PYTHONPATH="$RUNPOD_DIR:${PYTHONPATH:-}" $PY "$RUNPOD_DIR/smoke_test.py"
fi

# =============================================================================
# (7) the SINGLE DiffusionDrive inference dump over the split
# =============================================================================
STAGE="7-dump"
echo; echo "[7] single inference dump over $EVAL_SPLIT (the ONLY GPU step) ..."
# run_dump.py is resumable (it skips tokens already dumped and scenes with missing
# sensor data), so always invoke it: on a fresh run it dumps everything; on a
# resume it only fills the gaps. (Old behaviour skipped entirely if ANY .npz
# existed, which left an interrupted dump incomplete.)
_have=$(find "$DUMP_DIR" -maxdepth 1 -name '*.npz' 2>/dev/null | wc -l | tr -d ' ')
echo "    $_have dumps already present; running run_dump.py (resumes / fills gaps) ..."
if [[ "${VARIANT:-UP}" != "UP" ]]; then
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICE:-0}" PYTHONPATH="$RUNPOD_DIR:${PYTHONPATH:-}" \
    $PY "$RUNPOD_DIR/hb_run.py" "$RUNPOD_DIR/run_dump.py" --split "$EVAL_SPLIT" --out "$DUMP_DIR" --config "$CONFIG_ABS"
else
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICE:-0}" \
    $PY "$RUNPOD_DIR/run_dump.py" --split "$EVAL_SPLIT" --out "$DUMP_DIR" --config "$CONFIG_ABS"
fi
NDUMP=$(find "$DUMP_DIR" -maxdepth 1 -name '*.npz' 2>/dev/null | wc -l | tr -d ' ')
[[ "$NDUMP" -gt 0 ]] || { echo "no dumps produced; aborting."; exit 1; }
echo "    $NDUMP scene dumps ready."

# =============================================================================
# (8) CPU-only PDM scoring of the deployed plan (NO GPU)
# =============================================================================
STAGE="8-score"
echo; echo "[8] CPU-only PDM scoring of the deployed (argmax) plan ..."

# navsim PDM scoring = (a) a per-token METRIC CACHE (PDM simulation) + (b) the
# run_pdm_score.py entrypoint. We call the python directly (the shipped wrapper
# .sh has CRLF issues), with EXPLICIT data paths (our nested layout) and our
# ReplayAgent, which maps token <- scene.scene_metadata.initial_token and replays
# the precomputed per-method plan. CPU only.  === VERIFY override names ===
PDM_PY="$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py"
MC_PY="$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py"
[[ -f "$PDM_PY" ]] || { echo "run_pdm_score.py not found at $PDM_PY"; exit 1; }
WORKER="${PDM_WORKER:-single_machine_thread_pool}"
echo "    navsim worker = $WORKER  (set PDM_WORKER=ray_distributed to use all vCPUs)"
# ray_distributed parallelizes metric caching + scoring across all cores
# (single_machine_thread_pool is GIL-bound to ~1 core). Ensure ray is importable.
if [[ "$WORKER" == "ray_distributed" ]]; then
  $PY -c "import ray" 2>/dev/null || { echo "    installing ray ..."; $PY -m pip install --quiet ray || true; }
fi
# On >64-core pods the container pids limit (e.g. 16896) chokes ray with
# "can't start new thread" -> cap ray workers (2026-07-27, eu-cz-1 256 vCPU pod).
WORKER_ARGS=""
[[ "$WORKER" == "ray_distributed" ]] && WORKER_ARGS="worker.threads_per_node=${HB_RAY_THREADS:-64}"

# (8.0) build the metric cache once (per-token PDM-sim cache for $EVAL_SPLIT).
METRIC_CACHE="$NAVSIM_EXP_ROOT/metric_cache"
if [[ -z "$(ls -A "$METRIC_CACHE" 2>/dev/null)" ]]; then
  echo "    building metric cache (train_test_split=$EVAL_SPLIT; CPU, can be slow) ..."
  # metric caching builds from maps + logs (PDM simulation), not sensor images,
  # so its config has navsim_log_path but NOT sensor_blobs_path -- do not pass it.
  $PY "$MC_PY" \
      train_test_split="$EVAL_SPLIT" \
      cache.cache_path="$METRIC_CACHE" \
      navsim_log_path="$NAVSIM_LOG_PATH" \
      worker="$WORKER" $WORKER_ARGS \
    || { echo "metric caching failed (see log)."; exit 1; }
else
  echo "    metric cache present at $METRIC_CACHE (skipping)."
fi

# score one selection JSON with the ReplayAgent (CPU) and collect the aggregate.
# args: <name> <selection.json> <out.json>
score_selection() {
  local name="$1" sel="$2" out="$3"
  # resume: skip if this eval was already collected (lets a long navtest run be
  # re-launched after an interruption without re-scoring finished selections).
  [[ -s "$out" ]] && { echo "      $name already scored ($out); skip."; return 0; }
  [[ -s "$sel" ]] || { echo "selection $sel missing/empty"; return 1; }
  export MC_SELECTION="$sel"
  local edir="$NAVSIM_EXP_ROOT/$name"; mkdir -p "$edir"
  echo "      scoring $name ..."
  $PY "$PDM_PY" \
      agent=replay_agent \
      train_test_split="$EVAL_SPLIT" \
      metric_cache_path="$METRIC_CACHE" \
      navsim_log_path="$NAVSIM_LOG_PATH" \
      sensor_blobs_path="$NAVSIM_SENSOR_ROOT" \
      output_dir="$edir" \
      worker="$WORKER" $WORKER_ARGS \
    || { echo "navsim PDM scoring failed for $name"; return 1; }
  # navsim writes <timestamp>.csv under output_dir; collect_pdm picks the newest.
  $PY "$RUNPOD_DIR/collect_pdm.py" --csv "$edir" --method "$name" --out "$out"
}

# 8a. the deployed plan: argmax-logit anchor ----------------------------------
$PY "$RUNPOD_DIR/select_base.py" --dump-dir "$DUMP_DIR" --out "$SEL_DIR" --name base
score_selection "base" "$SEL_DIR/base.json" "$RESULTS/eval_base.json"

# =============================================================================
# =============================================================================
# (8b) HeavyBridge: forced-anchor selections (Phase-2 metrics)
# =============================================================================
STAGE="8b-heavybridge-forced"
echo; echo "[8b] forced-anchor selections (k = ${FORCED_KS:-1 2 3 5}) ..."
mkdir -p "$RESULTS/forced"
$PY "$RUNPOD_DIR/select_forced_anchor.py" --dumps "$DUMP_DIR" \
    --out "$RESULTS/forced" --ks "${FORCED_KS:-1 2 3 5}" --run-id "${VARIANT:-UP}"

# =============================================================================
# (10) assemble results/ + summary.md, then DONE
# =============================================================================
STAGE="10-summary"
echo; echo "[10] summary ..."
$PY "$RUNPOD_DIR/make_summary.py" --results "$RESULTS" --out "$RESULTS/summary.md" \
      --split "$EVAL_SPLIT" --variant "${VARIANT:-UP}"
cp -f run.log "$RESULTS/run.log" 2>/dev/null || true

echo "=============================================================="
echo "[remote_run] all stages complete $(date -u +%FT%TZ)"
echo "=============================================================="
touch DONE
