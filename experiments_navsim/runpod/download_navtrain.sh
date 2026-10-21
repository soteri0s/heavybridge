#!/usr/bin/env bash
# =============================================================================
# download_navtrain.sh <DATA_ROOT> — navtrain split for HeavyBridge training.
# Mirrors DiffusionDrive/download/download_navtrain.sh with fixes learned on
# the navmini dry run (2026-07-13):
#   * metadata tgz extracts to openscene-v1.1 (hyphen), not openscene_v1.1
#   * wget -c + per-file markers -> resumable / idempotent
#   * data lands next to the navmini data: $DATA_ROOT/navsim/
#   * leaves $DATA_ROOT/navtrain_sensors symlink = train_stage.sh sentinel
# =============================================================================
set -euo pipefail
DATA_ROOT="${1:?usage: download_navtrain.sh <DATA_ROOT>}"
NAVSIM_DATA="$DATA_ROOT/navsim"
mkdir -p "$NAVSIM_DATA"; cd "$NAVSIM_DATA"

# --- trainval metadata (navsim logs) -----------------------------------------
if [[ ! -d trainval_navsim_logs ]]; then
  echo "[navtrain] trainval metadata ..."
  wget -c -nv "https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_metadata_trainval.tgz"
  tar -xzf openscene_metadata_trainval.tgz
  rm -f openscene_metadata_trainval.tgz
  # layout differs between dump versions: handle hyphen AND underscore
  if   [[ -d openscene-v1.1/meta_datas ]]; then mv openscene-v1.1/meta_datas trainval_navsim_logs; rm -rf openscene-v1.1
  elif [[ -d openscene_v1.1/meta_datas ]]; then mv openscene_v1.1/meta_datas trainval_navsim_logs; rm -rf openscene_v1.1
  elif [[ -d meta_datas ]]; then mv meta_datas trainval_navsim_logs
  else echo "[navtrain] ERROR: metadata layout not recognized"; ls; exit 1; fi
else
  echo "[navtrain] trainval metadata already present (skipping)."
fi

# --- sensor blobs: 4x current + 4x history ------------------------------------
mkdir -p trainval_sensor_blobs/trainval .navtrain_done
fetch_split() {  # kind index
  local kind="$1" i="$2" tgz="navtrain_${1}_${2}.tgz" dir
  [[ "$kind" == current ]] && dir="current_split_${i}" || dir="history_split_${i}"
  if [[ -f ".navtrain_done/${kind}_${i}" ]]; then
    echo "[navtrain] ${tgz} already merged (skipping)."; return 0
  fi
  echo "[navtrain] ${tgz} ..."
  wget -c -nv "https://s3.eu-central-1.amazonaws.com/avg-projects-2/navsim/${tgz}"
  echo "[navtrain] extracting ${tgz}"
  tar -xzf "$tgz"; rm -f "$tgz"
  rsync -a "${dir}/" trainval_sensor_blobs/trainval/
  rm -rf "$dir"
  touch ".navtrain_done/${kind}_${i}"
}
for i in 1 2 3 4; do fetch_split current "$i"; done
# History splits SKIPPED: DiffusionDrive's agent uses SensorConfig.build_all_sensors(include=[3])
# (current frame only, verified in transfuser_agent.py on 2026-07-13) -> saves ~200 GB.
# To re-enable (for an agent that needs history sensors):
#   rm $DATA_ROOT/navsim/.navtrain_done/history_* ; uncomment:
# for i in 1 2 3 4; do fetch_split history "$i"; done

# --- sentinel expected by train_stage.sh --------------------------------------
ln -sfn "$NAVSIM_DATA/trainval_sensor_blobs" "$DATA_ROOT/navtrain_sensors"
echo "[navtrain] complete: logs=$NAVSIM_DATA/trainval_navsim_logs sensors=$NAVSIM_DATA/trainval_sensor_blobs/trainval"
