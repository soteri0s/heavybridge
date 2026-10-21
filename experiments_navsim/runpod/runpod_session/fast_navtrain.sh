#!/bin/bash
# fast_navtrain.sh — parallel (2 streams) mirror of runpod/download_navtrain.sh: same
# files, same layout, same .navtrain_done markers, so train_stage.sh's call of
# download_navtrain.sh then finds everything done. Run ON THE POD:
#   setsid nohup bash fast_navtrain.sh > /scratch/mc/fast_navtrain.log 2>&1 &
set -u
DATA_ROOT=/scratch/data; NAVSIM_DATA=$DATA_ROOT/navsim
export TAR_OPTIONS="--no-same-owner"
mkdir -p "$NAVSIM_DATA"; cd "$NAVSIM_DATA"
T=$(grep '^HF_TOKEN=' /scratch/mc/experiments_navsim/runpod/config.env | cut -d'"' -f2)
if [[ ! -d trainval_navsim_logs ]]; then
  echo "[navtrain] trainval metadata ... $(date -u +%T)"
  wget -c -nv --header="Authorization: Bearer $T" "https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_metadata_trainval.tgz"
  mkdir -p .meta_tv && tar -xzf openscene_metadata_trainval.tgz -C .meta_tv && rm -f openscene_metadata_trainval.tgz
  if   [[ -d .meta_tv/openscene-v1.1/meta_datas ]]; then mv .meta_tv/openscene-v1.1/meta_datas trainval_navsim_logs
  elif [[ -d .meta_tv/openscene_v1.1/meta_datas ]]; then mv .meta_tv/openscene_v1.1/meta_datas trainval_navsim_logs
  elif [[ -d .meta_tv/meta_datas ]]; then mv .meta_tv/meta_datas trainval_navsim_logs
  else echo "[navtrain] ERROR: metadata layout not recognized"; ls -R .meta_tv | head; exit 1; fi
  rm -rf .meta_tv; echo "[navtrain] metadata ok $(date -u +%T)"
fi
mkdir -p trainval_sensor_blobs/trainval .navtrain_done
merge_dir() {  # move per-log subdirs into trainval/ (same filesystem: instant); rsync only where a subdir already exists (2026-09-15)
  local src="$1" dst=trainval_sensor_blobs/trainval
  for sub in "$src"/*; do
    [[ -e "$sub" ]] || continue
    local b=$(basename "$sub")
    if [[ -e "$dst/$b" ]]; then rsync -a "$sub/" "$dst/$b/" && rm -rf "$sub"; else mv "$sub" "$dst/"; fi
  done
  rmdir "$src" 2>/dev/null || rm -rf "$src"
}
fetch_split() {  # kind index  (verbatim logic of download_navtrain.sh)
  local kind="$1" i="$2"; local tgz="navtrain_${kind}_${i}.tgz" dir
  [[ "$kind" == current ]] && dir="current_split_${i}" || dir="history_split_${i}"
  [[ -f ".navtrain_done/${kind}_${i}" ]] && { echo "[navtrain] ${tgz} already merged"; return 0; }
  echo "[navtrain] ${tgz} download $(date -u +%T)"
  for a in 1 2 3 4 5; do wget -c -q "https://s3.eu-central-1.amazonaws.com/avg-projects-2/navsim/${tgz}" && break; sleep 30; done
  [[ -s "$tgz" ]] || { echo "[navtrain] FAIL download $tgz"; return 1; }
  echo "[navtrain] ${tgz} extracting $(date -u +%T)"
  tar -xzf "$tgz" || { echo "[navtrain] FAIL extract $tgz"; rm -f "$tgz"; return 1; }
  rm -f "$tgz"
  merge_dir "$dir" && touch ".navtrain_done/${kind}_${i}"
  echo "[navtrain] ${tgz} merged $(date -u +%T)"
}
fetch_split current 1 & fetch_split current 2 & fetch_split current 3 & fetch_split current 4 & wait
ln -sfn "$NAVSIM_DATA/trainval_sensor_blobs" "$DATA_ROOT/navtrain_sensors"
echo "[navtrain] markers: $(ls .navtrain_done | wc -l)/4 ; complete $(date -u)"
[[ $(ls .navtrain_done | wc -l) -eq 4 ]] && touch NAVTRAIN_COMPLETE
