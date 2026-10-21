#!/bin/bash
# fast_mini.sh — parallel replacement for the devkit's sequential download_mini.sh
# (same URLs, same extraction layout, same final moves), plus maps and metadata so it
# works on a FRESH pod (2026-09-15; the 2026-09-13 version assumed a stock-script takeover).
# Run ON THE POD: setsid nohup bash fast_mini.sh > /scratch/mc/fast_mini.log 2>&1 &
# Idempotent: per-file markers in .mini_done/. Ends with MINI_COMPLETE marker.
set -u
mkdir -p /scratch/data/navsim; cd /scratch/data/navsim
export TAR_OPTIONS="--no-same-owner"
T=$(grep '^HF_TOKEN=' /scratch/mc/experiments_navsim/runpod/config.env | head -1 | cut -d'"' -f2)
BASE=https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1
mkdir -p .mini_done
# maps (nuplan-maps-v1.1.zip -> maps/), as the devkit's download_maps.sh does
if [[ ! -d maps ]]; then
  echo "[maps] $(date -u +%T)"; aria2c -c -x 8 -s 8 -q --timeout=30 --connect-timeout=20 --max-tries=0 --retry-wait=5 --lowest-speed-limit=100K -o nuplan-maps-v1.1.zip https://motional-nuplan.s3.amazonaws.com/public/nuplan-v1.1/nuplan-maps-v1.1.zip \
  && unzip -q nuplan-maps-v1.1.zip && rm -f nuplan-maps-v1.1.zip && mv nuplan-maps-v1.0 maps && echo "[maps] done $(date -u +%T)"
fi
# metadata (openscene_metadata_mini.tgz -> mini_navsim_logs), layout cases as in fast_test.sh
if [[ ! -d mini_navsim_logs ]]; then
  echo "[meta] $(date -u +%T)"; wget -c -q --header="Authorization: Bearer $T" "$BASE/openscene_metadata_mini.tgz" && mkdir -p .meta_m && tar -xzf openscene_metadata_mini.tgz -C .meta_m && rm -f openscene_metadata_mini.tgz
  if   [[ -d .meta_m/openscene-v1.1/meta_datas ]]; then mv .meta_m/openscene-v1.1/meta_datas .mini_navsim_logs_pending
  elif [[ -d .meta_m/openscene_v1.1/meta_datas ]]; then mv .meta_m/openscene_v1.1/meta_datas .mini_navsim_logs_pending
  elif [[ -d .meta_m/meta_datas ]]; then mv .meta_m/meta_datas .mini_navsim_logs_pending
  else echo "[meta] layout?"; ls -R .meta_m | head; fi
  rm -rf .meta_m; echo "[meta] done $(date -u +%T)"
fi
one() {  # kind idx
  local kind=$1 i=$2; local f="openscene_sensor_mini_${kind}_${i}.tgz"
  [[ -f ".mini_done/${kind}_${i}" ]] && return 0
  for attempt in 1 2 3 4; do
    aria2c -c -x 4 -s 4 -k 8M --file-allocation=none -q -o "$f" --header="Authorization: Bearer $T" "$BASE/openscene_sensor_mini_${kind}/$f" && break
    sleep 10
  done
  [[ -s "$f" ]] || { echo "FAIL download $f"; return 1; }
  if tar -xzf "$f"; then rm -f "$f" "$f.aria2"; touch ".mini_done/${kind}_${i}"; echo "done $kind $i $(date -u +%T)"
  else echo "FAIL extract $f"; rm -f "$f"; return 1; fi
}
jobs_list=(); for i in $(seq 0 31); do jobs_list+=("camera $i"); done; for i in $(seq 0 31); do jobs_list+=("lidar $i"); done
for j in "${jobs_list[@]}"; do
  while [[ $(jobs -rp | wc -l) -ge 6 ]]; do sleep 5; done
  one $j &
done
wait
n=$(ls .mini_done | wc -l)
echo "markers: $n / 64"
if [[ $n -eq 64 ]]; then
  [[ -d .mini_navsim_logs_pending ]] && mv .mini_navsim_logs_pending mini_navsim_logs
  [[ -d openscene-v1.1/meta_datas && ! -d mini_navsim_logs ]] && mv openscene-v1.1/meta_datas mini_navsim_logs
  [[ -d openscene-v1.1/sensor_blobs ]] && mv openscene-v1.1/sensor_blobs mini_sensor_blobs
  rmdir openscene-v1.1 2>/dev/null
  touch MINI_COMPLETE; echo "MINI_COMPLETE $(date -u)"
else
  echo "INCOMPLETE — rerun"
fi
