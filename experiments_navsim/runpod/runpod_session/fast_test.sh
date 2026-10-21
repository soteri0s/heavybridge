#!/bin/bash
# fast_test.sh — parallel mirror of the devkit's download_test.sh (navtest split) +
# download_maps.sh. Same URLs, same final layout (test_navsim_logs, test_sensor_blobs, maps).
# Run ON THE POD: setsid nohup bash fast_test.sh > /scratch/mc/fast_test.log 2>&1 &
set -u
cd /scratch/data/navsim
export TAR_OPTIONS="--no-same-owner"
T=$(grep '^HF_TOKEN=' /scratch/mc/experiments_navsim/runpod/config.env | cut -d'"' -f2)
BASE=https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1
mkdir -p .test_done
# maps (nuplan-maps-v1.1.zip -> maps/)
if [[ ! -d maps ]]; then
  echo "[maps] $(date -u +%T)"; aria2c -c -x 8 -s 8 -q --timeout=30 --connect-timeout=20 --max-tries=0 --retry-wait=5 --lowest-speed-limit=100K -o nuplan-maps-v1.1.zip https://motional-nuplan.s3.amazonaws.com/public/nuplan-v1.1/nuplan-maps-v1.1.zip \
  && unzip -q nuplan-maps-v1.1.zip && rm -f nuplan-maps-v1.1.zip && mv nuplan-maps-v1.0 maps && echo "[maps] done $(date -u +%T)"
fi
# metadata
if [[ ! -d test_navsim_logs ]]; then
  echo "[meta] $(date -u +%T)"; wget -c -q --header="Authorization: Bearer $T" "$BASE/openscene_metadata_test.tgz" && mkdir -p .meta_t && tar -xzf openscene_metadata_test.tgz -C .meta_t && rm -f openscene_metadata_test.tgz
  if   [[ -d .meta_t/openscene-v1.1/meta_datas ]]; then mv .meta_t/openscene-v1.1/meta_datas .test_navsim_logs_pending
  elif [[ -d .meta_t/openscene_v1.1/meta_datas ]]; then mv .meta_t/openscene_v1.1/meta_datas .test_navsim_logs_pending
  elif [[ -d .meta_t/meta_datas ]]; then mv .meta_t/meta_datas .test_navsim_logs_pending
  else echo "[meta] layout?"; ls -R .meta_t | head; fi
  rm -rf .meta_t; echo "[meta] done $(date -u +%T)"
fi
one() { local kind=$1 i=$2; local f="openscene_sensor_test_${kind}_${i}.tgz"
  [[ -f ".test_done/${kind}_${i}" ]] && return 0
  for attempt in 1 2 3 4; do
    aria2c -c -x 4 -s 4 -k 8M --file-allocation=none -q -o "$f" --header="Authorization: Bearer $T" "$BASE/openscene_sensor_test_${kind}/$f" && break
    sleep 10
  done
  [[ -s "$f" ]] || { echo "FAIL download $f"; return 1; }
  if tar -xzf "$f"; then rm -f "$f" "$f.aria2"; touch ".test_done/${kind}_${i}"; echo "done $kind $i $(date -u +%T)"
  else echo "FAIL extract $f"; rm -f "$f"; return 1; fi
}
jobs_list=(); for i in $(seq 0 31); do jobs_list+=("camera $i"); done; for i in $(seq 0 31); do jobs_list+=("lidar $i"); done
for j in "${jobs_list[@]}"; do
  while [[ $(jobs -rp | wc -l) -ge 6 ]]; do sleep 5; done
  one $j &
done
wait
n=$(ls .test_done | wc -l); echo "markers: $n / 64"
if [[ $n -eq 64 ]]; then
  [[ -d .test_navsim_logs_pending ]] && mv .test_navsim_logs_pending test_navsim_logs
  [[ -d openscene-v1.1/sensor_blobs ]] && mv openscene-v1.1/sensor_blobs test_sensor_blobs
  rmdir openscene-v1.1 2>/dev/null
  touch TEST_COMPLETE; echo "TEST_COMPLETE $(date -u)"
else echo "INCOMPLETE — rerun"; fi
