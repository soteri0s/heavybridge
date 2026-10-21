#!/bin/bash
# bootstrap_data.sh — fresh-pod data bootstrap for the alpha ablation (2026-09-15):
# maps + navmini (fast_mini.sh), navtrain (fast_navtrain.sh), navtest (fast_test.sh),
# sequentially; each script is internally parallel and idempotent and ends with its
# *_COMPLETE marker, which ablation_alpha.sh waits for.
# Run ON THE POD: setsid nohup bash bootstrap_data.sh > /scratch/mc/bootstrap.log 2>&1 &
set -u
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DEBIAN_FRONTEND=noninteractive
for tool in aria2c unzip wget rsync; do
  command -v "$tool" >/dev/null || { apt-get update -y >/dev/null 2>&1; apt-get install -y aria2 unzip wget rsync >/dev/null 2>&1; }
done
mkdir -p /scratch/data/navsim /scratch/mc/exp
cd /scratch/data/navsim
echo "[bootstrap] start $(date -u)"
[[ -f MINI_COMPLETE ]]     || bash "$D/fast_mini.sh"
[[ -f NAVTRAIN_COMPLETE ]] || bash "$D/fast_navtrain.sh"
[[ -f TEST_COMPLETE ]]     || bash "$D/fast_test.sh"
echo "[bootstrap] markers: $(ls MINI_COMPLETE NAVTRAIN_COMPLETE TEST_COMPLETE 2>/dev/null | tr '\n' ' ') $(date -u)"
df -h /scratch | tail -1
