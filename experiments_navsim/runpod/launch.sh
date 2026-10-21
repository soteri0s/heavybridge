#!/usr/bin/env bash
# =============================================================================
# launch.sh -- LOCAL one-command launcher for one HeavyBridge variant on a
# RunPod GPU pod (train -> navtest dump -> PDM scoring). From your laptop:
#
#     bash launch.sh --host <pod-ip> --port <pod-ssh-port> [--key ~/.ssh/id_ed25519]
#                    [--split navtest|navmini] [--workdir /workspace/mc] [--restart]
#
#   --restart : kill any running remote session + clear sentinels before launching
#               (use it to pick up updated scripts instead of re-attaching).
#
#   DRY RUN  (small navmini split):
#     bash launch.sh --host <ip> --port <port> --split navmini --workdir /workspace/mc-dry
#
# It: (1) checks SSH, (2) rsyncs the whole experiments_navsim/ to the pod,
# (3) launches remote_run.sh under nohup inside tmux (survives disconnect),
# (4) polls for the remote DONE / FAILED sentinel while streaming the log tail,
# (5) on DONE rsyncs results/ + logs + summary.md back into
#     ./runpod_results/<timestamp>/. Robust to disconnects: re-running re-attaches
# to the same remote run rather than starting a new one (remote_run.sh is
# idempotent). FAILS LOUDLY: a FAILED sentinel aborts with the remote reason.
# =============================================================================
set -euo pipefail

# --- defaults ----------------------------------------------------------------
HOST=""
PORT=""
KEY="$HOME/.ssh/id_ed25519"
REMOTE_WORKDIR="/workspace/mc"     # overridden by config.env's REMOTE_WORKDIR
SSH_USER="root"
SPLIT_OVERRIDE=""                  # --split navmini|navtest ; empty = use config.env
RESTART=0                          # --restart : kill any running session + clear sentinels first
POLL_SECS=30
HF_TOKEN_CLI=""                    # --hf-token : injected into the remote config.env (gated data)
DATA_VOLUME_CLI=""                 # --data-volume : remote network-volume mount for the dataset
WORKER_CLI=""                      # --worker : navsim worker (ray_distributed to use all cores)

HERE="$(cd "$(dirname "$0")" && pwd)"
EXP_DIR="$(dirname "$HERE")"        # experiments_navsim/
CONFIG_ENV="$HERE/config.env"

# --- args --------------------------------------------------------------------
usage() { sed -n '2,31p' "$0"; exit "${1:-0}"; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)    HOST="$2"; shift 2;;
    --port)    PORT="$2"; shift 2;;
    --key)     KEY="$2"; shift 2;;
    --workdir) REMOTE_WORKDIR="$2"; shift 2;;
    --user)    SSH_USER="$2"; shift 2;;
    --split)   SPLIT_OVERRIDE="$2"; shift 2;;
    --restart) RESTART=1; shift;;
    --hf-token) HF_TOKEN_CLI="$2"; shift 2;;
    --data-volume) DATA_VOLUME_CLI="$2"; shift 2;;
    --worker)  WORKER_CLI="$2"; shift 2;;
    -h|--help) usage 0;;
    *) echo "unknown arg: $1" >&2; usage 1;;
  esac
done

[[ -n "$HOST" ]] || { echo "ERROR: --host is required." >&2; usage 1; }
[[ -n "$PORT" ]] || { echo "ERROR: --port is required." >&2; usage 1; }
[[ -f "$CONFIG_ENV" ]] || { echo "ERROR: $CONFIG_ENV not found." >&2; exit 1; }
[[ -f "$KEY" ]] || { echo "ERROR: ssh key not found: $KEY" >&2; exit 1; }

# pull REMOTE_WORKDIR / SSH_USER from config.env, but let the CLI win: remember
# whether the user passed --workdir/--user before sourcing (which would clobber).
CLI_WORKDIR="$REMOTE_WORKDIR"; CLI_USER="$SSH_USER"
_DEF_WORKDIR="/workspace/mc"; _DEF_USER="root"
# shellcheck disable=SC1090
source "$CONFIG_ENV"
# CLI override (non-default) beats config.env; else use config.env; else default.
if [[ "$CLI_WORKDIR" != "$_DEF_WORKDIR" ]]; then REMOTE_WORKDIR="$CLI_WORKDIR"; fi
REMOTE_WORKDIR="${REMOTE_WORKDIR:-$_DEF_WORKDIR}"
if [[ "$CLI_USER" != "$_DEF_USER" ]]; then SSH_USER="$CLI_USER"; fi
SSH_USER="${SSH_USER:-$_DEF_USER}"

RUNSCRIPT="remote_run.sh"; SENT_DONE="DONE"; SENT_FAIL="FAILED"

SSH="ssh -p $PORT -i $KEY -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30"
SSH_TGT="$SSH_USER@$HOST"
REMOTE_EXP="$REMOTE_WORKDIR/experiments_navsim"
REMOTE_RUNPOD="$REMOTE_EXP/runpod"

echo "=============================================================="
echo " launch.sh  ->  $SSH_TGT:$PORT   workdir=$REMOTE_WORKDIR"
echo "=============================================================="

# --- (1) ssh check -----------------------------------------------------------
echo "[1] checking ssh ..."
$SSH $SSH_TGT "echo connected: \$(hostname); nvidia-smi -L 2>/dev/null | head -n1 || echo '(no nvidia-smi)'" \
  || { echo "ERROR: cannot ssh to $SSH_TGT:$PORT with $KEY." >&2; exit 1; }

# --- (2) ship the code -------------------------------------------------------
echo; echo "[2] rsync experiments_navsim/ -> pod ..."
$SSH $SSH_TGT "mkdir -p $REMOTE_EXP"
RSYNC_SSH="ssh -p $PORT -i $KEY -o StrictHostKeyChecking=accept-new"
rsync -az --no-owner --no-group --delete \
  --exclude 'runpod/runpod_results' \
  --exclude 'dumps' --exclude 'results' \
  --exclude '__pycache__' --exclude '.pytest_cache' \
  --exclude '*.pyc' --exclude '.git' \
  -e "$RSYNC_SSH" \
  "$EXP_DIR/" "$SSH_TGT:$REMOTE_EXP/"
# ship config.env explicitly (it is inside runpod/, included above, but make sure)
rsync -az --no-owner --no-group -e "$RSYNC_SSH" "$CONFIG_ENV" "$SSH_TGT:$REMOTE_RUNPOD/config.env"
# --split (if given) overrides EVAL_SPLIT by appending a winning assignment to the
# remote config.env (it is sourced, so the last assignment wins).
if [[ -n "$SPLIT_OVERRIDE" ]]; then
  case "$SPLIT_OVERRIDE" in
    navtest|navmini) ;;
    *) echo "ERROR: --split must be navtest or navmini (got '$SPLIT_OVERRIDE')." >&2; exit 1;;
  esac
  $SSH $SSH_TGT "printf '\n# --split override from launch.sh\nexport EVAL_SPLIT=%s\n' '$SPLIT_OVERRIDE' >> $REMOTE_RUNPOD/config.env"
  echo "    EVAL_SPLIT override -> $SPLIT_OVERRIDE"
fi
# inject the gated-data token and the network-volume mount (last assignment wins
# when config.env is sourced). The token is written only to the remote config.env.
if [[ -n "$HF_TOKEN_CLI" ]]; then
  $SSH $SSH_TGT "printf '\n# --hf-token from launch.sh\nexport HF_TOKEN=\"%s\"\n' '$HF_TOKEN_CLI' >> $REMOTE_RUNPOD/config.env"
  echo "    HF_TOKEN injected into remote config.env"
fi
if [[ -n "$DATA_VOLUME_CLI" ]]; then
  $SSH $SSH_TGT "printf '\n# --data-volume from launch.sh\nexport DATA_VOLUME=\"%s\"\n' '$DATA_VOLUME_CLI' >> $REMOTE_RUNPOD/config.env"
  echo "    DATA_VOLUME -> $DATA_VOLUME_CLI"
fi
# navsim worker: ray_distributed parallelizes metric caching + scoring across all
# vCPUs (single_machine_thread_pool is GIL-bound to ~1 core -- far too slow for navtest).
if [[ -n "$WORKER_CLI" ]]; then
  $SSH $SSH_TGT "printf '\n# --worker from launch.sh\nexport PDM_WORKER=\"%s\"\n' '$WORKER_CLI' >> $REMOTE_RUNPOD/config.env"
  echo "    PDM_WORKER -> $WORKER_CLI"
fi
echo "    code shipped to $REMOTE_EXP"

# --- (3) launch remotely under tmux + nohup (survives disconnect) ------------
echo; echo "[3] launching $RUNSCRIPT (tmux session 'mcrun', survives disconnect) ..."
# If a previous run is still going or finished, DO NOT start a second one.
REMOTE_LAUNCH=$(cat <<EOF
set -e
cd "$REMOTE_RUNPOD"
if [ "$RESTART" = "1" ]; then
  tmux kill-session -t mcrun 2>/dev/null || true   # kills the whole bash tree
  # bracket-trick patterns so pkill does NOT match this very launch script
  pkill -9 -f '[r]un_metric_caching' 2>/dev/null || true
  pkill -9 -f '[r]un_pdm_score' 2>/dev/null || true
  pkill -9 -f '[r]un_dump' 2>/dev/null || true
  pkill -f 'pip install' 2>/dev/null || true
  rm -f DONE FAILED NAVTEST_DONE NAVTEST_FAILED
  echo "RESTARTED (killed previous session + cleared sentinels)"
  sleep 2
fi
if [ -f $SENT_DONE ]; then echo "ALREADY_DONE"; exit 0; fi
if [ -f $SENT_FAIL ]; then echo "PREVIOUS_FAILED"; fi
if command -v tmux >/dev/null 2>&1; then
  if tmux has-session -t mcrun 2>/dev/null; then
    echo "REATTACH (mcrun already running)"; exit 0
  fi
  tmux new-session -d -s mcrun "bash $RUNSCRIPT > run.log 2>&1"
  echo "STARTED_TMUX"
else
  if pgrep -f 'bash $RUNSCRIPT' >/dev/null 2>&1; then
    echo "REATTACH (nohup already running)"; exit 0
  fi
  nohup bash $RUNSCRIPT > run.log 2>&1 &
  echo "STARTED_NOHUP"
fi
EOF
)
LAUNCH_OUT=$($SSH $SSH_TGT "$REMOTE_LAUNCH")
echo "    remote: $LAUNCH_OUT"

# --- (4) poll for DONE / FAILED, streaming the log tail ----------------------
echo; echo "[4] waiting for remote DONE/FAILED (polling every ${POLL_SECS}s) ..."
echo "    (safe to Ctrl-C and re-run launch.sh; it re-attaches, never restarts)"
LAST_LINE=0
while true; do
  STATUS=$($SSH $SSH_TGT "cd $REMOTE_RUNPOD 2>/dev/null && { [ -f $SENT_DONE ] && echo DONE; [ -f $SENT_FAIL ] && echo FAILED; } || echo MISSING" || echo "SSH_DROP")
  # stream new log lines
  NEW=$($SSH $SSH_TGT "tail -n +$((LAST_LINE+1)) $REMOTE_RUNPOD/run.log 2>/dev/null" || true)
  if [[ -n "$NEW" ]]; then
    echo "$NEW"
    ADDED=$(printf '%s\n' "$NEW" | wc -l | tr -d ' ')
    LAST_LINE=$((LAST_LINE + ADDED))
  fi
  case "$STATUS" in
    *DONE*)   echo; echo "[4] remote signalled DONE."; break;;
    *FAILED*)
      echo; echo "=============================================================="
      echo "[4] remote signalled FAILED. Reason:"
      $SSH $SSH_TGT "cat $REMOTE_RUNPOD/$SENT_FAIL $REMOTE_RUNPOD/FAILED 2>/dev/null" || true
      echo "    full log tail:"
      $SSH $SSH_TGT "tail -n 40 $REMOTE_RUNPOD/run.log" || true
      echo "=============================================================="
      exit 1;;
    SSH_DROP) echo "    (ssh hiccup; retrying in ${POLL_SECS}s)";;
  esac
  sleep "$POLL_SECS"
done

# --- (5) rsync results back --------------------------------------------------
STAMP=$(date +%Y%m%d_%H%M%S)
LOCAL_OUT="$HERE/runpod_results/$STAMP"
mkdir -p "$LOCAL_OUT"
echo; echo "[5] rsync results back -> $LOCAL_OUT ..."
rsync -az --no-owner --no-group -e "$RSYNC_SSH" \
  "$SSH_TGT:$REMOTE_RUNPOD/results/" "$LOCAL_OUT/results/" || true
rsync -az --no-owner --no-group -e "$RSYNC_SSH" \
  "$SSH_TGT:$REMOTE_RUNPOD/run.log" "$LOCAL_OUT/run.log" || true
# convenience: surface summary.md at the top
if [[ -f "$LOCAL_OUT/results/summary.md" ]]; then
  cp -f "$LOCAL_OUT/results/summary.md" "$LOCAL_OUT/summary.md"
fi

echo "=============================================================="
echo " DONE. Results in: $LOCAL_OUT"
echo "   - $LOCAL_OUT/summary.md   (one table per paper table/figure)"
echo "   - $LOCAL_OUT/results/     (eval_base.json, forced/ selections)"
echo "   - $LOCAL_OUT/run.log      (full remote log)"
echo "=============================================================="
