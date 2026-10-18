#!/usr/bin/env bash
# Runs inside tmux; a dead pane is visible to the launcher after clean shutdown.
set -uo pipefail
repo="$1"
log_file="$2"
session="$3"
shift 3
cd "$repo" || exit 1
tmux set-window-option -t "$session" remain-on-exit on
PYTHONUNBUFFERED=1 ./run_sonic_avp_teleop.sh "$@" 2>&1 \
  | grep --line-buffered -v 'No new image message' | tee -a "$log_file"
bridge_status=${PIPESTATUS[0]}
echo "bridge exited (status ${bridge_status}); log: ${log_file}"
exit "$bridge_status"
