#!/usr/bin/env bash
# Sponge-only F -> ] -> S -> T workflow, checked against actual simulator state.
# Launch explicitly in terminal three after scene and control are running.
# No physical hand DDS: run_sponge_teleop.sh supplies the simulation-only flags.
# Existing solo_teleop.sh is preserved for the other workflows.

set -u
# Task-specific copy: preserve the original solo_teleop.sh and key order.

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <VISION_PRO_IP> [extra bridge args, e.g. --no-mujoco-fpv]"
  exit 1
fi

AVP_IP="$1"
shift
SESSION=tele
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${SONIC_VENV_SIM:-$HOME/.config/visionpro-g1-inspire-teleop/venv_sonic_sim}/bin/python"
STATUS="$REPO/recordings/sponge_teleop_latest.json"
check() { "$PYTHON" "$REPO/tools/sponge_startup_check.py" "$@" --status "$STATUS"; }
require_controller() {
  check controller || { log "控制器已退出，停止自动按键；请在终端二重新启动 control。"; exit 1; }
}
WAIT_PUTON="${WAIT_PUTON:-6}"
BALANCE_SEC="${BALANCE_SEC:-6}"
TRIES=3
LOG="${LOG:-/tmp/solo_bridge.log}"

log() { echo "[solo $(date +%T)] $*"; }

# Number of times a line matching $1 appears in the bridge output so far.
# Read from a log file: the FPV image warnings flood the tmux scrollback.
count() { grep -c -- "$1" "$LOG" 2>/dev/null || true; }

# Wait up to $2 seconds for a new line matching $1 (beyond count $3).
wait_new() {
  local pattern="$1" timeout="$2" before="$3" i
  for ((i = 0; i < timeout * 2; i++)); do
    if (( i % 2 == 0 )); then require_controller; fi
    (( $(count "$pattern") > before )) && return 0
    sleep 0.5
  done
  return 1
}

# Press a calibration key and retry while the bridge reports FAILED.
calibrate() {
  local key="$1" done_pattern="$2" attempt done_before fail_before
  for ((attempt = 1; attempt <= TRIES; attempt++)); do
    done_before=$(count "$done_pattern")
    fail_before=$(count "Calibration FAILED")
    log "pressing $key (attempt $attempt/$TRIES) - hold still"
    tmux send-keys -t "$SESSION" "$key"
    for ((i = 0; i < 30; i++)); do
      if (( i % 2 == 0 )); then require_controller; fi
      if (( $(count "$done_pattern") > done_before )); then
        log "$key PASS"
        return 0
      fi
      if (( $(count "Calibration FAILED") > fail_before )); then
        log "$key FAILED (moved too much), retrying in 3 s"
        sleep 3
        continue 2
      fi
      sleep 0.5
    done
    log "$key: no answer from bridge within 15 s"
  done
  return 1
}

command -v tmux >/dev/null || { echo "tmux not installed: sudo apt install tmux"; exit 1; }
[[ -d "$REPO" ]] || { echo "Repo not found at $REPO (set REPO=...)"; exit 1; }

check preflight || exit 1

if tmux has-session -t "$SESSION" 2>/dev/null; then
  log "中断旧桥接并重连，保留终端二控制器（保持 Tracking Streamer 开启）"
  # Preflight confirms the sponge scene is suspended and the controller uses lo.
  # RawKeyboard uses cbreak with ISIG: Ctrl+C interrupts the bridge and runs its
  # finally cleanup. O instead sends stop=true, terminating the native controller.
  # Do not touch a dead pane; it may be left by an earlier failed launch.
  if [[ "$(tmux display-message -p -t "$SESSION" '#{pane_dead}' 2>/dev/null)" != 1 ]]; then
    tmux send-keys -t "$SESSION" C-c 2>/dev/null || true
  fi
  for ((i = 0; i < 20; i++)); do
    tmux has-session -t "$SESSION" 2>/dev/null || break
    [[ "$(tmux display-message -p -t "$SESSION" '#{pane_dead}' 2>/dev/null)" == 1 ]] && break
    sleep 0.25
  done
  if tmux has-session -t "$SESSION" 2>/dev/null && \
     [[ "$(tmux display-message -p -t "$SESSION" '#{pane_dead}' 2>/dev/null)" != 1 ]]; then
    # Also release a previous SDK stuck waiting for its first tracking frame.
    tmux send-keys -t "$SESSION" C-c 2>/dev/null || true
    sleep 1
  fi
  tmux kill-session -t "$SESSION" 2>/dev/null || true
fi
# Recheck after cleanup and while waiting; a process can exit asynchronously.
require_controller
: > "$LOG"
printf -v bridge_cmd '%q ' bash "$REPO/tools/solo_bridge_session.sh" "$REPO" "$LOG" "$SESSION" "$AVP_IP" "$@"
tmux new-session -d -s "$SESSION" -x 220 -y 50 "$bridge_cmd"
log "bridge started in tmux session '$SESSION', waiting for Vision Pro tracking..."
# Keys pressed before "AVP tracking locked" are ignored by the bridge, so wait for it.
wait_new "Press F (or c) for CALIB_FULL" 90 0 || { log "no Vision Pro tracking after 90 s - is Tracking Streamer open on $AVP_IP? tmux attach -t $SESSION"; exit 1; }
log "tracking locked"

for ((s = WAIT_PUTON; s > 0; s--)); do
  printf '\r[solo] put on the headset and stand still: %2d s ' "$s"
  sleep 1
done
echo

calibrate f "CALIB_FULL done" || { log "F failed $TRIES times - check tracking, then: tmux attach -t $SESSION"; exit 1; }

check controller || exit 1
engage_before=$(count "ENGAGE:")
log "pressing ] (start balance policy)"
tmux send-keys -t "$SESSION" "]"
wait_new "ENGAGE:" 10 "$engage_before" || { log "] not confirmed - check deploy terminal, then: tmux attach -t $SESSION"; exit 1; }
log "等待实际策略反馈、DDS 指令及仿真速度检查（最多 20 秒）"
check drop --wait 20 || { log "条件未满足，保持悬挂，不继续 S/T；查看终端一、二。"; exit 1; }
printf '\a'
log "检查通过：在 MuJoCo 窗口按 9；确认机器人落地后回到此终端按 Enter"
read -r -p "[solo] 落地后按 Enter... " || exit 1
for ((s = BALANCE_SEC; s > 0; s--)); do
  printf '\r[solo] waiting for the robot to balance, then back to the calibration pose: %2d s ' "$s"
  sleep 1
done
echo

check teleop --wait 20 || { log "机器人未稳定落地，不继续 S/T。"; exit 1; }
calibrate s "CALIB_SYNC done" || { log "S failed $TRIES times - tmux attach -t $SESSION"; exit 1; }

check teleop || exit 1
log "pressing T - teleop is live"
tmux send-keys -t "$SESSION" "t"
log "done. Pause: tmux send-keys -t $SESSION p   Stop: tmux send-keys -t $SESSION o"
