#!/usr/bin/env bash
set -euo pipefail

# First MuJoCo grasp experiment: right hand, no head walk or squat.
# Keep the normal pick-up launcher unchanged for later walking experiments.
if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <vision_pro_ip_or_room_code>" >&2
  exit 1
fi

exec "$(dirname "$0")/sonic-teleop-pick.sh" "$1" \
  --active-hands right \
  --no-head-locomotion \
  --no-head-height-squat \
  --no-hybrid-locomotion \
  --keyboard-loco-speed 0 \
  --loco-max-speed 0
