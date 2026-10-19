#!/usr/bin/env bash
set -euo pipefail

# Terminal 3 — Vision Pro teleop tuned for the sponge practice scene.
# Same solo_teleop.sh flow as the cube (F -> ] -> 9 -> S -> T); only bridge settings differ:
#   - no head walking / squat: the table docks within reach, so leaning must not walk the robot
#   - 1:1 forward/back and wrist mapping (cube default: forward 1.55x, backward 0.35x, wrist 0.9x);
#     sideways 0.7x so wide human reaches stay inside the shorter G1 arm's workspace
#   - hand targets kept inside the G1's reach: at most 0.48 m forward of the pelvis and at most
#     5 cm across the body midline; past that the robot hand waits at the edge instead of
#     chasing an unreachable target (which twisted the arm and stopped it following)
#   - faster, lighter smoothing for the arms
# Extra arguments are appended and override these (later flags win in argparse).
# Usage: bin/sonic-teleop-sponge.sh <VISION_PRO_IP> [extra bridge args]

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <VISION_PRO_IP> [extra bridge args]"
  exit 1
fi
AVP_IP="$1"
shift

exec bash "${ROOT}/solo_teleop.sh" "${AVP_IP}" --print-debug \
  --no-head-locomotion --no-head-height-squat \
  --hand-forward-scale 1.0 --hand-backward-scale 1.0 --wrist-rotation-scale 1.0 \
  --hand-lateral-scale 0.7 \
  --max-hand-x 0.48 --hand-cross-limit 0.05 \
  --hybrid-max-speed 0.8 --hybrid-smoothing-tau 0.04 \
  --arm-max-angular-speed 3.5 --arm-orientation-smoothing-tau 0.06 \
  "$@"
