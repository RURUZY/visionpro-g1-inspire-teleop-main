#!/usr/bin/env bash
set -euo pipefail

# Terminal 1: dry roller contact experiment in MuJoCo.
source "$(dirname "$0")/_common.sh"
g1_teleop_load_env

export GR00T_ROOT="${GR00T_ROOT:-/mnt/newssd/GR00T-WholeBodyControl}"
if [[ -n "${SONIC_PAINT_SCENE:-}" ]]; then
  SCENE_XML="$(python3 -c 'import pathlib, sys; print(pathlib.Path(sys.argv[1]).expanduser().resolve())' "${SONIC_PAINT_SCENE}")"
  if [[ ! -f "${SCENE_XML}" ]]; then
    echo "Paint scene not found: ${SCENE_XML}" >&2
    exit 1
  fi
else
  SCENE_XML="$(python3 "${G1_TELEOP_ROOT}/tools/generate_paint_scene.py")"
fi
export SONIC_ROBOT_SCENE="${SCENE_XML}"
export SONIC_HAND_MODEL=inspire
export SONIC_PAINT_DIAGNOSTICS=1
export SONIC_GRASP_DIAGNOSTICS=0
export G1_TELEOP_ENV_PRELOADED=1

cat <<EOF
SONIC Terminal 1 — dry roller + wall contact experiment (roller upright in right-side holder)
  Scene: ${SCENE_XML}
  Recording: paint_diagnostics_* under recordings/ (path printed on startup)
  Dropped brush: move hands clear, focus the MuJoCo window and press R to return it.
  Terminal 2: ./run_sonic_deploy.sh sim
  Terminal 3: bash bin/sonic-teleop-paint.sh --auto (VP: 192.168.2.5)
  After calibration: headset walking is active; tmux attach -t tele to view status.
  This scene models dry contact, not paint deposition or coating thickness.
EOF

exec bash "$(dirname "$0")/sonic-sim.sh" "$@"
