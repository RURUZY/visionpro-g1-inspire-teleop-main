#!/usr/bin/env bash
set -euo pipefail

# Terminal 1 — MuJoCo sponge scene (table + free sponge with handle).
# Same workflow as the cube scene; only the scene file differs.
# Pair with: run_sonic_deploy.sh sim + solo_teleop.sh <VP_IP> --print-debug

source "$(dirname "$0")/_common.sh"
g1_teleop_load_env

export GR00T_ROOT="${GR00T_ROOT:-/mnt/newssd/GR00T-WholeBodyControl}"
# Grasp/paint recorders are written for the cube and roller scenes.
export SONIC_GRASP_DIAGNOSTICS=0 SONIC_PAINT_DIAGNOSTICS=0
SIM_PYTHON="${SONIC_VENV_SIM:-${HOME}/.config/visionpro-g1-inspire-teleop/venv_sonic_sim}/bin/python"
SCENE_XML="$("${SIM_PYTHON}" "${G1_TELEOP_ROOT}/tools/generate_sponge_scene.py")"
export SONIC_ROBOT_SCENE="${SCENE_XML}"
export G1_TELEOP_ENV_PRELOADED=1

cat <<EOF
SONIC Terminal 1 — sponge scene
  Scene: ${SCENE_XML}
  Next: run_sonic_deploy.sh sim  |  solo_teleop.sh <VP_IP> --print-debug
EOF

exec "$(dirname "$0")/sonic-sim.sh" "$@"
