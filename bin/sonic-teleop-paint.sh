#!/usr/bin/env bash
set -euo pipefail

# Terminal 3: the recorded cube workflow's two-hand mapping, with headset walking.
usage() {
  echo "Usage: $0 [vision_pro_ip_or_room_code=192.168.2.5] [--auto] [--sim-thumb-full-range] [--no-mujoco-fpv]"
}
if [[ "${1:-}" == --help ]]; then
  usage
  exit 0
fi
AVP_ENDPOINT=192.168.2.5
if [[ $# -gt 0 && "$1" != --* ]]; then
  AVP_ENDPOINT="$1"
  shift
fi
AUTO_KEYS=0
EXTRA=()
for arg in "$@"; do
  case "$arg" in
    --auto) AUTO_KEYS=1 ;;
    --sim-thumb-full-range|--no-mujoco-fpv|--enable-mujoco-fpv) EXTRA+=("$arg") ;;
    *) echo "Unsupported option for the paint simulation: ${arg}" >&2; usage >&2; exit 2 ;;
  esac
done

source "$(dirname "$0")/_common.sh"
g1_teleop_load_env

PAINT_ARGS=(
  --active-hands both
  --head-locomotion
  --no-head-height-squat
  --no-hybrid-locomotion
  --keyboard-loco-speed 0
  --loco-max-speed 0.15
  --enable-inspire-hand-sim
  --enable-mujoco-fpv
)

cat <<EOF
SONIC Terminal 3 — dry roller experiment (same two-hand mapping as cube workflow)
  Endpoint: ${AVP_ENDPOINT}
  Flow: F -> ] -> S -> T; then grasp handle, lift, approach wall, make a short stroke.
  Both arms/hands active. Headset walking is active after T; no B unlock required.
  Walking command limit: 0.15 m/s. Keyboard movement (WASD, ,/.) is OFF.
  Stop moving the headset to stop translation. Head squat is OFF.
  Headset video is ON unless --no-mujoco-fpv is explicitly supplied.
  SONIC still balances the whole body; the feet are not fixed to the floor.
  Use Terminal 2 in sim mode. Physical hand DDS is not enabled by this launcher.
EOF

cd "${G1_TELEOP_ROOT}"
if [[ "${AUTO_KEYS}" == 1 ]]; then
  echo "Automatic F -> ] -> S -> T via solo_teleop.sh; confirm landing when prompted."
  echo "After calibration: headset walking is active. tmux attach -t tele to view the bridge."
  REPO="${G1_TELEOP_ROOT}" exec bash "${G1_TELEOP_ROOT}/solo_teleop.sh" \
    "${AVP_ENDPOINT}" "${PAINT_ARGS[@]}" "${EXTRA[@]}"
fi

g1_teleop_prepare_pythonpath
BRIDGE_PYTHON="$(g1_teleop_python)"
g1_teleop_require_python "${BRIDGE_PYTHON}"
exec "${BRIDGE_PYTHON}" scripts/g1_avp_sonic_teleop.py \
  --avp-endpoint "${AVP_ENDPOINT}" \
  "${PAINT_ARGS[@]}" \
  --print-debug \
  "${EXTRA[@]}"
