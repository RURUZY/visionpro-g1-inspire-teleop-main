#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
g1_teleop_load_env
g1_teleop_prepare_pythonpath
PYTHON="$(g1_teleop_python)"
g1_teleop_require_python "${PYTHON}"
if [[ $# -lt 1 ]]; then
  echo "Usage: bash bin/calibrate-thumb.sh <Vision_Pro_IP> [--side right|left]"
  exit 1
fi
AVP_ENDPOINT="$1"
shift
cd "${G1_TELEOP_ROOT}"
exec "${PYTHON}" scripts/calibrate_avp_thumb.py --avp-endpoint "${AVP_ENDPOINT}" "$@"
