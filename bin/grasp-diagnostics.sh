#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
g1_teleop_load_env
g1_teleop_prepare_pythonpath
PYTHON="$(g1_teleop_python)"
g1_teleop_require_python "${PYTHON}"
cd "${G1_TELEOP_ROOT}"
exec "${PYTHON}" tools/grasp_diagnostics_viewer.py "$@"
