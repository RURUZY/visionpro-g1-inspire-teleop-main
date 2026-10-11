#!/usr/bin/env bash
set -euo pipefail
export SONIC_GRASP_DIAGNOSTICS=1
exec bash "$(dirname "$0")/sonic-sim-pick.sh" "$@"
