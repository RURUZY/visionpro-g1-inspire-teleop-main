"""Compare the actual paint launch arguments against the recorded cube workflow."""

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

from g1_teleop.bridge.cli import parse_args

REPO = Path(__file__).resolve().parents[1]


def parse_entry_defaults(argv):
    spec = importlib.util.spec_from_file_location("sonic_entry", REPO / "scripts/g1_avp_sonic_teleop.py")
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    captured = []
    with patch.object(sys, "argv", argv), patch.object(
        entry.runpy, "run_path", lambda *a, **kw: captured.extend(sys.argv)
    ), contextlib.redirect_stdout(io.StringIO()):
        entry.main()
    with patch.object(sys, "argv", captured):
        return vars(parse_args())


@pytest.mark.parametrize("automatic", [False, True])
def test_paint_uses_cube_body_mapping_for_manual_and_automatic_launch(tmp_path, automatic):
    # Capture argv before any AVP/robot connection or tmux session can start.
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/g1_avp_sonic_teleop.py").write_text(
        "import json, sys\nprint(json.dumps(sys.argv))\n")
    (tmp_path / "solo_teleop.sh").write_text(
        'exec "$SONIC_PYTHON" "$G1_TELEOP_ROOT/scripts/g1_avp_sonic_teleop.py" --avp-endpoint "$@"\n')
    env = dict(os.environ, G1_TELEOP_ROOT=str(tmp_path), SONIC_PYTHON=sys.executable)
    command = ["bash", str(REPO / "bin/sonic-teleop-paint.sh")]
    if automatic:
        command.append("--auto")
    result = subprocess.run(command, env=env, check=True, text=True, capture_output=True)
    paint = parse_entry_defaults(json.loads(result.stdout.splitlines()[-1]))
    # Archived shell command and bridge log: both hands, this generic entry,
    # velocity/yaw gains zero, head squat disabled. No per-arm mapping override.
    cube = parse_entry_defaults([
        "bridge", "--avp-endpoint", "192.168.2.5", "--loco-velocity-gain", "0",
        "--loco-yaw-gain", "0", "--no-head-height-squat",
    ])
    movement_only = {
        "locomotion_clutch", "loco_velocity_gain", "loco_yaw_gain",
        "loco_max_speed", "keyboard_loco_speed", "print_debug",
    }
    assert {k: v for k, v in paint.items() if k not in movement_only} == {
        k: v for k, v in cube.items() if k not in movement_only}
    assert paint["active_hands"] == "both"
    assert not paint["locomotion_clutch"] and not paint["hybrid_locomotion"]
