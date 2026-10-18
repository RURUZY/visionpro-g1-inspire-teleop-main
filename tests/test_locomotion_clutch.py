"""Walking lock, heading continuity and head/keyboard resume without live I/O."""

import numpy as np
import pytest

from g1_teleop.locomotion.clutch import LocomotionClutch
from g1_teleop.locomotion.head import (
    HeadLocomotionConfig, HeadLocomotionState, SonicPlannerCommand,
    compute_sonic_planner_command, reset_head_locomotion_state,
)
from g1_teleop.locomotion.hybrid import KeyboardLocomotionController, keyboard_planner_command
from g1_teleop.transforms.frames import rotation_z
from g1_teleop.bridge.keyboard import BridgeState, HELD_KEYS, handle_key


def test_locked_blocks_translation_turning_and_height_even_without_tracking():
    clutch = LocomotionClutch(enabled=True)
    clutch.lock(0.7)
    incoming = SonicPlannerCommand(4, np.array([1., 0., 0.]), np.array([0., -1., 0.]), .15, .5)
    for command in (incoming, None):
        out = clutch.filter(command)
        assert out.mode == 0 and out.speed == -1 and out.height == -1
        np.testing.assert_array_equal(out.movement, [0., 0., 0.])
        np.testing.assert_allclose(out.facing, [np.cos(.7), np.sin(.7), 0.])
    assert incoming.mode == 4 and incoming.height == .5  # Input not mutated.


def test_disabled_keeps_existing_launchers_unchanged():
    clutch = LocomotionClutch()
    cmd = SonicPlannerCommand(1, np.array([1., 0., 0.]), np.array([1., 0., 0.]), .15)
    assert not clutch.locked
    assert clutch.filter(cmd) is cmd
    assert clutch.filter(None) is None


@pytest.mark.parametrize("facing", [0., 1.1, -2.9])
def test_resume_after_large_head_motion_does_not_walk_or_snap_heading(facing):
    clutch = LocomotionClutch(enabled=True)
    clutch.lock(facing)
    state = HeadLocomotionState()
    reset_head_locomotion_state(state, calib_yaw=0.)
    state.robot_base_yaw_at_calib = .2
    state.prev_head_pos = np.zeros(3)
    state.smooth_vx = .15
    pose = np.eye(4)
    pose[:3, :3] = rotation_z(-.9)
    pose[:3, 3] = [1., -.4, 1.6]  # Operator moved/looked around while locked.
    assert clutch.unlock(state, pose)
    assert state.robot_base_yaw_at_calib == .2
    cfg = HeadLocomotionConfig(max_speed=.15)
    robot_yaw = .2 + facing
    robot_quat = np.array([np.cos(robot_yaw / 2), 0., 0., np.sin(robot_yaw / 2)])
    for _ in range(60):
        cmd = compute_sonic_planner_command(
            pose, state, cfg, .02, calib_pos=pose[:3, 3].copy(), robot_base_quat=robot_quat)
        assert cmd.mode == 0
        np.testing.assert_allclose(cmd.facing, [np.cos(facing), np.sin(facing), 0.], atol=1e-12)
    old_pos = state.prev_head_pos.copy()
    assert not clutch.unlock(state, pose)  # B autorepeat cannot re-zero each frame.
    np.testing.assert_array_equal(state.prev_head_pos, old_pos)


@pytest.mark.parametrize("direction,key", [(1., "w"), (-1., "s")])
def test_head_and_keyboard_still_move_after_unlock_then_lock_stops_both(direction, key):
    clutch = LocomotionClutch(enabled=True)
    state = HeadLocomotionState()
    pose = np.eye(4)
    clutch.unlock(state, pose)
    cfg = HeadLocomotionConfig(max_speed=.15)
    for frame in range(60):
        pose[0, 3] = direction * .2 * frame * .02
        head = compute_sonic_planner_command(pose, state, cfg, .02, calib_pos=np.zeros(3))
    assert head.mode == 1 and head.movement[0] * direction > 0
    assert 0 < head.speed <= .15
    keyboard = KeyboardLocomotionController()
    for _ in range(60):
        kb = keyboard_planner_command(keyboard, {key}, cfg, .02, default_speed=.15)
    assert kb.mode == 1 and kb.movement[0] * direction > 0
    assert clutch.filter(head) is head and clutch.filter(kb) is kb
    clutch.lock(0.)
    assert clutch.filter(head).mode == 0
    assert clutch.filter(kb).mode == 0


@pytest.mark.parametrize("options", [
    {"keyboard_movement_enabled": False},
    {"keyboard_walk_speed": 0.0},
])
def test_disabled_keyboard_cannot_change_walk_or_heading_but_stop_still_works(options):
    state = BridgeState(facing_angle=.4)
    for key in HELD_KEYS | {"1", "2", "3"}:
        assert handle_key(state, key, head_locomotion=True, **options)
        assert state.mode == 0 and state.speed == -1
        assert state.facing_angle == .4
        np.testing.assert_array_equal(state.movement, np.zeros(3))
    for key in ("o", "O", "\x03"):
        assert not handle_key(state, key, head_locomotion=True, hybrid_locomotion=False,
                              **options)


def test_slow_head_input_accumulates_smoothing_and_stops_when_still():
    state = HeadLocomotionState()
    reset_head_locomotion_state(state, calib_yaw=0.)
    cfg = HeadLocomotionConfig(max_speed=.15, smooth_alpha=.18, velocity_deadzone=.06)
    pose = np.eye(4)
    for frame in range(60):
        pose[0, 3] = .1 * frame * .02
        cmd = compute_sonic_planner_command(pose, state, cfg, .02, calib_pos=np.zeros(3))
    assert cmd.mode == 1 and cmd.speed == pytest.approx(.1, abs=.001)
    for _ in range(60):
        cmd = compute_sonic_planner_command(pose, state, cfg, .02, calib_pos=np.zeros(3))
    assert cmd.mode == 0 and cmd.speed == -1
    assert state.smooth_vx == state.smooth_vy == 0.
