#!/usr/bin/env python3
"""Sponge-scene MuJoCo entry: the same sim loop as the cube scene, plus three sponge-only aids.

1. Lower hang height. The default elastic band holds the pelvis ~17 cm above standing
   height; dropping from there makes SONIC take a large recovery step toward the table.
   SONIC_BAND_LENGTH lowers the hang point (same as pressing 7 in the viewer before 9).
2. G = assisted pre-grasp. The sponge is placed at the right palm's grip pose and follows
   the hand while it stays closed; opening the hand drops it. This is a kinematic assist
   for teleop practice, NOT a finger-contact grasp: in this SONIC scene the same grip
   pose does not hold by contact alone (the scripted teacher scene does), cause not yet
   found. Do not use assisted episodes to judge grasp stability.
3. R = put the sponge back on the table. Robot, controller and calibration are untouched.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from threading import Event

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

HANG_OFFSET = float(os.environ.get('SONIC_BAND_LENGTH', '-0.10'))  # tested: -0.10 lands in place, -0.16 steps back
# Grip pose of the tool frame in right_base_link, same as the scripted teacher preview.
GRIP_POS = np.array([-.006, .025, .14])
GRIP_QUAT = np.array([.7071067812, 0, -.7071067812, 0])
CLOSE_JOINTS = ('index_1', 'middle_1', 'ring_1', 'little_1')
CLOSED_FRACTION = .6     # mean closure of the four fingers that counts as "closed"
OPEN_FRACTION = .3       # below this the hand counts as opened again
PIN_TIMEOUT_S = 8.       # time to close the hand after pressing G


class SpongeAids:
    """Runs on the simulation thread between steps; key callbacks only set events."""

    def __init__(self, model):
        import mujoco
        self.mj, self.model = mujoco, model
        joint = model.joint('sponge_free')
        self.qpos = slice(int(joint.qposadr[0]), int(joint.qposadr[0]) + 7)
        self.dofs = slice(int(joint.dofadr[0]), int(joint.dofadr[0]) + 6)
        self.tool = model.body('sponge_tool').id
        self.palm = model.body('right_base_link').id
        self.tool_geoms = set(np.flatnonzero(model.geom_bodyid == self.tool))
        fingers = [model.joint(f'right_hand_{j}_joint') for j in CLOSE_JOINTS]
        self.finger_qadr = np.array([int(j.qposadr[0]) for j in fingers])
        self.finger_low = np.array([j.range[0] for j in fingers])
        self.finger_span = np.array([j.range[1] - j.range[0] for j in fingers])
        self.grip_request, self.reset_request = Event(), Event()
        self.pin_started = self.closed_since = None
        self.scratch = mujoco.MjData(model)

    def closure(self, data):
        return float(np.mean((data.qpos[self.finger_qadr] - self.finger_low) / self.finger_span))

    def grip_pose(self, data):
        mj = self.mj
        quat = np.zeros(4)
        mj.mju_mulQuat(quat, data.xquat[self.palm], GRIP_QUAT)
        return np.r_[data.xpos[self.palm] + data.xmat[self.palm].reshape(3, 3) @ GRIP_POS, quat]

    def _hold(self, data, pose):
        data.qpos[self.qpos] = pose
        data.qvel[self.dofs] = 0
        data.qacc_warmstart[self.dofs] = 0

    def apply(self, data):
        """Call once before each mj_step."""
        if self.reset_request.is_set():
            self.reset_request.clear()
            self.pin_started = None
            self._return_to_table(data)
        if self.grip_request.is_set():
            self.grip_request.clear()
            self.pin_started, self.closed_since = data.time, None
            print('[sponge] Sponge placed in the right hand. Close the hand to hold it.', flush=True)
        if self.pin_started is None:
            return
        closure = self.closure(data)
        if self.closed_since is None:
            if closure >= CLOSED_FRACTION:
                self.closed_since = data.time
                print('[sponge] ASSISTED HOLD: sponge follows the closed hand. Open the hand to drop it.',
                      flush=True)
            elif data.time - self.pin_started > PIN_TIMEOUT_S:
                self.pin_started = None
                print('[sponge] Hand not closed within 8 s; sponge released. Press G again.', flush=True)
                return
        elif closure < OPEN_FRACTION:
            self.pin_started = self.closed_since = None
            print('[sponge] Hand opened; sponge released.', flush=True)
            return
        self._hold(data, self.grip_pose(data))

    @property
    def assisted(self):
        return self.closed_since is not None

    def _return_to_table(self, data):
        mj, s = self.mj, self.scratch
        s.qpos[:] = data.qpos
        s.qpos[self.qpos] = self.model.qpos0[self.qpos]
        mj.mj_forward(self.model, s)
        for c in s.contact:
            if ((int(c.geom1) in self.tool_geoms) != (int(c.geom2) in self.tool_geoms)) and c.dist < -1e-4:
                print('[sponge] Table spot occupied; move the hand away and press R again.', flush=True)
                return
        self._hold(data, self.model.qpos0[self.qpos])
        data.xfrc_applied[self.tool] = 0
        print('[sponge] Sponge back on the table.', flush=True)


def install():
    import mujoco.viewer
    import gear_sonic.utils.mujoco_sim.base_sim as base_sim
    from gear_sonic.utils.mujoco_sim.unitree_sdk2py_bridge import ElasticBand

    class LowerBand(ElasticBand):
        def __init__(self):
            super().__init__()
            self.length = HANG_OFFSET

    base_sim.ElasticBand = LowerBand
    original_env = base_sim.DefaultEnv
    original_launch = mujoco.viewer.launch_passive
    pending = {}

    def launch(*args, **kwargs):
        previous = kwargs.get('key_callback')

        def on_key(key):
            if key in (ord('G'), ord('g')):
                pending['grip'].set()
            elif key in (ord('R'), ord('r')):
                pending['reset'].set()
            elif previous is not None:
                previous(key)
        kwargs['key_callback'] = on_key
        return original_launch(*args, **kwargs)

    class SpongeEnv(original_env):
        def __init__(self, *args, **kwargs):
            pending['grip'], pending['reset'] = Event(), Event()
            mujoco.viewer.launch_passive = launch
            try:
                super().__init__(*args, **kwargs)
            finally:
                mujoco.viewer.launch_passive = original_launch
            self.sponge = SpongeAids(self.mj_model)
            self.sponge.grip_request, self.sponge.reset_request = pending['grip'], pending['reset']

        def sim_step(self):
            self.sponge.apply(self.mj_data)
            super().sim_step()

    base_sim.DefaultEnv = SpongeEnv


if __name__ == '__main__':
    install()
    import sonic_run_sim_loop

    print(f'[sponge] Hang offset {HANG_OFFSET:+.2f} m. MuJoCo window: G = sponge into right hand '
          '(assisted hold), R = sponge back on table, 9 = drop.', flush=True)
    sonic_run_sim_loop.main()
