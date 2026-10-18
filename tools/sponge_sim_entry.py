#!/usr/bin/env python3
"""Sponge-scene MuJoCo entry: the same sim loop as the cube scene, plus three sponge-only aids.

1. Lower hang height. The default elastic band holds the pelvis ~17 cm above standing
   height; dropping from there makes SONIC take a large recovery step toward the table.
   SONIC_BAND_LENGTH lowers the hang point (same as pressing 7 in the viewer before 9).
2. G = assisted pre-grasp. The sponge is placed at the right palm's grip point and follows the
   hand while it stays closed; opening the hand drops it. Its pad is kept level the whole time
   (only its heading follows the fingers), because the robot wrist cannot always reach the angle
   that makes a fist-held handle vertical. This is a kinematic assist
   for teleop practice, NOT a finger-contact grasp: in this SONIC scene the same grip
   pose does not hold by contact alone (the scripted teacher scene does), cause not yet
   found. Do not use assisted episodes to judge grasp stability.
3. R = put the sponge back on the table. Robot, controller and calibration are untouched.
   It also happens automatically when the sponge has lain tipped over (> 60 deg) or off the table
   for 1.5 s with the hand more than 12 cm away: a lying handle cannot be picked up with this hand.
   While the sponge is held off the table, its tilt, the handle direction in the palm frame and
   the right wrist angles are logged at 5 Hz to recordings/sponge_grip_<time>.jsonl, to design a
   handle angle that comes out level under the operator's natural grip.
4. Table docking. The table (a mocap body) starts parked TABLE_PARK_OFFSET farther away so the
   arm swing at ] and the drop at 9 cannot touch it; once the band is released and the robot has
   stood still for DOCK_AFTER_S, the table and the sponge on it slide in to their final place.
   A practice convenience, as if someone pushed the table over after the robot stood up.
5. Faster sim loop. Profiling on the Jetson (68 steps/s = 0.34x real time) showed the desktop
   viewer sync costs ~17 ms per call at 50 Hz, and the upstream per-step rate limiter sleeps on
   fast steps without ever catching up after slow ones. Here the viewer refreshes at
   SONIC_VIEWER_HZ (default 15; 0 = every step); the headset image keeps its own rate.
   SONIC_SIM_CATCHUP=1 additionally replaces the loop with one that sleeps only when ahead of
   real time (68 -> ~123 steps/s, ~1.0x with the controller attached). It is OFF by default:
   in 3 of ~10 runs the sim stalled for ~2 s, the controller lost LowState and the robot fell;
   the cause was not reproduced or confirmed. Publishing robot state less often did not help.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from threading import Event

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

HANG_OFFSET = float(os.environ.get('SONIC_BAND_LENGTH', '-0.10'))  # tested: -0.10 lands in place, -0.16 steps back
# Grip point of the handle in right_base_link (same point as the scripted teacher's grasp).
GRIP_POS = np.array([-.006, .025, .14])
CLOSE_JOINTS = ('index_1', 'middle_1', 'ring_1', 'little_1')
CLOSED_FRACTION = .6     # mean closure of the four fingers that counts as "closed"
OPEN_FRACTION = .3       # below this the hand counts as opened again
PIN_TIMEOUT_S = 8.       # time to close the hand after pressing G
DOCK_AFTER_S = 2.        # standing time after the band is released before the table moves in
DOCK_DURATION_S = 1.5
DOCK_MAX_PELVIS_X = .12  # the robot must not stand closer than this to dock the table
DOCK_MAX_SPEED = .05     # m/s pelvis speed that counts as standing still
TIP_ANGLE_DEG = 60.      # handle tilt from vertical that counts as tipped over
TIP_HOLD_S = 1.5         # tipped this long before the automatic return
TIP_HAND_CLEAR_M = .12   # only when the palm is at least this far from the sponge
VIEWER_HZ = float(os.environ.get('SONIC_VIEWER_HZ', '15'))
MAX_BACKLOG_S = .1       # when further behind than this, stop trying to catch up
CATCHUP = os.environ.get('SONIC_SIM_CATCHUP', '0') == '1'


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
        self.grip_rotation = np.eye(3)
        self.home = model.qpos0[self.qpos].copy()  # sponge return pose; moves with the table
        table = model.body('sponge_table')
        self.mocap = int(table.mocapid[0])
        self.table_park = model.body_pos[table.id].copy()
        from generate_sponge_scene import TABLE_PARK_OFFSET
        self.dock_shift = np.array([-TABLE_PARK_OFFSET, 0., 0.])
        self.standing_since = self.dock_started = None
        self.docked = False
        self.tipped_since = None
        hand = [model.joint(f'right_wrist_{n}_joint') for n in ('roll', 'pitch', 'yaw')]
        self.wrist_qadr = np.array([int(j.qposadr[0]) for j in hand])
        self.hand_bodies = set()
        for body in range(model.nbody):
            ancestor = body
            while ancestor and ancestor != self.palm:
                ancestor = int(model.body_parentid[ancestor])
            if ancestor == self.palm:
                self.hand_bodies.add(body)
        self.grip_log = None
        self.grip_log_last = -1.
        self.warned_close = False

    def dock(self, data, band_enabled):
        """Slide the table (and the sponge resting on it) in after the robot has stood for a while."""
        if self.docked:
            return
        if self.dock_started is None:
            still = (not band_enabled and float(np.linalg.norm(data.qvel[:2])) < DOCK_MAX_SPEED)
            self.standing_since = (self.standing_since or data.time) if still else None
            if self.standing_since is None or data.time - self.standing_since < DOCK_AFTER_S:
                return
            if data.qpos[0] > DOCK_MAX_PELVIS_X:
                if not self.warned_close:
                    print('[sponge] Robot stands too far forward; table stays parked. Step back to dock it.', flush=True)
                    self.warned_close = True
                return
            self.dock_started = data.time
            self.sponge_start = data.qpos[self.qpos].copy()
            on_table = abs(self.sponge_start[2] - self.home[2]) < .02
            self.carry_sponge = on_table and self.pin_started is None
            print('[sponge] Robot is standing; moving the table in.', flush=True)
        u = min(1., (data.time - self.dock_started) / DOCK_DURATION_S)
        u = u * u * (3 - 2 * u)
        data.mocap_pos[self.mocap] = self.table_park + u * self.dock_shift
        if self.carry_sponge:
            pose = self.sponge_start.copy()
            pose[:3] += u * self.dock_shift
            self._hold(data, pose)
        if u >= 1.:
            self.docked = True
            self.home[:3] += self.dock_shift
            print('[sponge] Table in place. G = sponge into the right hand, R = sponge back on the table.',
                  flush=True)

    def closure(self, data):
        return float(np.mean((data.qpos[self.finger_qadr] - self.finger_low) / self.finger_span))

    def level_grip_rotation(self, data):
        """Tool rotation in the palm frame that makes the pad level for the hand pose right now.

        Tool z (handle axis, pad below) points straight up; tool x follows the horizontal
        direction the fingers point, so the sponge faces where the hand faces.
        """
        palm = data.xmat[self.palm].reshape(3, 3)
        forward = palm[:, 2].copy()          # finger direction
        forward[2] = 0
        if np.linalg.norm(forward) < 1e-3:  # fingers pointing straight up/down
            forward = palm[:, 0].copy()
            forward[2] = 0
        x = forward / np.linalg.norm(forward)
        z = np.array([0., 0., 1.])
        tool = np.column_stack([x, np.cross(z, x), z])
        return palm.T @ tool

    def grip_pose(self, data):
        palm = data.xmat[self.palm].reshape(3, 3)
        self.grip_rotation = self.level_grip_rotation(data)  # keep the pad level while held
        quat = np.zeros(4)
        self.mj.mju_mat2Quat(quat, (palm @ self.grip_rotation).flatten())
        return np.r_[data.xpos[self.palm] + palm @ GRIP_POS, quat]

    def _hold(self, data, pose):
        data.qpos[self.qpos] = pose
        data.qvel[self.dofs] = 0
        data.qacc_warmstart[self.dofs] = 0

    def tipped(self, data):
        handle_axis = data.xmat[self.tool].reshape(3, 3)[:, 2]
        return float(np.degrees(np.arccos(np.clip(handle_axis[2], -1, 1)))) > TIP_ANGLE_DEG

    def auto_upright(self, data):
        """Return a tipped-over sponge to its spot once the hand is clear."""
        if not self.docked or self.pin_started is not None:
            self.tipped_since = None
            return
        far = float(np.linalg.norm(data.xpos[self.palm] - data.xpos[self.tool])) > TIP_HAND_CLEAR_M
        fell_off = data.xpos[self.tool][2] < self.home[2] - .10
        if not ((self.tipped(data) or fell_off) and far):
            self.tipped_since = None
            return
        if self.tipped_since is None:
            self.tipped_since = data.time
        elif data.time - self.tipped_since > TIP_HOLD_S:
            self.tipped_since = None
            print('[sponge] Sponge tipped over or fell off; returning it upright.', flush=True)
            self._return_to_table(data)

    def hand_contacts(self, data):
        n = 0
        for i in range(data.ncon):
            c = data.contact[i]
            a, b = self.model.geom_bodyid[c.geom1], self.model.geom_bodyid[c.geom2]
            if (a == self.tool and b in self.hand_bodies) or (b == self.tool and a in self.hand_bodies):
                n += 1
        return n

    def log_grip(self, data):
        """Record how the sponge sits in the hand while it is carried (5 Hz)."""
        if data.time - self.grip_log_last < .2:
            return
        lifted = data.xpos[self.tool][2] > self.home[2] + .02
        assisted = self.pin_started is not None
        if not lifted or not (assisted or self.hand_contacts(data) >= 2):
            return
        self.grip_log_last = data.time
        if self.grip_log is None:
            import json, time as _time
            self._json = json
            path = Path(os.environ.get('SONIC_SPONGE_GRIP_LOG') or Path(__file__).resolve().parents[1]
                        / 'recordings' / _time.strftime('sponge_grip_%Y%m%d_%H%M%S.jsonl'))
            path.parent.mkdir(parents=True, exist_ok=True)
            self.grip_log = path.open('a')
            print(f'[sponge] Logging how the sponge sits in the hand to {path}', flush=True)
        palm = data.xmat[self.palm].reshape(3, 3)
        handle = data.xmat[self.tool].reshape(3, 3)[:, 2]
        self.grip_log.write(self._json.dumps({
            'sim_time': round(float(data.time), 3), 'mode': 'assisted' if assisted else 'contact',
            'tilt_deg': round(float(np.degrees(np.arccos(np.clip(handle[2], -1, 1)))), 2),
            'handle_in_palm': np.round(palm.T @ handle, 4).tolist(),
            'palm_axes_world': np.round(palm, 4).tolist(),
            'wrist_roll_pitch_yaw': np.round(data.qpos[self.wrist_qadr], 4).tolist(),
        }) + '\n')
        self.grip_log.flush()

    def apply(self, data):
        """Call once before each mj_step."""
        self.auto_upright(data)
        self.log_grip(data)
        if self.reset_request.is_set():
            self.reset_request.clear()
            self.pin_started = None
            self._return_to_table(data)
        if self.grip_request.is_set():
            self.grip_request.clear()
            self.pin_started, self.closed_since = data.time, None
            self.grip_rotation = self.level_grip_rotation(data)  # pad level for the hand pose at G
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
        s.qpos[self.qpos] = self.home
        s.mocap_pos[:] = data.mocap_pos
        mj.mj_forward(self.model, s)
        for c in s.contact:
            if ((int(c.geom1) in self.tool_geoms) != (int(c.geom2) in self.tool_geoms)) and c.dist < -1e-4:
                print('[sponge] Table spot occupied; move the hand away and press R again.', flush=True)
                return
        self._hold(data, self.home)
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
            band = bool(self.elastic_band and self.elastic_band.enable)
            self.sponge.dock(self.mj_data, band)
            self.sponge.apply(self.mj_data)
            super().sim_step()

    base_sim.DefaultEnv = SpongeEnv
    if CATCHUP:
        base_sim.BaseSimulator.start = _start
    elif VIEWER_HZ > 0:
        original_init = base_sim.BaseSimulator.__init__

        def init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            self.viewer_dt = max(self.sim_dt, 1. / VIEWER_HZ)  # upstream loop, slower desktop viewer

        base_sim.BaseSimulator.__init__ = init


def _start(self):
    """BaseSimulator.start with a slower desktop viewer and real-time catch-up scheduling."""
    import pickle
    import time
    every = lambda dt: max(1, int(round(dt / self.sim_dt)))  # noqa: E731
    viewer_every = every(1. / VIEWER_HZ) if VIEWER_HZ > 0 else 1
    reward_every, image_every = every(self.reward_dt), every(self.image_dt)
    sim_cnt, last_redis = 0, time.time()
    deadline = time.monotonic()
    try:
        while self._running and ((self.sim_env.viewer and self.sim_env.viewer.is_running())
                                 or self.sim_env.viewer is None):
            self.sim_env.sim_step()
            if self.redis_client is not None and time.time() - last_redis > .1:
                head_pose = self.sim_env.get_head_pose()
                self.redis_client.set('head_pos', pickle.dumps(head_pose[:3]))
                self.redis_client.set('head_quat', pickle.dumps(head_pose[3:]))
                last_redis = time.time()
            if sim_cnt % viewer_every == 0:
                self.sim_env.update_viewer()
            if sim_cnt % reward_every == 0:
                self.sim_env.update_reward()
            if sim_cnt % image_every == 0:
                self.sim_env.update_render_caches()
            sim_cnt += 1
            deadline += self.sim_dt
            ahead = deadline - time.monotonic()
            if ahead > 0:
                time.sleep(ahead)
            elif ahead < -MAX_BACKLOG_S:
                deadline = time.monotonic()  # too far behind: run at full speed, no unbounded backlog
    except KeyboardInterrupt:
        print('Simulator interrupted by user.')
    finally:
        self.close()


if __name__ == '__main__':
    install()
    import sonic_run_sim_loop

    print(f'[sponge] Hang offset {HANG_OFFSET:+.2f} m. The table moves in about {DOCK_AFTER_S:.0f} s after the '
          'robot stands. MuJoCo window: G = sponge into right hand (assisted hold), R = sponge back, 9 = drop.',
          flush=True)
    sonic_run_sim_loop.main()
