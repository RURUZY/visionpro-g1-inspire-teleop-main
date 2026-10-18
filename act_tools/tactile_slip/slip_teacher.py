#!/usr/bin/env python3
"""Tactile + slip data for the original handled-sponge teacher (topic 1, step 1-2).

Reuses wipe_teacher unchanged: same scene, the sponge held by finger contact only (no weld),
the comfortable "left" region. Each episode randomizes conditions that make the grasp slip:
grip-force limit, tool mass, hand-tool friction, wiping press depth, and (half of the
episodes) a short push on the tool mid-wipe. Every frame records 17-pad tactile forces,
finger actuator forces, the true tool pose in the palm, and a slip label derived from it.
Failed episodes are kept: slips and drops are the point.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'wipe_teacher'))
sys.path.insert(0, str(HERE))
import wipe_teacher as W  # noqa: E402
from tactile import FEATURES, TactileMixin  # noqa: E402

GRIP_SCALE = (.03, 1.)        # fraction of the nominal 1.4 Nm finger torque limit (log-uniform)
MASS_SCALE = (.5, 2.2)        # tool mass and inertia (nominal 85 g); >~2.5x made MuJoCo unstable
HAND_FRICTION = (.3, 1.)      # handle/backplate friction against the fingers (nominal 1.0)
PRESS = (-.004, 0.)           # pad target relative to the table top (m)
PUSH_PROB = .8
PUSH_FORCE_N = (2., 8.)
TWIST_PROB = .85              # side pushes are mostly absorbed by table friction; twists cause slips
TWIST_NM = (.15, .6)
PUSH_DURATION_S = .3
PUSH_WINDOW_S = (.8, 9.)      # any phase; on the table, friction absorbs most of a push
# Normal wiping wobble is ~1.2 cm/s and ~20 deg/s (95th percentile); real slips reach >10 cm/s.
SLIP_SPEED = .03              # m/s of the tool relative to the palm
SLIP_ANGULAR = np.radians(45)  # rad/s
SLIP_MIN_FRAMES = 2           # sustained over at least 100 ms

LABEL_FEATURES = {
    'label.slip': {'dtype': 'float32', 'shape': (1,), 'names': ['slip']},
    'label.slip_speed': {'dtype': 'float32', 'shape': (2,), 'names': ['linear_m_s', 'angular_rad_s']},
}


class _StepWithPush:
    """mujoco module proxy: applies the scheduled push to the tool during mj_step."""

    def __init__(self, mujoco, teacher):
        self._mj, self._t = mujoco, teacher

    def __getattr__(self, name):
        return getattr(self._mj, name)

    def mj_step(self, m, d):
        push = self._t.push
        d.xfrc_applied[self._t.tool] = 0
        if push is not None and push['t'] <= d.time < push['t'] + PUSH_DURATION_S:
            if push['kind'] == 'force':
                d.xfrc_applied[self._t.tool, :3] = push['vector']
            else:  # torque about the tool's handle axis
                d.xfrc_applied[self._t.tool, 3:] = d.xmat[self._t.tool].reshape(3, 3)[:, 2] * push['vector']
        self._mj.mj_step(m, d)


class SlipTeacher(TactileMixin, W.WipeTeacher):
    def __init__(self, scene_path):
        super().__init__(scene_path)
        self.setup_tactile()
        m = self.model
        self.nominal = dict(
            hand_forcerange=m.actuator_forcerange[self.hand_act].copy(),
            mass=float(m.body_mass[self.tool]), inertia=m.body_inertia[self.tool].copy(),
            tool_geoms=[g for g in range(m.ngeom) if m.geom_bodyid[g] == self.tool and m.geom(g).name != 'sponge_pad'],
        )
        self.nominal['friction'] = m.geom_friction[self.nominal['tool_geoms']].copy()
        self.push = None
        self.mj = _StepWithPush(self.mj, self)

    def randomize(self, rng):
        m, n = self.model, self.nominal
        grip = float(np.exp(rng.uniform(*np.log(GRIP_SCALE))))
        mass = rng.uniform(*MASS_SCALE)
        mu = rng.uniform(*HAND_FRICTION)
        press = rng.uniform(*PRESS)
        m.actuator_forcerange[self.hand_act] = n['hand_forcerange'] * grip
        m.body_mass[self.tool] = n['mass'] * mass
        m.body_inertia[self.tool] = n['inertia'] * mass
        m.geom_friction[n['tool_geoms'], 0] = mu
        W.WIPE_Z = W.TABLE_Z + press
        self.push = None
        if rng.random() < PUSH_PROB:
            if rng.random() < TWIST_PROB:
                self.push = {'kind': 'twist', 'vector': float(rng.choice([-1, 1]) * rng.uniform(*TWIST_NM))}
            else:
                direction = rng.normal(size=3)
                direction[2] = -abs(direction[2])  # sideways or downward along the handle
                direction /= np.linalg.norm(direction)
                self.push = {'kind': 'force', 'vector': direction * rng.uniform(*PUSH_FORCE_N)}
            self.push['t'] = float(rng.uniform(*PUSH_WINDOW_S))
        return {'grip_scale': round(grip, 3), 'mass_scale': round(mass, 3), 'hand_friction': round(mu, 3),
                'press_m': round(press, 4),
                'push': None if self.push is None else {
                    'kind': self.push['kind'], 't': round(self.push['t'], 2),
                    'value': np.round(self.push['vector'], 3).tolist()}}

    def run_episode(self, rng, side, *, record=True, viewer=None):
        params = self.randomize(rng)
        frames, info = super().run_episode(rng, side, record=record, viewer=viewer)
        info.update(conditions=params, grip='contact_only')
        if frames:
            pose = np.array([f['privileged.tool_in_palm'] for f in frames])
            lin = np.r_[0, np.linalg.norm(np.diff(pose[:, :3], axis=0), axis=1) * W.FPS]
            ang = np.r_[0, np.linalg.norm(np.diff(pose[:, 3:], axis=0), axis=1) * W.FPS]
            raw = (lin > SLIP_SPEED) | (ang > SLIP_ANGULAR)
            raw[:W.FPS // 2] = False  # settling right after the pre-grasp release
            slip = raw.copy()
            for k in range(len(raw)):  # keep only runs of at least SLIP_MIN_FRAMES
                run = raw[max(0, k - SLIP_MIN_FRAMES + 1):k + 1]
                if not (len(run) == SLIP_MIN_FRAMES and run.all()):
                    slip[k] = raw[k] and k + 1 < len(raw) and raw[k + 1:k + SLIP_MIN_FRAMES].all()
            for f, s, v, w in zip(frames, slip, lin, ang):
                f['label.slip'] = np.array([s], np.float32)
                f['label.slip_speed'] = np.array([v, w], np.float32)
            onset = np.flatnonzero(slip)
            info.update(slip_frames=int(slip.sum()),
                        first_slip_s=None if not len(onset) else round(onset[0] / W.FPS, 2))
        return frames, info


def configure():
    W.WipeTeacher = SlipTeacher
    W.EXTRA_FEATURES = {**FEATURES, **LABEL_FEATURES}


if __name__ == '__main__':
    configure()
    defaults = {'--output': str(HERE / 'output'), '--repo-id': 'local/g1_inspire_wipe_slip_sim',
                '--side': 'left'}
    for flag, value in defaults.items():
        if flag not in sys.argv and not (flag == '--output' and '--viewer' in sys.argv):
            sys.argv += [flag, value]
    if '--keep-all' not in sys.argv and '--viewer' not in sys.argv:
        sys.argv.append('--keep-all')
    W.main()
