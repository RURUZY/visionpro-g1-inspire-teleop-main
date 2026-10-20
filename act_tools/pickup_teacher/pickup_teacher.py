#!/usr/bin/env python3
"""Scripted pick-up-then-wipe teacher: the sponge starts standing on the table, not in the hand.

Each episode: open hand above and behind-right of the handle -> down beside it ->
slide in to the handle -> close the fingers -> lift -> wipe the instructed region (same strokes
as wipe_teacher) -> lift away. The tool is a free body held only by finger contact.
After the lift the wrist turns the tool back to zero yaw (like the pre-grasped teacher).
Randomized per episode: tool position (+-1.5 cm x, +-2 cm y) and yaw (+-15 deg), region offset,
start pose, approach timing and stroke speed. Coverage and in-hand drift are measured only after
the lift, so the resting tool or the grasp itself never counts as wiping or slipping.

Same dataset format as wipe_teacher (13-D state/action; the 6 hand actions now open and close).
Simulation only, fixed pelvis; not a real-robot interface.

  python pickup_teacher/pickup_teacher.py --episodes 200 --side left --output pickup_teacher/runs/left200
  add --randomize light,color,camera,physics for domain randomization (wipe_teacher/randomization.py)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'wipe_teacher'))
import wipe_teacher as W  # noqa: E402

TOOL_XY = np.array([.19, -.18])     # handle position on the table, between and in front of the regions
TOOL_XY_RANGE = np.array([.015, .02])
TOOL_YAW_DEG = 15.
APPROACH = np.array([-.10, -.06, 0.])  # slide in from behind-right (collision-checked along the whole path)
HOVER = .12                          # height above the approach point before descending
START_RANGE = np.array([.02, .02, .03])  # start pose around the hover point (z: upward only)
LIFT = .10
GRASP_Z = .0005                      # pad bottom just above the table at the grasp
CLOSE_S = .7                         # finger closing ramp
REST_Z = .0985                       # tool origin above the table top when standing on its pad
PRESS = .005                         # extra press while wiping: a contact grip yields upward on the table


def yaw_matrix(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


class PickupPlan:
    """Site target and hand closure (0 open .. 1 closed) over time."""

    def __init__(self, points, times, closure, grasped_at, wipe_from, yaw, turn_from):
        self.path = W.Path3D(points, times)
        self.yaw, self.turn_from = yaw, turn_from
        self.closure, self.duration = closure, self.path.duration
        self.grasped_at = grasped_at    # after the lift: grasp drift and coverage are measured from here
        self.wipe_from = wipe_from
        self.offset = np.zeros(3)       # applied to wiping targets; set by PickupTeacher.measure_grip

    def __call__(self, t):
        return self.path(t) + (self.offset if t >= self.wipe_from else 0)

    def rot(self, t):
        """Target orientation: the tool's table yaw until the lift, then turned back to zero yaw
        (wiping with a yawed pad pushes the upper arm into the torso near the body midline)."""
        u = W.smooth((t - self.turn_from) / (self.wipe_from - self.turn_from))
        return yaw_matrix(self.yaw * (1 - u))

    def hand(self, t, closed):
        return closed * float(np.clip(self.closure(t), 0, 1))


class PickupTeacher(W.WipeTeacher):
    RANDOMIZE = ()      # domain randomization kinds for data generation (see --randomize)
    RANDOMIZE_SEED = 0

    def __init__(self, scene_path):
        super().__init__(scene_path)
        self.closed_hand = self.qref[self.control_qadr][self.hand_act].copy()
        self.target_rot = np.eye(3)
        self.randomizer = None
        if self.RANDOMIZE:
            from randomization import Perturber
            self.randomizer = Perturber(self, self.RANDOMIZE, self.RANDOMIZE_SEED)

    def solve(self, target):
        """wipe_teacher.solve with a yawed target orientation (the tool keeps its table yaw)."""
        mj, m, ik = self.mj, self.model, self.ik
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for _ in range(80):
            mj.mj_forward(m, ik)
            rot = ik.site_xmat[self.site].reshape(3, 3)
            orientation_error = .5 * sum(np.cross(rot[:, i], self.target_rot[:, i]) for i in range(3))
            error = np.r_[target - ik.site_xpos[self.site], .2 * orientation_error]
            if np.linalg.norm(error) < 2e-5:
                break
            mj.mj_jacSite(m, ik, jp, jr, self.site)
            jac = np.vstack([jp[:, self.va], .2 * jr[:, self.va]])
            step = jac.T @ np.linalg.solve(jac @ jac.T + 2e-5 * np.eye(6), error)
            step += .06 * (np.eye(7) - np.linalg.pinv(jac) @ jac) @ (self.preferred - ik.qpos[self.qa])
            ik.qpos[self.qa] = np.clip(ik.qpos[self.qa] + np.clip(step, -.08, .08),
                                       self.lower + .03, self.upper - .03)
        mj.mj_forward(m, ik)
        return ik.qpos.copy(), float(np.linalg.norm(target - ik.site_xpos[self.site]))

    def measure_grip(self, plan):
        """After the lift: the tool sits in the hand slightly differently every time, so shift the
        wiping targets by the measured offset between the IK site and the real pad centre."""
        surface = self.model.site('sponge_surface').id
        offset = self.data.site_xpos[self.site] - self.data.site_xpos[surface]
        # Horizontal offset only: hanging, the tool sags in the hand; pressed on the table it is
        # pushed back up, so a fixed extra press (PRESS) replaces a vertical correction.
        plan.offset = np.r_[offset[:2], -PRESS]
        return offset

    def reach_from(self, via, start):
        """IK error at start when walked there from via (see reset_on_table)."""
        self.ik.qpos[:] = self.qref
        for u in np.r_[np.zeros(3), np.linspace(.1, 1, 10), np.ones(3)]:
            q, err = self.solve(via * (1 - u) + start * u)
        return err

    def reset_on_table(self, start, tool_xy, yaw, via):
        """Arm at the start pose with an open hand; tool standing on the table; 0.5 s settling.

        IK from qref straight to a start close to the chest can stall in a local minimum, so it is
        solved at an easy point (via) first and then walked to the start in small steps."""
        mj, m, d = self.mj, self.model, self.data
        mj.mj_resetData(m, d)
        self.target_rot = yaw_matrix(yaw)
        err = self.reach_from(via, start)
        q = self.ik.qpos.copy()
        if err > .004:
            raise RuntimeError(f'start pose unreachable: {err:.4f} m')
        d.qpos[:] = q
        d.qpos[self.right_hand_qadr] = 0
        quat = np.zeros(4)
        mj.mju_mat2Quat(quat, yaw_matrix(yaw).flatten())
        d.qpos[self.tool_qa:self.tool_qa + 7] = np.r_[tool_xy, W.TABLE_Z + REST_Z, quat]
        d.ctrl[:] = q[self.control_qadr]
        d.ctrl[self.hand_act] = 0
        for _ in range(250):
            mj.mj_step(m, d)
        mj.mj_forward(m, d)
        d.time = 0

    def begin_episode(self, rng, side, region_offset=None):
        """Place regions and tool, reset, and return (region centre, PickupPlan).
        region_offset overrides the drawn region offset (generalization tests)."""
        offset = self.place_regions(rng)
        if region_offset is not None:
            offset = np.asarray(region_offset, float)
            for s, site in self.region_site.items():
                self.model.site_pos[site][:2] = W.REGIONS[s] + offset
                self.grids[s] = self.grid + self.model.site_pos[site][:2]
        center = W.REGIONS[side] + offset
        tool_xy = TOOL_XY + rng.uniform(-TOOL_XY_RANGE, TOOL_XY_RANGE)
        yaw = np.radians(rng.uniform(-TOOL_YAW_DEG, TOOL_YAW_DEG))
        rot = yaw_matrix(yaw)
        wipe = W.plan_episode(rng, center, .12)  # pad turned back to zero yaw: 12 cm along x
        grasp = np.r_[tool_xy, W.TABLE_Z + GRASP_Z]
        approach = grasp + rot @ APPROACH
        hover = approach + [0, 0, HOVER]
        lifted = grasp + [0, 0, LIFT]
        # Start near the tool, high enough that the open hand never sweeps over the handle. Some
        # corners of this box are outside the wrist's range with an upright tool; redraw those
        # (deterministic for a given seed).
        self.target_rot = rot
        for _ in range(20):
            start = hover + rng.uniform(-START_RANGE * [1, 1, 0], START_RANGE)
            if self.reach_from(lifted, start) <= .004:
                break
        pause = rng.uniform(0, .3)
        seg = [(start, 0.), (hover, .8 + pause), (approach, .8), (grasp, .8), (grasp, CLOSE_S + .3), (lifted, .8)]
        points, times = [p for p, _ in seg], list(np.cumsum([d for _, d in seg]))
        close_start, lifted_at = times[3], times[5]
        # Wiping: from the lifted pose straight into the usual strokes (skip the wipe start point).
        wipe_t0 = lifted_at + .6
        for p, tt in zip(wipe.points[1:], wipe.times[1:]):
            points.append(p)
            times.append(wipe_t0 + tt)
        closure = lambda t: (t - close_start) / CLOSE_S
        self.reset_on_table(start, tool_xy, yaw, via=lifted)
        self.tool_yaw = float(np.degrees(yaw))
        return center, PickupPlan(points, times, closure, grasped_at=wipe_t0, wipe_from=wipe_t0,
                                  yaw=yaw, turn_from=times[4])

    def run_episode(self, rng, side, *, record=True, viewer=None):
        if self.randomizer is not None:
            self.randomizer.apply()
        center, plan = self.begin_episode(rng, side)
        covered = {s: np.zeros(len(c), bool) for s, c in self.grids.items()}
        frames, max_ik, max_drift, max_rot, weak = [], 0., 0., 0., 0
        rel0, lifted_ok, grip_offset = None, False, None
        n_frames = int(np.ceil(plan.duration * W.FPS)) + W.FPS // 2
        grasp_frame = int(np.ceil(plan.grasped_at * W.FPS))
        for k in range(n_frames):
            t = min(k / W.FPS, plan.duration)
            self.target_rot = plan.rot(t)
            target_q, err = self.solve(plan(t))
            max_ik = max(max_ik, err)
            action = np.r_[target_q[self.qa], plan.hand(t, self.closed_hand)].astype(np.float32)
            if record:
                frames.append({'observation.images.head': self.image(),
                               'observation.state': self.state(), 'action': action, **self.extra_obs()})
            self.data.ctrl[self.arm_act] = action[:7]
            self.data.ctrl[self.hand_act] = action[7:]
            for i in range(W.SUBSTEPS):
                self.mj.mj_step(self.model, self.data)
                if viewer is not None and i % 5 == 0:
                    viewer.sync()
            if not np.isfinite(self.data.qpos).all() or any(w.number for w in self.data.warning):
                return frames, {'side': side, 'success': False, 'failure': 'physics_instability'}
            self.on_frame(k)
            if k == grasp_frame:
                rel0 = self.relative_tool()
                grip_offset = self.measure_grip(plan)
                lifted_ok = (self.data.xpos[self.tool][2] > W.TABLE_Z + REST_Z + .5 * LIFT
                             and len(self.hand_tool_contacts()) >= 2)
            if rel0 is not None:
                self.mark_coverage(covered)
                pos, rot = self.relative_tool()
                max_drift = max(max_drift, float(np.linalg.norm(pos - rel0[0])))
                max_rot = max(max_rot, float(np.degrees(np.arccos(np.clip((np.trace(rot @ rel0[1].T) - 1) / 2, -1, 1)))))
                weak += len(self.hand_tool_contacts()) < 2
            if viewer is not None:
                import time
                time.sleep(1 / W.FPS)
        other = 'left' if side == 'right' else 'right'
        info = {
            'side': side, 'task': W.TASKS[side], 'frames': n_frames, 'region_center': center.round(4).tolist(),
            'tool_yaw_deg': round(self.tool_yaw, 1), 'picked_up': bool(lifted_ok),
            'target_coverage': round(float(covered[side].mean()), 4),
            'other_coverage': round(float(covered[other].mean()), 4),
            'max_ik_error_m': round(max_ik, 5),
            'max_grasp_drift_m': round(max_drift, 4), 'max_grasp_rotation_deg': round(max_rot, 2),
            'frames_with_weak_grasp': weak,
            'grip_offset_m': np.round(grip_offset, 4).tolist() if rel0 is not None else None,
        }
        s = W.SUCCESS
        failures = [name for name, bad in (
            ('pickup_failed', not lifted_ok),
            ('low_target_coverage', info['target_coverage'] < s['target_coverage']),
            ('wiped_other_region', info['other_coverage'] > s['other_coverage']),
            ('tool_slipped', max_drift > s['max_grasp_drift_m']),
            ('tool_lost', len(self.hand_tool_contacts()) < 2)) if bad]
        info['success'] = not failures
        info['failure'] = ','.join(failures)
        return frames, info


def main():
    # --randomize light,color,camera,physics (domain randomization, redrawn every episode); the
    # remaining arguments go to wipe_teacher.main.
    if '--randomize' in sys.argv:
        i = sys.argv.index('--randomize')
        kinds = tuple(k for k in sys.argv[i + 1].split(',') if k)
        del sys.argv[i:i + 2]
        bad = set(kinds) - {'light', 'color', 'camera', 'physics'}
        if bad:
            raise SystemExit(f'--randomize: unknown or unsupported kinds {sorted(bad)}')
        seed = int(sys.argv[sys.argv.index('--seed') + 1]) if '--seed' in sys.argv else 0
        PickupTeacher.RANDOMIZE, PickupTeacher.RANDOMIZE_SEED = kinds, seed + 104729
    W.WipeTeacher = PickupTeacher
    W.main()
    out = Path(sys.argv[sys.argv.index('--output') + 1]) if '--output' in sys.argv else W.HERE / 'output'
    summary = out / 'summary.json'
    if summary.exists():
        data = json.loads(summary.read_text())
        data['scope'] = ('Scripted simulated teacher, fixed pelvis, tool picked up from the table by finger '
                         'contact, then wiping. Not real-robot data.')
        data['randomize'] = list(PickupTeacher.RANDOMIZE)
        summary.write_text(json.dumps(data, indent=2) + '\n')


if __name__ == '__main__':
    main()
