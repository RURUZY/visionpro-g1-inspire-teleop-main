#!/usr/bin/env python3
"""Scripted wiping teacher for G1 + Inspire in MuJoCo, recorded as a LeRobot v3 dataset.

Fixed pelvis with gravity compensation, right hand pre-grasping a free-body sponge
(held by finger contact, no weld). Each episode picks "left" or "right", wipes that
region in parallel strokes and lifts away. Only episodes that pass the success check
go into the dataset; every episode is logged to episodes.jsonl.

This is simulated teacher data for an ACT pipeline check, not a real-robot interface.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'sponge_demo'))
import sponge_demo  # noqa: E402  (scene + tool model are reused from the preview)

TABLE_Z = .78               # Reachable band for a horizontal pad with the upright, fixed torso.
WIPE_Z = TABLE_Z - .002     # Soft-contact approximation, not a force controller.
REGIONS = {'left': np.array([.34, -.09]), 'right': np.array([.34, -.27])}
REGION_HALF = np.array([.05, .05])
TASKS = {'left': 'wipe the left area', 'right': 'wipe the right area'}
FPS = 20
SUBSTEPS = 25               # 25 x 2 ms physics steps per 50 ms control frame.
IMAGE_SIZE = (240, 320)     # height, width
HEAD_FOVY = 58              # Vertical FOV close to the G1 head RealSense D435; real camera still to calibrate.
WAIST_YAW = -.25            # Turn the torso/head ~14 deg toward the right-hand workspace.
LEFT_ARM_REST = {'shoulder_pitch': .3, 'shoulder_roll': .25, 'elbow': 1.2}  # out of the head view
CELL = .005                 # Coverage grid resolution (m).
EXTRA_FEATURES = {}         # variants add per-frame fields here (see WipeTeacher.extra_obs)
SUCCESS = dict(target_coverage=.90, other_coverage=.05, max_grasp_drift_m=.02)
HAND_ACTIVE = sponge_demo.HAND_ACTIVE
STATE_NAMES = ([f'right_{j}_joint' for j in sponge_demo.ARM]
               + [f'right_hand_{j}_joint' for j in HAND_ACTIVE])


def build_scene(output: Path) -> Path:
    """Preview scene with two identical-looking target regions (language picks one)."""
    sponge_demo.TABLE_Z = TABLE_Z
    path = sponge_demo.generate(output)
    tree = ET.parse(path)
    root = tree.getroot()
    root.set('model', 'g1_sponge_wipe_scripted_teacher')
    world = root.find('worldbody')
    world.remove(world.find("site[@name='wipe_target']"))
    root.find(".//camera[@name='head_camera']").set('fovy', str(HEAD_FOVY))
    for side, center in REGIONS.items():
        ET.SubElement(world, 'site', {
            'name': f'region_{side}', 'type': 'box',
            'pos': f'{center[0]} {center[1]} {TABLE_Z + .0005}',
            'size': f'{REGION_HALF[0]} {REGION_HALF[1]} .0003', 'rgba': '.12 .70 .68 .30'})
    tree.write(path, encoding='utf-8')
    return path


def smooth(u):
    u = np.clip(u, 0, 1)
    return u * u * (3 - 2 * u)


class Path3D:
    """Piecewise smooth-step interpolation between waypoints at given times."""

    def __init__(self, points, times):
        self.points, self.times = np.asarray(points, float), np.asarray(times, float)
        self.duration = float(self.times[-1])

    def __call__(self, t):
        i = int(np.clip(np.searchsorted(self.times, t, side='right') - 1, 0, len(self.times) - 2))
        u = smooth((t - self.times[i]) / (self.times[i + 1] - self.times[i]))
        return self.points[i] * (1 - u) + self.points[i + 1] * u


def plan_episode(rng, center, pad_width_x):
    """Approach -> parallel strokes covering the region -> lift. Randomized within small ranges."""
    hx, hy = REGION_HALF
    start = np.array([.30, -.12, TABLE_Z + .12]) + rng.uniform([-.03, -.05, -.02], [.03, .05, .03])
    speed = rng.uniform(.05, .07)  # stroke speed m/s
    # Stroke along y; step along x so pad passes overlap and reach both x edges.
    n = max(2, int(np.ceil((2 * hx) / (pad_width_x * .8))) + 1)
    xs = np.linspace(center[0] - hx + pad_width_x * .35, center[0] + hx - pad_width_x * .35, n)
    y0, y1 = center[1] - hy - .01, center[1] + hy + .01
    if rng.random() < .5:
        y0, y1 = y1, y0
    pts = [start, [xs[0], y0, TABLE_Z + .06], [xs[0], y0, WIPE_Z]]
    for k, x in enumerate(xs):
        a, b = (y0, y1) if k % 2 == 0 else (y1, y0)
        if k:
            pts.append([x, a, WIPE_Z])
        pts.append([x, b, WIPE_Z])
    pts += [[pts[-1][0], pts[-1][1], TABLE_Z + .08], start + [0, 0, .02]]
    pts = np.array(pts, float)
    times = [0.]
    for i in range(1, len(pts)):
        dist = np.linalg.norm(pts[i] - pts[i - 1])
        on_table = pts[i][2] <= WIPE_Z + 1e-6 and pts[i - 1][2] <= WIPE_Z + 1e-6
        times.append(times[-1] + max(.6, dist / (speed if on_table else .12)))
    times[1] += rng.uniform(0, .4)  # small hesitation before descending
    return Path3D(pts, times)


class WipeTeacher:
    def __init__(self, scene_path):
        import mujoco
        self.mj = mujoco
        m = self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data, self.ik = mujoco.MjData(m), mujoco.MjData(m)
        arm = np.array([m.joint('right_' + n + '_joint').id for n in sponge_demo.ARM])
        self.qa, self.va = m.jnt_qposadr[arm], m.jnt_dofadr[arm]
        self.lower, self.upper = m.jnt_range[arm].T
        self.site = m.site('sponge_reference').id
        self.tool = m.body('sponge_tool').id
        self.palm = m.body('right_base_link').id
        self.tool_qa = int(m.joint('sponge_free').qposadr[0])
        self.tool_va = int(m.joint('sponge_free').dofadr[0])
        self.pad = m.geom('sponge_pad').id
        self.table = m.geom('table_top').id
        self.hand_bodies = {b for b in range(m.nbody) if self._under(b, self.palm)}
        self.control_qadr = m.jnt_qposadr[m.actuator_trnid[:, 0]]
        names = [m.joint(j).name for j in m.actuator_trnid[:, 0]]
        self.arm_act = [names.index(f'right_{j}_joint') for j in sponge_demo.ARM]
        self.hand_act = [names.index(f'right_hand_{j}_joint') for j in HAND_ACTIVE]
        self.hand_qadr = np.array([m.joint(f'right_hand_{j}_joint').qposadr[0] for j in HAND_ACTIVE])
        self.right_hand_qadr = [int(m.jnt_qposadr[j]) for j in range(m.njnt)
                                if m.joint(j).name.startswith('right_hand_')]
        self.preferred = np.array([-.6, -.35, 0., 1.1, 0., -.5, 0.])
        q = m.qpos0.copy()
        q[self.qa] = self.preferred
        q[m.joint('waist_yaw_joint').qposadr[0]] = WAIST_YAW
        for name, angle in LEFT_ARM_REST.items():
            q[m.joint(f'left_{name}_joint').qposadr[0]] = angle
        for name, angle in [('thumb_1', 1.1641), ('thumb_2', .5864), ('thumb_3', .5864 * .8024),
                            ('thumb_4', .5864 * .8024 * .9487), ('index_1', 1.4381), ('middle_1', 1.4381),
                            ('ring_1', 1.4381), ('little_1', 1.4381), ('index_2', 1.4381 * 1.0843),
                            ('middle_2', 1.4381 * 1.0843), ('ring_2', 1.4381 * 1.0843),
                            ('little_2', 1.4381 * 1.0843)]:
            q[m.joint('right_hand_' + name + '_joint').qposadr[0]] = angle
        self.qref = q
        self.renderer = None
        self.region_site = {s: m.site(f'region_{s}').id for s in REGIONS}
        xs = np.arange(-REGION_HALF[0] + CELL / 2, REGION_HALF[0], CELL)
        ys = np.arange(-REGION_HALF[1] + CELL / 2, REGION_HALF[1], CELL)
        self.grid = np.stack(np.meshgrid(xs, ys), -1).reshape(-1, 2)
        self.grids = {}

    def _under(self, body, ancestor):
        while body and body != ancestor:
            body = self.model.body_parentid[body]
        return body == ancestor

    def solve(self, target):
        mj, m, ik = self.mj, self.model, self.ik
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for _ in range(80):
            mj.mj_forward(m, ik)
            rot = ik.site_xmat[self.site].reshape(3, 3)
            orientation_error = .5 * sum(np.cross(rot[:, i], np.eye(3)[:, i]) for i in range(3))
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

    def hand_tool_contacts(self):
        m, bodies = self.model, set()
        for c in self.data.contact:
            a, b = m.geom_bodyid[[c.geom1, c.geom2]]
            if self.tool in (a, b):
                other = b if a == self.tool else a
                if other in self.hand_bodies:
                    bodies.add(m.body(other).name)
        return bodies

    def relative_tool(self):
        d = self.data
        r = d.xmat[self.palm].reshape(3, 3)
        return r.T @ (d.xpos[self.tool] - d.xpos[self.palm]), r.T @ d.xmat[self.tool].reshape(3, 3)

    def pad_width_x(self):
        """Pad extent along world x when held (pad yaw depends on the grasp)."""
        rot = self.data.geom_xmat[self.pad].reshape(3, 3)
        half = self.model.geom_size[self.pad]
        return float(2 * (abs(rot[0, 0]) * half[0] + abs(rot[0, 1]) * half[1]))

    def reset(self, start):
        """Pre-grasp: hold the tool in the closing hand for 1 s, then 1 s unsupported settling."""
        mj, m, d = self.mj, self.model, self.data
        mj.mj_resetData(m, d)
        self.ik.qpos[:] = self.qref
        q, err = self.solve(start)
        if err > .004:
            raise RuntimeError(f'start pose unreachable: {err:.4f} m')
        d.qpos[:] = q
        d.qpos[self.right_hand_qadr] = 0
        d.ctrl[:] = q[self.control_qadr]
        mj.mj_forward(m, d)
        ref = m.body('tool_reference_frame').id
        held = np.r_[d.xpos[ref], d.xquat[ref]]
        for _ in range(500):
            d.qpos[self.tool_qa:self.tool_qa + 7] = held
            d.qvel[self.tool_va:self.tool_va + 6] = 0
            mj.mj_step(m, d)
        d.qpos[self.tool_qa:self.tool_qa + 7] = held
        d.qvel[self.tool_va:self.tool_va + 6] = 0
        for _ in range(500):
            mj.mj_step(m, d)
        mj.mj_forward(m, d)
        d.time = 0

    def extra_obs(self):
        """Additional per-frame fields; none in the base teacher."""
        return {}

    def on_frame(self, k):
        """Called after each control frame is simulated; variants add monitoring/feedback here."""

    def state(self):
        return np.r_[self.data.qpos[self.qa], self.data.qpos[self.hand_qadr]].astype(np.float32)

    def image(self):
        if self.renderer is None:
            self.renderer = self.mj.Renderer(self.model, height=IMAGE_SIZE[0], width=IMAGE_SIZE[1])
        self.renderer.update_scene(self.data, camera='head_camera')
        return self.renderer.render().copy()

    def mark_coverage(self, covered):
        """Mark grid cells under the pad while the pad touches the table."""
        d = self.data
        if not any({int(c.geom1), int(c.geom2)} == {self.pad, self.table} for c in d.contact):
            return
        rot = d.geom_xmat[self.pad].reshape(3, 3)
        half = self.model.geom_size[self.pad][:2]
        for side, cells in self.grids.items():
            local = (cells - d.geom_xpos[self.pad][:2]) @ rot[:2, :2]
            covered[side] |= np.all(np.abs(local) <= half, axis=1)

    def place_regions(self, rng):
        """Shift both drawn regions together by a small random offset (visible in the image)."""
        offset = rng.uniform(-.015, .015, 2)
        for side, site in self.region_site.items():
            self.model.site_pos[site][:2] = REGIONS[side] + offset
            self.grids[side] = self.grid + self.model.site_pos[site][:2]
        return offset

    def run_episode(self, rng, side, *, record=True, viewer=None):
        offset = self.place_regions(rng)
        center = REGIONS[side] + offset
        # Plan needs the pad width, which is known after the grasp settles.
        start = plan_episode(rng, center, .07)(0)
        self.reset(start)
        plan = plan_episode(rng, center, self.pad_width_x())
        plan.points[0] = start
        rel0 = self.relative_tool()
        covered = {s: np.zeros(len(c), bool) for s, c in self.grids.items()}
        frames, max_ik, max_drift, max_rot, contacts_lost = [], 0., 0., 0., 0
        hand_target = self.qref[self.control_qadr][self.hand_act]
        n_frames = int(np.ceil(plan.duration * FPS)) + FPS // 2
        for k in range(n_frames):
            t = min(k / FPS, plan.duration)
            target_q, err = self.solve(plan(t))
            max_ik = max(max_ik, err)
            action = np.r_[target_q[self.qa], hand_target].astype(np.float32)
            if record:
                frames.append({'observation.images.head': self.image(),
                               'observation.state': self.state(), 'action': action, **self.extra_obs()})
            self.data.ctrl[self.arm_act] = action[:7]
            self.data.ctrl[self.hand_act] = action[7:]
            for _ in range(SUBSTEPS):
                self.mj.mj_step(self.model, self.data)
                if viewer is not None and _ % 5 == 0:
                    viewer.sync()
            if not np.isfinite(self.data.qpos).all() or any(w.number for w in self.data.warning):
                return frames, {'side': side, 'success': False, 'failure': 'physics_instability'}
            self.mark_coverage(covered)
            self.on_frame(k)
            pos, rot = self.relative_tool()
            max_drift = max(max_drift, float(np.linalg.norm(pos - rel0[0])))
            max_rot = max(max_rot, float(np.degrees(np.arccos(np.clip((np.trace(rot @ rel0[1].T) - 1) / 2, -1, 1)))))
            contacts_lost += len(self.hand_tool_contacts()) < 2
            if viewer is not None:
                time.sleep(max(0, 1 / FPS - .0))
        other = 'left' if side == 'right' else 'right'
        info = {
            'side': side, 'task': TASKS[side], 'frames': n_frames,
            'region_center': center.round(4).tolist(),
            'target_coverage': round(float(covered[side].mean()), 4),
            'other_coverage': round(float(covered[other].mean()), 4),
            'max_ik_error_m': round(max_ik, 5),
            'max_grasp_drift_m': round(max_drift, 4), 'max_grasp_rotation_deg': round(max_rot, 2),
            'frames_with_weak_grasp': contacts_lost,
        }
        failures = []
        if info['target_coverage'] < SUCCESS['target_coverage']:
            failures.append('low_target_coverage')
        if info['other_coverage'] > SUCCESS['other_coverage']:
            failures.append('wiped_other_region')
        if max_drift > SUCCESS['max_grasp_drift_m']:
            failures.append('tool_slipped')
        if len(self.hand_tool_contacts()) < 2:
            failures.append('tool_lost')
        info['success'] = not failures
        info['failure'] = ','.join(failures)
        return frames, info


def create_dataset(root, repo_id):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    h, w = IMAGE_SIZE
    return LeRobotDataset.create(
        repo_id=repo_id, root=root, fps=FPS, robot_type='g1_inspire_sim_fixed_pelvis',
        features={
            'observation.images.head': {'dtype': 'video', 'shape': (h, w, 3),
                                        'names': ['height', 'width', 'channels']},
            'observation.state': {'dtype': 'float32', 'shape': (13,), 'names': STATE_NAMES},
            'action': {'dtype': 'float32', 'shape': (13,), 'names': [n + '.pos_target' for n in STATE_NAMES]},
            **EXTRA_FEATURES,
        },
        video_backend='pyav', vcodec='h264', encoder_threads=2,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--episodes', type=int, default=10, help='number of successful episodes to keep')
    parser.add_argument('--output', type=Path, default=HERE / 'output')
    parser.add_argument('--repo-id', default='local/g1_inspire_wipe_sim')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--side', choices=['left', 'right', 'both'], default='both')
    parser.add_argument('--max-attempts', type=int, default=0, help='default: 3 x episodes')
    parser.add_argument('--viewer', action='store_true', help='watch episodes in a window, nothing is recorded')
    parser.add_argument('--keep-all', action='store_true',
                        help='also store failed episodes (success flag is in episodes.jsonl)')
    args = parser.parse_args()

    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('HF_DATASETS_OFFLINE', '1')
    if args.viewer:
        sponge_demo.configure_viewer_display(None)
    else:
        os.environ.setdefault('MUJOCO_GL', 'egl')
    args.output.mkdir(parents=True, exist_ok=True)
    scene = build_scene(args.output / 'scene')
    teacher = WipeTeacher(scene)
    rng = np.random.default_rng(args.seed)
    sides = ['left', 'right'] if args.side == 'both' else [args.side]

    if args.viewer:
        import mujoco.viewer
        with mujoco.viewer.launch_passive(teacher.model, teacher.data) as viewer:
            viewer.cam.lookat[:] = [.25, -.12, .95]
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.5, 135, -25
            k = 0
            while viewer.is_running():
                side = sides[k % len(sides)]
                _, info = teacher.run_episode(rng, side, record=False, viewer=viewer)
                print(json.dumps(info, ensure_ascii=False), flush=True)
                k += 1
        return

    import av
    av.logging.set_level(av.logging.ERROR)  # silence per-episode libx264 statistics
    root = args.output / 'dataset'
    if root.exists():
        parser.error(f'{root} already exists; choose another --output or remove it first')
    dataset = create_dataset(root, args.repo_id)
    log = (args.output / 'episodes.jsonl').open('a')
    kept, attempt = 0, 0
    max_attempts = args.max_attempts or 3 * args.episodes
    t0 = time.time()
    while kept < args.episodes and attempt < max_attempts:
        side = sides[attempt % len(sides)]
        frames, info = teacher.run_episode(rng, side)
        keep = bool(frames) and (info['success'] or (args.keep_all and info['failure'] != 'physics_instability'))
        info.update(attempt=attempt, seed=args.seed, kept_index=kept if keep else None)
        if keep:
            for frame in frames:
                dataset.add_frame({**frame, 'task': TASKS[side]})
            dataset.save_episode()
            kept += 1
        log.write(json.dumps(info) + '\n')
        log.flush()
        print(f"[{attempt}] {side:5s} success={info['success']} target={info.get('target_coverage', 0):.2f} "
              f"other={info.get('other_coverage', 0):.2f} drift={info.get('max_grasp_drift_m', 0)*100:.1f}cm "
              f"{info['failure']}  ({time.time()-t0:.0f}s)", flush=True)
        attempt += 1
    dataset.finalize()
    summary = {'dataset_root': str(root), 'repo_id': args.repo_id, 'kept_episodes': kept,
               'attempts': attempt, 'fps': FPS, 'image': list(IMAGE_SIZE),
               'state_action_names': STATE_NAMES, 'success_thresholds': SUCCESS,
               'scope': 'Scripted simulated teacher, fixed pelvis, pre-grasped tool. Not real-robot data.'}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    if kept < args.episodes:
        raise SystemExit(f'only {kept}/{args.episodes} successful episodes within {attempt} attempts')


if __name__ == '__main__':
    main()
