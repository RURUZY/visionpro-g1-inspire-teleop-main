#!/usr/bin/env python3
"""Closed-loop evaluation in the wiping scene: the policy produces every action.

Policies:
  teacher     scripted teacher (upper bound; checks the evaluator itself)
  replay      replays the actions of a recorded dataset episode open-loop (checks recording/execution)
  checkpoint  a trained LeRobot policy (e.g. ACT) from --checkpoint DIR

Each trial uses the same scene, region placement and start pose as data generation for its
seed, then scores: target/other coverage, success, tool lost, max in-hand drift, slip frames,
peak fingertip force, inference time. Results: trials.jsonl and summary.json (Wilson 95% CI).
Simulation only; not a real-robot evaluation.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for sub in ('wipe_teacher', 'wipe_block', 'tactile_slip'):
    sys.path.insert(0, str(ROOT / sub))
import wipe_teacher as W  # noqa: E402
from tactile import TactileMixin  # noqa: E402

VARIANTS = ('handle', 'block')
SLIP_SPEED, SLIP_ANGULAR = .03, np.radians(45)


def make_teacher(variant, scene_dir):
    if variant == 'block':
        import wipe_block
        wipe_block.configure()
        base = wipe_block.BlockTeacher
    else:
        class base(TactileMixin, W.WipeTeacher):
            def __init__(self, path):
                super().__init__(path)
                self.setup_tactile()
    return base(W.build_scene(scene_dir))


class TeacherPolicy:
    def __init__(self, teacher):
        self.t = teacher

    def start(self, plan, task):
        self.plan, self.k = plan, 0

    def act(self, obs):
        q, _ = self.t.solve(self.plan(min(self.k / W.FPS, self.plan.duration)))
        self.k += 1
        hand = self.t.qref[self.t.control_qadr][self.t.hand_act]
        return np.r_[q[self.t.qa], hand].astype(np.float32)


class ReplayPolicy:
    def __init__(self, dataset_root, repo_id):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        self.ds = LeRobotDataset(repo_id, root=dataset_root, video_backend='pyav')
        self.episode = 0

    def start(self, plan, task):
        ep = self.ds.meta.episodes[self.episode % self.ds.num_episodes]
        a, b = int(ep['dataset_from_index']), int(ep['dataset_to_index'])
        self.actions = [self.ds.hf_dataset[i]['action'] for i in range(a, b)]
        self.actions = [np.asarray(x, np.float32) for x in self.actions]
        self.episode += 1
        self.k = 0

    def act(self, obs):
        a = self.actions[min(self.k, len(self.actions) - 1)]
        self.k += 1
        return a


class CheckpointPolicy:
    def __init__(self, path, device):
        import torch
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors
        from lerobot.configs.policies import PreTrainedConfig
        self.torch = torch
        cfg = PreTrainedConfig.from_pretrained(path)
        cfg.device = device
        self.policy = get_policy_class(cfg.type).from_pretrained(path, config=cfg)
        self.policy.eval()
        self.pre, self.post = make_pre_post_processors(cfg, pretrained_path=path)
        self.keys = set(cfg.input_features)

    def start(self, plan, task):
        self.task = task
        self.policy.reset()

    def act(self, obs):
        torch = self.torch
        batch = {'task': self.task}
        for key, value in obs.items():
            if key not in self.keys:
                continue
            value = torch.from_numpy(np.ascontiguousarray(value))
            if key.startswith('observation.images.'):
                value = value.permute(2, 0, 1).float() / 255.
            batch[key] = value
        with torch.inference_mode():
            action = self.policy.select_action(self.pre(batch))
            action = self.post(action)
        return action.squeeze(0).cpu().numpy().astype(np.float32)


def wilson(k, n, z=1.96):
    if n == 0:
        return [0., 0.]
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(max(0, centre - half), 3), round(min(1, centre + half), 3)]


def run_trial(t, policy, rng, side, max_seconds):
    offset = t.place_regions(rng)
    center = W.REGIONS[side] + offset
    start = W.plan_episode(rng, center, .07)(0)
    t.reset(start)
    plan = W.plan_episode(rng, center, t.pad_width_x())
    plan.points[0] = start
    policy.start(plan, W.TASKS[side])
    rel0 = t.relative_tool()
    covered = {s: np.zeros(len(c), bool) for s, c in t.grids.items()}
    prev_pose, slip_frames, max_drift, peak_tip, infer = None, 0, 0., 0., []
    tips = [i for i, b in enumerate(__import__('tactile').TACTILE_BODIES) if b.endswith(('_3', '_4'))]
    n = int(max_seconds * W.FPS)
    for k in range(n):
        obs = {'observation.images.head': t.image(), 'observation.state': t.state(), **t.extra_obs()}
        t0 = time.perf_counter()
        action = policy.act(obs)
        infer.append(time.perf_counter() - t0)
        t.data.ctrl[t.arm_act] = action[:7]
        t.data.ctrl[t.hand_act] = action[7:13]
        for _ in range(W.SUBSTEPS):
            t.mj.mj_step(t.model, t.data)
        if not np.isfinite(t.data.qpos).all():
            return {'success': False, 'failure': 'physics_instability'}
        t.mark_coverage(covered)
        pose = t.tool_in_palm()
        max_drift = max(max_drift, float(np.linalg.norm(pose[:3] - rel0[0])))
        if prev_pose is not None and k > W.FPS // 2:
            lin = np.linalg.norm(pose[:3] - prev_pose[:3]) * W.FPS
            ang = np.linalg.norm(pose[3:] - prev_pose[3:]) * W.FPS
            slip_frames += int(lin > SLIP_SPEED or ang > SLIP_ANGULAR)
        prev_pose = pose
        peak_tip = max(peak_tip, float(obs['observation.tactile'][tips].max()))
    other = 'left' if side == 'right' else 'right'
    result = {
        'side': side, 'task': W.TASKS[side],
        'target_coverage': round(float(covered[side].mean()), 4),
        'other_coverage': round(float(covered[other].mean()), 4),
        'tool_lost': len(t.hand_tool_contacts()) < 2,
        'max_grasp_drift_m': round(max_drift, 4), 'slip_frames': slip_frames,
        'peak_fingertip_force_N': round(peak_tip, 2),
        'inference_ms_mean': round(1000 * float(np.mean(infer)), 2),
        'inference_ms_p95': round(1000 * float(np.percentile(infer, 95)), 2),
    }
    s = W.SUCCESS
    fails = [name for name, bad in (
        ('low_target_coverage', result['target_coverage'] < s['target_coverage']),
        ('wiped_other_region', result['other_coverage'] > s['other_coverage']),
        ('tool_slipped', max_drift > s['max_grasp_drift_m']), ('tool_lost', result['tool_lost'])) if bad]
    result.update(success=not fails, failure=','.join(fails))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--policy', choices=['teacher', 'replay', 'checkpoint'], required=True)
    p.add_argument('--checkpoint', help='pretrained_model directory of a LeRobot policy')
    p.add_argument('--dataset', help='dataset root, for --policy replay')
    p.add_argument('--repo-id', default='local/g1_inspire_wipe_sim')
    p.add_argument('--variant', choices=VARIANTS, default='handle')
    p.add_argument('--side', choices=['left', 'right', 'both'], default='both')
    p.add_argument('--trials', type=int, default=10)
    p.add_argument('--seed', type=int, default=1000, help='use seeds not used for training data')
    p.add_argument('--max-seconds', type=float, default=14.)
    p.add_argument('--device', default='cpu')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    os.environ.setdefault('MUJOCO_GL', 'egl')
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    args.output.mkdir(parents=True, exist_ok=True)
    teacher = make_teacher(args.variant, args.output / 'scene')
    if args.policy == 'teacher':
        policy = TeacherPolicy(teacher)
    elif args.policy == 'replay':
        policy = ReplayPolicy(args.dataset, args.repo_id)
    else:
        policy = CheckpointPolicy(args.checkpoint, args.device)
    sides = ['left', 'right'] if args.side == 'both' else [args.side]
    rng = np.random.default_rng(args.seed)
    results = []
    with (args.output / 'trials.jsonl').open('w') as log:
        for i in range(args.trials):
            r = run_trial(teacher, policy, rng, sides[i % len(sides)], args.max_seconds)
            r['trial'] = i
            results.append(r)
            log.write(json.dumps(r) + '\n')
            print(f"[{i}] {r.get('side', ''):5s} success={r['success']} target={r.get('target_coverage', 0):.2f} "
                  f"other={r.get('other_coverage', 0):.2f} slip={r.get('slip_frames')} "
                  f"tip={r.get('peak_fingertip_force_N')}N {r['failure']}", flush=True)
    k = sum(r['success'] for r in results)
    summary = {'policy': args.policy, 'checkpoint': args.checkpoint, 'variant': args.variant,
               'trials': len(results), 'successes': k, 'success_rate': round(k / max(1, len(results)), 3),
               'success_rate_wilson95': wilson(k, len(results)),
               'tool_lost': sum(bool(r.get('tool_lost')) for r in results),
               'mean_target_coverage': round(float(np.mean([r.get('target_coverage', 0) for r in results])), 3),
               'scope': 'Closed-loop simulation evaluation; not a real-robot result.'}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
