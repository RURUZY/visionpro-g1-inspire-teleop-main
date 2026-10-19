#!/usr/bin/env python3
"""Watch a trained policy (or the scripted teacher) wipe in a MuJoCo window.

Same scenes, start poses and scoring as eval_policy.py, but shown live and not saved. The window
runs slower than real time because every frame renders the head camera and runs the policy.
Close the window to stop. Simulation only.

  .venv/bin/python eval/view_policy.py --checkpoint runs/act_left200_60k/ckpt_060000/pretrained_model
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_policy as E  # noqa: E402
import sponge_demo  # noqa: E402  (on the path via eval_policy -> wipe_teacher)

W = E.W


def run(t, policy, viewer, rng, side, max_seconds):
    offset = t.place_regions(rng)
    center = W.REGIONS[side] + offset
    start = W.plan_episode(rng, center, .07)(0)
    t.reset(start)
    plan = W.plan_episode(rng, center, t.pad_width_x())
    plan.points[0] = start
    policy.start(plan, W.TASKS[side])
    covered = {s: np.zeros(len(c), bool) for s, c in t.grids.items()}
    for k in range(int(max_seconds * W.FPS)):
        if not viewer.is_running():
            return None
        tick = time.perf_counter()
        obs = {'observation.images.head': t.image(), 'observation.state': t.state(), **t.extra_obs()}
        action = policy.act(obs)
        t.data.ctrl[t.arm_act] = action[:7]
        t.data.ctrl[t.hand_act] = action[7:13]
        for _ in range(W.SUBSTEPS):
            t.mj.mj_step(t.model, t.data)
        t.mark_coverage(covered)
        viewer.sync()
        if k % W.FPS == 0:
            print(f'  {k / W.FPS:4.0f} s  coverage {covered[side].mean():.0%}', flush=True)
        time.sleep(max(0., 1 / W.FPS - (time.perf_counter() - tick)))
    return covered[side].mean(), len(t.hand_tool_contacts()) >= 2


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', help='pretrained_model directory; omit to watch the scripted teacher')
    p.add_argument('--side', choices=['left', 'right'], default='left')
    p.add_argument('--seed', type=int, default=3000)
    p.add_argument('--max-seconds', type=float, default=14.)
    p.add_argument('--output', type=Path, default=Path('runs/view_scene'))
    args = p.parse_args()
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('MUJOCO_GL', 'egl')  # head-camera renderer; the window itself uses GLFW
    sponge_demo.configure_viewer_display(None)
    teacher = E.make_teacher('handle', args.output)
    policy = E.CheckpointPolicy(args.checkpoint, 'cpu') if args.checkpoint else E.TeacherPolicy(teacher)
    rng = np.random.default_rng(args.seed)
    import mujoco.viewer
    with mujoco.viewer.launch_passive(teacher.model, teacher.data) as viewer:
        viewer.cam.lookat[:] = [.25, -.12, .95]
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.5, 135, -25
        trial = 0
        while viewer.is_running():
            print(f'trial {trial} ({W.TASKS[args.side]})', flush=True)
            result = run(teacher, policy, viewer, rng, args.side, args.max_seconds)
            if result is None:
                break
            coverage, holding = result
            ok = coverage >= W.SUCCESS['target_coverage'] and holding
            print(f'  -> coverage {coverage:.0%}, {"holding" if holding else "TOOL LOST"}, '
                  f'{"SUCCESS" if ok else "fail"}', flush=True)
            trial += 1
            time.sleep(1.5)


if __name__ == '__main__':
    main()
