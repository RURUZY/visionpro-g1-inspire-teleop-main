#!/usr/bin/env python3
"""With vs without tactile feedback on the same randomized episodes (topic 1, steps 4-5).

Conditions (identical seeds, so grip limit, tool mass, friction, press and disturbance match):
  open_loop     randomized (often low) grip limit, no feedback
  feedback      same, but when the slip detector fires the grip limit is raised to the nominal maximum
  max_grip      grip limit always at the nominal maximum (reference: stability vs force)

Metrics per condition: success, tool lost, labeled slip frames, mean and peak fingertip force,
detector firings and latency from the first labeled slip. Simulation only.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import slip_teacher as T  # noqa: E402
from slip_detector import Detector, features  # noqa: E402
from tactile import TACTILE_BODIES  # noqa: E402

TIPS = [i for i, b in enumerate(TACTILE_BODIES) if b.endswith(('_3', '_4'))]
W = T.W


class FeedbackTeacher(T.SlipTeacher):
    mode = 'open_loop'
    detector = None

    def randomize(self, rng):
        params = super().randomize(rng)
        if self.mode == 'max_grip':
            self.model.actuator_forcerange[self.hand_act] = self.nominal['hand_forcerange']
        self.history, self.fired_at, self.firings, self.tip_force = [], None, 0, []
        return params

    def image(self):  # images are not needed here; skip rendering for speed
        return np.zeros((2, 2, 3), np.uint8)

    def on_frame(self, k):
        tactile = self.tactile()
        force = self.data.actuator_force[self.hand_act].astype(np.float32)
        self.tip_force.append(float(tactile[TIPS].max()))
        self.history.append((tactile, force))
        if self.mode != 'feedback' or len(self.history) < 4 or k < W.FPS // 2:
            return
        recent = self.history[-4:]
        x = features(np.array([h[0] for h in recent]), np.array([h[1] for h in recent]))[-1:]
        fire = self.detector.prob(x)[0] > self.detector.threshold
        if fire:
            self.firings += 1
            if self.fired_at is None:
                self.fired_at = k
            self.model.actuator_forcerange[self.hand_act] = self.nominal['hand_forcerange']


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(c - h, 3), round(c + h, 3)]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--detector', type=Path, required=True)
    p.add_argument('--episodes', type=int, default=30)
    p.add_argument('--seed', type=int, default=500, help='seeds not used for detector training')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    os.environ.setdefault('MUJOCO_GL', 'egl')
    T.configure()
    W.WipeTeacher = FeedbackTeacher
    args.output.mkdir(parents=True, exist_ok=True)
    teacher = FeedbackTeacher(W.build_scene(args.output / 'scene'))
    teacher.detector = Detector.from_json(json.loads(args.detector.read_text()))
    rows = []
    log = (args.output / 'episodes.jsonl').open('w')
    for i in range(args.episodes):
        for mode in ('open_loop', 'feedback', 'max_grip'):
            teacher.mode = mode
            frames, info = teacher.run_episode(np.random.default_rng(args.seed + i), 'left', record=True)
            if info['failure'] == 'physics_instability':
                row = {'episode': i, 'mode': mode, 'unstable': True}
            else:
                first_slip = None if info.get('first_slip_s') is None else int(info['first_slip_s'] * W.FPS)
                latency = None
                if teacher.fired_at is not None and first_slip is not None and teacher.fired_at >= first_slip - 2:
                    latency = round((teacher.fired_at - first_slip) / W.FPS, 3)
                row = {'episode': i, 'mode': mode, 'unstable': False, 'success': info['success'],
                       'tool_lost': 'tool_lost' in info['failure'], 'slip_frames': info.get('slip_frames', 0),
                       'mean_tip_force_N': round(float(np.mean(teacher.tip_force)), 3),
                       'peak_tip_force_N': round(float(np.max(teacher.tip_force)), 3),
                       'firings': teacher.firings, 'first_fire_s': None if teacher.fired_at is None
                       else round(teacher.fired_at / W.FPS, 2), 'detection_latency_s': latency,
                       'conditions': info['conditions']}
            rows.append(row)
            log.write(json.dumps(row) + '\n')
            log.flush()
        done = [r for r in rows if r['episode'] == i]
        print(f"[{i}] " + '  '.join(f"{r['mode']}: " + ('unstable' if r['unstable'] else
              f"succ={int(r['success'])} lost={int(r['tool_lost'])} slipF={r['slip_frames']} fire={r['firings']}")
              for r in done), flush=True)
    # Compare only episodes stable in all three conditions.
    stable = {i for i in range(args.episodes)
              if all(not r['unstable'] for r in rows if r['episode'] == i)}
    summary = {'episodes_compared': len(stable), 'conditions': {}}
    for mode in ('open_loop', 'feedback', 'max_grip'):
        rs = [r for r in rows if r['mode'] == mode and r['episode'] in stable]
        k = sum(r['success'] for r in rs)
        lat = [r['detection_latency_s'] for r in rs if r['detection_latency_s'] is not None]
        summary['conditions'][mode] = {
            'success': k, 'success_rate': round(k / max(1, len(rs)), 3),
            'success_wilson95': wilson(k, len(rs)) if rs else None,
            'tool_lost': sum(r['tool_lost'] for r in rs),
            'episodes_with_slip': sum(r['slip_frames'] > 0 for r in rs),
            'slip_frames_total': sum(r['slip_frames'] for r in rs),
            'mean_tip_force_N': round(float(np.mean([r['mean_tip_force_N'] for r in rs])), 3),
            'peak_tip_force_N_mean': round(float(np.mean([r['peak_tip_force_N'] for r in rs])), 3),
            'episodes_with_firing': sum(r['firings'] > 0 for r in rs),
            'detection_latency_s_median': None if not lat else round(float(np.median(lat)), 3),
        }
    summary['scope'] = ('Simulation with a logistic-regression slip detector on simulated contact forces; '
                        'not a hardware result. Grip boost = raise finger torque limit to nominal.')
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
