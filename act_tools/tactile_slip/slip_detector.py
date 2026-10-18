#!/usr/bin/env python3
"""Causal slip detector from tactile + finger-force signals (topic 1, step 3).

Inputs per frame are only signals a real hand can provide: 17-pad tactile forces and 6 finger
actuator forces, plus their changes over the last 1 and 3 frames. The simulation-only tool pose
is used for labels, never as input. Model: logistic regression (numpy), trained on whole
episodes of one dataset and tested on another.

Event metrics: a slip event is a run of labeled slip frames; it counts as detected if the
detector fires between 2 frames before onset and 10 frames (0.5 s) after; latency is measured
from onset. False alarms are firings in frames more than 10 frames away from any slip.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

FPS = 20
EARLY, LATE = 2, 10


def load(root: Path):
    frames = pd.concat([pd.read_parquet(p) for p in sorted((root / 'data').rglob('*.parquet'))])
    episodes = []
    for _, ep in frames.groupby('episode_index'):
        ep = ep.sort_values('frame_index')
        episodes.append({
            'tactile': np.stack(ep['observation.tactile'].to_numpy()).astype(np.float32),
            'force': np.stack(ep['observation.hand_force'].to_numpy()).astype(np.float32),
            'slip': np.stack(ep['label.slip'].to_numpy()).reshape(len(ep), -1)[:, 0] > .5,
        })
    return episodes


def features(tactile, force):
    """Causal features for every frame of one episode."""
    x = np.concatenate([tactile, force], axis=1)
    d1 = np.diff(x, axis=0, prepend=x[:1])
    d3 = x - np.concatenate([np.repeat(x[:1], 3, 0), x[:-3]])
    total = tactile.sum(1, keepdims=True)
    dtotal = np.diff(total, axis=0, prepend=total[:1])
    return np.concatenate([x, np.abs(d1), np.abs(d3), total, np.abs(dtotal)], axis=1)


class Detector:
    def fit(self, X, y, epochs=3000, lr=.05, l2=1e-3):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Z = (X - self.mu) / self.sd
        w, b = np.zeros(Z.shape[1]), 0.
        pos = max(y.mean(), 1e-3)
        sw = np.where(y, .5 / pos, .5 / (1 - pos))  # balance the rare slip frames
        for _ in range(epochs):
            p = 1 / (1 + np.exp(-(Z @ w + b)))
            g = sw * (p - y)
            w -= lr * (Z.T @ g / len(y) + l2 * w)
            b -= lr * g.mean()
        self.w, self.b = w, b
        return self

    def prob(self, X):
        return 1 / (1 + np.exp(-(((X - self.mu) / self.sd) @ self.w + self.b)))

    def to_json(self, threshold):
        return {'mu': self.mu.tolist(), 'sd': self.sd.tolist(), 'w': self.w.tolist(), 'b': float(self.b),
                'threshold': threshold}

    @classmethod
    def from_json(cls, d):
        det = cls()
        det.mu, det.sd, det.w, det.b = (np.array(d['mu']), np.array(d['sd']), np.array(d['w']), d['b'])
        det.threshold = d['threshold']
        return det


def events(slip):
    starts = np.flatnonzero(slip & ~np.r_[False, slip[:-1]])
    ends = np.flatnonzero(slip & ~np.r_[slip[1:], False])
    return list(zip(starts, ends))


def evaluate(episodes, fire_fn):
    detected, latencies, n_events, false_alarm_frames, quiet_frames = 0, [], 0, 0, 0
    for ep in episodes:
        fire = fire_fn(ep)
        near = np.zeros(len(fire), bool)
        for s, e in events(ep['slip']):
            n_events += 1
            near[max(0, s - LATE):e + LATE + 1] = True
            window = np.flatnonzero(fire[max(0, s - EARLY):s + LATE + 1])
            if len(window):
                detected += 1
                latencies.append((window[0] + max(0, s - EARLY) - s) / FPS)
        quiet_frames += int((~near).sum())
        # count firing onsets (not every frame) away from slips as false alarms
        onsets = fire & ~np.r_[False, fire[:-1]]
        false_alarm_frames += int((onsets & ~near).sum())
    return {'slip_events': n_events, 'detected': detected,
            'recall': round(detected / max(1, n_events), 3),
            'latency_s_median': None if not latencies else round(float(np.median(latencies)), 3),
            'false_alarms_per_min': round(false_alarm_frames / max(1, quiet_frames / FPS / 60), 3)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--train', type=Path, required=True, help='LeRobot dataset root from slip_teacher.py')
    p.add_argument('--test', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True, help='detector JSON path')
    p.add_argument('--max-false-alarms', type=float, default=1.0, help='per minute, chooses the threshold on train')
    args = p.parse_args()
    train, test = load(args.train), load(args.test)
    X = np.concatenate([features(e['tactile'], e['force']) for e in train])
    y = np.concatenate([e['slip'] for e in train]).astype(float)
    det = Detector().fit(X, y)

    def learned(threshold):
        return lambda ep: det.prob(features(ep['tactile'], ep['force'])) > threshold

    # Threshold: highest recall on the TRAIN set within the false-alarm budget.
    best = None
    for th in np.linspace(.05, .99, 95):
        m = evaluate(train, learned(th))
        if m['false_alarms_per_min'] <= args.max_false_alarms and (best is None or m['recall'] > best[1]['recall']):
            best = (th, m)
    threshold = float(best[0]) if best else .5

    # Baseline: a single threshold on the change of total tactile force.
    def baseline(th):
        return lambda ep: np.abs(np.diff(ep['tactile'].sum(1), prepend=ep['tactile'][0].sum())) > th
    base_best = None
    for th in np.linspace(.1, 10, 100):
        m = evaluate(train, baseline(th))
        if m['false_alarms_per_min'] <= args.max_false_alarms and (base_best is None or m['recall'] > base_best[1]['recall']):
            base_best = (th, m)
    base_th = float(base_best[0]) if base_best else 5.

    report = {
        'train_episodes': len(train), 'test_episodes': len(test),
        'train_slip_frame_fraction': round(float(y.mean()), 4),
        'learned': {'threshold': round(threshold, 3), 'train': best[1] if best else None,
                    'test': evaluate(test, learned(threshold))},
        'baseline_delta_total_tactile': {'threshold_N': round(base_th, 2),
                                         'train': base_best[1] if base_best else None,
                                         'test': evaluate(test, baseline(base_th))},
        'inputs': '17 tactile pads + 6 finger actuator forces and their 1/3-frame changes (no privileged state)',
        'scope': 'Simulated contact forces; thresholds must be re-fit on real tactile data.',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(det.to_json(threshold)) + '\n')
    report_path = args.output.with_name(args.output.stem + '_report.json')
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
