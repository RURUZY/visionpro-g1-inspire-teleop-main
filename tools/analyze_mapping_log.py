#!/usr/bin/env python3
"""Analyse a teleop mapping log written with the bridge's --mapping-log PATH.

For the live-teleop frames it reports, per wrist and axis (robot frame):
  - tracking error between the targets sent to SONIC and the robot's actual wrist (FK of body_q)
  - lag of the actual wrist behind the sent target, and of the sent target behind the raw
    (pre-smoothing) target, from cross-correlation of velocities
  - motion gain: how far the sent target moved per metre of operator wrist motion
    (operator wrist relative to the head, path length)
and saves a plot of raw target, sent target and actual wrist for both arms.

  python tools/analyze_mapping_log.py recordings/mapping_XXXX.jsonl [--phase live]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
AXES = "xyz"


def load(path, phase):
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    if phase != "all":
        rows = [r for r in rows if r["phase"] == phase]
    return rows


def lag_s(a, b, dt, max_lag=1.0):
    """Delay of b behind a (s) from cross-correlation of their velocities."""
    va, vb = np.diff(a), np.diff(b)
    if va.std() < 1e-5 or vb.std() < 1e-5:
        return None
    va, vb = (va - va.mean()) / va.std(), (vb - vb.mean()) / vb.std()
    n = int(max_lag / dt)
    best, best_k = -np.inf, 0
    for k in range(0, n + 1):
        c = np.mean(va[: len(va) - k] * vb[k:]) if k < len(va) - 5 else -np.inf
        if c > best:
            best, best_k = c, k
    return best_k * dt


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("log", type=Path)
    p.add_argument("--phase", default="live", help="live (default), stand_hold, calib, paused or all")
    p.add_argument("--plot", type=Path, help="default: next to the log, .png")
    args = p.parse_args()
    rows = load(args.log, args.phase)
    if len(rows) < 20:
        raise SystemExit(f"only {len(rows)} '{args.phase}' frames in {args.log}")
    t = np.array([r["t"] for r in rows]) - rows[0]["t"]
    dt = float(np.median(np.diff(t)))
    raw = np.array([r["target_raw"] for r in rows], float)
    sent = np.array([r["target_sent"] for r in rows], float)
    print(f"{args.log.name}: {len(rows)} '{args.phase}' frames, {t[-1]:.1f} s, {1 / dt:.0f} Hz")

    actual = None
    have_q = [r["body_q"] is not None for r in rows]
    if all(have_q):
        os.environ.setdefault("GR00T_ROOT", str(ROOT.parent / "GR00T-WholeBodyControl"))
        from g1_teleop.calibration.session import FkRobotReference
        fk = FkRobotReference()
        if fk.available:
            actual = np.array([fk.vr_targets_from_body_q(np.array(r["body_q"]))[0] for r in rows])
        else:
            print(f"FK unavailable ({fk._load_error}); skipping actual-wrist metrics")
    else:
        print(f"body_q missing in {have_q.count(False)}/{len(rows)} frames; skipping actual-wrist metrics")

    for side, sl in (("left", slice(0, 3)), ("right", slice(3, 6))):
        print(f"\n== {side} wrist ==")
        for i, ax in enumerate(AXES):
            j = sl.start + i
            line = f"  {ax}: sent range {np.ptp(sent[:, j]) * 100:5.1f} cm"
            lag = lag_s(raw[:, j], sent[:, j], dt)
            line += f" | smoothing lag {lag * 1000:4.0f} ms" if lag is not None else " | smoothing lag  n/a"
            if actual is not None:
                err = actual[:, j] - sent[:, j]
                line += (f" | actual-sent error mean {err.mean() * 100:+5.1f} cm, "
                         f"|p95| {np.percentile(np.abs(err), 95) * 100:4.1f} cm")
                lag = lag_s(sent[:, j], actual[:, j], dt)
                line += f", robot lag {lag * 1000:4.0f} ms" if lag is not None else ", robot lag n/a"
            print(line)
        key = f"avp_{side}"
        if all(r[key] and r["avp_head"] for r in rows):
            op = np.array([np.array(r[key][:3]) - np.array(r["avp_head"][:3]) for r in rows])
            op_path = np.linalg.norm(np.diff(op, axis=0), axis=1).sum()
            tg_path = np.linalg.norm(np.diff(sent[:, sl], axis=0), axis=1).sum()
            line = f"  motion gain: sent target path / operator wrist path = {tg_path / max(op_path, 1e-9):.2f}"
            if actual is not None:
                ac_path = np.linalg.norm(np.diff(actual[:, sl], axis=0), axis=1).sum()
                line += f", robot wrist path / operator = {ac_path / max(op_path, 1e-9):.2f}"
            print(line)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
    for c, (side, off) in enumerate((("left", 0), ("right", 3))):
        for i, ax in enumerate(AXES):
            a = axs[i, c]
            a.plot(t, raw[:, off + i], lw=.8, label="raw target")
            a.plot(t, sent[:, off + i], lw=1.2, label="sent target")
            if actual is not None:
                a.plot(t, actual[:, off + i], lw=1.2, label="robot wrist (FK)")
            a.set_ylabel(f"{side} {ax} (m)")
            a.grid(alpha=.3)
    axs[0, 0].legend(fontsize=8)
    axs[-1, 0].set_xlabel("t (s)")
    axs[-1, 1].set_xlabel("t (s)")
    out = args.plot or args.log.with_suffix(".png")
    fig.suptitle(f"{args.log.name} ({args.phase})")
    fig.tight_layout()
    fig.savefig(out, dpi=90)
    print(f"\nplot: {out}")


if __name__ == "__main__":
    main()
