"""Per-frame teleop mapping recorder (JSONL), for analysing how operator motion maps to the robot.

One line per published frame:
  t                 monotonic time (s)
  phase             stand_hold / calib / paused / live
  avp_head/left/right   operator poses from the AVP stream: position xyz (m) + quaternion wxyz
  target_raw        VR 3-point positions (9) after calibration mapping, before smoothing
  target_sent       VR 3-point positions (9) and orientations (12) actually sent to SONIC
  body_q            measured robot joint positions (29) from deploy feedback, if available
Robot hand positions are computed offline from body_q (see tools/analyze_mapping_log.py), so
recording adds no FK cost to the control loop.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


def _pose(matrix) -> list[float] | None:
    if matrix is None:
        return None
    m = np.asarray(matrix, dtype=np.float64).reshape(-1, 4, 4)[0]
    r = m[:3, :3]
    # Rotation matrix -> quaternion wxyz (Shepperd's method, numerically safe).
    tr = np.trace(r)
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        q = [0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(r)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(1.0 + r[i, i] - r[j, j] - r[k, k])
        q = [0.0] * 4
        q[0] = (r[k, j] - r[j, k]) / s
        q[i + 1] = 0.25 * s
        q[j + 1] = (r[j, i] + r[i, j]) / s
        q[k + 1] = (r[k, i] + r[i, k]) / s
    return [round(float(x), 5) for x in (*m[:3, 3], *q)]


def _vec(x, n=None) -> list[float] | None:
    if x is None:
        return None
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    if n is not None:
        a = a[:n]
    return [round(float(v), 5) for v in a]


class MappingLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = self.path.open("a", encoding="utf-8")
        self._n = 0
        print(f"[mapping_log] recording to {self.path}")

    def write(self, *, phase: str, tracking, target_raw, target_sent_pos, target_sent_orn, body_q) -> None:
        tracking = tracking or {}
        row = {
            "t": round(time.monotonic(), 4),
            "phase": phase,
            "avp_head": _pose(tracking.get("head")),
            "avp_left": _pose(tracking.get("left_wrist")),
            "avp_right": _pose(tracking.get("right_wrist")),
            "target_raw": _vec(target_raw, 9),
            "target_sent": _vec(target_sent_pos, 9),
            "target_sent_orn": _vec(target_sent_orn, 12),
            "body_q": _vec(body_q, 29),
        }
        self._f.write(json.dumps(row) + "\n")
        self._n += 1
        if self._n % 50 == 0:
            self._f.flush()

    def close(self) -> None:
        self._f.close()
