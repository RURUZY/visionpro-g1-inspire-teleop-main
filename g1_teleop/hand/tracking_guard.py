"""Reject unusable skeletons before they become simulated finger commands."""

import numpy as np

from g1_teleop.hand.mapping import FINGER_CHAINS, FINGER_ORDER, _get_point


def validate_hand_skeleton(hand):
    if hand is None:
        raise ValueError("hand missing")
    for finger in (*FINGER_ORDER, "thumb"):
        spec = FINGER_CHAINS[finger]
        points = [_get_point(hand, spec[key]) for key in ("knuckle", "prox", "mid", "tip")]
        if finger != "thumb" and hasattr(hand, spec["metacarpal"]):
            points.insert(0, _get_point(hand, spec["metacarpal"]))
        points = np.asarray(points)
        if points.shape[1:] != (3,) or not np.isfinite(points).all():
            raise ValueError(f"{finger}: invalid joint positions")
        if np.any(np.linalg.norm(np.diff(points, axis=0), axis=1) < 1e-6):
            raise ValueError(f"{finger}: collapsed bone")
    wrist = _get_point(hand, "wrist")
    index = _get_point(hand, "indexMetacarpal", fallback="indexKnuckle")
    little = _get_point(hand, "littleMetacarpal", fallback="littleKnuckle")
    middle = _get_point(hand, "middleMetacarpal", fallback="middleKnuckle")
    normal = np.cross(index - little, middle - wrist)
    if not np.isfinite(normal).all() or np.linalg.norm(normal) < 1e-10:
        raise ValueError("degenerate palm")


class SimHandTrackingGuard:
    """One guard per hand; briefly hold invalid input, then release on timeout.

    This validates geometry, not packet freshness. Valid-looking frozen SDK
    samples cannot be distinguished from a stationary hand here.
    """

    def __init__(self, mapper, open_command, timeout):
        self.mapper = mapper
        self.open_command = np.asarray(open_command, dtype=float).copy()
        self.command = self.open_command.copy()
        self.timeout = max(float(timeout), 0.0)
        self.last_valid = None
        self.status = None

    def update(self, hand, now):
        debug = None
        reason = ""
        try:
            validate_hand_skeleton(hand)
            command, debug = self.mapper.build_command(hand)
            if not np.isfinite(command).all():
                raise ValueError("non-finite command")
        except (AttributeError, ValueError, IndexError, TypeError) as exc:
            reason = str(exc)
            if self.last_valid is not None and now - self.last_valid < self.timeout:
                status = "holding"
            else:
                status = "open_after_loss"
                self.command = self.open_command.copy()
                self.mapper.reset()
        else:
            self.command = np.asarray(command).copy()
            self.last_valid = now
            status = "tracking"
        changed = status != self.status
        self.status = status
        return self.command.copy(), debug, status, changed, reason
