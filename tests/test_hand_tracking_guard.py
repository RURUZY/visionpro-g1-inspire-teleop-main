"""A lost skeleton must not be mistaken for the operator opening a hand."""

from copy import deepcopy
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from g1_teleop.bridge.cli import parse_args
from g1_teleop.hand.mapping import FINGER_CHAINS, FINGER_ORDER, InspireHandMapper
from g1_teleop.hand.tracking_guard import SimHandTrackingGuard


def skeleton(bend=0):
    hand = SimpleNamespace()

    def joint(name, point):
        transform = np.eye(4)
        transform[:3, 3] = point
        setattr(hand, name, transform)

    joint("wrist", [0, 0, 0])
    for i, finger in enumerate((*FINGER_ORDER, "thumb")):
        spec = FINGER_CHAINS[finger]
        point = np.array([(i - 1.5) * .02, .04, 0.])
        keys = ("metacarpal", "knuckle", "prox", "mid", "tip") if finger != "thumb" else ("knuckle", "prox", "mid", "tip")
        for j, key in enumerate(keys):
            joint(spec[key], point)
            angle = np.deg2rad(j * bend)
            point = point + [0, .02 * np.cos(angle), .02 * np.sin(angle)]
    return hand


@pytest.fixture
def guard(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["bridge", "--enable-inspire-hand-sim"])
    mapper = InspireHandMapper.from_args(parse_args())
    return SimHandTrackingGuard(mapper, [1000] * 6, 1.0)


@pytest.mark.parametrize("failure", ["missing", "collapsed", "partial", "nan", "missing_joint"])
def test_closed_hand_survives_short_invalid_input_then_times_out(guard, failure):
    closed, *_ = guard.update(skeleton(70), 1.0)
    assert np.all(closed[:4] < 50)
    bad = skeleton()
    if failure == "missing":
        bad = None
    elif failure == "collapsed":
        for name in vars(bad):
            setattr(bad, name, np.eye(4))
    elif failure == "partial":
        bad.indexTip = bad.indexIntermediateTip.copy()
    elif failure == "nan":
        bad.thumbTip[0, 3] = np.nan
    else:
        del bad.ringTip
    smoothed = deepcopy(guard.mapper.smoothed)
    for now in (1.2, 1.7):
        command, _, status, _, _ = guard.update(bad, now)
        np.testing.assert_array_equal(command, closed)
        assert status == "holding"
        assert guard.mapper.smoothed == smoothed
    command, _, status, _, _ = guard.update(bad, 2.01)
    np.testing.assert_array_equal(command, [1000] * 6)
    assert status == "open_after_loss"
    command, _, status, _, _ = guard.update(skeleton(70), 2.1)
    assert status == "tracking"
    assert np.all(command[:4] < 50)


def test_real_straight_fingers_are_valid_and_release_grasp(guard):
    guard.update(skeleton(70), 1.0)
    for i in range(40):
        command, _, status, _, _ = guard.update(skeleton(), 1.1 + i * .02)
    assert status == "tracking"
    assert np.all(command[:4] > 980)


def test_other_hand_does_not_extend_missing_hand_timeout(guard):
    other = SimHandTrackingGuard(InspireHandMapper.from_args(guard.mapper.args), [1000] * 6, 1.0)
    guard.update(skeleton(70), 1.0)
    other.update(skeleton(70), 1.0)
    for now in (1.5, 2.1):
        other.update(skeleton(70), now)
        command, _, status, _, _ = guard.update(None, now)
    assert status == "open_after_loss"
    np.testing.assert_array_equal(command, [1000] * 6)
    assert other.status == "tracking"
    assert np.all(other.command[:4] < 50)
