from copy import deepcopy
import sys

import numpy as np
import pytest

from g1_teleop.bridge.cli import parse_args
from g1_teleop.hand.mapping import HandCalibration, InspireHandMapper, FINGER_ORDER
from g1_teleop.hand.thumb_calibration import fit_thumb_calibration


def samples():
    return {"thumb_bend_open": [8.] * 40, "thumb_bend_close": [51.] * 40,
            "thumb_rot_open": [32.] * 40, "thumb_rot_close": [7.] * 40}


def test_saved_thumb_fit_preserves_effective_sim_fingers(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["bridge", "--enable-inspire-hand-sim"])
    args = parse_args()
    before = InspireHandMapper.from_args(args, tmp_path / "absent.json")
    original = deepcopy(before.calibration.channels)
    fitted, _ = fit_thumb_calibration(before.calibration, samples())
    path = tmp_path / "calibration.json"
    fitted.save(str(path))
    after = InspireHandMapper.from_args(args, path)
    assert before.calibration.channels == original
    for finger in FINGER_ORDER:
        assert after.calibration.range_for(finger) == (10., 70.)
    for pose in ("open", "close"):
        raw = dict(zip(FINGER_ORDER, [25., 35., 45., 55.]))
        raw.update({key: samples()[f"{key}_{pose}"][0] for key in ("thumb_bend", "thumb_rot")})
        before.measure_raw = lambda hand: (raw, {})
        after.measure_raw = lambda hand: (raw, {})
        before.reset(); after.reset()
        a, _ = before.build_command(None)
        b, _ = after.build_command(None)
        np.testing.assert_array_equal(a[:4], b[:4])
        np.testing.assert_array_equal(b[4:], [800, 800] if pose == "open" else [200, 200])


@pytest.mark.parametrize("kind", ["few", "nan", "same", "noisy"])
def test_bad_capture_does_not_change_base(kind):
    base = HandCalibration()
    original = deepcopy(base.channels)
    data = samples()
    if kind == "few": data["thumb_rot_open"] = [32.] * 5
    if kind == "nan": data["thumb_rot_open"][0] = float("nan")
    if kind == "same": data["thumb_rot_close"] = [32.] * 40
    if kind == "noisy": data["thumb_rot_open"] = list(np.linspace(0, 64, 40))
    with pytest.raises(ValueError): fit_thumb_calibration(base, data)
    assert base.channels == original


def test_preserves_personal_finger_ranges():
    base = HandCalibration()
    base.channels["index"] = {"open": 7., "close": 83.}
    fitted, _ = fit_thumb_calibration(base, samples())
    for finger in FINGER_ORDER:
        assert fitted.channels[finger] == base.channels[finger]
