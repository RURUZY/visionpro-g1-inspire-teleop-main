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


@pytest.mark.parametrize("options", [[], ["--enable-inspire-hand-dds"],
    ["--enable-inspire-hand-sim", "--enable-inspire-hand-dds"]])
def test_full_range_rejects_non_sim_output(monkeypatch, options):
    monkeypatch.setattr(sys, "argv", ["bridge", "--sim-thumb-full-range", *options])
    with pytest.raises(SystemExit) as exc:
        parse_args()
    assert exc.value.code == 2


def test_full_range_preserves_calibration_and_fingers_and_opens_on_tracking_loss(monkeypatch, tmp_path):
    from g1_teleop.hand.tracking_guard import SimHandTrackingGuard

    monkeypatch.setattr(sys, "argv", ["bridge", "--enable-inspire-hand-sim"])
    old_args = parse_args()
    monkeypatch.setattr(sys, "argv", ["bridge", "--enable-inspire-hand-sim", "--sim-thumb-full-range"])
    new_args = parse_args()
    fitted, _ = fit_thumb_calibration(HandCalibration(), samples())
    path = tmp_path / "personal.json"
    fitted.save(path)
    contents = path.read_bytes()
    old = InspireHandMapper.from_args(old_args, path)
    new = InspireHandMapper.from_args(new_args, path)
    for pose in ("open", "close", "open"):
        raw = dict(zip(FINGER_ORDER, [25., 35., 45., 55.]))
        raw.update({key: samples()[f"{key}_{pose}"][0] for key in ("thumb_bend", "thumb_rot")})
        old.measure_raw = lambda hand: (raw, {})
        new.measure_raw = lambda hand: (raw, {})
        for _ in range(40):
            a, _ = old.build_command(None)
            b, _ = new.build_command(None)
        np.testing.assert_array_equal(a[:4], b[:4])
        np.testing.assert_allclose(b[4:], [1000, 1000] if pose == "open" else [0, 0], atol=20)
    assert path.read_bytes() == contents
    # Match runtime's open-command construction: a lost skeleton must not close
    # the thumb root as a side effect of the old inversion convention.
    opening = [new_args.open_angle] * 4 + [new_args.thumb_bend_open_angle, new_args.thumb_rotation_open_angle]
    guard = SimHandTrackingGuard(new, opening, new_args.hand_tracking_timeout)
    guard.update(None, 1.0)
    np.testing.assert_array_equal(guard.command[4:], [1000, 1000])
