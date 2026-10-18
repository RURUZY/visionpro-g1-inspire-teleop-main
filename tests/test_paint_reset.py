"""Tool resets preserve robot state, time, and the ongoing recording."""

from pathlib import Path
import json

import mujoco
import numpy as np
import pytest

from g1_teleop.sim.paint_diagnostics import PaintRecorder
from g1_teleop.sim.paint_reset import PaintToolReset, reset_key_callback, viewer_reset_key
from tools.generate_paint_scene import generate_scene, ROBOT_MODEL


@pytest.fixture
def scene(tmp_path):
    if not ROBOT_MODEL.is_file():
        pytest.skip("Prepare local G1 assets first")
    model = mujoco.MjModel.from_xml_path(str(generate_scene(tmp_path / "scene")))
    return model, mujoco.MjData(model)


def test_dropped_tool_reset_keeps_robot_time_and_commands(scene, tmp_path):
    model, data = scene
    reset = PaintToolReset(model)
    data.qpos[reset.qpos[:3]] = [1.5, -.9, .05]
    data.qvel[:] = .1
    data.qacc_warmstart[:] = 2
    data.qfrc_applied[:] = .2
    data.xfrc_applied[:] = .3
    data.ctrl[:] = .4
    data.time = 7.3
    saved = {name: getattr(data, name).copy() for name in
             ("qpos", "qvel", "qacc_warmstart", "qfrc_applied", "xfrc_applied", "ctrl")}
    assert reset.apply_pending(data) is None
    reset.request()
    event = reset.apply_pending(data)
    assert event and event["reason"] == "manual_tool_reset"
    np.testing.assert_array_equal(data.qpos[reset.qpos], model.qpos0[reset.qpos])
    keep_qpos = np.setdiff1d(np.arange(model.nq), reset.qpos)
    keep_dofs = np.setdiff1d(np.arange(model.nv), reset.dofs)
    np.testing.assert_array_equal(data.qpos[keep_qpos], saved["qpos"][keep_qpos])
    for name in ("qvel", "qacc_warmstart", "qfrc_applied"):
        np.testing.assert_array_equal(getattr(data, name)[keep_dofs], saved[name][keep_dofs])
        assert np.all(getattr(data, name)[reset.dofs] == 0)
    np.testing.assert_array_equal(data.ctrl, saved["ctrl"])
    other_bodies = sorted(set(range(model.nbody)) - reset.bodies)
    np.testing.assert_array_equal(data.xfrc_applied[other_bodies], saved["xfrc_applied"][other_bodies])
    assert data.time == 7.3 and reset.apply_pending(data) is None
    recorder = PaintRecorder(model, output_dir=tmp_path / "run")
    try:
        recorder.record_manual_reset(event)
        before = recorder.begin_step(data)
        assert before["episode"] == 1 and before["sim_time"] == 7.3
        recorded = json.loads((recorder.path / "events.jsonl").read_text())
        assert recorded["reason"] == "manual_tool_reset" and recorded["episode"] == 1
    finally:
        recorder.close()


def test_occupied_return_position_is_rejected_without_state_changes(scene):
    model, data = scene
    reset = PaintToolReset(model)
    # Put the pelvis into the return location, while the brush is on the floor.
    data.qpos[:3] = model.qpos0[reset.qpos[:3]]
    data.qpos[reset.qpos[:3]] = [2, 0, .05]
    saved = data.qpos.copy()
    reset.request()
    assert reset.apply_pending(data) is None
    np.testing.assert_array_equal(data.qpos, saved)


def test_viewer_key_queues_reset_and_preserves_elastic_band_keys(monkeypatch):
    import mujoco.viewer
    received, resets, launches = [], [], []

    def original(*args, **kwargs):
        launches.append(kwargs)
        return "viewer"

    monkeypatch.setattr(mujoco.viewer, "launch_passive", original)
    with viewer_reset_key(lambda: resets.append(True)):
        assert mujoco.viewer.launch_passive(None, None, key_callback=received.append) == "viewer"
    assert mujoco.viewer.launch_passive is original
    callback = launches[0]["key_callback"]
    callback(ord("R"))
    callback(ord("9"))
    assert resets == [True] and received == [ord("9")]
    reset_key_callback(lambda: resets.append(True))(ord("r"))
    assert len(resets) == 2
