"""Contact-load and lifecycle checks without DDS, sockets, or robot hardware."""

import csv
import hashlib
import json
import sys
import types

import mujoco
import numpy as np
import pytest

from g1_teleop.sim.paint_diagnostics import PaintRecorder


def support_scene(support="wall", height=.2):
    """0.3 kg tool; translation-only mount lets known lateral load test friction."""
    wall_x = 0 if support == "wall" else 4
    table_x = 0 if support == "table" else 8
    hand_x = 0 if support == "hand" else -4
    rest = '<geom name="paint_handle_rest" type="box" size="1 1 .1"/>' if support == "rest" else ""
    xml = f'''<mujoco><option timestep="0.002"/><default>
      <geom friction="1 .01 .001" solref=".01 1"/></default><worldbody>
      <geom name="paint_wall" type="box" size="1 1 .1" pos="{wall_x} 0 0"/>
      <geom name="paint_table_top" type="box" size="1 1 .1" pos="{table_x} 0 0"/>
      {rest}
      <site name="paint_target_patch" type="box" size=".2 .3 .001" pos="0 0 .1"/>
      <body name="right_wrist_yaw_link" pos="{hand_x} 0 0">
        <geom name="hand_pad" type="box" size="1 1 .1"/></body>
      <body name="paint_tool_body" pos="0 0 {height}">
        <joint name="paint_tool_free" type="slide" axis="1 0 0"/>
        <joint name="tool_slide_y" type="slide" axis="0 1 0"/>
        <joint name="tool_slide_z" type="slide" axis="0 0 1"/>
        <inertial mass=".1" pos="0 0 0" diaginertia=".001 .001 .001"/>
        <body name="paint_roller_body"><joint name="paint_roller_hinge" axis="0 1 0"/>
          <geom name="paint_roller" type="sphere" size=".1" mass=".2"/></body>
      </body></worldbody></mujoco>'''
    model = mujoco.MjModel.from_xml_string(xml)
    return model, mujoco.MjData(model)


def solved_sample(recorder, model, data, *, elastic_band=False):
    before = recorder.begin_step(data)
    mujoco.mj_step(model, data)
    return recorder.end_step(data, before, elastic_band)


def test_wall_normal_and_friction_match_known_load_and_are_readonly(tmp_path):
    model, data = support_scene()
    tool = model.body("paint_tool_body").id
    data.xfrc_applied[tool, 1] = .4
    recorder = PaintRecorder(model, output_dir=tmp_path / "run")
    try:
        for _ in range(1000):
            mujoco.mj_step(model, data)
        before = recorder.begin_step(data)
        mujoco.mj_step(model, data)
        watched = {name: getattr(data, name).copy() for name in
                   ("qpos", "qvel", "qacc", "ctrl", "xfrc_applied", "qfrc_applied", "xpos", "xquat")}
        sample = recorder.end_step(data, before, elastic_band=True)
        for name, value in watched.items():
            np.testing.assert_array_equal(getattr(data, name), value)
        assert sample["roller_wall_contact"]
        assert sample["elastic_band"]
        assert sample["roller_wall_normal_sum_n"] == pytest.approx(.3 * 9.81, abs=.01)
        assert sample["roller_wall_tangent_magnitude_sum_n"] == pytest.approx(.4, abs=.01)
        force = [sample[f"roller_wall_force_on_tool_world_{axis}_n"] for axis in "xyz"]
        np.testing.assert_allclose(force, [0, -.4, .3 * 9.81], atol=.01)
        assert not sample["tool_table_contact"]
        assert not sample["right_hand_tool_contact"]
    finally:
        recorder.close()
    with (recorder.path / "samples.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["relative_position_delta_m"] == ""
    metadata = json.loads((recorder.path / "metadata.json").read_text())
    assert metadata["model_sha256"] == hashlib.sha256(recorder.model_path.read_bytes()).hexdigest()
    assert metadata["target_patch_geometry"]["size_m"] == [.2, .3, .001]
    status = json.loads((recorder.path / "recording_status.json").read_text())
    assert status["samples_saved"] == 1 and status["error"] is None


@pytest.mark.parametrize(("support", "group"), [
    ("table", "tool_table"), ("rest", "tool_table"), ("hand", "right_hand_tool")])
def test_tool_subtree_support_contacts_include_roller(tmp_path, support, group):
    model, data = support_scene(support)
    recorder = PaintRecorder(model, output_dir=tmp_path / support)
    try:
        for _ in range(1000):
            mujoco.mj_step(model, data)
        sample = solved_sample(recorder, model, data)
        assert sample[group + "_contact"]
        assert sample[group + "_normal_sum_n"] == pytest.approx(.3 * 9.81, abs=.01)
        assert sample[group + "_force_on_tool_world_z_n"] == pytest.approx(.3 * 9.81, abs=.01)
        assert sample["roller_wall_normal_sum_n"] == 0
    finally:
        recorder.close()


def test_no_contact_zero_force_episode_reset_and_relative_motion(tmp_path):
    model, data = support_scene(height=2.)
    model.opt.gravity[:] = 0
    recorder = PaintRecorder(model, output_dir=tmp_path / "run")
    try:
        first = solved_sample(recorder, model, data)
        assert first["episode"] == 0
        assert first["relative_position_delta_m"] is None
        assert recorder.begin_step(data) is None
        for key, value in first.items():
            if key.endswith("_n") or key.endswith("_contact_count"):
                assert value == 0
        data.qvel[0] = .1
        for _ in range(10):
            mujoco.mj_step(model, data)
        second = solved_sample(recorder, model, data)
        assert second["relative_position_delta_m"] == pytest.approx(.002, abs=1e-9)
        assert second["relative_rotation_delta_rad"] == 0
        assert second["relative_sample_interval_s"] == pytest.approx(.022)
        recorder.episode += 1
        mujoco.mj_resetData(model, data)
        third = solved_sample(recorder, model, data)
        assert third["episode"] == 1
        assert third["sim_time"] == 0
        assert third["relative_position_delta_m"] is None
    finally:
        recorder.close()
    recorder.close()
    assert recorder.begin_step(data) is None
    assert recorder.end_step(data, {}) is None


def test_recording_does_not_change_dynamics(tmp_path):
    model, data = support_scene()
    plain = mujoco.MjData(model)
    recorder = PaintRecorder(model, output_dir=tmp_path / "run")
    try:
        for _ in range(100):
            before = recorder.begin_step(data)
            mujoco.mj_step(model, data)
            mujoco.mj_step(model, plain)
            recorder.end_step(data, before)
        np.testing.assert_array_equal(data.qpos, plain.qpos)
        np.testing.assert_array_equal(data.qvel, plain.qvel)
        np.testing.assert_array_equal(data.qacc, plain.qacc)
        assert recorder.saved == 10  # 0.2 s sampled at 50 Hz.
    finally:
        recorder.close()


def test_write_failure_is_reported(tmp_path):
    model, data = support_scene()
    recorder = PaintRecorder(model, output_dir=tmp_path / "run")
    class BrokenWriter:
        def writerow(self, row):
            raise OSError("test disk failure")
    recorder.writer = BrokenWriter()
    try:
        with pytest.raises(RuntimeError, match="Paint recording failed.*test disk failure"):
            solved_sample(recorder, model, data)
        assert recorder.error and recorder.saved == 0
    finally:
        recorder.close()
    assert "test disk failure" in json.loads((recorder.path / "recording_status.json").read_text())["error"]


def test_conflicting_modes_fail_before_importing_simulator(monkeypatch):
    from g1_teleop.sim.inspire_hand import install_inspire_sim
    monkeypatch.setenv("SONIC_PAINT_DIAGNOSTICS", "1")
    monkeypatch.setenv("SONIC_GRASP_DIAGNOSTICS", "1")
    with pytest.raises(ValueError, match="mutually exclusive"):
        install_inspire_sim()


def test_hook_captures_before_fall_reset_without_dds(monkeypatch, tmp_path):
    import g1_teleop.sim.inspire_hand as hand_module
    import g1_teleop.sim.paint_reset as reset_module
    # This recorder fixture uses slide joints, not the real free roller tool.
    monkeypatch.setattr(reset_module, "PaintToolReset", lambda *args: types.SimpleNamespace(
        apply_pending=lambda data: None))
    for name in ("gear_sonic", "gear_sonic.utils", "gear_sonic.utils.mujoco_sim",
                 "gear_sonic.utils.mujoco_sim.base_sim", "gear_sonic.utils.mujoco_sim.simulator_factory"):
        module = types.ModuleType(name)
        monkeypatch.setitem(sys.modules, name, module)
        if "." in name:
            parent, attr = name.rsplit(".", 1)
            setattr(sys.modules[parent], attr, module)
    class Env:
        def __init__(self):
            self.mj_model, self.mj_data = support_scene()
            self.sim_dt = self.mj_model.opt.timestep
            self.elastic_band = None
        def sim_step(self):
            mujoco.mj_step(self.mj_model, self.mj_data)
            self.check_fall()
        def check_fall(self):
            self.reset()
        def reset(self):
            mujoco.mj_resetData(self.mj_model, self.mj_data)
    class Controller:
        def __init__(self, *args):
            self.closed = False
        def reset(self):
            pass
        def close(self):
            self.closed = True
    base = sys.modules["gear_sonic.utils.mujoco_sim.base_sim"]
    base.DefaultEnv = Env
    base.BaseSimulator = type("Base", (), {"close": lambda self: None})
    factory = sys.modules["gear_sonic.utils.mujoco_sim.simulator_factory"]
    factory.SimulatorFactory = type("Factory", (), {"_create_mujoco_simulator": staticmethod(lambda *a, **k: None)})
    monkeypatch.setattr(hand_module, "InspireHandController", Controller)
    monkeypatch.setenv("SONIC_PAINT_DIAGNOSTICS", "1")
    monkeypatch.delenv("SONIC_GRASP_DIAGNOSTICS", raising=False)
    monkeypatch.delenv("SONIC_ROBOT_SCENE", raising=False)
    monkeypatch.setenv("SONIC_PAINT_OUTPUT_DIR", str(tmp_path / "hook"))
    hand_module.install_inspire_sim()
    env = base.DefaultEnv()
    sim = base.BaseSimulator()
    sim.sim_env = env
    try:
        env.sim_step()
        assert env.mj_data.time == 0
        assert env.grasp_diagnostics.seq == 1
        assert env.grasp_diagnostics.episode == 1
    finally:
        sim.close()
    assert env.inspire.closed
    with (tmp_path / "hook/samples.csv").open() as stream:
        sample = next(csv.DictReader(stream))
    assert float(sample["post_step_time"]) > 0
    assert sample["episode"] == "0"
