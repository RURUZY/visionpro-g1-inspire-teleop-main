"""Physics-backed recorder tests; never starts DDS or a robot controller."""
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import zmq

from g1_teleop.sim.grasp_diagnostics import GraspRecorder
from g1_teleop.sim.inspire_hand import InspireHandController

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "assets/mujoco/g1_runtime/scene_43dof_inspire_hand_pnp_cube.xml"


def support_scene():
    model = mujoco.MjModel.from_xml_string('''<mujoco><option timestep="0.002"/>
    <worldbody><geom name="table_top" type="box" size="1 1 .1"/>
      <body name="pick_cube_body" pos="0 0 .15"><freejoint/>
        <geom name="pick_cube" type="box" size=".05 .05 .05" mass=".2"/></body>
      <body name="left_wrist_yaw_link" pos="3 0 0"><joint name="left"/><geom size=".01"/></body>
      <body name="right_wrist_yaw_link" pos="-3 0 0"><joint name="right"/><geom size=".01"/></body>
    </worldbody></mujoco>''')
    controller = SimpleNamespace(active={s: np.array([model.joint(s).id]) for s in ("left", "right")},
                                 trace={"left": {}, "right": {}}, last_received=None, timed_out=False)
    return model, mujoco.MjData(model), controller


def test_contact_force_direction_and_weight_support(tmp_path):
    model, data, controller = support_scene()
    rec = GraspRecorder(model, controller, output_dir=tmp_path / "run", endpoint=None)
    try:
        for _ in range(1000): mujoco.mj_step(model, data)
        before = rec.begin_step(data); mujoco.mj_step(model, data)
        q = data.qpos.copy(); v = data.qvel.copy(); ctrl = data.ctrl.copy()
        s = rec.end_step(data, before)
        np.testing.assert_array_equal(data.qpos, q)
        np.testing.assert_array_equal(data.qvel, v)
        np.testing.assert_array_equal(data.ctrl, ctrl)
        np.testing.assert_allclose(s["cube"]["contact_force_world_n"], [0, 0, .2 * 9.81], atol=.01)
        assert s["cube"]["table_contact"]
        assert abs(s["cube"]["bottom_clearance_m"]) < .002
        assert s["hands"]["right"]["normal_contact_sum_n"] == 0
    finally:
        rec.close()
    stored = json.loads((rec.path / "samples.jsonl").read_text())
    assert stored["seq"] == 0
    assert json.loads((rec.path / "recording_status.json").read_text())["samples_saved"] == 1
    assert mujoco.MjModel.from_binary_path(str(rec.model_path)).nq == model.nq


def test_live_transport_and_reset_time(tmp_path):
    model, data, controller = support_scene()
    rec = GraspRecorder(model, controller, output_dir=tmp_path / "run", endpoint="tcp://127.0.0.1:*")
    ctx = zmq.Context(); sub = ctx.socket(zmq.SUB); sub.setsockopt(zmq.SUBSCRIBE, b"")
    sub.connect(rec.socket.getsockopt(zmq.LAST_ENDPOINT).decode())
    try:
        import time
        time.sleep(.05)
        b = rec.begin_step(data); mujoco.mj_step(model, data); rec.end_step(data, b)
        assert sub.poll(1000)
        s = sub.recv_json()
        assert s["sim_time"] == 0 and s["post_step_time"] == model.opt.timestep
        assert rec.begin_step(data) is None
        rec.last_time = 1.
        mujoco.mj_resetData(model, data)
        assert rec.begin_step(data) is not None
    finally:
        sub.close(0); ctx.term(); rec.close()


@pytest.mark.skipif(not SCENE.exists(), reason="generated Inspire scene absent")
def test_diagnostics_does_not_change_hand_or_cube_dynamics(tmp_path):
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    a, b = mujoco.MjData(model), mujoco.MjData(model)
    ca, cb = InspireHandController(model, a, connect=False), InspireHandController(model, b, connect=False)
    rec = GraspRecorder(model, cb, output_dir=tmp_path / "run", endpoint=None)
    try:
        for i in range(60):
            before = rec.begin_step(b)
            for data, c in ((a, ca), (b, cb)):
                c.commands = {side: np.full(6, 400.) for side in ("left", "right")}
                tau = c.torques(model.opt.timestep)
                data.ctrl[c.indices["left"] - 1] = tau[:12]
                data.ctrl[c.indices["right"] - 1] = tau[12:]
                mujoco.mj_step(model, data)
            if before is not None: rec.end_step(b, before)
        np.testing.assert_array_equal(a.qpos, b.qpos)
        np.testing.assert_array_equal(a.qvel, b.qvel)
        np.testing.assert_array_equal(a.ctrl, b.ctrl)
        assert len(cb.trace["right"]["target_rad"]) == 6
    finally:
        rec.close(); ca.close(); cb.close()


@pytest.mark.skipif(not SCENE.exists(), reason="generated Inspire scene absent")
def test_launcher_hook_records_before_fall_reset(monkeypatch, tmp_path):
    """Exercise install_inspire_sim without importing DDS or binding real robot ports."""
    import sys
    import types
    import g1_teleop.sim.grasp_diagnostics as diagnostics
    import g1_teleop.sim.inspire_hand as hand_module
    from g1_teleop.sim.inspire_hand import install_inspire_sim
    for name in ("gear_sonic", "gear_sonic.utils", "gear_sonic.utils.mujoco_sim",
                 "gear_sonic.utils.mujoco_sim.base_sim", "gear_sonic.utils.mujoco_sim.simulator_factory"):
        module = types.ModuleType(name)
        monkeypatch.setitem(sys.modules, name, module)
        if "." in name:
            parent, attr = name.rsplit(".", 1)
            setattr(sys.modules[parent], attr, module)
    class Env:
        def __init__(self):
            self.mj_model = mujoco.MjModel.from_xml_path(str(SCENE))
            self.mj_data = mujoco.MjData(self.mj_model)
            self.sim_dt = self.mj_model.opt.timestep
            self.elastic_band = None
        def sim_step(self):
            tau = self.compute_hand_torques()
            self.mj_data.ctrl[self.inspire.indices["left"] - 1] = tau[:12]
            self.mj_data.ctrl[self.inspire.indices["right"] - 1] = tau[12:]
            mujoco.mj_step(self.mj_model, self.mj_data)
            self.check_fall()
        def check_fall(self):
            self.reset()
        def reset(self):
            mujoco.mj_resetData(self.mj_model, self.mj_data)
    base = sys.modules["gear_sonic.utils.mujoco_sim.base_sim"]
    base.DefaultEnv = Env
    base.BaseSimulator = type("Base", (), {"close": lambda self: None})
    factory = sys.modules["gear_sonic.utils.mujoco_sim.simulator_factory"]
    factory.SimulatorFactory = type("Factory", (), {"_create_mujoco_simulator": staticmethod(lambda *a, **k: None)})
    original = diagnostics.GraspRecorder
    monkeypatch.setattr(diagnostics, "GraspRecorder", lambda m, c, **kw: original(m, c, output_dir=tmp_path / "hook", endpoint=None))
    monkeypatch.setattr(hand_module, "InspireHandController", lambda m, d, *a, **kw: InspireHandController(m, d, connect=False))
    monkeypatch.setenv("SONIC_GRASP_DIAGNOSTICS", "1")
    install_inspire_sim()
    env = base.DefaultEnv()
    try:
        env.sim_step()
        assert env.mj_data.time == 0  # The fake fall reset has executed.
        assert env.grasp_diagnostics.seq == 1
        assert env.grasp_diagnostics.episode == 1
    finally:
        env.grasp_diagnostics.close(); env.inspire.close()
    sample = json.loads((tmp_path / "hook/samples.jsonl").read_text())
    assert sample["post_step_time"] > 0  # Captured before reset, not a fabricated zero-time state.
    assert sample["episode"] == 0
