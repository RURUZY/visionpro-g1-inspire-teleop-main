"""Exercise the patched upstream environment with mocked I/O and real MuJoCo."""
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from test_sponge_teleop import generator, ROBOT


def test_task_integration_blocks_early_drop_and_freezes_once(tmp_path, monkeypatch):
    import yaml
    import mujoco
    import gear_sonic.utils.mujoco_sim.base_sim as base
    import g1_teleop.calibration.session as session
    import sponge_teleop_sim as task
    import g1_teleop.sim.inspire_hand as inspire
    from gear_sonic.utils.mujoco_sim.simulator_factory import SimulatorFactory
    original_create = SimulatorFactory._create_mujoco_simulator
    hand_class = inspire.InspireHandController
    monkeypatch.setattr(inspire, "InspireHandController", lambda m,d,*a: hand_class(m,d,connect=False))
    monkeypatch.setenv("SONIC_GRASP_DIAGNOSTICS", "0")
    monkeypatch.setenv("SONIC_PAINT_DIAGNOSTICS", "0")
    original_env, original_start, original_close = base.DefaultEnv, base.BaseSimulator.start, base.BaseSimulator.close
    class Feedback:
        def __init__(self, *args): pass
        def latest(self): return None
        def close(self): pass
    monkeypatch.setattr(session, 'ZmqFeedbackClient', Feedback)
    monkeypatch.setenv('SONIC_SPONGE_STATUS', str(tmp_path/'status.json'))
    groot = Path('/home/epcap/projects/GR00T-WholeBodyControl')
    config = yaml.safe_load((groot/'gear_sonic/utils/mujoco_sim/wbc_configs/g1_29dof_sonic_model12.yaml').read_text())
    config.update(ROBOT_SCENE=str(generator.generate_scene(tmp_path, robot_model=ROBOT)),
                  enable_waist=True, NUM_HAND_JOINTS=12, NUM_HAND_MOTORS=0, PRINT_SCENE_INFORMATION=False)
    # Prevent tests from leaving class patches behind for other scenes.
    try:
        task.install_sponge_task()
        inspire.install_inspire_sim()
        env = base.DefaultEnv(config=config, onscreen=False, offscreen=False)
        published = []
        env.unitree_bridge = SimpleNamespace(low_cmd=None, joystick=False,
            PublishLowState=lambda obs: published.append(obs))
        env.compute_body_torques = lambda: np.zeros(29)
        env.band_pending.set()
        env.sim_step()
        assert env.elastic_band.enable  # No controller or policy: 9 is rejected.
        assert env.mj_data.time == pytest.approx(.005)
        assert len(published) == 1
        env.elastic_band.enable = False
        env.sim_step()
        frozen = env.mj_data.qpos.copy()
        frozen_time = env.mj_data.time
        assert env.guard.halted
        env.band_pending.set()
        env.sponge_reset.request()
        for _ in range(3): env.sim_step()
        np.testing.assert_array_equal(env.mj_data.qpos, frozen)
        assert env.mj_data.time == frozen_time
        assert len(published) == 1  # No frozen-state heartbeats to controller.
        assert len(list(tmp_path.glob('sponge_fault_*.json'))) == 1
        env._write_status(task.time.monotonic(), force=True)
        import json
        status = json.loads((tmp_path/'status.json').read_text())
        assert status['halted'] and not status['ready_drop'] and not status['ready_teleop']
    finally:
        SimulatorFactory._create_mujoco_simulator = staticmethod(original_create)
        base.DefaultEnv, base.BaseSimulator.start, base.BaseSimulator.close = original_env, original_start, original_close
