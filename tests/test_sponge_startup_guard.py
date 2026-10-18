"""No headsets, DDS channels, GUI windows or native controllers are started."""
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import json
import sys

import mujoco
import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))
from sponge_runtime_guard import StartupGuard, realtime_loop
from sponge_startup_check import controller_error, status_error
from test_sponge_teleop import scene, startup_samples


def steady(guard, data, *, policy=True, band=True, speed=1., until=3.):
    data.qpos[:3] = [0, 0, .8]
    data.qpos[3:7] = [1, 0, 0, 0]
    data.qvel[:] = 0
    mujoco.mj_forward(guard.model, data)
    for now in np.arange(0, until+.001, .02):
        data.time = float(now*speed)
        guard.observe_command(object(), now)
        state = guard.measure(now, band, now if policy else None)
    return state


def test_drop_requires_actual_policy_and_realtime_physics(scene):
    m, d = scene
    guard = StartupGuard(m, d, now=0)
    assert not steady(guard, d, policy=False)['ready_drop']
    guard = StartupGuard(m, d, now=0)
    assert not steady(guard, d, speed=.435)['ready_drop']
    guard = StartupGuard(m, d, now=0)
    assert steady(guard, d)['ready_drop']


def test_stale_dds_cannot_be_disguised_by_policy_feedback(scene):
    m, d = scene
    guard = StartupGuard(m, d, now=0)
    steady(guard, d)
    state = guard.measure(3.3, True, 3.3)
    assert not state['ready_drop']
    assert 'DDS' in state['ready_reason']


def test_lost_controller_freezes_without_reset_or_fake_recovery(scene):
    m, d = scene
    guard = StartupGuard(m, d, now=0)
    steady(guard, d)
    before = d.qpos.copy()
    guard.measure(3.3, False, 3.)
    assert '中断' in guard.fault(3.3, False)
    np.testing.assert_array_equal(d.qpos, before)
    guard.observe_command(object(), 3.31)
    guard.measure(3.31, False, 3.31)
    assert guard.fault(3.31, False)
    assert not guard.last_state['ready_teleop']


def test_fall_latches_before_upstream_point_two_metre_reset(scene):
    m, d = scene
    guard = StartupGuard(m, d, now=0)
    steady(guard, d)
    d.qpos[3:7] = [np.cos(np.deg2rad(20)), 0, np.sin(np.deg2rad(20)), 0]
    guard.measure(3.01, False, 3.01)
    assert '倾斜' in guard.fault(3.01, False)
    assert d.qpos[2] == .8
    assert len(guard.history) > 20


def test_ground_readiness_waits_for_stable_pose(scene):
    m, d = scene
    guard = StartupGuard(m, d, now=0)
    steady(guard, d)
    for now in (3.01, 4., 5.02):
        d.time = now
        guard.observe_command(object(), now)
        s = guard.measure(now, False, now)
        assert s['ready_teleop'] == (now == 5.02)
    d.qvel[0] = .5
    assert not guard.measure(5.03, False, 5.03)['ready_teleop']


def test_render_overruns_do_not_permanently_slow_physics():
    class Clock:
        now = 0.
        def __call__(self): return self.now
        def sleep(self, seconds): self.now += seconds
    clock = Clock()
    class Env:
        viewer = None
        steps = 0
        images = 0
        def sim_step(self):
            self.steps += 1
            clock.now += .0025
            if clock.now >= 5:
                sim._running = False
        def update_viewer(self): clock.now += .020
        def update_render_caches(self):
            self.images += 1
            clock.now += .012
        def clock_overrun(self, lag): pytest.fail(f'unexpected lag {lag}')
    env = Env()
    sim = SimpleNamespace(sim_env=env, _running=True, sim_dt=.005, close=lambda: None)
    realtime_loop(sim, clock=clock, sleep=clock.sleep)
    assert env.steps*.005 / clock.now > .97
    assert env.images > 20


def test_clock_pause_is_reported_instead_of_large_catchup_burst():
    now = [0.]
    overruns = []
    env = SimpleNamespace(viewer=None, sim_step=lambda: now.__setitem__(0, now[0]+.5),
        update_viewer=lambda: None, update_render_caches=lambda: None)
    sim = SimpleNamespace(sim_env=env, _running=True, sim_dt=.005, close=lambda: None)
    def overrun(lag):
        overruns.append(lag)
        sim._running = False
    env.clock_overrun = overrun
    realtime_loop(sim, clock=lambda: now[0], sleep=lambda s: now.__setitem__(0, now[0]+s))
    assert overruns == [.495]


def test_proc_guard_rejects_missing_duplicate_or_hardware_controllers(tmp_path):
    assert controller_error(tmp_path)
    def add(pid, interface):
        p = tmp_path / str(pid)
        p.mkdir()
        (p/'cmdline').write_bytes(b'/test/g1_deploy_onnx_ref\0'+interface+b'\0')
    add(1, b'lo')
    assert not controller_error(tmp_path)
    add(2, b'eno1')
    assert '非回环' in controller_error(tmp_path)
    (tmp_path/'2/cmdline').write_bytes(b'/test/g1_deploy_onnx_ref\0lo\0')
    assert controller_error(tmp_path)


def test_old_frozen_or_stale_status_never_releases_startup():
    status = dict(updated_unix=100., guard_version=1, elastic_band=True, ready_drop=True,
                  ready_teleop=True, halted='')
    assert not status_error(status, 'drop', now=101.)
    assert status_error(status, 'drop', now=103.)
    assert status_error({**status, 'guard_version':0}, 'drop', now=101.)
    assert status_error({**status, 'halted':'test'}, 'teleop', now=101.)
    assert status_error({**status, 'elastic_band':False}, 'preflight', now=101.)


def test_gui_nine_is_queued_instead_of_bypassing_drop_guard(monkeypatch):
    import mujoco.viewer
    from sponge_teleop_sim import task_keys
    captured, calls = {}, []
    def launch(*args, **kwargs):
        captured.update(kwargs)
        return object()
    monkeypatch.setattr(mujoco.viewer, 'launch_passive', launch)
    with task_keys(lambda: calls.append('reset'), lambda: calls.append('request_drop')):
        mujoco.viewer.launch_passive(key_callback=lambda key: calls.append(key))
    callback = captured['key_callback']
    callback(ord('9')); callback(ord('R')); callback(ord('7'))
    assert calls == ['request_drop', 'reset', ord('7')]
    assert mujoco.viewer.launch_passive is launch
