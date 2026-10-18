#!/usr/bin/env python3
"""Task-only extensions to the existing SONIC/Inspire simulator; no motion script."""
from contextlib import contextmanager
from pathlib import Path
from threading import Event
import json
import os
import time

import mujoco
import numpy as np


class SpongeReset:
    """R resets only the free tool. Reject a return pose occupied by the robot."""
    def __init__(self, model, pending=None):
        self.model = model
        joint = model.joint('sponge_free')
        self.qpos = slice(int(joint.qposadr[0]), int(joint.qposadr[0]) + 7)
        self.dofs = slice(int(joint.dofadr[0]), int(joint.dofadr[0]) + 6)
        self.body = model.body('sponge_tool').id
        self.geoms = set(np.flatnonzero(model.geom_bodyid == self.body))
        self.pending = pending if pending is not None else Event()
        self.scratch = mujoco.MjData(model)

    def request(self):
        self.pending.set()

    def apply(self, data):
        if not self.pending.is_set():
            return False
        self.pending.clear()
        self.scratch.qpos[:] = data.qpos
        self.scratch.qpos[self.qpos] = self.model.qpos0[self.qpos]
        mujoco.mj_forward(self.model, self.scratch)
        for c in self.scratch.contact:
            if ((int(c.geom1) in self.geoms) != (int(c.geom2) in self.geoms)) and c.dist < -1e-4:
                print('[sponge] Return position occupied; move hands away and press R again.', flush=True)
                return False
        data.qpos[self.qpos] = self.model.qpos0[self.qpos]
        for name in ('qvel', 'qacc_warmstart', 'qfrc_applied'):
            getattr(data, name)[self.dofs] = 0
        data.xfrc_applied[self.body] = 0
        print('[sponge] Tool returned to table. Robot and teleop calibration preserved.', flush=True)
        return True


@contextmanager
def task_keys(reset_request, band_request):
    # Viewer callback runs on the GUI thread: queue requests for the physics thread.
    import mujoco.viewer
    original = mujoco.viewer.launch_passive
    def launch(*args, **kwargs):
        previous = kwargs.get('key_callback')
        def on_key(key):
            if key in (ord('r'), ord('R')):
                reset_request()
            elif key == ord('9'):
                band_request()
            elif previous:
                previous(key)
        kwargs['key_callback'] = on_key
        return original(*args, **kwargs)
    mujoco.viewer.launch_passive = launch
    try:
        yield
    finally:
        mujoco.viewer.launch_passive = original


def install_sponge_task():
    import gear_sonic.utils.mujoco_sim.base_sim as base_sim
    from g1_teleop.calibration.session import ZmqFeedbackClient
    from sponge_runtime_guard import StartupGuard, realtime_loop
    original = base_sim.DefaultEnv
    original_close = base_sim.BaseSimulator.close

    class SpongeEnv(original):
        def __init__(self, *args, **kwargs):
            pending = Event()
            self.band_pending = Event()
            with task_keys(pending.set, self.band_pending.set):
                super().__init__(*args, **kwargs)
            self.sponge_reset = SpongeReset(self.mj_model, pending)
            self.guard = StartupGuard(self.mj_model, self.mj_data)
            # g1_debug is published only by the C++ CONTROL loop, unlike the
            # bridge's ENGAGE log, which merely confirms sending a start command.
            self.policy_feedback = ZmqFeedbackClient('127.0.0.1', 5557)
            self.sponge_steps = 0
            self.sponge_physics_ms = 0.
            self.sponge_render_ms = 0.
            self.sponge_last_status = time.monotonic()
            self.sponge_status_path = Path(os.environ['SONIC_SPONGE_STATUS'])
            self.sponge_status_path.parent.mkdir(parents=True, exist_ok=True)
            self.fault_written = False
            if self.viewer is not None:
                with self.viewer.lock():
                    self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                    self.viewer.cam.lookat[:] = [.25, -.12, .95]
                    self.viewer.cam.distance = 1.85
                    self.viewer.cam.azimuth = 130
                    self.viewer.cam.elevation = -25
            print('[sponge] R 复位海绵；9 请求落地（检查通过才解除悬挂）。', flush=True)
            print('[sponge] 真实手指接触抓取；异常冻结并保存记录，需重新启动三终端。', flush=True)

        def _measure(self):
            now = time.monotonic()
            bridge = self.unitree_bridge
            self.guard.observe_command(getattr(bridge, 'low_cmd', None), now)
            feedback = self.policy_feedback.latest()
            band = bool(self.elastic_band and self.elastic_band.enable)
            self.guard.measure(now, band, feedback.monotonic_time if feedback else None)
            return now, band

        def _write_fault(self):
            if not self.guard.halted or self.fault_written:
                return
            self.fault_written = True
            path = self.sponge_status_path.parent / f'sponge_fault_{time.time_ns()}.json'
            path.write_text(json.dumps(dict(reason=self.guard.halted,
                history=list(self.guard.history), qpos=self.mj_data.qpos.tolist(),
                qvel=self.mj_data.qvel.tolist(), sim_time=float(self.mj_data.time)), ensure_ascii=False)+'\n')
            print(f'[sponge] 已冻结：{self.guard.halted}。记录：{path}', flush=True)
            print('[sponge] 请结束终端三、二、一并重新启动；不会自动复位后继续运动。', flush=True)

        def _write_status(self, now, force=False):
            if not force and now-self.sponge_last_status < .5:
                return
            hand = getattr(self, 'inspire', None)
            status = dict(pid=os.getpid(), updated_unix=time.time(), sim_time=float(self.mj_data.time),
                steps_per_second=self.sponge_steps/max(.001, now-self.sponge_last_status),
                physics_step_ms=self.sponge_physics_ms, render_ms=self.sponge_render_ms,
                hand_packets=hand.received if hand else 0,
                hand_timed_out=hand.timed_out if hand else False,
                robot_qpos=self.mj_data.qpos[:self.sponge_reset.qpos.start].tolist(),
                sponge_qpos=self.mj_data.qpos[self.sponge_reset.qpos].tolist(),
                elastic_band=bool(self.elastic_band and self.elastic_band.enable),
                **self.guard.last_state)
            status['halted'] = self.guard.halted
            if self.guard.halted:
                status.update(ready_drop=False, ready_teleop=False, ready_reason=self.guard.halted)
            tmp = self.sponge_status_path.with_suffix('.tmp')
            tmp.write_text(json.dumps(status, ensure_ascii=False)+'\n')
            tmp.replace(self.sponge_status_path)
            self.sponge_last_status, self.sponge_steps = now, 0

        def sim_step(self):
            start = time.monotonic()
            now, band = self._measure()
            self.guard.fault(now, band)
            if self.band_pending.is_set():
                self.band_pending.clear()
                if self.guard.halted:
                    print('[sponge] 已冻结，请重新启动三终端。', flush=True)
                elif not band:
                    self.elastic_band.enable = True
                    print('[sponge] 已重新启用悬挂。', flush=True)
                elif self.guard.last_state['ready_drop']:
                    self.elastic_band.enable = False
                    print('[sponge] 检查通过，已解除悬挂。', flush=True)
                else:
                    print('[sponge] 保持悬挂：'+self.guard.last_state['ready_reason'], flush=True)
            if not self.guard.halted:
                self.sponge_reset.apply(self.mj_data)
                super().sim_step()
                self.sponge_steps += 1
            # A halted experiment does not publish LowState; the native controller
            # will also see its existing missing-state timeout, rather than stale
            # poses being advertised as a running experiment.
            self.sponge_physics_ms = 1000*(time.monotonic()-start)
            self._write_fault()
            self._write_status(time.monotonic())

        def check_fall(self):
            # Called by the upstream step before it could reset the fallen robot.
            now, band = self._measure()
            self.guard.fault(now, band)

        def clock_overrun(self, lag):
            if not self.elastic_band.enable:
                self.guard.halted = f'物理时钟延迟 {lag:.3f} 秒'
                self._write_fault()

        def reset(self):
            # A whole-robot reset invalidates controller/VR calibration.
            if hasattr(self, 'guard'):
                self.guard.halted = '请求了整机复位，需要重新启动三终端并校准'
                self._write_fault()
            else:
                super().reset()

        def update_render_caches(self):
            start = time.monotonic()
            result = super().update_render_caches()
            self.sponge_render_ms = 1000*(time.monotonic()-start)
            return result

        def handle_keyboard_button(self, key):
            if key.lower() == 'r':
                self.sponge_reset.request()
            elif key == '9':
                self.band_pending.set()
            else:
                super().handle_keyboard_button(key)

    def close(sim):
        feedback = getattr(getattr(sim, 'sim_env', None), 'policy_feedback', None)
        if feedback is not None:
            feedback.close()
            sim.sim_env.policy_feedback = None
        original_close(sim)

    base_sim.DefaultEnv = SpongeEnv
    base_sim.BaseSimulator.start = realtime_loop
    base_sim.BaseSimulator.close = close


if __name__ == '__main__':
    import sonic_run_sim_loop
    install_sponge_task()
    sonic_run_sim_loop.main()
