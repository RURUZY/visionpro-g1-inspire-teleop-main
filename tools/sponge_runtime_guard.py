"""Sponge-scene readiness, latched fault reporting, and wall-clock scheduling.

No DDS writers, calibration keys, or robot motion commands live in this module.
"""
from collections import deque
import math
import time

import mujoco
import numpy as np


class StartupGuard:
    def __init__(self, model, data, now=None):
        self.model, self.data = model, data
        now = time.monotonic() if now is None else now
        self.samples = deque([(now, float(data.time))])
        self.history = deque(maxlen=400)
        self.last_history = -1.
        self.last_command = None
        self.command_at = None
        self.feedback_at = None
        self.policy_since = None
        self.ground_since = None
        self.slow_since = None
        self.halted = ''
        self.table = model.body('sponge_table').id
        self.robot_end = self.table
        self.real_time_factor = 0.
        self.last_state = {}

    def observe_command(self, command, now):
        # UnitreeSdk2Bridge.LowCmdHandler replaces this object on every receipt.
        if command is not None and command is not self.last_command:
            self.last_command = command
            self.command_at = now

    def contacts(self):
        result = []
        for contact in self.data.contact:
            a, b = map(int, self.model.geom_bodyid[[contact.geom1, contact.geom2]])
            if ((0 < a < self.robot_end <= b) or (0 < b < self.robot_end <= a)):
                result.append(dict(a=self.model.geom(contact.geom1).name,
                                   b=self.model.geom(contact.geom2).name,
                                   distance=float(contact.dist)))
        return result

    def measure(self, now, band, policy_at):
        self.feedback_at = policy_at
        command_age = None if self.command_at is None else now - self.command_at
        policy_age = None if policy_at is None else now - policy_at
        command_ok = command_age is not None and command_age < .25
        policy_ok = policy_age is not None and 0 <= policy_age < .25
        if policy_ok:
            if self.policy_since is None:
                self.policy_since = now
        else:
            self.policy_since = None
        if now > self.samples[-1][0]:
            self.samples.append((now, float(self.data.time)))
        while len(self.samples) > 2 and self.samples[1][0] <= now - 2:
            self.samples.popleft()
        elapsed = now - self.samples[0][0]
        self.real_time_factor = ((self.data.time - self.samples[0][1]) / elapsed
                                 if elapsed > .001 else 0.)
        rate_ready = elapsed >= 1.9 and self.real_time_factor >= .9
        q = self.data.qpos[3:7]
        tilt = math.degrees(math.acos(float(np.clip(1 - 2*(q[1]**2 + q[2]**2), -1, 1))))
        linear_speed = float(np.linalg.norm(self.data.qvel[:3]))
        angular_speed = float(np.linalg.norm(self.data.qvel[3:6]))
        contacts = self.contacts()
        reason = ''
        if self.halted:
            reason = self.halted
        elif not command_ok:
            reason = '未收到新鲜的 DDS 电机指令，请检查终端二'
        elif not policy_ok:
            reason = '未收到实际策略运行反馈，请检查终端二是否退出或尚未按 ]'
        elif self.policy_since is None or now - self.policy_since < 1:
            reason = '等待策略连续运行至少 1 秒'
        elif not rate_ready:
            reason = f'仿真实时倍率 {self.real_time_factor:.2f}，落地要求至少 0.90'
        elif tilt > 15 or linear_speed > .2 or angular_speed > 1:
            reason = '机器人尚未稳定，请保持悬挂并检查姿态'
        elif band and any(c['distance'] < -1e-4 for c in contacts):
            reason = '落地前机器人已接触桌面或工具，请停止并检查场景'
        if not band and not reason:
            if self.ground_since is None:
                self.ground_since = now
        else:
            self.ground_since = None
        self.last_state = dict(guard_version=1, halted=self.halted, ready_reason=reason,
            ready_drop=bool(band and not reason),
            ready_teleop=bool(self.ground_since is not None and now-self.ground_since >= 2),
            real_time_factor=float(self.real_time_factor), command_age_s=command_age,
            policy_age_s=policy_age, tilt_deg=tilt, base_speed_m_s=linear_speed,
            base_angular_speed_rad_s=angular_speed, robot_prop_contacts=contacts)
        if float(self.data.time) - self.last_history >= .02:
            self.history.append(dict(wall_time=time.time(), sim_time=float(self.data.time),
                elastic_band=band, **self.last_state,
                robot_qpos=self.data.qpos[:60].tolist(), robot_qvel=self.data.qvel[:59].tolist()))
            self.last_history = float(self.data.time)
        return self.last_state

    def fault(self, now, band):
        """Freeze this experiment on loss of control; never reset into old commands."""
        if self.halted or band:
            return self.halted
        s = self.last_state
        reason = ''
        if s['command_age_s'] is None or s['command_age_s'] >= .25:
            reason = '落地后 DDS 电机指令中断'
        elif s['policy_age_s'] is None or s['policy_age_s'] >= .25:
            reason = '落地后策略反馈中断'
        elif not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            reason = '仿真状态出现非有限数值'
        elif self.data.qpos[2] < .45 or s['tilt_deg'] > 35:
            reason = '机器人高度或倾斜超出本次站立操作范围'
        if self.real_time_factor < .85:
            if self.slow_since is None:
                self.slow_since = now
            if now - self.slow_since > 1 and not reason:
                reason = '仿真持续跟不上实时控制'
        else:
            self.slow_since = None
        if reason:
            self.halted = reason
        return self.halted


def realtime_loop(sim, *, clock=time.monotonic, sleep=time.sleep):
    """Keep physics deadlines across rendering, skipping graphics while behind.

    The upstream loop sleeps relative to each individual step, permanently losing
    simulation time whenever viewer.sync/render exceeds one 5 ms physics step.
    Graphics here run on wall-clock deadlines and adapt to their measured cost.
    Physics dt and the native controller's frequencies remain unchanged.
    """
    env = sim.sim_env
    deadline = clock()
    viewer_due = image_due = deadline
    viewer_cost = image_cost = .005
    try:
        while sim._running and (env.viewer is None or env.viewer.is_running()):
            env.sim_step()
            deadline += sim.sim_dt
            now = clock()
            lag = now - deadline
            if lag > .25:
                # Do not run an unbounded catch-up burst after an OS suspension.
                env.clock_overrun(lag)
                deadline = now
            if now <= deadline:
                if now >= viewer_due:
                    start = clock()
                    env.update_viewer()
                    viewer_cost = .8*viewer_cost + .2*(clock()-start)
                    viewer_due = clock() + max(.05, 4*viewer_cost)
                # Never stack a second costly draw onto a missed physics deadline.
                now = clock()
                if now <= deadline and now >= image_due:
                    start = clock()
                    env.update_render_caches()
                    image_cost = .8*image_cost + .2*(clock()-start)
                    image_due = clock() + max(1/15, 4*image_cost)
                remaining = deadline - clock()
                if remaining > 0:
                    sleep(remaining)
    except KeyboardInterrupt:
        print('Simulator interrupted by user.', flush=True)
    finally:
        sim.close()
