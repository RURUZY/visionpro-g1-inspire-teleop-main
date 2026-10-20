#!/usr/bin/env python3
"""Run a trained LeRobot policy (ACT) on the G1 + Inspire through the Unitree SDK topics.

The same program drives the MuJoCo stand-in (sdk_sim/g1_sdk_sim.py, DDS domain 1, default)
and, later, the real robot (domain 0). It only talks to SDK topics:
  reads   rt/lowstate (right arm q), rt/inspire_hand/state/r (finger angles), head image (ZMQ)
  writes  rt/arm_sdk (default; weight-blended over Unitree's own controller) or rt/lowcmd
          (--mode debug, robot must be held by a stand), and rt/inspire_hand/ctrl/r

Action = 13-D: 7 right-arm joint targets (hardware order 22-28) + 6 Inspire channels as joint
angles (converted to angle_set 0..1000). Observation = the same 13 joints + head image, as in the
simulated training data.

Safety layer between the policy and the robot:
  - hand-over ramp: arm_sdk weight 0 -> 1 over --ramp-s while holding the current pose; at the
    end weight 1 -> 0 (Unitree's controller takes the arms back) - no snap at start or stop
  - per-joint speed limit (--max-joint-speed rad/s) and joint-limit clamp on every 250 Hz command
  - watchdog: stale state (>0.1 s) or image (>0.3 s), or a non-finite action -> hold the pose
  - only the right arm and hand move; waist and left arm hold their start pose

In simulation, --trials N runs N episodes (reset and scoring through the simulator's control
port) and writes trials.jsonl / summary.json.

  .venv/bin/python sdk_deploy/run_policy_sdk.py --checkpoint CKPT --trials 10 --output runs/sdk_eval
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for sub in ('eval', 'wipe_teacher', 'pickup_teacher', 'tactile_slip', 'sdk_sim'):
    sys.path.insert(0, str(ROOT / sub))
sys.path.insert(0, os.environ.get('INSPIRE_SDKPY', '/home/epcap/projects/xr_deps/inspire_hand_sdk/inspire_sdkpy'))

RIGHT_ARM = list(range(22, 29))
ARM_SDK_MOTORS = list(range(12, 29))
KP = {'arm': 80.0, 'wrist': 40.0}
KD = {'arm': 3.0, 'wrist': 1.5}
CMD_HZ, POLICY_HZ = 250.0, 20.0
STATE_MAX_AGE, IMAGE_MAX_AGE = 0.1, 0.3


def joint_limits():
    """G1 arm and Inspire channel joint ranges from the MuJoCo robot model."""
    import mujoco
    import sponge_demo
    from g1_sdk_sim import G1_MOTORS, HAND_CHANNELS
    m = mujoco.MjModel.from_xml_path(str(sponge_demo.ROBOT))
    body = np.array([m.jnt_range[m.joint(f'{n}_joint').id] for n in G1_MOTORS])
    hand = np.array([m.jnt_range[m.joint(f'right_hand_{c}_joint').id] for c in HAND_CHANNELS])
    return body, hand


class Robot:
    """SDK side: latest state, image, and a 250 Hz arm command loop with the safety layer."""

    def __init__(self, args):
        import zmq
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
        from unitree_sdk2py.utils.crc import CRC
        import inspire_dds
        from g1_sdk_sim import hand_angles_to_q, hand_q_to_angles
        self.inspire_dds, self.to_q, self.to_angles = inspire_dds, hand_angles_to_q, hand_q_to_angles
        self.args = args
        self.body_lim, self.hand_lim = joint_limits()
        self.lock = threading.Lock()
        self.q = self.q_time = None
        self.hand_angles = self.hand_time = None
        self.frame = self.frame_time = None
        ChannelFactoryInitialize(args.domain, args.iface)
        ChannelSubscriber('rt/lowstate', LowState_).Init(self._on_state, 10)
        ChannelSubscriber('rt/inspire_hand/state/r', inspire_dds.inspire_hand_state).Init(self._on_hand, 10)
        topic = 'rt/arm_sdk' if args.mode == 'arm_sdk' else 'rt/lowcmd'
        self.cmd_pub = ChannelPublisher(topic, LowCmd_); self.cmd_pub.Init()
        self.hand_pub = ChannelPublisher('rt/inspire_hand/ctrl/r', inspire_dds.inspire_hand_ctrl); self.hand_pub.Init()
        self.cmd, self.crc = unitree_hg_msg_dds__LowCmd_(), CRC()
        self.zmq = zmq.Context()
        self.img_sock = self.zmq.socket(zmq.SUB)
        self.img_sock.setsockopt(zmq.RCVHWM, 2)  # CONFLATE does not support multipart frames
        self.img_sock.setsockopt(zmq.SUBSCRIBE, b'')
        self.img_sock.connect(args.image_endpoint)
        # command state
        self.active = False
        self.weight = 0.0
        self.weight_goal = 0.0
        self.hold = None           # 29 joint targets held for non-policy joints
        self.cmd_q = None          # last commanded 29 joint targets
        self.goal_q = None         # right-arm goal from the policy (7)
        self.stats = {'held_stale': 0, 'speed_clipped': 0, 'limit_clipped': 0}
        threading.Thread(target=self._image_loop, daemon=True).start()
        threading.Thread(target=self._command_loop, daemon=True).start()

    # --- inputs ---
    def _on_state(self, msg):
        q = np.array([msg.motor_state[i].q for i in range(29)])
        with self.lock:
            self.q, self.q_time = q, time.monotonic()

    def _on_hand(self, msg):
        with self.lock:
            self.hand_angles, self.hand_time = np.array(msg.angle_act, float), time.monotonic()

    def _image_loop(self):
        while True:
            parts = self.img_sock.recv_multipart()
            header = json.loads(parts[1])
            frame = np.frombuffer(parts[2], np.uint8).reshape(header['shape'])
            with self.lock:
                self.frame, self.frame_time = frame, time.monotonic()

    def observation(self):
        """(state13, image) or None if anything is stale."""
        now = time.monotonic()
        with self.lock:
            if (self.q is None or self.hand_angles is None or self.frame is None
                    or now - self.q_time > STATE_MAX_AGE or now - self.hand_time > STATE_MAX_AGE * 3
                    or now - self.frame_time > IMAGE_MAX_AGE):
                return None
            hand_q = self.to_q(self.hand_angles, self.hand_lim[:, 0], self.hand_lim[:, 1])
            return np.r_[self.q[RIGHT_ARM], hand_q].astype(np.float32), self.frame.copy()

    def wait_state(self, timeout=5.0):
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            if self.observation() is not None:
                return True
            time.sleep(0.02)
        return False

    # --- outputs ---
    def engage(self):
        """Take the arms over at the current pose (arm_sdk weight ramps up in the command loop)."""
        with self.lock:
            self.hold = self.q.copy()
            self.cmd_q = self.q.copy()
            self.goal_q = self.q[RIGHT_ARM].copy()
            self.weight, self.weight_goal = 0.0, 1.0
            self.active = True

    def release(self, wait=True):
        with self.lock:
            self.weight_goal = 0.0
            if self.args.mode == 'debug' and self.hold is not None:
                self.goal_q = self.hold[RIGHT_ARM].copy()   # debug mode: return to the start pose
        if wait:
            time.sleep(self.args.ramp_s + 0.2)
        with self.lock:
            self.active = self.args.mode == 'debug'

    def set_goal(self, arm7, hand6_q):
        with self.lock:
            if self.active:
                self.goal_q = np.asarray(arm7, float).copy()
        angles = self.to_angles(hand6_q, self.hand_lim[:, 0], self.hand_lim[:, 1])
        self.hand_pub.Write(self.inspire_dds.inspire_hand_ctrl(
            pos_set=[0] * 6, angle_set=[int(a) for a in angles], force_set=[0] * 6, speed_set=[0] * 6,
            mode=0b0001))

    def hold_pose(self):
        """Watchdog: keep the right arm where it is."""
        with self.lock:
            if self.active and self.q is not None:
                self.goal_q = self.q[RIGHT_ARM].copy()
                self.stats['held_stale'] += 1

    def _command_loop(self):
        dt = 1.0 / CMD_HZ
        ramp_rate = dt / max(self.args.ramp_s, 1e-3)
        max_step = self.args.max_joint_speed * dt
        while True:
            t0 = time.monotonic()
            with self.lock:
                if self.active and self.cmd_q is not None:
                    self.weight += float(np.clip(self.weight_goal - self.weight, -ramp_rate, ramp_rate))
                    q = self.cmd_q.copy()
                    goal = np.clip(self.goal_q, self.body_lim[RIGHT_ARM, 0], self.body_lim[RIGHT_ARM, 1])
                    self.stats['limit_clipped'] += int(np.any(goal != self.goal_q))
                    step = goal - q[RIGHT_ARM]
                    clipped = np.clip(step, -max_step, max_step)
                    self.stats['speed_clipped'] += int(np.any(clipped != step))
                    q[RIGHT_ARM] += clipped
                    others = [i for i in ARM_SDK_MOTORS if i not in RIGHT_ARM]
                    q[others] = self.hold[others]
                    self.cmd_q = q
                    self._write(q, self.weight)
            time.sleep(max(0.0, dt - (time.monotonic() - t0)))

    def _write(self, q, weight):
        cmd = self.cmd
        motors = ARM_SDK_MOTORS
        for i in motors:
            wrist = i in (19, 20, 21, 26, 27, 28)
            mc = cmd.motor_cmd[i]
            mc.mode, mc.q, mc.dq, mc.tau = 1, float(q[i]), 0.0, 0.0
            mc.kp = KP['wrist' if wrist else 'arm']
            mc.kd = KD['wrist' if wrist else 'arm']
        if self.args.mode == 'arm_sdk':
            cmd.motor_cmd[29].q = float(weight)
        cmd.crc = self.crc.Crc(cmd)
        self.cmd_pub.Write(cmd)


class AsyncChunkPolicy:
    """Predict whole action chunks in a background thread while the robot executes the last one.

    ACT on the Jetson CPU needs ~200 ms per chunk, longer than one 50 ms control step. A chunk is
    predicted from the observation at step s0; when it arrives at step s1, its first s1 - s0
    actions (already in the past) are skipped and the rest replaces the queue. The worker then
    replans immediately, so the robot keeps 20 Hz without pausing.
    """

    def __init__(self, checkpoint_policy, observe):
        import torch
        self.cp, self.observe, self.torch = checkpoint_policy, observe, torch
        self.lock = threading.Lock()
        self.queue, self.step, self.task = [], 0, ''
        self.running, self.generation = False, 0
        self.latencies = []

    def start(self, task):
        with self.lock:
            self.queue, self.step, self.task, self.latencies = [], 0, task, []
            self.generation += 1
            generation = self.generation
        self.running = True
        threading.Thread(target=self._worker, args=(generation,), daemon=True).start()

    def stop(self):
        self.running = False

    def _infer(self, state, image):
        torch = self.torch
        batch = {'task': self.task,
                 'observation.state': torch.from_numpy(state),
                 'observation.images.head': torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float() / 255.}
        with torch.inference_mode():
            chunk = self.cp.post(self.cp.policy.predict_action_chunk(self.cp.pre(batch)))
        return chunk[0].cpu().numpy().astype(np.float32)

    def _worker(self, generation):
        while self.running and generation == self.generation:
            obs = self.observe()
            if obs is None:
                time.sleep(0.01)
                continue
            with self.lock:
                s0 = self.step
            t0 = time.perf_counter()
            chunk = self._infer(*obs)
            self.latencies.append(time.perf_counter() - t0)
            with self.lock:
                if generation == self.generation:   # a late chunk from a finished episode is dropped
                    self.queue = list(chunk[self.step - s0:])

    def next_action(self):
        """Called once per control step; None while no prediction is available."""
        with self.lock:
            self.step += 1
            return self.queue.pop(0) if self.queue else None


def control_call(sock, req):
    sock.send(json.dumps(req).encode())
    return json.loads(sock.recv())


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(max(0, c - h), 3), round(min(1, c + h), 3)]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--mode', choices=['arm_sdk', 'debug'], default='arm_sdk')
    p.add_argument('--domain', type=int, default=1, help='DDS domain: 1 = simulator (default), 0 = real robot')
    p.add_argument('--iface', default='lo')
    p.add_argument('--image-endpoint', default='tcp://127.0.0.1:5560')
    p.add_argument('--control-endpoint', default='tcp://127.0.0.1:5561', help='simulator only')
    p.add_argument('--side', choices=['left', 'right'], default='left')
    p.add_argument('--seed', type=int, default=2000)
    p.add_argument('--trials', type=int, default=1)
    p.add_argument('--seconds', type=float, default=18.0)
    p.add_argument('--ramp-s', type=float, default=2.0)
    p.add_argument('--max-joint-speed', type=float, default=1.5, help='rad/s per joint')
    p.add_argument('--sync', action='store_true', help='blocking select_action per step (no async chunks)')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.domain == 0:
        raise SystemExit('Real-robot use (domain 0) is disabled until the safety checklist is done.')
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    import zmq
    import eval_policy as E

    args.output.mkdir(parents=True, exist_ok=True)
    policy = E.CheckpointPolicy(args.checkpoint, args.device)
    robot = Robot(args)
    runner = AsyncChunkPolicy(policy, robot.observation)
    ctl = robot.zmq.socket(zmq.REQ)
    ctl.connect(args.control_endpoint)
    task = E.W.TASKS[args.side]
    results = []
    with (args.output / 'trials.jsonl').open('w') as log:
        for i in range(args.trials):
            robot.release(wait=False)
            with robot.lock:
                robot.active = False
            control_call(ctl, {'cmd': 'reset', 'seed': args.seed + i, 'side': args.side})
            time.sleep(0.5)
            if not robot.wait_state():
                raise SystemExit('no robot state / image from the simulator')
            policy.start(None, task)
            robot.stats = {k: 0 for k in robot.stats}      # per-episode safety-layer counters
            robot.engage()
            time.sleep(args.ramp_s)                       # hand-over at the current pose
            if not args.sync:
                runner.start(task)
            t0, steps, infer, empty = time.monotonic(), 0, [], 0
            while time.monotonic() - t0 < args.seconds:
                tick = time.monotonic()
                obs = robot.observation()
                if obs is None:
                    robot.hold_pose()
                else:
                    if args.sync:
                        ti = time.perf_counter()
                        action = policy.act({'observation.state': obs[0], 'observation.images.head': obs[1]})
                        infer.append(time.perf_counter() - ti)
                    else:
                        action = runner.next_action()
                        empty += action is None
                    if action is None or not np.all(np.isfinite(action)):
                        robot.hold_pose()
                    else:
                        robot.set_goal(action[:7], action[7:13])
                    steps += 1
                time.sleep(max(0.0, 1 / POLICY_HZ - (time.monotonic() - tick)))
            if not args.sync:
                runner.stop()
                infer = runner.latencies
            m = control_call(ctl, {'cmd': 'metrics'})
            ok = m['target_coverage'] >= E.W.SUCCESS['target_coverage'] and \
                m['other_coverage'] <= E.W.SUCCESS['other_coverage'] and m['tool_in_hand']
            r = {'trial': i, 'seed': args.seed + i, 'success': bool(ok), **{k: v for k, v in m.items() if k != 'ok'},
                 'policy_steps': steps, 'steps_without_action': empty, 'inference_ms_mean': round(1000 * float(np.mean(infer)), 1) if infer else None,
                 **robot.stats}
            results.append(r)
            log.write(json.dumps(r) + '\n'); log.flush()
            print(f"[{i}] success={ok} coverage={m['target_coverage']:.2f} other={m['other_coverage']:.2f} "
                  f"in_hand={m['tool_in_hand']} steps={steps} stats={robot.stats}", flush=True)
            robot.release()
    k = sum(r['success'] for r in results)
    summary = {'checkpoint': args.checkpoint, 'mode': args.mode, 'trials': len(results), 'successes': k,
               'success_rate_wilson95': wilson(k, len(results)), 'max_joint_speed': args.max_joint_speed,
               'scope': 'Policy driven through Unitree SDK topics against the MuJoCo stand-in (sdk_sim).'}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
