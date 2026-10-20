#!/usr/bin/env python3
"""MuJoCo stand-in for the real G1 + Inspire SDK interface (Unitree DDS), on the wiping scene.

Programs written against the robot's SDK topics (xr_teleoperate, sdk_deploy/run_policy_sdk.py)
can run unchanged against this simulator, then against the real robot:

  subscribes  rt/lowcmd             debug-mode joint commands (motors 0-28, hardware order)
              rt/arm_sdk            motion-mode arm commands, blended by weight = motor_cmd[29].q
              rt/inspire_hand/ctrl/{l,r}   Inspire angle_set (0 closed .. 1000 open)
  publishes   rt/lowstate           motor q/dq/tau_est (500 Hz)
              rt/inspire_hand/state/{l,r}  angle_act (100 Hz)
              rt/inspire_hand/touch/r      17 tactile groups from the simulated pads (50 Hz)
  ZMQ         tcp://*:5560          head camera frames (stands in for the robot's image service)
              tcp://*:5561          control: reset an episode, read task metrics

Approximations (simulation only): the pelvis is fixed (like a robot held in a stand) and links
are gravity-compensated, so commanded kp/kd/tau are not applied; each commanded joint goes to its
position actuator target. Tactile values are the simulated pad normal force (0.01 N units) spread
evenly over the group's taxels, not a calibrated sensor model.
DDS domain 1 by default (the real robot uses 0), so this cannot drive a robot by accident.

  .venv/bin/python sdk_sim/g1_sdk_sim.py --task pickup [--viewer] [--perturb light,color,camera,physics]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for sub in ('eval', 'wipe_teacher', 'pickup_teacher', 'tactile_slip'):
    sys.path.insert(0, str(ROOT / sub))
INSPIRE_SDK = os.environ.get('INSPIRE_SDKPY', '/home/epcap/projects/xr_deps/inspire_hand_sdk/inspire_sdkpy')
sys.path.insert(0, INSPIRE_SDK)  # the inspire_dds message package only (no Qt GUI imports)

# 29 body motors in Unitree hardware order (rt/lowcmd and rt/lowstate indices 0-28).
G1_MOTORS = (
    [f'{s}_{j}' for s in ('left', 'right') for j in
     ('hip_pitch', 'hip_roll', 'hip_yaw', 'knee', 'ankle_pitch', 'ankle_roll')]
    + ['waist_yaw', 'waist_roll', 'waist_pitch']
    + [f'{s}_{j}' for s in ('left', 'right') for j in
       ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow', 'wrist_roll', 'wrist_pitch', 'wrist_yaw')]
)
ARM_SDK_MOTORS = range(12, 29)  # waist + both arms, as rt/arm_sdk controls them
ARM_SDK_WEIGHT = 29              # motor_cmd[29].q carries the arm_sdk blend weight
HAND_CHANNELS = ('little_1', 'ring_1', 'middle_1', 'index_1', 'thumb_2', 'thumb_1')
# rt/inspire_hand/touch field per simulated pad (tactile.TACTILE_BODIES order).
TOUCH_FIELDS = (
    [f'finger{n}_{part}_touch' for n in ('one', 'two', 'three', 'four') for part in ('palm', 'top', 'tip')]
    + ['fingerfive_palm_touch', 'fingerfive_middle_touch', 'fingerfive_top_touch', 'fingerfive_tip_touch',
       'palm_touch']
)
CMD_TIMEOUT_S = 0.5


def hand_angles_to_q(angles, lo, hi):
    """Inspire convention: 1000 = open (joint lower limit), 0 = closed (upper limit)."""
    return lo + (1.0 - np.clip(np.asarray(angles, float), 0, 1000) / 1000.0) * (hi - lo)


def hand_q_to_angles(q, lo, hi):
    return np.clip(np.round(1000.0 * (1.0 - (np.asarray(q, float) - lo) / (hi - lo))), 0, 1000).astype(int)


class SdkSim:
    def __init__(self, args):
        import eval_policy as E
        self.E, self.W = E, E.W
        self.args = args
        self.t = E.make_teacher(args.task if args.task == 'pickup' else 'handle', Path(args.scene_dir))
        self.pickup = args.task == 'pickup'
        self.perturber = E.Perturber(self.t, args.perturb.split(','), args.seed + 7919) if args.perturb else None
        m = self.t.model
        names = [m.joint(j).name for j in m.actuator_trnid[:, 0]]
        self.motor_act = np.array([names.index(f'{n}_joint') for n in G1_MOTORS])
        self.motor_qadr = np.array([m.joint(f'{n}_joint').qposadr[0] for n in G1_MOTORS])
        self.motor_vadr = np.array([m.joint(f'{n}_joint').dofadr[0] for n in G1_MOTORS])
        self.hand = {}
        for side in ('l', 'r'):
            full = 'left' if side == 'l' else 'right'
            joints = [m.joint(f'{full}_hand_{c}_joint') for c in HAND_CHANNELS]
            self.hand[side] = dict(
                act=np.array([names.index(j.name) for j in joints]),
                qadr=np.array([j.qposadr[0] for j in joints]),
                lo=m.jnt_range[[j.id for j in joints], 0], hi=m.jnt_range[[j.id for j in joints], 1])
        self.lock = threading.Lock()
        self.rng = np.random.default_rng(args.seed)
        self.side = args.side
        self.reset_episode(args.seed, args.side)

    # ---- episodes -------------------------------------------------------------------------
    def reset_episode(self, seed, side):
        t, W = self.t, self.W
        rng = np.random.default_rng(seed)
        with self.lock:
            shift = self.perturber.apply() if self.perturber else None
            if self.pickup:
                center, plan = t.begin_episode(rng, side, region_offset=shift)
            else:
                offset = t.place_regions(rng)
                if shift is not None:
                    offset = shift
                    for s, site in t.region_site.items():
                        t.model.site_pos[site][:2] = W.REGIONS[s] + offset
                        t.grids[s] = t.grid + t.model.site_pos[site][:2]
                center = W.REGIONS[side] + offset
                t.reset(W.plan_episode(rng, center, .07)(0))
            self.side = side
            self.targets = t.data.ctrl.copy()
            self.hold_arm = self.targets[self.motor_act].copy()   # what the robot holds without arm_sdk
            self.lowcmd = None                                    # (time, q[29], mask[29])
            self.armsdk = None                                    # (time, q[29], weight)
            self.covered = {s: np.zeros(len(c), bool) for s, c in t.grids.items()}
            self.rel0 = None
            self.episode_start = time.monotonic()
            self.sim_time0 = t.data.time
        print(f'[sdk_sim] reset: seed {seed}, task {self.args.task}, side {side}', flush=True)

    def metrics(self):
        t = self.t
        with self.lock:
            other = 'left' if self.side == 'right' else 'right'
            in_hand = len(t.hand_tool_contacts()) >= 2
            lifted = bool(t.data.xpos[t.tool][2] > self.W.TABLE_Z + .15)
            return {'side': self.side,
                    'target_coverage': round(float(self.covered[self.side].mean()), 4),
                    'other_coverage': round(float(self.covered[other].mean()), 4),
                    'tool_in_hand': in_hand, 'tool_lifted': lifted,
                    'sim_time': round(float(t.data.time - self.sim_time0), 3)}

    # ---- DDS callbacks --------------------------------------------------------------------
    def on_lowcmd(self, msg):
        q = np.array([msg.motor_cmd[i].q for i in range(29)])
        mask = np.array([msg.motor_cmd[i].kp > 0 or msg.motor_cmd[i].mode != 0 for i in range(29)])
        with self.lock:
            self.lowcmd = (time.monotonic(), q, mask)

    def on_armsdk(self, msg):
        q = np.array([msg.motor_cmd[i].q for i in range(29)])
        w = float(np.clip(msg.motor_cmd[ARM_SDK_WEIGHT].q, 0.0, 1.0))
        with self.lock:
            self.armsdk = (time.monotonic(), q, w)

    def on_hand(self, side, msg):
        h = self.hand[side]
        q = hand_angles_to_q(list(msg.angle_set), h['lo'], h['hi'])
        with self.lock:
            self.targets[h['act']] = q

    # ---- stepping -------------------------------------------------------------------------
    def apply_commands(self, now):
        targets = self.targets
        if self.lowcmd is not None and now - self.lowcmd[0] < CMD_TIMEOUT_S:
            _, q, mask = self.lowcmd
            act = self.motor_act[mask]
            targets[act] = q[mask]
        if self.armsdk is not None and now - self.armsdk[0] < CMD_TIMEOUT_S:
            _, q, w = self.armsdk
            idx = np.array(list(ARM_SDK_MOTORS))
            hold = self.hold_arm[idx]
            targets[self.motor_act[idx]] = w * q[idx] + (1 - w) * hold
        self.t.data.ctrl[:] = targets

    def step(self, n):
        mj, m, d = self.t.mj, self.t.model, self.t.data
        with self.lock:
            self.apply_commands(time.monotonic())
            for _ in range(n):
                mj.mj_step(m, d)
            self.t.mark_coverage(self.covered)

    def fill_lowstate(self, msg, tick):
        d = self.t.data
        with self.lock:
            q, dq = d.qpos[self.motor_qadr], d.qvel[self.motor_vadr]
            tau = d.actuator_force[self.motor_act]
        for i in range(29):
            ms = msg.motor_state[i]
            ms.q, ms.dq, ms.tau_est = float(q[i]), float(dq[i]), float(tau[i])
        msg.imu_state.quaternion = [1.0, 0.0, 0.0, 0.0]
        msg.tick = tick
        msg.mode_machine = 5

    def hand_angles(self, side):
        h = self.hand[side]
        with self.lock:
            return hand_q_to_angles(self.t.data.qpos[h['qadr']], h['lo'], h['hi'])

    def touch_values(self):
        with self.lock:
            return self.t.tactile()  # newtons per pad, right hand

    def image(self):
        with self.lock:
            return self.t.image()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--task', choices=['pickup', 'pregrasp'], default='pickup')
    p.add_argument('--side', choices=['left', 'right'], default='left')
    p.add_argument('--seed', type=int, default=2000)
    p.add_argument('--perturb', default='', help='light,color,camera,physics (randomized each reset)')
    p.add_argument('--domain', type=int, default=1, help='DDS domain (real robot: 0)')
    p.add_argument('--iface', default='lo')
    p.add_argument('--image-port', type=int, default=5560)
    p.add_argument('--control-port', type=int, default=5561)
    p.add_argument('--image-hz', type=float, default=20.)
    p.add_argument('--viewer', action='store_true', help='MuJoCo window (needs a display)')
    p.add_argument('--scene-dir', default=str(ROOT / 'sdk_sim' / 'scene'))
    args = p.parse_args()
    os.environ.setdefault('MUJOCO_GL', 'egl')
    os.environ.setdefault('HF_HUB_OFFLINE', '1')

    import zmq
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
    from unitree_sdk2py.utils.crc import CRC
    import inspire_dds

    sim = SdkSim(args)
    ChannelFactoryInitialize(args.domain, args.iface)
    ChannelSubscriber('rt/lowcmd', LowCmd_).Init(sim.on_lowcmd, 10)
    ChannelSubscriber('rt/arm_sdk', LowCmd_).Init(sim.on_armsdk, 10)
    for side in ('l', 'r'):
        ChannelSubscriber(f'rt/inspire_hand/ctrl/{side}', inspire_dds.inspire_hand_ctrl).Init(
            lambda msg, s=side: sim.on_hand(s, msg), 10)
    low_pub = ChannelPublisher('rt/lowstate', LowState_); low_pub.Init()
    hand_pub = {s: ChannelPublisher(f'rt/inspire_hand/state/{s}', inspire_dds.inspire_hand_state) for s in 'lr'}
    for pub in hand_pub.values():
        pub.Init()
    touch_pub = ChannelPublisher('rt/inspire_hand/touch/r', inspire_dds.inspire_hand_touch); touch_pub.Init()

    ctx = zmq.Context()
    img_sock = ctx.socket(zmq.PUB); img_sock.setsockopt(zmq.SNDHWM, 2); img_sock.bind(f'tcp://*:{args.image_port}')
    ctl_sock = ctx.socket(zmq.REP); ctl_sock.bind(f'tcp://*:{args.control_port}')

    viewer = None
    if args.viewer:
        import mujoco.viewer
        import sponge_demo
        sponge_demo.configure_viewer_display(None)
        viewer = mujoco.viewer.launch_passive(sim.t.model, sim.t.data)

    crc, low = CRC(), unitree_hg_msg_dds__LowState_()
    dt = sim.t.model.opt.timestep
    steps_per_tick = 2                       # lowstate at 250 Hz of simulated time
    tick, next_img, next_hand, next_touch = 0, 0., 0., 0.
    wall0, sim0, last_report = time.monotonic(), sim.t.data.time, time.monotonic()
    print(f'[sdk_sim] DDS domain {args.domain} on {args.iface}; images tcp://*:{args.image_port}; '
          f'control tcp://*:{args.control_port}', flush=True)
    try:
        while viewer is None or viewer.is_running():
            sim.step(steps_per_tick)
            tick += 1
            sim.fill_lowstate(low, tick)
            low.crc = crc.Crc(low)
            low_pub.Write(low)
            st = sim.t.data.time
            if st >= next_hand:
                next_hand = st + .01
                for s in 'lr':
                    hs = inspire_dds.inspire_hand_state(
                        pos_act=[0] * 6, angle_act=[int(a) for a in sim.hand_angles(s)], force_act=[0] * 6,
                        current=[0] * 6, err=[0] * 6, status=[0] * 6, temperature=[0] * 6)
                    hand_pub[s].Write(hs)
            if st >= next_touch:
                next_touch = st + .02
                pads = sim.touch_values()
                touch_pub.Write(inspire_dds.inspire_hand_touch(**{
                    f: [int(round(100 * float(force) / TOUCH_SIZES[f]))] * TOUCH_SIZES[f]
                    for f, force in zip(TOUCH_FIELDS, pads)}))
            if st >= next_img:
                next_img = st + 1 / args.image_hz
                frame = sim.image()
                header = json.dumps({'sim_time': st, 'shape': frame.shape}).encode()
                img_sock.send_multipart([b'head', header, frame.tobytes()])
                if viewer is not None:
                    viewer.sync()
            while ctl_sock.poll(0):
                req = json.loads(ctl_sock.recv())
                if req.get('cmd') == 'reset':
                    sim.reset_episode(int(req.get('seed', args.seed)), req.get('side', sim.side))
                    next_img = next_hand = next_touch = sim.t.data.time
                    # The reset zeroes simulated time: re-anchor real-time pacing to it.
                    wall0, sim0 = time.monotonic(), sim.t.data.time
                    reply = {'ok': True}
                elif req.get('cmd') == 'metrics':
                    reply = {'ok': True, **sim.metrics()}
                else:
                    reply = {'ok': False, 'error': f"unknown cmd {req.get('cmd')}"}
                ctl_sock.send(json.dumps(reply).encode())
            # Pace simulated time to wall time (the sim may also fall behind on a busy machine).
            ahead = (sim.t.data.time - sim0) - (time.monotonic() - wall0)
            if ahead > 0:
                time.sleep(ahead)
            if time.monotonic() - last_report > 10:
                last_report = time.monotonic()
                rtf = (sim.t.data.time - sim0) / (time.monotonic() - wall0)
                print(f'[sdk_sim] real-time factor {rtf:.2f}', flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        if viewer is not None:
            viewer.close()


TOUCH_SIZES = {
    **{f'finger{n}_tip_touch': 9 for n in ('one', 'two', 'three', 'four', 'five')},
    **{f'finger{n}_top_touch': 96 for n in ('one', 'two', 'three', 'four', 'five')},
    **{f'finger{n}_palm_touch': 80 for n in ('one', 'two', 'three', 'four')},
    'fingerfive_middle_touch': 9, 'fingerfive_palm_touch': 96, 'palm_touch': 112,
}

if __name__ == '__main__':
    main()
