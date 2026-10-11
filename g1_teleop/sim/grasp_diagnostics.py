"""Read-only diagnostics after a solved MuJoCo step; no actuation or state edits."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import queue
import threading
import time

import mujoco
import numpy as np
import zmq

from g1_teleop.sim.inspire_hand import CHANNEL_JOINTS


class GraspRecorder:
    def __init__(self, model, controller, *, output_dir=None,
                 endpoint="tcp://127.0.0.1:5558", rate=50):
        self.model = model
        self.controller = controller
        self.rate = float(rate)
        if not 1 <= self.rate <= 200:
            raise ValueError("diagnostic sample rate must be 1..200 Hz")
        root = Path(__file__).resolve().parents[2] / "recordings"
        self.path = Path(output_dir) if output_dir else root / ("grasp_diagnostics_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
        self.path.mkdir(parents=True, exist_ok=False)
        self.path = self.path.resolve()
        self.model_path = self.path / "model.mjb"
        mujoco.mj_saveModel(model, str(self.model_path))
        self.seq = 0
        self.episode = 0
        self.dropped = 0
        self.saved = 0
        self.error = None
        self.closed = False
        self.next_time = 0.
        self.last_time = -1.
        self.cube = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pick_cube_body")
        self.table = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table_top")
        self.wrists = {s: model.body(f"{s}_wrist_yaw_link").id for s in ("left", "right")}
        self.geom_side = {}
        for geom in range(model.ngeom):
            body = int(model.geom_bodyid[geom])
            while body:
                for side, wrist in self.wrists.items():
                    if body == wrist:
                        self.geom_side[geom] = side
                body = int(model.body_parentid[body])
        self.socket = None
        self.context = None
        if endpoint:
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.PUB)
            self.socket.setsockopt(zmq.SNDHWM, 2)
            self.socket.setsockopt(zmq.LINGER, 0)
            self.socket.bind(endpoint)
        self.metadata = {
            "schema": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
            "mujoco_version": mujoco.__version__, "model_sha256": hashlib.sha256(self.model_path.read_bytes()).hexdigest(),
            "timestep_s": float(model.opt.timestep), "integrator": int(model.opt.integrator),
            "sample_rate_hz": rate, "endpoint": endpoint,
            "channels": list(CHANNEL_JOINTS),
            "controller": {"kp": 2., "kd": .08, "torque_limit_nm": 1.4, "target_slew_rad_s": 3.},
            "timing": "qpos/qvel and control angles are pre-step; forces are the solver result of that step. Euler scene: same evaluation pose. No extra mj_forward is run in the live simulator.",
            "limits": "Sampled diagnostics, not every contact event. Contact force is simulation output, not hardware measurement. Network preview may skip frames. Snapshot replay is visual only, not a re-run of dynamics.",
        }
        (self.path / "metadata.json").write_text(json.dumps(self.metadata, indent=2))
        self.items = queue.Queue(maxsize=256)
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._write, daemon=True)
        self.thread.start()
        controller.trace_enabled = True
        print(f"[grasp_diagnostics] recording {self.path} | preview {endpoint}", flush=True)

    def begin_step(self, data):
        if self.closed:
            return None
        t = float(data.time)
        if t < self.last_time:
            self.next_time = t
        self.last_time = t
        if t + 1e-9 < self.next_time:
            return None
        self.next_time = t + 1. / self.rate
        return {"sim_time": t, "qpos": data.qpos.copy().tolist(), "qvel": data.qvel.copy().tolist()}

    def snapshot(self, data, before, elastic_band=False):
        now = time.monotonic()
        contacts = []
        normal_sum = {"left": 0., "right": 0.}
        table_contact = False
        net_cube_force = np.zeros(3)
        for index in range(data.ncon):
            c = data.contact[index]
            g1, g2 = int(c.geom1), int(c.geom2)
            b1, b2 = int(self.model.geom_bodyid[g1]), int(self.model.geom_bodyid[g2])
            if self.cube < 0 or self.cube not in (b1, b2):
                continue
            other = g2 if b1 == self.cube else g1
            side = self.geom_side.get(other)
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, data, index, force)
            # Contact frame x-axis points from geom1 to geom2: force is on geom2.
            world = np.asarray(c.frame).reshape(3, 3).T @ force[:3]
            if b1 == self.cube:
                world *= -1
            net_cube_force += world
            table_contact |= other == self.table and force[0] > 0
            if side:
                normal_sum[side] += max(float(force[0]), 0.)
            contacts.append({
                "geom1": g1, "geom2": g2,
                "other_name": self.model.geom(other).name or self.model.body(int(self.model.geom_bodyid[other])).name,
                "side": side, "position": c.pos.copy().tolist(), "distance_m": float(c.dist),
                "normal_n": float(force[0]), "tangent_n": float(np.linalg.norm(force[1:3])),
                "force_on_cube_world_n": world.tolist(),
            })
        hands = {}
        for side, active in self.controller.active.items():
            hands[side] = dict(self.controller.trace.get(side, {}))
            dof = self.model.jnt_dofadr[active]
            hands[side]["actuator_torque_nm"] = data.qfrc_actuator[dof].copy().tolist()
            hands[side]["normal_contact_sum_n"] = normal_sum[side]
        cube = None
        if self.cube >= 0:
            pos = data.xpos[self.cube].copy()
            table_z = (float(data.geom_xpos[self.table, 2] + self.model.geom_size[self.table, 2])
                       if self.table >= 0 else None)
            cube_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "pick_cube")
            bottom = None
            if cube_geom >= 0:
                rotation = data.geom_xmat[cube_geom].reshape(3, 3)
                bottom = float(data.geom_xpos[cube_geom, 2] - np.abs(rotation[2]) @ self.model.geom_size[cube_geom])
            cube = {"position": pos.tolist(), "table_contact": bool(table_contact),
                    "bottom_clearance_m": bottom - table_z if bottom is not None and table_z is not None else None,
                    "contact_force_world_n": net_cube_force.tolist(),
                    "relative_wrist": {s: (data.xmat[w].reshape(3, 3).T @ (pos - data.xpos[w])).tolist()
                                       for s, w in self.wrists.items()}}
        age = None if self.controller.last_received is None else max(0., now - self.controller.last_received)
        return {**before, "seq": self.seq, "episode": self.episode, "wall_time": time.time(),
                "monotonic_time": now, "post_step_time": float(data.time),
                "ctrl": data.ctrl.copy().tolist(), "hands": hands, "cube": cube, "contacts": contacts,
                "command_age_s": age, "communication_timeout": bool(self.controller.timed_out),
                "elastic_band": bool(elastic_band), "recording_drops": self.dropped,
                "recording_error": self.error, "session_dir": str(self.path), "model_path": str(self.model_path)}

    def end_step(self, data, before, elastic_band=False):
        if self.closed:
            return None
        sample = self.snapshot(data, before, elastic_band)
        self.seq += 1
        payload = json.dumps(sample, separators=(",", ":"), allow_nan=False)
        if self.socket is not None:
            try:
                self.socket.send_string(payload, zmq.NOBLOCK)
            except zmq.Again:
                pass
        try:
            self.items.put_nowait(payload)
        except queue.Full:
            self.dropped += 1
        return sample

    def _write(self):
        try:
            with (self.path / "samples.jsonl").open("w", buffering=65536) as stream:
                last_flush = time.monotonic()
                while not self.stop.is_set() or not self.items.empty():
                    try:
                        line = self.items.get(timeout=.1)
                    except queue.Empty:
                        continue
                    stream.write(line + "\n")
                    self.saved += 1
                    if time.monotonic() - last_flush > 1:
                        stream.flush()
                        last_flush = time.monotonic()
        except Exception as exc:
            self.error = repr(exc)
            print(f"[grasp_diagnostics] recording failed: {exc}", flush=True)

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.stop.set()
        self.thread.join(timeout=5)
        if self.socket is not None:
            self.socket.close(0)
            self.context.term()
        (self.path / "recording_status.json").write_text(json.dumps({
            "samples_produced": self.seq, "samples_saved": self.saved, "dropped": self.dropped,
            "writer_finished": not self.thread.is_alive(), "error": self.error}, indent=2))
