"""Read-only, sampled MuJoCo contact diagnostics for the dry paint roller task."""

import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import mujoco
import numpy as np


class PaintRecorder:
    """Record solved contacts without changing simulation state or actuation."""

    def __init__(self, model, controller=None, *, output_dir=None, rate=50, model_source=None):
        self.model = model
        self.rate = float(rate)
        if not 1 <= self.rate <= 200:
            raise ValueError("paint diagnostic sample rate must be 1..200 Hz")
        self.bodies = {label: model.body(name).id for label, name in (
            ("tool", "paint_tool_body"), ("roller", "paint_roller_body"),
            ("right_wrist", "right_wrist_yaw_link"))}
        # Base motion distinguishes whole-body sway from wrist tracking error.
        pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        if pelvis >= 0:
            self.bodies["pelvis"] = pelvis
        self.wall = model.geom("paint_wall").id
        # Tool supports: the pedestal and cup holder, or the legacy table/rest.
        # Contacts with them are still reported in the tool_table columns.
        stand = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "paint_stand_body")
        if stand >= 0:
            self.support_geoms = {g for g in range(model.ngeom) if model.geom_bodyid[g] == stand}
        else:
            self.support_geoms = {model.geom("paint_table_top").id}
            rest = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "paint_handle_rest")
            if rest >= 0:
                self.support_geoms.add(rest)
        self.roller = model.geom("paint_roller").id
        self.hinge = model.joint("paint_roller_hinge").id
        self.tool_geoms = self._descendant_geoms(self.bodies["tool"])
        if self.roller not in self.tool_geoms:
            raise ValueError("paint_roller must belong to the paint_tool_body subtree")
        self.hand_geoms = {}
        for side in ("left", "right"):
            wrist = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_wrist_yaw_link")
            self.hand_geoms[side] = self._descendant_geoms(wrist) if wrist >= 0 else set()
        self.seq = self.saved = self.episode = 0
        self.closed = False
        self.error = None
        self.next_time = 0.
        self.last_time = -1.
        self.last_episode = 0
        self.previous = None
        self.last_flush = time.monotonic()
        root = Path(__file__).resolve().parents[2] / "recordings"
        output_dir = output_dir or os.environ.get("SONIC_PAINT_OUTPUT_DIR")
        self.path = Path(output_dir) if output_dir else root / (
            "paint_diagnostics_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
        self.path = self.path.resolve()
        self.model_path = self.path / "model.mjb"
        self.stream = None
        try:
            self.path.mkdir(parents=True, exist_ok=False)
            mujoco.mj_saveModel(model, str(self.model_path))
            source = Path(model_source).resolve() if model_source else None
            target = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "paint_target_patch")
            self.metadata = {
                "schema": 2, "created_utc": datetime.now(timezone.utc).isoformat(),
                "pose_bodies": {label: model.body(body).name for label, body in self.bodies.items()},
                "mujoco_version": mujoco.__version__,
                "model_sha256": hashlib.sha256(self.model_path.read_bytes()).hexdigest(),
                "model_source": str(source) if source else None,
                "source_scene_sha256": hashlib.sha256(source.read_bytes()).hexdigest() if source else None,
                "sample_rate_hz": self.rate, "timestep_s": float(model.opt.timestep),
                "integrator": int(model.opt.integrator),
                "wall_geometry": self._geometry("geom", self.wall),
                "tool_table_support_geometries": [self._geometry("geom", geom) for geom in sorted(self.support_geoms)],
                "target_patch_geometry": self._geometry("site", target) if target >= 0 else None,
                "quaternion_order": "wxyz",
                "timing": "sim_time is pre-step; post_step_time is after integration. Body poses and contacts are the final solver evaluation stored by mj_step (Euler: pre-integration pose). Roller angle and velocity are copied before the step. No mj_forward or state edits are performed by this recorder.",
                "contact_convention": "Normal columns sum nonnegative scalar normal forces; tangent columns sum each contact's tangential-force magnitude (not magnitude of the net force). World force vectors act on the tool/roller. Counts and contact flags include only positive normal force.",
                "relative_motion": "tool_in_right_wrist pose is the tool body frame expressed in the wrist body frame. Position/rotation deltas compare consecutive samples within an episode. These raw relative motions do not establish grasp success or distinguish deliberate regrasping from slip.",
                "manual_resets": "R returns only the tool to its initial pose. Each accepted reset starts a new episode and is recorded in events.jsonl; it is not a grasp or recovery success.",
                "limits": "Forces are sampled simulation solver outputs, not hardware measurements or evidence of precise constant-force control. No stroke start/end labels are recorded, so no success rate or paint coverage is inferred. Geometric contact does not model paint deposition or coating quality. Enabled elastic-band assistance must be accounted for in analysis.",
            }
            self._write_json("metadata.json", self.metadata)
            self.stream = (self.path / "samples.csv").open("w", newline="", buffering=65536)
            self.writer = None
        except Exception as exc:
            if self.stream is not None:
                self.stream.close()
            raise RuntimeError(f"Cannot initialize paint recording at {self.path}: {exc}") from exc
        print(f"[paint_diagnostics] recording {self.path}", flush=True)

    def _descendant_geoms(self, ancestor):
        result = set()
        for geom in range(self.model.ngeom):
            body = int(self.model.geom_bodyid[geom])
            while body:
                if body == ancestor:
                    result.add(geom)
                    break
                body = int(self.model.body_parentid[body])
        return result

    def _geometry(self, kind, index):
        return {
            "name": getattr(self.model, kind)(index).name,
            "body": self.model.body(int(getattr(self.model, kind + "_bodyid")[index])).name,
            "type": int(getattr(self.model, kind + "_type")[index]),
            "size_m": getattr(self.model, kind + "_size")[index].tolist(),
            "position_in_body_m": getattr(self.model, kind + "_pos")[index].tolist(),
            "quaternion_in_body_wxyz": getattr(self.model, kind + "_quat")[index].tolist(),
        }

    def _write_json(self, name, value):
        (self.path / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")

    def begin_step(self, data):
        if self.closed:
            return None
        t = float(data.time)
        if t < self.last_time or self.episode != self.last_episode:
            self.next_time = t
            self.previous = None
        self.last_time = t
        self.last_episode = self.episode
        if t + 1e-9 < self.next_time:
            return None
        self.next_time = t + 1. / self.rate
        return {"sim_time": t, "episode": self.episode, "wall_time_unix_s": time.time(),
                "roller_angle_rad": float(data.qpos[self.model.jnt_qposadr[self.hinge]]),
                "roller_velocity_rad_s": float(data.qvel[self.model.jnt_dofadr[self.hinge]])}

    def record_manual_reset(self, event):
        self.episode += 1
        row = {**event, "episode": self.episode, "wall_time_unix_s": time.time()}
        try:
            with (self.path / "events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        except (OSError, ValueError) as exc:
            self.error = repr(exc)
            raise RuntimeError(f"Cannot record manual tool reset at {self.path}: {exc}") from exc

    @staticmethod
    def _vector(row, prefix, values, axes="xyz", unit="m"):
        for axis, value in zip(axes, values):
            row[f"{prefix}_{axis}_{unit}"] = float(value)

    def snapshot(self, data, before, elastic_band=False):
        row = {"seq": self.seq, **before, "post_step_time": float(data.time),
               "elastic_band": bool(elastic_band)}
        for label, body in self.bodies.items():
            self._vector(row, label, data.xpos[body])
            self._vector(row, label + "_quat", data.xquat[body], "wxyz", "unitless")
        tool, wrist = self.bodies["tool"], self.bodies["right_wrist"]
        wrist_rotation = data.xmat[wrist].reshape(3, 3)
        position = wrist_rotation.T @ (data.xpos[tool] - data.xpos[wrist])
        rotation = wrist_rotation.T @ data.xmat[tool].reshape(3, 3)
        quaternion = np.empty(4)
        mujoco.mju_mat2Quat(quaternion, rotation.reshape(9))
        self._vector(row, "tool_in_right_wrist", position)
        self._vector(row, "tool_in_right_wrist_quat", quaternion, "wxyz", "unitless")
        row.update(relative_position_delta_m=None, relative_rotation_delta_rad=None,
                   relative_sample_interval_s=None)
        if self.previous is not None:
            episode, t, old_position, old_quaternion = self.previous
            dt = before["sim_time"] - t
            if episode == before["episode"] and dt > 0:
                row["relative_position_delta_m"] = float(np.linalg.norm(position - old_position))
                row["relative_rotation_delta_rad"] = float(2 * np.arccos(np.clip(abs(quaternion @ old_quaternion), 0, 1)))
                row["relative_sample_interval_s"] = dt
        groups = {key: {"count": 0, "normal": 0., "tangent": 0., "world": np.zeros(3)}
                  for key in ("roller_wall", "tool_table", "left_hand_tool", "right_hand_tool")}
        for index in range(data.ncon):
            contact = data.contact[index]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            if (g1 in self.tool_geoms) == (g2 in self.tool_geoms):
                continue
            tool_geom, other = (g1, g2) if g1 in self.tool_geoms else (g2, g1)
            key = None
            if tool_geom == self.roller and other == self.wall:
                key = "roller_wall"
            elif other in self.support_geoms:
                key = "tool_table"
            else:
                for side, geoms in self.hand_geoms.items():
                    if other in geoms:
                        key = side + "_hand_tool"
                        break
            if key is None:
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, data, index, force)
            group = groups[key]
            group["count"] += int(force[0] > 0)
            group["normal"] += max(float(force[0]), 0.)
            group["tangent"] += float(np.linalg.norm(force[1:3]))
            world = np.asarray(contact.frame).reshape(3, 3).T @ force[:3]
            group["world"] += world if tool_geom == g2 else -world
        for key, group in groups.items():
            row[key + "_contact"] = group["count"] > 0
            row[key + "_contact_count"] = group["count"]
            row[key + "_normal_sum_n"] = group["normal"]
            row[key + "_tangent_magnitude_sum_n"] = group["tangent"]
            self._vector(row, key + "_force_on_tool_world", group["world"], unit="n")
        return row, position.copy(), quaternion.copy()

    def end_step(self, data, before, elastic_band=False):
        if self.closed or before is None:
            return None
        row, position, quaternion = self.snapshot(data, before, elastic_band)
        try:
            if self.writer is None:
                self.writer = csv.DictWriter(self.stream, fieldnames=list(row))
                self.writer.writeheader()
            self.writer.writerow(row)
            self.saved += 1
            if time.monotonic() - self.last_flush >= 1:
                self.stream.flush()
                self.last_flush = time.monotonic()
        except (OSError, ValueError) as exc:
            self.error = repr(exc)
            raise RuntimeError(f"Paint recording failed at {self.path}: {exc}") from exc
        self.previous = (before["episode"], before["sim_time"], position, quaternion)
        self.seq += 1
        return row

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.stream.close()
            self._write_json("recording_status.json", {
                "samples_produced": self.seq, "samples_saved": self.saved,
                "writer_finished": True, "error": self.error})
        except (OSError, ValueError) as exc:
            self.error = repr(exc)
            raise RuntimeError(f"Cannot finalize paint recording at {self.path}: {exc}") from exc
