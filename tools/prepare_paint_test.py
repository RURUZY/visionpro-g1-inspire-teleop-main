#!/usr/bin/env python3
"""Check a dry-roller scene and create an unfilled five-trial worksheet.

No network connections, robot commands, policy execution, or grasp trials.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE = ROOT / "assets/mujoco/g1_runtime/scene_43dof_inspire_hand_paint.xml"


def inspect_scene(scene: Path) -> dict:
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    tool = model.body("paint_tool_body").id
    roller = model.body("paint_roller_body").id
    free = model.joint("paint_tool_free").id
    hinge = model.joint("paint_roller_hinge").id
    if model.jnt_type[free] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError("The roller tool must have a free joint")
    if model.jnt_type[hinge] != mujoco.mjtJoint.mjJNT_HINGE:
        raise ValueError("The roller must have a passive hinge")
    descendants = set()
    for body in range(model.nbody):
        ancestor = body
        while ancestor not in (0, tool):
            ancestor = int(model.body_parentid[ancestor])
        if ancestor == tool:
            descendants.add(body)
    if model.body_parentid[tool] != 0 or roller not in descendants:
        raise ValueError("Tool must be independent of the robot and contain the roller")
    if any(model.body_mocapid[b] >= 0 for b in descendants):
        raise ValueError("Tool must not be driven as a mocap body")
    for i in range(model.neq):
        if model.eq_type[i] in (mujoco.mjtEq.mjEQ_WELD, mujoco.mjtEq.mjEQ_CONNECT):
            if descendants.intersection((int(model.eq_obj1id[i]), int(model.eq_obj2id[i]))):
                raise ValueError("Tool has a body equality constraint")
        if model.eq_type[i] == mujoco.mjtEq.mjEQ_JOINT:
            if {free, hinge}.intersection((int(model.eq_obj1id[i]), int(model.eq_obj2id[i]))):
                raise ValueError("Tool joints have an equality constraint")
    if model.nu != 53:
        raise ValueError(f"Expected 53 robot actuators, found {model.nu}")
    # SONIC expects all robot joints before independently appended object joints.
    for i in range(model.nu):
        if (model.actuator_trntype[i] != mujoco.mjtTrn.mjTRN_JOINT
                or int(model.actuator_trnid[i, 0]) != i + 1):
            raise ValueError(f"SONIC joint-to-actuator mapping changed at actuator {i}")
    if min(free, hinge) <= 53:
        raise ValueError("Tool joints must be appended after all robot joints")
    wall = model.geom("paint_wall").id
    column = model.geom("paint_stand_column").id
    roller_geom = model.geom("paint_roller").id
    if not ((int(model.geom_contype[wall]) & int(model.geom_conaffinity[roller_geom]))
            or (int(model.geom_contype[roller_geom]) & int(model.geom_conaffinity[wall]))):
        raise ValueError("Roller and wall cannot collide")
    if not np.isfinite(data.qpos).all():
        raise ValueError("Invalid initial model state")
    target = model.site("paint_target_patch").id
    return {
        "scope": "Model preflight only; no policy, reach, grasp, or stroke trial executed",
        "scene": str(scene.resolve()),
        "scene_sha256": hashlib.sha256(scene.read_bytes()).hexdigest(),
        "mujoco_version": mujoco.__version__,
        "robot_actuators": model.nu,
        "sonic_actuator_mapping_preserved": True,
        "tool_free_joint": True,
        "roller_passive_hinge": True,
        "tool_mass_kg": float(sum(model.body_mass[b] for b in descendants)),
        "tool_initial_position_m": data.xpos[tool].tolist(),
        "roller_initial_position_m": data.xpos[roller].tolist(),
        "wall_center_m": data.geom_xpos[wall].tolist(),
        "wall_half_size_m": model.geom_size[wall].tolist(),
        "holder_floor_z_m": float(data.geom_xpos[column, 2] + model.geom_size[column, 2]),
        "grip_initial_position_m": data.site_xpos[model.site("paint_handle_grip").id].tolist(),
        "target_patch_center_m": data.site_xpos[target].tolist(),
        "target_patch_half_size_m": model.site_size[target].tolist(),
        "completed_trials": 0,
        "physical_parameters_calibrated": False,
        "paint_deposition_simulated": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    report = inspect_scene(args.scene)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    out = args.output_dir or ROOT / "recordings" / ("paint_test_" + stamp)
    out.mkdir(parents=True, exist_ok=False)
    (out / "preflight.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    fields = ["trial", "status", "recording_dir", "episode", "stroke_start_sim_s",
              "stroke_end_sim_s", "grasp_lift_hold_3s", "completed_10cm_strokes",
              "tool_dropped", "robot_fell", "manual_intervention", "elastic_band_enabled",
              "result", "failure_reason", "scene_sha256", "notes"]
    with (out / "trials.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for trial in range(1, 6):
            writer.writerow({"trial": trial, "status": "not_run", "scene_sha256": report["scene_sha256"]})
    print(out)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
