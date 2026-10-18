"""Offline scene physics checks; no robot/network connections are opened."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

REPO = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO / "assets/mujoco/g1_runtime/g1_29dof_with_inspire_hand.xml"
spec = importlib.util.spec_from_file_location("generate_paint_scene", REPO / "tools/generate_paint_scene.py")
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)


@pytest.fixture
def scene(tmp_path):
    if not MODEL_PATH.is_file():
        pytest.skip("Prepare local G1 assets with tools/setup_inspire_sim.py first")
    path = generator.generate_scene(tmp_path)
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def test_robot_indices_and_constraints_are_unchanged(scene):
    model, _ = scene
    robot = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    assert model.nu == robot.nu == 53
    assert model.nq == robot.nq + 8  # Free tool (7), passive roller (1).
    assert model.nv == robot.nv + 7
    assert model.njnt == robot.njnt + 2
    np.testing.assert_array_equal(model.actuator_trnid, robot.actuator_trnid)
    np.testing.assert_array_equal(model.jnt_qposadr[:robot.njnt], robot.jnt_qposadr)
    np.testing.assert_array_equal(model.jnt_dofadr[:robot.njnt], robot.jnt_dofadr)
    for index in range(robot.njnt):
        assert model.joint(index).name == robot.joint(index).name
    assert model.neq == robot.neq  # No grasp weld or other artificial attachment.
    np.testing.assert_array_equal(model.eq_type, robot.eq_type)
    np.testing.assert_array_equal(model.eq_obj1id, robot.eq_obj1id)
    np.testing.assert_array_equal(model.eq_obj2id, robot.eq_obj2id)
    assert model.body("paint_tool_body").parentid[0] == 0
    assert model.joint("paint_tool_free").type[0] == mujoco.mjtJoint.mjJNT_FREE
    assert model.joint("paint_roller_hinge").type[0] == mujoco.mjtJoint.mjJNT_HINGE
    assert model.site("paint_target_patch").type[0] == mujoco.mjtGeom.mjGEOM_BOX
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "paint_target_patch") == -1


def test_initial_scene_has_no_robot_prop_penetration(scene):
    model, data = scene
    tool_joint = model.joint("paint_tool_free").id
    robot_body_limit = model.body("paint_wall_body").id
    assert tool_joint == 54
    for contact in data.contact:
        body1, body2 = model.geom_bodyid[[contact.geom1, contact.geom2]]
        robot1 = 0 < body1 < robot_body_limit
        robot2 = 0 < body2 < robot_body_limit
        prop1 = model.geom(contact.geom1).name.startswith("paint_")
        prop2 = model.geom(contact.geom2).name.startswith("paint_")
        assert not (((robot1 and prop2) or (robot2 and prop1)) and contact.dist < 0)


def test_recorded_cube_startup_and_landing_clear_paint_props(scene):
    model, data = scene
    fixture = json.loads((REPO / "tests/fixtures/cube_startup_clearance.json").read_text())
    robot_limit = model.body("paint_wall_body").id
    robot_nq = int(model.joint("paint_tool_free").qposadr[0])
    for sample in fixture["samples"]:
        assert len(sample["robot_qpos"]) == robot_nq
        data.qpos[:robot_nq] = sample["robot_qpos"]
        mujoco.mj_forward(model, data)
        for contact in data.contact:
            a, b = model.geom_bodyid[[contact.geom1, contact.geom2]]
            robot_prop = (0 < a < robot_limit <= b) or (0 < b < robot_limit <= a)
            assert not (robot_prop and contact.dist < 0), (
                sample["sim_time"], model.geom(contact.geom1).name,
                model.geom(contact.geom2).name, contact.dist)


def test_upright_tool_in_right_holder_and_free_standing_wall(scene):
    model, data = scene
    grip = data.site("paint_handle_grip").xpos
    roller = data.body("paint_roller_body").xpos
    axis = data.body("paint_tool_body").xmat.reshape(3, 3)
    # Handle vertical with the roller on top; the axle is already parallel to the wall.
    assert axis[2, 0] > 0.99
    assert abs(axis[1, 1]) > 0.99
    assert roller[2] - grip[2] > 0.15
    # The bent frame points outward, away from the centreline.
    assert data.geom("paint_frame_arm").xpos[1] < grip[1] - 0.05
    # Holder in front of the right hand at waist height, reachable without walking.
    assert grip[1] < -0.20
    assert 0.30 < grip[0] - data.qpos[0] < 0.45
    assert 0.85 < grip[2] < 1.00
    # Nothing stands between the robot and the wall; the pedestal ends in front of it.
    wall = model.geom("paint_wall")
    wall_front = data.geom("paint_wall").xpos[0] - wall.size[0]
    column = model.geom("paint_stand_column")
    assert data.geom("paint_stand_column").xpos[0] + column.size[0] < wall_front - 0.05
    names = [model.geom(i).name for i in range(model.ngeom)]
    assert not any(name.startswith("paint_table") for name in names)
    target = data.site("paint_target_patch").xpos
    size = model.site("paint_target_patch").size
    assert abs(target[0] - wall_front) < 0.003
    assert abs(target[1]) + size[1] <= wall.size[1]
    assert abs(target[2] - data.geom("paint_wall").xpos[2]) + size[2] <= wall.size[2]


def _right_arm_ik(model, data, target, axis, iterations=400):
    """Damped least squares on the 7 right-arm joints; returns (error_m, min joint margin)."""
    joints = [model.joint(f"right_{name}_joint").id for name in (
        "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
        "wrist_roll", "wrist_pitch", "wrist_yaw")]
    qadr, vadr = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    low, high = model.jnt_range[joints].T
    palm = model.body("right_base_link").id
    wrap = np.array([-0.006, 0.035, 0.13])  # centre of the closed finger wrap
    for _ in range(iterations):
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)  # mj_jac needs current cdof/subtree_com
        rot = data.xmat[palm].reshape(3, 3)
        point = data.xpos[palm] + rot @ wrap
        jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jp, jr, point, palm)
        jac = np.vstack([jp[:, vadr], 0.3 * jr[:, vadr]])
        err = np.r_[target - point, 0.3 * np.cross(-rot[:, 0], axis)]
        step = jac.T @ np.linalg.solve(jac @ jac.T + 1e-4 * np.eye(6), err)
        # Null-space pull toward mid-range avoids solutions pinned at a limit.
        null = np.eye(len(joints)) - np.linalg.pinv(jac) @ jac
        step += 0.01 * null @ ((low + high) / 2 - data.qpos[qadr])
        data.qpos[qadr] = np.clip(data.qpos[qadr] + 0.5 * step, low, high)
    mujoco.mj_kinematics(model, data)
    rot = data.xmat[palm].reshape(3, 3)
    error = np.linalg.norm(target - data.xpos[palm] - rot @ wrap)
    margin = np.min(np.minimum(data.qpos[qadr] - low, high - data.qpos[qadr]))
    return error, margin


def test_grip_and_wall_stroke_reachable_from_standing_pose(scene):
    """Static arm IK only: no balance, collision-free path or grasp is implied."""
    model, data = scene
    fixture = json.loads((REPO / "tests/fixtures/cube_startup_clearance.json").read_text())
    standing = np.array(fixture["samples"][-1]["robot_qpos"])
    standing[0:2] = data.qpos[0:2]  # initial base position, no walking
    standing[3:7] = (1, 0, 0, 0)
    qadr = model.joint("paint_tool_free").qposadr[0]
    grip = data.site("paint_handle_grip").xpos.copy()
    target = data.site("paint_target_patch").xpos.copy()
    data.qpos[:qadr] = standing
    error, margin = _right_arm_ik(model, data, grip, np.array([0.0, 0.0, 1.0]))
    assert error < 0.01 and margin > 0.2
    wall_front = target[0]
    for dz in (-0.10, 0.0, 0.10):  # 20 cm vertical stroke through the target
        roller = np.array([wall_front - 0.032, target[1], target[2] + dz])
        results = []
        for tilt in np.radians((15, 25, 35)):  # handle leaning back toward the robot
            axis = np.array([np.sin(tilt), 0.0, np.cos(tilt)])
            data.qpos[:qadr] = standing
            results.append(_right_arm_ik(model, data, roller - 0.205 * axis, axis))
        assert any(error < 0.01 and margin > 0.15 for error, margin in results), (dz, results)


def test_head_view_contains_target_and_grip(scene):
    model, data = scene
    fixture = json.loads((REPO / "tests/fixtures/cube_startup_clearance.json").read_text())
    for pose in (model.qpos0[:60], fixture["samples"][-1]["robot_qpos"]):
        data.qpos[:60] = pose
        mujoco.mj_forward(model, data)
        camera = data.camera("head_camera")
        frame = camera.xmat.reshape(3, 3)
        half_height = np.tan(np.deg2rad(model.camera("head_camera").fovy[0] / 2))
        target = data.site("paint_target_patch").xpos
        size = model.site("paint_target_patch").size
        points = [data.site("paint_handle_grip").xpos]
        points += [target + [0, sy * size[1], sz * size[2]]
                   for sy in (-1, 1) for sz in (-1, 1)]
        for point in points:
            local = frame.T @ (point - camera.xpos)
            depth = -local[2]
            assert depth > 0
            assert abs(local[1]) < depth * half_height
            assert abs(local[0]) < depth * half_height * (640 / 480)


def test_tool_settles_upright_in_holder_without_attachment(scene):
    model, data = scene
    qadr = model.joint("paint_tool_free").qposadr[0]
    vadr = model.joint("paint_tool_free").dofadr[0]
    robot_pose = data.qpos[:qadr].copy()
    initial = data.site("paint_handle_grip").xpos.copy()
    halfway = None
    # Hold only the robot state to isolate the tool test from walking control.
    # Tool free joint and roller hinge evolve under gravity/contact throughout.
    steps = round(4.0 / model.opt.timestep)
    for frame in range(steps):
        data.qpos[:qadr] = robot_pose
        data.qvel[:vadr] = 0
        mujoco.mj_step(model, data)
        if frame == steps // 2:
            halfway = data.body("paint_tool_body").xpos.copy()
    assert np.isfinite(data.qpos).all()
    assert not any(warning.number for warning in data.warning)
    final = data.body("paint_tool_body").xpos
    assert np.linalg.norm(final - halfway) < 0.001
    assert np.linalg.norm(data.qvel[vadr:]) < 0.001
    axis = data.body("paint_tool_body").xmat.reshape(3, 3)[:, 0]
    assert axis[2] > np.cos(np.radians(10))  # stays nearly upright in the cup
    assert np.linalg.norm(data.site("paint_handle_grip").xpos - initial) < 0.02
    stand = model.body("paint_stand_body").id
    supports = {model.geom(c.geom1).name if model.geom_bodyid[c.geom1] == stand else model.geom(c.geom2).name
                for c in data.contact
                if stand in (model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2])}
    assert "paint_stand_column" in supports  # handle tip rests on the cup floor


def test_wall_contact_turns_the_unactuated_roller(scene):
    model, data = scene
    qadr = model.joint("paint_tool_free").qposadr[0]
    vadr = model.joint("paint_tool_free").dofadr[0]
    hinge_vadr = model.joint("paint_roller_hinge").dofadr[0]
    assert model.joint("paint_roller_hinge").id not in model.actuator_trnid[:, 0]
    robot_pose = data.qpos[:qadr].copy()
    wall_front = data.geom("paint_wall").xpos[0] - model.geom("paint_wall").size[0]
    target = data.site("paint_target_patch").xpos.copy()
    # A test fixture prescribes the handle's upward movement and 1 mm wall
    # compression. Roller spin remains passive; there is no hinge command.
    for frame in range(200):
        data.qpos[:qadr] = robot_pose
        data.qvel[:vadr] = 0
        data.qpos[qadr:qadr + 7] = [wall_front - 0.032 - 0.195 + 0.001,
                                   target[1], target[2] + 0.05 * frame * model.opt.timestep,
                                   1, 0, 0, 0]
        data.qvel[vadr:vadr + 6] = [0, 0, 0.05, 0, 0, 0]
        mujoco.mj_step(model, data)
    pairs = {frozenset((model.geom(c.geom1).name, model.geom(c.geom2).name)) for c in data.contact}
    assert frozenset(("paint_roller", "paint_wall")) in pairs
    normal_force = 0.0
    for index, contact in enumerate(data.contact):
        names = {model.geom(contact.geom1).name, model.geom(contact.geom2).name}
        if names == {"paint_roller", "paint_wall"}:
            wrench = np.zeros(6)
            mujoco.mj_contactForce(model, data, index, wrench)
            normal_force += wrench[0]
    assert normal_force > 0
    assert abs(data.qvel[hinge_vadr]) > 0.1
    assert np.isfinite(data.qpos).all()


def test_missing_assets_explain_setup_without_downloading(tmp_path):
    with pytest.raises(FileNotFoundError, match="tools/setup_inspire_sim.py"):
        generator.generate_scene(tmp_path, robot_model=tmp_path / "missing.xml")


def test_diagnostics_capture_base_sway_without_changing_robot(scene, tmp_path):
    from g1_teleop.sim.paint_diagnostics import PaintRecorder
    model, data = scene
    recorder = PaintRecorder(model, output_dir=tmp_path / "base_motion")
    try:
        data.qpos[:3] = [.1, -.02, .78]
        data.qpos[3:7] = [np.cos(.1), np.sin(.1), 0., 0.]
        mujoco.mj_forward(model, data)
        before = recorder.begin_step(data)
        expected = data.qpos.copy()
        row, _, _ = recorder.snapshot(data, before)
        np.testing.assert_array_equal(data.qpos, expected)
        np.testing.assert_allclose([row[f"pelvis_{a}_m"] for a in "xyz"], expected[:3])
        np.testing.assert_allclose([row[f"pelvis_quat_{a}_unitless"] for a in "wxyz"], expected[3:7])
        assert row["wall_time_unix_s"] > 0
    finally:
        recorder.close()
