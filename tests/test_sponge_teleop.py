"""Offline scene/hand/control compatibility. No robot DDS or headset required."""
from pathlib import Path
import importlib.util
import json
import os

import mujoco
import numpy as np
import pytest

ROOT = Path(os.environ.get('SPONGE_TEST_REPO', Path(__file__).resolve().parents[1]))
TOOLS = Path(__file__).resolve().parents[1] / 'tools'


def load(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


generator = load('generate_sponge_scene')
SpongeReset = load('sponge_teleop_sim').SpongeReset
ROBOT = ROOT / 'assets/mujoco/g1_runtime/g1_29dof_with_inspire_hand.xml'


@pytest.fixture
def scene(tmp_path):
    path = generator.generate_scene(tmp_path, robot_model=ROBOT)
    model = mujoco.MjModel.from_xml_path(str(path))
    model.opt.timestep = .005
    return model, mujoco.MjData(model)


def startup_samples():
    return json.loads((ROOT / 'tests/fixtures/cube_startup_clearance.json').read_text())['samples']


def test_sonic_robot_and_actuator_layout_preserved(scene):
    m, _ = scene
    robot = mujoco.MjModel.from_xml_path(str(ROBOT))
    assert m.nq == robot.nq + 7 and m.nv == robot.nv + 6
    assert m.nu == robot.nu == 53 and m.neq == robot.neq
    assert m.body('sponge_tool').parentid[0] == 0
    assert m.joint('sponge_free').type[0] == mujoco.mjtJoint.mjJNT_FREE
    assert m.body_gravcomp[m.body('sponge_tool').id] == 0
    for attr in ('actuator_trnid', 'actuator_gainprm', 'actuator_biasprm', 'actuator_ctrlrange',
                 'eq_type', 'eq_obj1id', 'eq_obj2id'):
        np.testing.assert_array_equal(getattr(m, attr), getattr(robot, attr))
    np.testing.assert_array_equal(m.jnt_qposadr[:robot.njnt], robot.jnt_qposadr)


def test_recorded_startup_clears_all_props(scene):
    m, d = scene
    first_prop = m.body('sponge_table').id
    for sample in startup_samples():
        d.qpos[:60] = sample['robot_qpos']
        mujoco.mj_forward(m, d)
        for c in d.contact:
            a, b = m.geom_bodyid[[c.geom1, c.geom2]]
            robot_prop = (0 < a < first_prop <= b) or (0 < b < first_prop <= a)
            assert not (robot_prop and c.dist < -1e-5), (sample['sim_time'], m.body(a).name, m.body(b).name, c.dist)


def test_free_sponge_settles_on_table(scene):
    m, d = scene
    initial = d.qpos[:60].copy()
    for _ in range(600):
        d.qpos[:60] = initial
        d.qvel[:59] = 0
        mujoco.mj_step(m, d)
    assert np.isfinite(d.qpos).all() and not any(w.number for w in d.warning)
    assert np.linalg.norm(d.qvel[59:]) < .001
    assert abs(d.qpos[62] - (generator.TABLE_HEIGHT + .098)) < .002
    names = {(m.geom(c.geom1).name, m.geom(c.geom2).name) for c in d.contact}
    assert any({'sponge_pad', 'sponge_table_top'} == set(pair) for pair in names)


def test_head_camera_sees_handle_and_wipe_target(scene):
    m, d = scene
    for sample in (startup_samples()[0], startup_samples()[-1]):
        d.qpos[:60] = sample['robot_qpos']
        mujoco.mj_forward(m, d)
        cam = d.camera('head_camera')
        half = np.tan(np.deg2rad(m.camera('head_camera').fovy[0] / 2))
        target = d.site('sponge_wipe_target').xpos
        hx, hy = generator.WIPE_TARGET_HALF
        points = [d.site('sponge_handle_grip').xpos] + [target + [x, y, 0] for x in (-hx, hx) for y in (-hy, hy)]
        for point in points:
            local = cam.xmat.reshape(3, 3).T @ (point - cam.xpos)
            assert -local[2] > 0
            assert abs(local[0]) < -local[2]*half*640/480
            assert abs(local[1]) < -local[2]*half


def test_reset_preserves_robot_commands_and_time(scene):
    m, d = scene
    reset = SpongeReset(m)
    d.qpos[reset.qpos.start:reset.qpos.start+3] = [1, -.5, .05]
    d.qvel[:] = .1
    d.ctrl[:] = .2
    d.time = 4.2
    saved = d.qpos.copy()
    reset.request()
    assert reset.apply(d)
    np.testing.assert_array_equal(d.qpos[:60], saved[:60])
    np.testing.assert_array_equal(d.qpos[reset.qpos], m.qpos0[reset.qpos])
    assert d.time == 4.2 and np.all(d.ctrl == .2)
    assert np.all(d.qvel[reset.dofs] == 0) and np.all(d.qvel[:59] == .1)
    assert not reset.apply(d)


def test_grip_lift_and_wipe_reachable_after_landing(scene):
    # Geometric arm IK only, not a claim of learned grasping or balance success.
    # With the lowered hang (tools/sponge_sim_entry.py) recorded runs land in place, pelvis x ~-0.02 m.
    m, d = scene
    joints = [m.joint(f'right_{s}_joint').id for s in (
        'shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow',
        'wrist_roll', 'wrist_pitch', 'wrist_yaw')]
    qa, va = m.jnt_qposadr[joints], m.jnt_dofadr[joints]
    low, high = m.jnt_range[joints].T
    palm = m.body('right_base_link').id
    standing = np.array(startup_samples()[-1]['robot_qpos'])
    standing[:2] = [-.02, 0]
    standing[3:7] = [1, 0, 0, 0]
    sx, sy = generator.SPONGE_XY
    (wx, wy), (hx, hy) = generator.WIPE_TARGET_XY, generator.WIPE_TARGET_HALF
    grip = generator.TABLE_HEIGHT + .098
    wipe = [(wx + a, wy + b, grip) for a in (-hx, hx) for b in (-hy, hy)]
    for x, y, z in [(sx, sy, grip), (sx, sy, grip + .1)] + wipe:
        d.qpos[:60] = standing
        target = np.array([x, y, z])
        for _ in range(500):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            rot = d.xmat[palm].reshape(3, 3)
            point = d.xpos[palm] + rot @ np.array([-.006, .025, .14])
            jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
            mujoco.mj_jac(m, d, jp, jr, point, palm)
            jac = np.vstack([jp[:, va], .3*jr[:, va]])
            err = np.r_[target-point, .3*np.cross(-rot[:, 0], [0, 0, 1])]
            step = jac.T @ np.linalg.solve(jac@jac.T + 1e-4*np.eye(6), err)
            step += .01*(np.eye(7)-np.linalg.pinv(jac)@jac)@((low+high)/2-d.qpos[qa])
            d.qpos[qa] = np.clip(d.qpos[qa]+.5*step, low, high)
        assert np.linalg.norm(target-point) < .01, (target, point)
        assert np.min(np.minimum(d.qpos[qa]-low, high-d.qpos[qa])) > .1


def test_reset_rejects_occupied_return_position(scene):
    m, d = scene
    reset = SpongeReset(m)
    d.qpos[:3] = m.qpos0[60:63]
    d.qpos[60:63] = [2, 0, .05]
    before = d.qpos.copy()
    reset.request()
    assert not reset.apply(d)
    np.testing.assert_array_equal(d.qpos, before)


def packet(left, right):
    from g1_teleop.bridge.zmq_pub import PackedPublisher
    pub = object.__new__(PackedPublisher)
    messages = []
    class Sink:
        def send(self, data):
            messages.append(data)
    pub.socket = Sink()
    pub.send_inspire_hand(np.asarray(left), np.asarray(right))
    return messages[0]


def test_existing_bridge_drives_both_hands_and_timeout(scene):
    from g1_teleop.sim.inspire_hand import InspireHandController
    m, d = scene
    m.opt.gravity[:] = 0
    controller = InspireHandController(m, d, connect=False)
    for command in ([0]*6, [1000]*6):
        for i in range(300):
            d.qpos[:7] = m.qpos0[:7]
            d.qvel[:6] = 0
            assert controller.receive(packet(command, command), i*.005)
            tau = controller.torques(.005, now=i*.005)
            d.ctrl[controller.indices['left']-1] = tau[:12]
            d.ctrl[controller.indices['right']-1] = tau[12:]
            mujoco.mj_step(m, d)
        for active in controller.active.values():
            target = m.jnt_range[active, 1 if command[0] == 0 else 0]
            np.testing.assert_allclose(d.qpos[m.jnt_qposadr[active]], target, atol=.08)
    controller.torques(.005, now=4)
    assert controller.timed_out
    assert np.isfinite(d.qpos).all() and not any(w.number for w in d.warning)


def test_table_docks_after_standing_and_carries_sponge(scene):
    import sys
    sys.path.insert(0, str(TOOLS))
    from sponge_sim_entry import SpongeAids, DOCK_AFTER_S, DOCK_DURATION_S, LAYOUT_PELVIS_XY
    m, d = scene
    aids = SpongeAids(m)
    d.qpos[:60] = startup_samples()[-1]['robot_qpos']
    d.qpos[:2] = LAYOUT_PELVIS_XY
    mujoco.mj_forward(m, d)
    start = d.qpos[aids.qpos].copy()
    aids.dock(d, band_enabled=True)  # still hanging: nothing moves
    assert not aids.docked and aids.dock_started is None
    d.qvel[:] = 0
    t = 0.
    while not aids.docked and t < DOCK_AFTER_S + DOCK_DURATION_S + 1:
        d.time = t
        aids.dock(d, band_enabled=False)
        t += .005
    assert aids.docked
    np.testing.assert_allclose(d.mocap_pos[aids.mocap], aids.table_park + aids.dock_shift)
    np.testing.assert_allclose(d.qpos[aids.qpos][:3], start[:3] + aids.dock_shift, atol=1e-9)
    np.testing.assert_allclose(aids.home[:3], start[:3] + aids.dock_shift)
    assert abs(aids.home[0] - generator.SPONGE_XY[0]) < 1e-9


def test_table_docks_relative_to_where_the_robot_stands(scene):
    import sys
    sys.path.insert(0, str(TOOLS))
    from sponge_sim_entry import SpongeAids, DOCK_AFTER_S, DOCK_DURATION_S, LAYOUT_PELVIS_XY
    m, d = scene
    aids = SpongeAids(m)
    d.qpos[:60] = startup_samples()[-1]['robot_qpos']
    d.qpos[:2] = [.30, -.08]  # a long forward landing step, as seen in a recorded run
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    t = 0.
    while not aids.docked and t < DOCK_AFTER_S + DOCK_DURATION_S + 1:
        d.time = t
        aids.dock(d, band_enabled=False)
        t += .005
    assert aids.docked
    stance = np.array([.30, -.08]) - LAYOUT_PELVIS_XY
    np.testing.assert_allclose(aids.home[:2], np.array(generator.SPONGE_XY) + stance, atol=1e-9)
    table_front = d.mocap_pos[aids.mocap][0] - .22
    assert abs(table_front - (generator.TABLE_FRONT_X + stance[0])) < 1e-9

def test_tipped_sponge_returns_upright_after_docking(scene):
    import sys
    sys.path.insert(0, str(TOOLS))
    from sponge_sim_entry import SpongeAids, TIP_HOLD_S
    m, d = scene
    aids = SpongeAids(m)
    d.qpos[:60] = startup_samples()[-1]['robot_qpos']
    d.qpos[:2] = 0
    aids.docked = True
    aids.home[:3] += aids.dock_shift
    d.mocap_pos[aids.mocap] = aids.table_park + aids.dock_shift
    tipped = aids.home.copy()
    tipped[:3] = aids.home[:3] + [0, .05, -.07]
    tipped[3:] = [.7071068, .7071068, 0, 0]  # handle lying along y
    d.qpos[aids.qpos] = tipped
    mujoco.mj_forward(m, d)
    assert aids.tipped(d)
    t = 0.
    while t < TIP_HOLD_S + .2:
        d.time = t
        aids.apply(d)
        t += .005
    mujoco.mj_forward(m, d)
    np.testing.assert_allclose(d.qpos[aids.qpos], aids.home)
    assert not aids.tipped(d)



def test_g_key_levels_the_pad_for_any_hand_pose(scene, tmp_path, monkeypatch):
    import sys
    monkeypatch.setenv('SONIC_SPONGE_GRIP_LOG', str(tmp_path / 'grip.jsonl'))
    sys.path.insert(0, str(TOOLS))
    from sponge_sim_entry import SpongeAids
    m, d = scene
    aids = SpongeAids(m)
    rng = np.random.default_rng(0)
    for _ in range(5):
        d.qpos[:60] = startup_samples()[-1]['robot_qpos']
        arm = [m.joint(f'right_{n}_joint') for n in ('shoulder_roll', 'shoulder_yaw', 'wrist_roll', 'wrist_pitch', 'wrist_yaw')]
        for j in arm:
            d.qpos[j.qposadr[0]] = rng.uniform(j.range[0] * .5, j.range[1] * .5)
        mujoco.mj_forward(m, d)
        aids.grip_request.set()
        aids.apply(d)
        mujoco.mj_forward(m, d)
        handle_axis = d.xmat[aids.tool].reshape(3, 3)[:, 2]
        assert handle_axis[2] > .999  # handle vertical, pad level, whatever the wrist pose
        aids.pin_started = None


def test_hand_lateral_scale_only_scales_sideways(monkeypatch):
    import sys
    from g1_teleop.bridge.cli import parse_args
    from g1_teleop.bridge.vr_targets import official_hand_delta
    rel, calib = np.array([.1, .2, .3]), np.zeros(3)
    out = {}
    for scale in ('1.0', '0.5'):
        monkeypatch.setattr(sys, 'argv', ['bridge', '--avp-endpoint', 'x', '--hand-lateral-scale', scale])
        out[scale] = official_hand_delta(rel, calib, parse_args(), side='right')
    np.testing.assert_allclose(out['0.5'][[0, 2]], out['1.0'][[0, 2]])
    np.testing.assert_allclose(out['0.5'][1], .5 * out['1.0'][1])


def test_grip_log_records_carried_sponge(scene, tmp_path, monkeypatch):
    import json, sys
    sys.path.insert(0, str(TOOLS))
    monkeypatch.setenv('SONIC_SPONGE_GRIP_LOG', str(tmp_path / 'grip.jsonl'))
    from sponge_sim_entry import SpongeAids
    m, d = scene
    aids = SpongeAids(m)
    d.qpos[:60] = startup_samples()[-1]['robot_qpos']
    mujoco.mj_forward(m, d)
    aids.grip_request.set()
    for k in range(3):
        d.time = k * .25
        aids.apply(d)
        mujoco.mj_forward(m, d)
    rows = [json.loads(line) for line in (tmp_path / 'grip.jsonl').read_text().splitlines()]
    assert rows and rows[0]['mode'] == 'assisted' and rows[0]['tilt_deg'] < 1.
    assert len(rows[0]['handle_in_palm']) == 3 and len(rows[0]['wrist_roll_pitch_yaw']) == 3
