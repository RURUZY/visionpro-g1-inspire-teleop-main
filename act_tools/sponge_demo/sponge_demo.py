#!/usr/bin/env python3
"""Offline G1 sponge-wiping preview with a contact-held, free-body tool.

This is a scripted, pre-grasped tool-use check, not an ACT/VLA policy,
autonomous pickup, deformable sponge model, or cleaning validation.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as ET

import numpy as np

ROBOT = Path('/home/epcap/projects/visionpro-g1-inspire-teleop-main/assets/mujoco/g1_runtime/g1_29dof_with_inspire_hand.xml')
ARM = ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow', 'wrist_roll', 'wrist_pitch', 'wrist_yaw')
TABLE_Z = 0.78
WIPE_Z = TABLE_Z - 0.002  # Soft-contact approximation, not a force controller.
DURATION = 12.0
HAND_ACTIVE = ('little_1', 'ring_1', 'middle_1', 'index_1', 'thumb_2', 'thumb_1')
TOOL_OFFSET = '-0.006 0.025 0.14'
TOOL_QUAT = '0.7071067812 0 -0.7071067812 0'


def configure_viewer_display(requested=None):
    """Use an explicit display or the user's only accessible X11 desktop."""
    authority = Path.home() / '.Xauthority'
    if authority.is_file():
        os.environ.setdefault('XAUTHORITY', str(authority))
    explicit = requested or os.environ.get('DISPLAY')
    candidates = [explicit] if explicit else [
        ':' + p.name[1:] for p in sorted(Path('/tmp/.X11-unix').glob('X*'))
        if p.name[1:].isdigit()
    ]
    accessible = []
    for display in candidates:
        try:
            probe = subprocess.run(['xdpyinfo', '-display', display],
                env=dict(os.environ, DISPLAY=display), stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, timeout=3)
        except FileNotFoundError:
            raise RuntimeError('缺少 xdpyinfo；请安装 x11-utils 后检查图形显示连接。') from None
        except subprocess.TimeoutExpired:
            continue
        if probe.returncode == 0:
            accessible.append(display)
    if len(accessible) != 1:
        if len(accessible) > 1:
            raise RuntimeError('找到多个可访问的桌面：' + ', '.join(accessible)
                + '。请用 --display :编号 指定窗口位置。')
        raise RuntimeError('没有可访问的图形桌面。请登录 Jetson 桌面后重试，'
            '或用 --render 生成视频。SSH/VS Code 终端本身不是图形桌面。')
    os.environ['DISPLAY'] = accessible[0]
    os.environ['MUJOCO_GL'] = 'glfw'
    print(f'[窗口模式] DISPLAY={accessible[0]}，窗口将在 Jetson 对应桌面打开。关闭窗口即可停止。', flush=True)
    return accessible[0]


def add(parent, tag, **attributes):
    return ET.SubElement(parent, tag, {k: str(v) for k, v in attributes.items()})


def generate(output: Path) -> Path:
    tree = ET.parse(ROBOT)
    root = tree.getroot()
    root.set('model', 'g1_sponge_wipe_pregrasped_contact_preview')
    world = root.find('worldbody')
    pelvis = world.find("body[@name='pelvis']")
    pelvis.remove(pelvis.find("joint[@type='free']"))
    for body in root.findall('.//body'):
        body.set('gravcomp', '1')
    # The fixed pelvis and gravity compensation isolate the upper-body preview.
    for joint in root.findall('.//body/joint'):
        joint.set('damping', '0.03' if '_hand_' in joint.get('name', '') else '2')
    actuators = root.find('actuator')
    for motor in list(actuators):
        joint = root.find(f".//body/joint[@name='{motor.get('joint')}']")
        hand = '_hand_' in motor.get('joint', '')
        if hand and not any(motor.get('joint').endswith('_hand_'+s+'_joint') for s in HAND_ACTIVE):
            # The other six joints follow the original linkage equalities.
            actuators.remove(motor)
            continue
        motor.set('forcerange', motor.get('ctrlrange'))
        motor.tag = 'position'
        motor.set('kp', '2' if hand else '180')
        motor.set('kv', '0.08' if hand else '12')
        motor.set('ctrlrange', joint.get('range'))
    add(root, 'option', timestep='0.002', integrator='implicitfast', cone='elliptic')
    visual = add(root, 'visual')
    add(visual, 'global', offwidth='960', offheight='720')
    add(visual, 'headlight', ambient='0.45 0.45 0.45', diffuse='0.65 0.65 0.65')
    add(visual, 'quality', shadowsize='2048')
    add(world, 'light', pos='0 -1 3', dir='0 0 -1', diffuse='0.8 0.8 0.8')
    add(world, 'geom', name='floor', type='plane', size='3 3 0.05', rgba='0.16 0.20 0.25 1')
    table = add(world, 'body', name='wipe_table', pos=f'0.40 -0.05 {TABLE_Z-0.025}')
    add(table, 'geom', name='table_top', type='box', size='0.30 0.29 0.025', rgba='0.85 0.88 0.88 1',
        friction='0.5 0.005 0.0001', contype='1', conaffinity='15')
    for x in (-0.24, 0.24):
        for y in (-0.23, 0.23):
            add(table, 'geom', type='box', pos=f'{x} {y} {-TABLE_Z/2}',
                size=f'0.02 0.02 {TABLE_Z/2-0.025}', rgba='0.25 0.30 0.34 1', contype='0', conaffinity='0')
    # Visual target only, deliberately not a dirt-removal/success simulation.
    add(world, 'site', name='wipe_target', type='box', pos=f'0.37 -0.16 {TABLE_Z+0.0005}',
        size='0.100 0.085 0.0003', rgba='0.12 0.70 0.68 0.30')
    palm = root.find(".//body[@name='right_base_link']")
    reference = add(palm, 'body', name='tool_reference_frame', pos=TOOL_OFFSET, quat=TOOL_QUAT)
    add(reference, 'site', name='sponge_reference', pos='0 0 -0.098', size='0.002', rgba='0 0 0 0')
    # Independent body: no weld, no gravity compensation, no attachment to the hand.
    tool = add(world, 'body', name='sponge_tool', pos='0.3 -0.2 1.0')
    add(tool, 'freejoint', name='sponge_free')
    add(tool, 'geom', name='short_handle', type='capsule', fromto='0 0 -0.05 0 0 0.05',
        size='0.012', mass='0.025', rgba='0.2 0.25 0.28 1', contype='8', conaffinity='5',
        friction='1 0.005 0.0001')
    add(tool, 'geom', name='sponge_backplate', type='box', pos='0 0 -0.068', size='0.060 0.035 0.006',
        mass='0.035', rgba='0.16 0.25 0.28 1', contype='8', conaffinity='5')
    add(tool, 'geom', name='sponge_pad', type='box', pos='0 0 -0.086', size='0.060 0.035 0.012',
        mass='0.025', rgba='0.98 0.78 0.15 1', contype='8', conaffinity='5',
        friction='0.55 0.005 0.0001', solref='0.02 1', solimp='0.8 0.95 0.01')
    add(tool, 'site', name='sponge_surface', pos='0 0 -0.098', size='0.002', rgba='0 0 0 0')
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'scene_sponge_wipe.xml'
    tree.write(path, encoding='utf-8')
    return path


def smooth(u):
    u = np.clip(u, 0, 1)
    return u*u*(3-2*u)


def waypoint(t):
    # Move along adjacent stripes while keeping the rigid pad horizontal.
    points = np.array([
        [.33, -.21, TABLE_Z+.09], [.33, -.21, WIPE_Z],
        [.33, -.11, WIPE_Z], [.37, -.11, WIPE_Z],
        [.37, -.21, WIPE_Z], [.41, -.21, WIPE_Z],
        [.41, -.11, WIPE_Z], [.41, -.11, TABLE_Z+.09],
    ])
    times = np.array([0, 2, 4, 5, 7, 8, 10, 12])
    idx = min(np.searchsorted(times, t, side='right')-1, len(times)-2)
    idx = max(idx, 0)
    u = smooth((t-times[idx])/(times[idx+1]-times[idx]))
    return points[idx]*(1-u)+points[idx+1]*u


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'output')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--viewer', action='store_true', help='Open an interactive desktop window (default)')
    mode.add_argument('--render', action='store_true', help='Explicitly generate an MP4 without opening a window')
    mode.add_argument('--check-only', action='store_true')
    mode.add_argument('--dynamics-only', action='store_true', help='Check the scripted motion without rendering')
    mode.add_argument('--grasp-check', action='store_true', help='Check unsupported holding and deliberate release')
    parser.add_argument('--display', help='X11 desktop, for example :1; otherwise detect the only accessible desktop')
    parser.add_argument('--close-up', action='store_true', help='Focus the camera on the hand and sponge')
    args = parser.parse_args()
    args.viewer = args.viewer or not (args.render or args.check_only or args.dynamics_only or args.grasp_check)
    if args.display and not args.viewer:
        parser.error('--display 只适用于窗口模式。')
    if args.viewer:
        try:
            configure_viewer_display(args.display)
        except RuntimeError as error:
            parser.exit(2, str(error)+'\n')
    else:
        os.environ.setdefault('MUJOCO_GL', 'egl')
        if args.render:
            print(f'[视频模式] 不打开窗口，视频将写入 {args.output / "sponge-wipe-preview.mp4"}', flush=True)
    import mujoco
    path = generate(args.output)
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    ik = mujoco.MjData(model)
    arm = np.array([model.joint('right_'+name+'_joint').id for name in ARM])
    qa, va = model.jnt_qposadr[arm], model.jnt_dofadr[arm]
    lower, upper = model.jnt_range[arm].T
    site = model.site('sponge_reference').id
    actual_site = model.site('sponge_surface').id
    tool_body = model.body('sponge_tool').id
    palm_body = model.body('right_base_link').id
    hand_bodies = set()
    for body_id in range(model.nbody):
        ancestor = body_id
        while ancestor and ancestor != palm_body:
            ancestor = model.body_parentid[ancestor]
        if ancestor == palm_body:
            hand_bodies.add(body_id)
    tool_qa = int(model.joint('sponge_free').qposadr[0])
    tool_va = int(model.joint('sponge_free').dofadr[0])
    qref = model.qpos0.copy()
    # Seed away from a straight arm and bias the redundant elbow posture.
    preferred = np.array([-.6, -.35, 0., 1.1, 0., -.5, 0.])
    qref[qa] = preferred
    # Targets close around the handle; torque caps and contacts limit actual closure.
    for name, angle in [('thumb_1', 1.1641), ('thumb_2', .5864), ('thumb_3', .5864*.8024),
                        ('thumb_4', .5864*.8024*.9487), ('index_1', 1.4381), ('middle_1', 1.4381),
                        ('ring_1', 1.4381), ('little_1', 1.4381), ('index_2', 1.4381*1.0843),
                        ('middle_2', 1.4381*1.0843), ('ring_2', 1.4381*1.0843), ('little_2', 1.4381*1.0843)]:
        qref[model.joint('right_hand_'+name+'_joint').qposadr[0]] = angle
    ik.qpos[:] = qref
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))

    def solve(target):
        for _ in range(80):
            mujoco.mj_forward(model, ik)
            rot = ik.site_xmat[site].reshape(3, 3)
            orientation_error = .5*sum((np.cross(rot[:, i], np.eye(3)[:, i]) for i in range(3)))
            error = np.r_[target-ik.site_xpos[site], .2*orientation_error]
            if np.linalg.norm(error) < 2e-5:
                break
            mujoco.mj_jacSite(model, ik, jp, jr, site)
            jac = np.vstack([jp[:, va], .2*jr[:, va]])
            step = jac.T @ np.linalg.solve(jac@jac.T+2e-5*np.eye(6), error)
            step += .06*(np.eye(7)-np.linalg.pinv(jac)@jac)@(preferred-ik.qpos[qa])
            ik.qpos[qa] = np.clip(ik.qpos[qa]+np.clip(step, -.08, .08), lower+.03, upper-.03)
        mujoco.mj_forward(model, ik)
        error = float(np.linalg.norm(target-ik.site_xpos[site]))
        if error > .004:
            raise RuntimeError(f'Unreachable preview target {target.tolist()}: {error:.5f} m')
        return ik.qpos.copy(), error

    qref, _ = solve(waypoint(0))
    control_qadr = model.jnt_qposadr[model.actuator_trnid[:, 0]]
    right_hand_actuators = [i for i, j in enumerate(model.actuator_trnid[:, 0])
        if model.joint(j).name.startswith('right_hand_')]
    right_hand_qadr = [int(model.jnt_qposadr[j]) for j in range(model.njnt)
        if model.joint(j).name.startswith('right_hand_')]
    if args.check_only:
        errors = []
        for t in np.linspace(0, DURATION, 121):
            _, error = solve(waypoint(t))
            errors.append(error)
        print(json.dumps({'scene':str(path),'max_ik_error_m':max(errors),'targets':len(errors)}))
        return

    def hand_contacts():
        bodies = set()
        for contact in data.contact:
            a, b = model.geom_bodyid[[contact.geom1, contact.geom2]]
            if tool_body not in (a, b):
                continue
            other = b if a == tool_body else a
            if other in hand_bodies:
                bodies.add(model.body(other).name)
        return sorted(bodies)

    def relative_tool_pose():
        palm_rotation = data.xmat[palm_body].reshape(3, 3)
        return (palm_rotation.T @ (data.xpos[tool_body]-data.xpos[palm_body]),
                palm_rotation.T @ data.xmat[tool_body].reshape(3, 3))

    def initialize():
        mujoco.mj_resetData(model, data)
        ik.qpos[:] = qref
        initial, _ = solve(waypoint(0))
        data.qpos[:] = initial
        data.qpos[right_hand_qadr] = 0
        data.ctrl[:] = initial[control_qadr]
        mujoco.mj_forward(model, data)
        reference_body = model.body('tool_reference_frame').id
        held_pose = np.r_[data.xpos[reference_body], data.xquat[reference_body]]
        # Pre-grasp initialization only: place the tool in the palm while fingers close.
        # All subsequent simulation steps leave the free joint to physics.
        for _ in range(500):
            data.qpos[tool_qa:tool_qa+7] = held_pose
            data.qvel[tool_va:tool_va+6] = 0
            mujoco.mj_step(model, data)
        data.qpos[tool_qa:tool_qa+7] = held_pose
        data.qvel[tool_va:tool_va+6] = 0
        for _ in range(500):
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        data.time = 0
        return data.qpos.copy(), relative_tool_pose()

    initial_state, initial_relative = initialize()
    if args.grasp_check:
        held_start = data.xpos[tool_body].copy()
        hold_drift, counts, thumb_counts, hand_torque = [], [], [], []
        for step in range(2500):
            mujoco.mj_step(model, data)
            if step % 25 == 0:
                mujoco.mj_forward(model, data)
                hold_drift.append(float(np.linalg.norm(data.xpos[tool_body]-held_start)))
                contacts = hand_contacts()
                counts.append(len(contacts))
                thumb_counts.append(any('thumb' in name for name in contacts))
                hand_torque.append(float(np.max(np.abs(data.actuator_force[right_hand_actuators]))))
        before_release_z = float(data.xpos[tool_body, 2])
        data.ctrl[right_hand_actuators] = 0
        for _ in range(1000):
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        report = {'initialization': 'Tool positioned during 1 s of finger closure, followed by 1 s unsupported settling.',
            'hold_seconds': 5, 'tool_free_joint': True, 'tool_gravity_compensation': float(model.body_gravcomp[tool_body]),
            'tool_welds': sum(int(model.eq_type[i] == mujoco.mjtEq.mjEQ_WELD
                and tool_body in (model.eq_obj1id[i],model.eq_obj2id[i])) for i in range(model.neq)),
            'gravity_m_s2': model.opt.gravity.tolist(),
            'max_hold_drift_m': max(hold_drift), 'min_contacting_hand_bodies': min(counts),
            'samples_with_thumb_contact': sum(thumb_counts), 'hold_samples': len(counts),
            'max_hand_actuator_torque_Nm': max(hand_torque),
            'release_drop_m': before_release_z-float(data.xpos[tool_body, 2]),
            'contacts_after_release': hand_contacts(),
            'physics_warnings': [int(w.number) for w in data.warning],
            'scope': 'Single-pose simulated pre-grasp hold/release check, not autonomous pickup or hardware validation.'}
        report['passed'] = (report['max_hold_drift_m'] < .01 and min(counts) >= 3
            and all(thumb_counts) and report['release_drop_m'] > .04 and not hand_contacts()
            and report['tool_welds'] == 0 and not any(report['physics_warnings']))
        (args.output/'grasp-check.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))
        if not report['passed']:
            raise SystemExit(1)
        return

    logs = []
    sponge_geom, table_geom = model.geom('sponge_pad').id, model.geom('table_top').id
    def expected_contact(contact):
        if {int(contact.geom1),int(contact.geom2)} == {sponge_geom, table_geom}:
            return True
        a, b = model.geom_bodyid[[contact.geom1,contact.geom2]]
        return (a == tool_body and b in hand_bodies) or (b == tool_body and a in hand_bodies)

    def step_frame(t):
        qref, error = solve(waypoint(t))
        data.ctrl[:] = qref[control_qadr]
        for _ in range(25):
            mujoco.mj_step(model, data)
        assert np.isfinite(data.qpos).all()
        if any(w.number for w in data.warning):
            raise RuntimeError('MuJoCo instability warning')
        mujoco.mj_forward(model, data)
        contacts = [c for c in data.contact if {int(c.geom1),int(c.geom2)}=={sponge_geom,table_geom}]
        relative_pos, relative_rot = relative_tool_pose()
        hand_contact_bodies = hand_contacts()
        logs.append({'time':round(t,3),'target':waypoint(t).tolist(),
            'actual':data.site_xpos[actual_site].tolist(),'ik_error_m':error,
            'tracking_error_m':float(np.linalg.norm(data.site_xpos[actual_site]-waypoint(t))),
            'grasp_translation_drift_m':float(np.linalg.norm(relative_pos-initial_relative[0])),
            'grasp_rotation_drift_deg':float(np.degrees(np.arccos(np.clip((np.trace(relative_rot @ initial_relative[1].T)-1)/2,-1,1)))),
            'hand_tool_contact_bodies':hand_contact_bodies,
            'arm_target':qref[qa].tolist(),'arm_actual':data.qpos[qa].tolist(),
            'other_contacts':[[model.geom(c.geom1).name or str(c.geom1), model.geom(c.geom2).name or str(c.geom2)]
                for c in data.contact if not expected_contact(c)],
            'sponge_table_contacts':len(contacts)})

    camera = mujoco.MjvCamera()
    camera.lookat[:] = [.20, -.06, .94]
    camera.distance = 1.65
    camera.azimuth = 125
    camera.elevation = -24
    close_camera = mujoco.MjvCamera()
    close_camera.lookat[:] = [.36, -.16, .87]
    close_camera.distance = .53
    close_camera.azimuth = 125
    close_camera.elevation = -15
    if args.close_up:
        camera = close_camera
    if args.dynamics_only:
        for frame_index in range(int(DURATION*20)+1):
            step_frame(frame_index/20)
    elif args.viewer:
        import mujoco.viewer
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.lookat[:]=camera.lookat
            viewer.cam.distance=camera.distance
            viewer.cam.azimuth=camera.azimuth
            viewer.cam.elevation=camera.elevation
            while viewer.is_running():
                start=time.monotonic()
                t=min(data.time,DURATION)
                step_frame(t)
                viewer.sync()
                if data.time > DURATION+1:
                    initial_state, initial_relative = initialize()
                    logs.clear()
                time.sleep(max(0,.05-(time.monotonic()-start)))
        print('窗口已关闭。完整接触记录可使用 --dynamics-only 或 --render 生成。')
        return
    else:
        import av
        from PIL import Image, ImageDraw, ImageFont
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 19)
        small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 15)
        video = args.output/'sponge-wipe-preview.mp4'
        with mujoco.Renderer(model, height=600, width=800) as renderer, av.open(str(video),'w') as container:
            stream=container.add_stream('libx264',rate=20)
            stream.width,stream.height,stream.pix_fmt=800,600,'yuv420p'
            stream.options={'crf':'21'}
            for frame_index in range(int(DURATION*20)+1):
                t=frame_index/20
                step_frame(t)
                renderer.update_scene(data,camera=camera)
                img=Image.fromarray(renderer.render())
                draw=ImageDraw.Draw(img)
                draw.rectangle((0,0,800,75),fill=(19,26,35))
                draw.text((16,10),'G1 + INSPIRE | SPONGE WIPE | SCRIPTED PREVIEW',font=font,fill='white')
                phase='Approach' if t<2 else ('Wipe one region' if t<10 else 'Lift away')
                draw.text((16,43),f'{phase}   {t:04.1f}s   |   free tool held by finger contacts',font=small,fill=(185,207,218))
                draw.rectangle((0,567,800,600),fill=(19,26,35))
                draw.text((16,575),'Scripted pre-grasp: no ACT/VLA, autonomous pickup or cleaning validation.',font=small,fill=(236,202,124))
                if frame_index in [0,100,200,240]:
                    img.save(args.output/f'preview-{frame_index:03d}.png')
                    renderer.update_scene(data,camera=close_camera)
                    Image.fromarray(renderer.render()).save(args.output/f'hand-closeup-{frame_index:03d}.png')
                frame=av.VideoFrame.from_ndarray(np.array(img),format='rgb24')
                for packet in stream.encode(frame):container.mux(packet)
            for packet in stream.encode():container.mux(packet)
        print('Video:',video,flush=True)
    wiping = [x for x in logs if 2.2 <= x['time'] <= 9.8]
    summary={'scene':str(path),'steps_recorded':len(logs),
        'max_ik_error_m':max(x['ik_error_m'] for x in logs),
        'max_tracking_error_m':max(x['tracking_error_m'] for x in logs),
        'frames_with_sponge_contact':sum(x['sponge_table_contacts']>0 for x in logs),
        'sampled_wiping_frames':len(wiping),
        'sampled_wiping_frames_with_contact':sum(x['sponge_table_contacts']>0 for x in wiping),
        'other_contact_frames':sum(bool(x['other_contacts']) for x in logs),
        'frames_with_hand_tool_contact':sum(bool(x['hand_tool_contact_bodies']) for x in logs),
        'frames_with_thumb_contact':sum(any('thumb' in b for b in x['hand_tool_contact_bodies']) for x in logs),
        'max_grasp_translation_drift_m':max(x['grasp_translation_drift_m'] for x in logs),
        'max_grasp_rotation_drift_deg':max(x['grasp_rotation_drift_deg'] for x in logs),
        'contact_note':'Contact and slip are measured, not enforced; force control remains future work.',
        'scope':'Scripted fixed-base, pre-grasped free-body tool with finger contacts. No ACT/VLA, autonomous pickup or cleaning validation.'}
    (args.output/'trajectory.json').write_text(json.dumps(logs,indent=2)+'\n')
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
