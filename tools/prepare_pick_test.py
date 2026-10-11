#!/usr/bin/env python3
"""Headless scene checks and a blank ten-trial worksheet; no DDS or robot control.

Contact probes deliberately place the cube in overlap in a separate MjData.
They validate collision configuration only, never count as a successful grasp.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE = ROOT / 'assets/mujoco/g1_runtime/scene_43dof_inspire_hand_pnp_cube.xml'


def inspect_scene(scene: Path) -> dict:
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    cube = model.body('pick_cube_body').id
    cube_geom = model.geom('pick_cube').id
    joint = model.joint('pick_cube_joint').id
    table = model.geom('table_top').id
    if model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE or model.body_mocapid[cube] != -1:
        raise ValueError('Cube must be a free, non-mocap body')
    for eq in range(model.neq):
        if model.eq_type[eq] in (mujoco.mjtEq.mjEQ_WELD, mujoco.mjtEq.mjEQ_CONNECT):
            if cube in (model.eq_obj1id[eq], model.eq_obj2id[eq]):
                raise ValueError('Cube is bound by a body equality constraint')
        if model.eq_type[eq] == mujoco.mjtEq.mjEQ_JOINT:
            if joint in (model.eq_obj1id[eq], model.eq_obj2id[eq]):
                raise ValueError('Cube is bound by a joint equality constraint')
    for act in range(model.nu):
        if model.actuator_trntype[act] in (mujoco.mjtTrn.mjTRN_JOINT, mujoco.mjtTrn.mjTRN_JOINTINPARENT):
            if model.actuator_trnid[act, 0] == joint:
                raise ValueError('Cube has a direct joint actuator')
    probe_counts = {}
    right_geoms = []
    wrist = model.body('right_wrist_yaw_link').id
    for g in range(model.ngeom):
        body_id = int(model.geom_bodyid[g])
        body_name = model.body(body_id).name
        ancestor = body_id
        while ancestor not in (0, wrist):
            ancestor = int(model.body_parentid[ancestor])
        if ancestor != wrist or body_id == wrist:
            continue
        compatible = ((int(model.geom_contype[g]) & int(model.geom_conaffinity[cube_geom]))
                      or (int(model.geom_contype[cube_geom]) & int(model.geom_conaffinity[g])))
        if not compatible:
            continue
        right_geoms.append(g)
        if 'index' not in body_name and 'thumb' not in body_name:
            continue
        probe = mujoco.MjData(model)
        qadr = int(model.jnt_qposadr[joint])
        # Avoid exactly coincident convex centers, a degenerate contact query.
        probe.qpos[qadr:qadr + 3] = data.geom_xpos[g] + np.array([.025, .002, .003])
        mujoco.mj_forward(model, probe)
        n = sum(set((int(c.geom1), int(c.geom2))) == {cube_geom, g} for c in probe.contact)
        probe_counts[body_name + ':' + str(g)] = n
    if not right_geoms or not all(
        any(name.startswith('right_' + digit) and n > 0 for name, n in probe_counts.items())
        for digit in ('thumb', 'index')
    ):
        raise ValueError('Both right thumb and index must contact cube in overlap probes')

    # Isolate the existing table and cube, retaining their geometry/materials.
    # No arm movement or grasp occurs in this gravity-only settling check.
    xml = ET.parse(scene).getroot()
    for include in xml.findall('include'):
        xml.remove(include)
    isolated = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding='unicode'))
    isolated.opt.timestep = .002
    state = mujoco.MjData(isolated)
    for _ in range(1000):
        mujoco.mj_step(isolated, state)
    mujoco.mj_forward(isolated, state)
    cgeom = isolated.geom('pick_cube').id
    tgeom = isolated.geom('table_top').id
    contacts = [c for c in state.contact if set((int(c.geom1), int(c.geom2))) == {cgeom, tgeom}]
    if not contacts or not np.isfinite(state.qpos).all():
        raise ValueError('Isolated cube did not settle on table')
    table_height = float(data.geom_xpos[table, 2] + model.geom_size[table, 2])
    settled_height = float(state.geom_xpos[cgeom, 2])
    nominal_height = table_height + float(model.geom_size[cube_geom, 2])
    if abs(settled_height - nominal_height) > .003 or np.linalg.norm(state.qvel) > .01:
        raise ValueError('Cube resting pose is not stable within the diagnostic tolerance')
    return {
        'scope': 'Scene and collision preflight only; no reach or grasp success established',
        'scene': str(scene.resolve()),
        'scene_sha256': hashlib.sha256(scene.read_bytes()).hexdigest(),
        'robot_model_sha256': hashlib.sha256((scene.parent / 'g1_29dof_with_inspire_hand.xml').read_bytes()).hexdigest(),
        'mujoco_version': mujoco.__version__,
        'cube_free_joint': True,
        'cube_body_equality_or_direct_joint_actuator': False,
        'cube_size_m': (2 * model.geom_size[cube_geom]).tolist(),
        'cube_mass_kg': float(model.body_mass[cube]),
        'table_top_z_m': table_height,
        'cube_initial_center_m': data.xpos[cube].tolist(),
        'cube_resting_center_z_m': settled_height,
        'isolated_table_contact_count': len(contacts),
        'isolated_settling_duration_s': 2.0,
        'right_hand_collision_compatible_geoms': len(right_geoms),
        'artificial_overlap_contact_probes_not_grasps': probe_counts,
        'cube_friction_slide_torsion_roll': model.geom_friction[cube_geom].tolist(),
        'cube_contact_dimension': int(model.geom_condim[cube_geom]),
        'cube_free_joint_damping': model.dof_damping[int(model.jnt_dofadr[joint]):int(model.jnt_dofadr[joint]) + 6].tolist(),
        'physical_parameters_calibrated': False,
        'completed_grasp_trials': 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, default=DEFAULT_SCENE)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    report = inspect_scene(args.scene)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out = args.output_dir or ROOT / 'recordings' / ('right_hand_pick_' + stamp)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'preflight.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    fields = ['trial', 'status', 'max_lift_cm', 'continuous_hold_s', 'right_finger_contact_seen',
              'left_hand_used', 'stable_release_on_table', 'result', 'failure_reason',
              'video_or_log', 'elastic_band_enabled', 'scene_sha256', 'notes']
    with (out / 'trials.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for trial in range(1, 11):
            writer.writerow({'trial': trial, 'status': 'not_run', 'scene_sha256': report['scene_sha256']})
    (out / '操作步骤.txt').write_text('''单右手抓方块：先导仿真试验（10 次尚未执行）

目的：检查够不够得到、手指方向是否正确、物体是否靠接触抓住。
准备检查不代表已经完成抓取；不要把自动手指开合测试计入这 10 次。

三个终端都先进入项目目录：
  cd /home/epcap/projects/visionpro-g1-inspire-teleop-main
终端 1：./run_sonic_sim_loop_pnp.sh
终端 2：./run_sonic_deploy.sh sim
终端 3：./bin/sonic-teleop-pick-right.sh <Vision_Pro_IP或房间码>
沿用 F → ] → S → T 校准/接入流程，以各终端实际提示为准。
新入口只跟踪右手；左手保持初始目标。关闭头部走动/蹲起，键盘平移速度为零。
它仍是全身平衡策略，不是把机器人底盘焊死。不要按额外行走/转向键。
本入口不启用实体 Inspire 手 DDS 输出；deploy 明确使用 sim 模式。

先练习，练习不计入正式 10 次。固定方块、桌子、摩擦和弹力带设置。
每次：右手张开靠近 → 对准 → 缓慢闭合 → 抬起 5–10 cm → 连续保持 3 s → 放回桌面并松开。
高度以方块在桌面静止时为零，不以手掌高度为零；3 秒按仿真时间判断。
成功：右手有可见接触、方块离桌至少 5 cm、连续保持至少 3 s，最后松手后稳定留在桌面。
超过 10 cm 单独备注，不据此自动计失败；若只完成抬起，没有完成放回，整次不记成功。
失败原因选：够不到 / 对不准 / 夹不住 / 抬起后滑落 / 放回或松开失败 / 跟踪丢失 / 机器人失稳。
不要用鼠标外力提方块、直接改物体位姿或增加绑定来获得成功。
每次开始前用原有重置流程恢复场景，等方块落稳；复位后检查是否需重新校准。
保持相同弹力带设置，并在 trials.csv 记录。练习/参数更改后另开一组，不挑最好 10 次。
保存视频或日志；记录最大抬高、连续保持时间、接触情况和失败原因。
表中 not_run 是未执行，不是失败，不纳入成功率。完成 10 次后报成功数/10，并保留失败记录。

当前模型注意事项：
方块边长 7 cm、质量由模型计算；桌面实际高度约 0.74 m。
原场景使用 friction="0.9 0.35 0.08"、condim=4、自由关节 damping=0.06。
这些物理参数未实测标定；仿真成功不能直接当真实材料抓持能力。
MuJoCo 接触允许有限软接触压入；观察持续明显穿透与掉落，不能只要求像素级零重叠。
preflight.json 中的 overlap probes 是故意制造几何重叠来确认碰撞生效，不是抓取。
场景参数和检查过程没有给方块添加绑定。
接触/摩擦定义参考：https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-geom

后续：方块 → 滚筒柄 → 持滚筒空中走直线 → 墙面接触涂刷。
''', encoding='utf-8')
    print(out)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
