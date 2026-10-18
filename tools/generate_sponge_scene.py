#!/usr/bin/env python3
"""Build a SONIC-compatible free sponge scene from the installed Inspire model."""
from pathlib import Path
import argparse
import xml.etree.ElementTree as ET

REPO = Path(__file__).resolve().parents[1]
ROBOT_MODEL = REPO / 'assets/mujoco/g1_runtime/g1_29dof_with_inspire_hand.xml'
TABLE_HEIGHT = .72
# Checked against recorded F -> ] -> 9 -> S -> T runs of the cube workflow:
# after the elastic band is released the robot steps forward to pelvis
# x ~0.38 m; the front edge leaves room for that step instead of catching the
# hips. The sponge sits outside the right-arm swing at ], and all targets are
# reachable from the landed stance (otherwise step forward as with the cube).
TABLE_FRONT_X = .56
TABLE_CENTER_Y, TABLE_HALF_Y = -.16, .30
SPONGE_XY = (.63, -.40)
WIPE_TARGET_XY, WIPE_TARGET_HALF = (.66, -.23), (.04, .05)


def generate_scene(output_dir=None, *, robot_model=ROBOT_MODEL):
    robot_model = Path(robot_model).resolve()
    tree = ET.parse(robot_model)
    root = tree.getroot()
    root.set('model', 'g1_inspire_sponge_visionpro_teleop')
    # Keep all body joints, torque actuators, linkage constraints and their order.
    # Resolve assets so a generated scene also works outside the model directory.
    compiler = root.find('compiler')
    directories = {}
    if compiler is not None:
        for key in ('meshdir', 'texturedir', 'assetdir'):
            if compiler.get(key):
                directories[key] = (robot_model.parent / compiler.get(key)).resolve()
                compiler.set(key, str(directories[key]))
    for element in root.iter():
        if element.get('file'):
            base = directories.get({'mesh': 'meshdir', 'texture': 'texturedir'}.get(element.tag),
                                   directories.get('assetdir', robot_model.parent))
            if element.tag == 'include':
                base = robot_model.parent
            element.set('file', str((base / element.get('file')).resolve()))
    # head_camera is left exactly as in the robot model (same headset view as the cube scene).

    def add(parent, tag, **attrs):
        return ET.SubElement(parent, tag, {k: str(v) for k, v in attrs.items()})

    add(root, 'statistic', center='0.50 -0.16 0.85', extent='1.6')
    visual = add(root, 'visual')
    add(visual, 'headlight', ambient='0.4 0.4 0.4', diffuse='0.65 0.65 0.65')
    add(visual, 'global', azimuth='130', elevation='-25', offwidth='960', offheight='720')
    world = root.find('worldbody')
    add(world, 'light', pos='0 -1 2.5', dir='0 0 -1', directional='true')
    add(world, 'geom', name='floor', type='plane', size='0 0 0.05', rgba='.16 .20 .25 1')
    add(world, 'site', name='com_marker', size='.015', rgba='1 0 0 .4')
    table = add(world, 'body', name='sponge_table', pos=f'{TABLE_FRONT_X+.22} {TABLE_CENTER_Y} {TABLE_HEIGHT-.025}')
    add(table, 'geom', name='sponge_table_top', type='box', size=f'.22 {TABLE_HALF_Y} .025',
        rgba='.80 .85 .86 1', friction='.5 .005 .0001', contype='1', conaffinity='15')
    # Legs are set back from the front edge, away from the knees.
    for x in (-.03, .18):
        for y in (-(TABLE_HALF_Y-.04), TABLE_HALF_Y-.04):
            add(table, 'geom', name=f'sponge_table_leg_{x}_{y}', type='box',
                pos=f'{x} {y} {-TABLE_HEIGHT/2}',
                size=f'.02 .02 {(TABLE_HEIGHT-.05)/2}', rgba='.27 .31 .34 1')
    (wx, wy), (hx, hy) = WIPE_TARGET_XY, WIPE_TARGET_HALF
    add(world, 'site', name='sponge_wipe_target', type='box', pos=f'{wx} {wy} {TABLE_HEIGHT+.001}',
        size=f'{hx} {hy} .0005', rgba='.08 .65 .60 .35')
    tool = add(world, 'body', name='sponge_tool', pos=f'{SPONGE_XY[0]} {SPONGE_XY[1]} {TABLE_HEIGHT+.0985}')
    add(tool, 'freejoint', name='sponge_free')
    common = dict(contype='8', conaffinity='5', condim='4')
    add(tool, 'geom', name='sponge_handle', type='capsule', fromto='0 0 -.05 0 0 .05',
        size='.012', mass='.025', rgba='.20 .25 .28 1', friction='1 .005 .0001', **common)
    add(tool, 'geom', name='sponge_backplate', type='box', pos='0 0 -.068', size='.06 .035 .006',
        mass='.035', rgba='.16 .25 .28 1', **common)
    add(tool, 'geom', name='sponge_pad', type='box', pos='0 0 -.086', size='.06 .035 .012',
        mass='.025', rgba='.98 .78 .15 1', friction='.55 .005 .0001',
        solref='.02 1', solimp='.8 .95 .01', **common)
    add(tool, 'site', name='sponge_handle_grip', size='.004', rgba='.1 .9 .3 .7')
    add(tool, 'site', name='sponge_surface', pos='0 0 -.098', size='.002', rgba='0 0 0 0')
    output = Path(output_dir or ROBOT_MODEL.parent / 'sponge_teleop').resolve()
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'scene_sponge_teleop.xml'
    if path == robot_model:
        raise ValueError('Output must not overwrite the source model')
    ET.indent(tree, space='  ')
    tree.write(path, encoding='utf-8')
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--robot-model', type=Path, default=ROBOT_MODEL)
    print(generate_scene(**vars(parser.parse_args())))
