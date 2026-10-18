#!/usr/bin/env python3
"""Palm-down sponge-block wiping teacher with tactile recording (variant of wipe_teacher).

The right hand rests on top of a plain sponge block (no handle), fingers curled over its
front edge, and pushes it across the table: the sponge never leaves the table, like a
person wiping. wipe_teacher stays unchanged; this file swaps the tool, grasp, table,
regions and motion plan, and adds per-frame tactile fields.

ASSISTED GRIP: the block is attached to the palm by a soft weld. Contact-only holding of a
palm-down block with this Inspire hand model was not stable in MuJoCo (the block slipped out
within a second in most placements). Episodes are labeled grip='assisted_weld'; they show the
wiping motion and produce contact/tactile signals, but cannot show slip or grasp failure.
"""
from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'wipe_teacher'))
import wipe_teacher as W  # noqa: E402

sys.path.insert(0, str(HERE.parent / 'tactile_slip'))
from tactile import FEATURES as EXTRA_FEATURES, TACTILE_BODIES, TactileMixin  # noqa: E402,F401

_base_scene = W.build_scene

# Right-hand frame: fingers +z, palm normal +y, thumb side -x. The palm collision box
# ends at y=+0.0415 and spans z in [0, 0.17]; MCP joints sit at z~0.156.
BLOCK_SIZE = (.10, .09, .05)                          # along fingers, width, thickness (m)
BLOCK_IN_HAND = (0., .0415 + BLOCK_SIZE[2] / 2 + .001, .105)  # top face against the palm
# Block axes in the hand frame: x forward (=hand z), y = -hand x, z up (= -hand y).
BLOCK_ROT = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], float)
WELD_SOLREF = '.02 1'         # soft attachment: the block can give a little under load
HAND_FRICTION = 1.5           # sponge against the hand
TABLE_FRICTION = .4           # sponge against the smooth table top

TABLE_Z = .74                 # reach band for this grasp: x 0.18-0.38 m, y +0.06 to -0.34 m
TABLE_CENTER_X = .44          # front edge at x=0.14
REGIONS = {'left': np.array([.31, -.08]), 'right': np.array([.31, -.26])}
REGION_HALF = np.array([.06, .06])  # 12 x 12 cm, a bit larger than the 10 x 9 cm sponge
WIPE_HEIGHT = float(__import__('os').environ.get('WIPE_HEIGHT', '0'))  # sponge-bottom target above the table; deeper presses jam the arm
HOME = np.array([.195, -.17])  # sponge starts here, in front of and between the regions
WAIST_YAW = -.15

def block_quat():
    import mujoco
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, BLOCK_ROT.flatten())
    return q


def build_scene(output: Path) -> Path:
    path = _base_scene(output)
    tree = ET.parse(path)
    root = tree.getroot()
    root.set('model', 'g1_sponge_block_wipe_teacher')
    world = root.find('worldbody')
    world.remove(world.find("body[@name='sponge_tool']"))
    palm = root.find(".//body[@name='right_base_link']")
    palm.remove(palm.find("body[@name='tool_reference_frame']"))
    length, width, thick = BLOCK_SIZE
    quat = ' '.join(f'{v:.10f}' for v in block_quat())
    pos = ' '.join(map(str, BLOCK_IN_HAND))
    ref = ET.SubElement(palm, 'body', {'name': 'tool_reference_frame', 'pos': pos, 'quat': quat})
    # IK target: centre of the block's bottom face, axes aligned with the world.
    ET.SubElement(ref, 'site', {'name': 'sponge_reference', 'pos': f'0 0 {-thick / 2}',
                                'size': '.002', 'rgba': '0 0 0 0'})
    tool = ET.SubElement(world, 'body', {'name': 'sponge_tool', 'pos': '0.3 -0.2 1.2'})
    ET.SubElement(tool, 'freejoint', {'name': 'sponge_free'})
    ET.SubElement(tool, 'geom', {
        'name': 'sponge_pad', 'type': 'box', 'size': f'{length / 2} {width / 2} {thick / 2}',
        'mass': '.04', 'rgba': '.15 .45 .90 1', 'contype': '8', 'conaffinity': '5',
        'friction': f'{HAND_FRICTION} .005 .0001', 'solref': '.02 1', 'solimp': '.8 .95 .01'})
    ET.SubElement(tool, 'site', {'name': 'sponge_surface', 'pos': f'0 0 {-thick / 2}',
                                 'size': '.002', 'rgba': '0 0 0 0'})
    # Sponge-table friction set explicitly (MuJoCo would otherwise use the larger of the two).
    contact = root.find('contact')
    if contact is None:
        contact = ET.SubElement(root, 'contact')
    ET.SubElement(contact, 'pair', {'geom1': 'sponge_pad', 'geom2': 'table_top', 'condim': '3',
                                    'friction': f'{TABLE_FRICTION} {TABLE_FRICTION} .005 .0001 .0001'})
    equality = root.find('equality')
    if equality is None:
        equality = ET.SubElement(root, 'equality')
    ET.SubElement(equality, 'weld', {'name': 'assisted_grip', 'body1': 'right_base_link',
                                     'body2': 'sponge_tool', 'relpose': f'{pos} {quat}',
                                     'solref': WELD_SOLREF})
    table = world.find("body[@name='wipe_table']")
    _, y, _ = map(float, table.get('pos').split())
    table.set('pos', f'{TABLE_CENTER_X} {y} {TABLE_Z - .025}')
    for leg in table.findall('geom')[1:]:
        lx, ly, _ = map(float, leg.get('pos').split())
        leg.set('pos', f'{lx} {ly} {-TABLE_Z / 2}')
        sx, sy, _ = map(float, leg.get('size').split())
        leg.set('size', f'{sx} {sy} {TABLE_Z / 2 - .025}')
    for side, center in REGIONS.items():
        world.find(f"site[@name='region_{side}']").set('pos', f'{center[0]} {center[1]} {TABLE_Z + .0005}')
    tree.write(path, encoding='utf-8')
    return path


def plan_on_table(rng, center, pad_width_x):
    """Sponge stays on the table: slide from home, parallel strokes over the region, slide back."""
    hx, hy = W.REGION_HALF
    home = np.r_[HOME + rng.uniform(-.02, .02, 2), W.WIPE_Z]
    speed = rng.uniform(.05, .07)
    n = max(2, int(np.ceil((2 * hx) / (pad_width_x * .8))) + 1)
    xs = np.linspace(center[0] - hx + pad_width_x * .35, center[0] + hx - pad_width_x * .35, n)
    reach_y = hy - BLOCK_SIZE[1] / 2 + .01  # pad edge, not pad centre, must pass the region edge
    y0, y1 = center[1] - reach_y, center[1] + reach_y
    if abs(home[1] - y1) < abs(home[1] - y0):
        y0, y1 = y1, y0  # start on the edge nearer home
    # Transit in front of the regions (x = home), then straight in, so the pad never crosses the other region.
    pts = [home, [home[0], y0, W.WIPE_Z], [xs[0], y0, W.WIPE_Z]]
    for k, x in enumerate(xs):
        a, b = (y0, y1) if k % 2 == 0 else (y1, y0)
        if k:
            pts.append([x, a, W.WIPE_Z])
        pts.append([x, b, W.WIPE_Z])
    pts.append([home[0], pts[-1][1], W.WIPE_Z])
    pts.append([home[0], home[1], W.WIPE_Z])
    pts = np.array(pts, float)
    times = [0.]
    for i in range(1, len(pts)):
        times.append(times[-1] + max(.6, np.linalg.norm(pts[i] - pts[i - 1]) / speed))
    times[1] += rng.uniform(0, .4)
    return W.Path3D(pts, times)


class BlockTeacher(TactileMixin, W.WipeTeacher):
    def __init__(self, scene_path):
        super().__init__(scene_path)
        self.setup_tactile()

    def run_episode(self, rng, side, *, record=True, viewer=None):
        frames, info = super().run_episode(rng, side, record=record, viewer=viewer)
        info['grip'] = 'assisted_weld'
        if frames:
            arm = np.array([f['action'][:7] for f in frames])
            jump = float(np.abs(np.diff(arm, axis=0)).max())
            info['max_arm_target_step_rad'] = round(jump, 4)
            if jump > MAX_ARM_STEP and info['success']:
                info['success'], info['failure'] = False, 'jerky_arm_motion'
        return frames, info


MAX_ARM_STEP = .15  # rad per 50 ms frame; larger steps mean an IK branch switch


def configure():
    """Point the shared teacher at this variant's tool, table, regions, plan and fields."""
    W.TABLE_Z, W.WIPE_Z = TABLE_Z, TABLE_Z + WIPE_HEIGHT
    W.REGIONS = REGIONS
    W.REGION_HALF = REGION_HALF
    W.WAIST_YAW = WAIST_YAW
    W.build_scene = build_scene
    W.plan_episode = plan_on_table
    W.EXTRA_FEATURES = EXTRA_FEATURES
    W.WipeTeacher = BlockTeacher


if __name__ == '__main__':
    configure()
    if '--output' not in sys.argv and '--viewer' not in sys.argv:
        sys.argv += ['--output', str(HERE / 'output')]
    if '--repo-id' not in sys.argv:
        sys.argv += ['--repo-id', 'local/g1_inspire_wipe_block_sim']
    W.main()
