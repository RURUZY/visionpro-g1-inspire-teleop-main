#!/usr/bin/env python3
"""Generate a dry-roller contact scene from already-installed G1 assets.

This is a grasp/contact experiment, not a paint deposition or fluid simulation.
Generation uses only the standard library and never downloads model assets.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape


REPO_ROOT = Path(__file__).resolve().parents[1]
ROBOT_MODEL = REPO_ROOT / "assets/mujoco/g1_runtime/g1_29dof_with_inspire_hand.xml"
TEMPLATE = REPO_ROOT / "assets/mujoco/scene_43dof_inspire_hand_paint.xml.in"
SCENE_NAME = "scene_43dof_inspire_hand_paint.xml"
# Handle axis (tool +X) up, roller on top, roller axle along world Y so it is
# already parallel to the wall when lifted; the bent frame points outward (-Y),
# away from the hands' landing sweep.
TOOL_QUAT = (0.0, math.sqrt(0.5), 0.0, math.sqrt(0.5))
# Holder geometry (must match the template): 60 mm deep cup, 2 mm wall clearance.
CUP_DEPTH = 0.06
CUP_OFFSET = 0.021
COLUMN_HALF = 0.06
# Tool frame distances along the handle axis.
HANDLE_TIP = 0.069 + 0.016   # capsule half-length + radius
GRIP_OFFSET = 0.025          # paint_handle_grip site, above the handle centre


def _write_paint_camera_model(robot_model: Path, destination: Path) -> Path:
    """Keep the shared robot intact; change only this scene's simulated view."""
    tree = ET.parse(robot_model)
    camera = tree.find(".//camera[@name='head_camera']")
    if camera is None:
        raise ValueError("Prepared G1 model is missing head_camera")
    for attribute in ("quat", "axisangle", "xyaxes", "zaxis"):
        camera.attrib.pop(attribute, None)
    # About 21 degrees below horizontal, wide enough for the tabletop and the
    # eye-height target. These are simulation viewing settings, not calibration.
    camera.set("euler", "0 -1.2 -1.57079632679")
    camera.set("fovy", "80")
    compiler = tree.find("compiler")
    asset_dirs = {}
    if compiler is not None:
        for attribute in ("meshdir", "texturedir", "assetdir"):
            if compiler.get(attribute):
                path = (robot_model.parent / compiler.get(attribute)).resolve()
                compiler.set(attribute, str(path))
                asset_dirs[attribute] = path
    # The supplied runtime model normally has absolute asset paths. Resolve
    # relative ones too, since --output-dir can put this copy elsewhere.
    for element in tree.iter():
        if element.get("file"):
            directory = asset_dirs.get({"mesh": "meshdir", "texture": "texturedir"}.get(element.tag),
                                       asset_dirs.get("assetdir", robot_model.parent))
            if element.tag == "include":
                directory = robot_model.parent
            element.set("file", str((directory / element.get("file")).resolve()))
    path = destination / "g1_inspire_paint_view.xml"
    if path == robot_model:
        raise ValueError("Paint camera copy must not overwrite the source robot model")
    tree.write(path, encoding="utf-8")
    return path


def generate_scene(
    output_dir: Path | str | None = None,
    *,
    robot_model: Path | str = ROBOT_MODEL,
    wall_x: float = 0.60,
    wall_half_width: float = 0.50,
    target_height: float = 1.22,
    target_y: float = -0.15,
    stand_x: float = 0.38,
    stand_y: float = -0.30,
    grip_height: float = 0.92,
) -> Path:
    """Return the generated scene; ``wall_x`` is its robot-facing surface.

    The roller stands upright in a cup on a pedestal in front of the robot's
    right hand: the handle is vertical, the roller is on top and its axle is
    already parallel to the wall. ``grip_height`` is the world height of the
    handle grip marker. The defaults were chosen with right-arm IK from the
    recorded standing pose: the grip and a roller stroke on the target are
    reachable without walking and away from joint limits. The wall is free
    standing, with nothing between it and the robot.
    """
    robot_model = Path(robot_model).expanduser().resolve()
    if not robot_model.is_file():
        raise FileNotFoundError(
            f"Prepared G1 Inspire model not found: {robot_model}\n"
            "First set GR00T_ROOT to your GR00T-WholeBodyControl checkout and run "
            "tools/setup_inspire_sim.py using the simulation Python environment. "
            "This generator does not download assets."
        )
    values = (wall_x, wall_half_width, target_height, target_y, stand_x, stand_y, grip_height)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Scene parameters must be finite")
    stand_top = grip_height - HANDLE_TIP - GRIP_OFFSET
    if stand_top < 0.30:
        raise ValueError("grip_height is too low for the pedestal")
    if target_height <= 0.34:
        raise ValueError("target_height must leave room for the panel supports above the floor")
    if wall_half_width < 0.30 or abs(target_y) + 0.12 > wall_half_width:
        raise ValueError("The target patch must lie on the panel")
    if stand_x + COLUMN_HALF + 0.05 > wall_x:
        raise ValueError("Pedestal must stand at least 5 cm in front of the wall")
    destination = Path(output_dir or ROBOT_MODEL.parent).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    scene_path = destination / SCENE_NAME
    if scene_path == robot_model:
        raise ValueError("Scene output must not overwrite the robot model")
    paint_robot_model = _write_paint_camera_model(robot_model, destination)
    post_y = wall_half_width + 0.03
    # 0.5 mm above the cup floor so the tool settles instead of starting in contact.
    tool_pos = (stand_x, stand_y, stand_top + HANDLE_TIP + 0.0005)
    substitutions = {
        "G1_ROBOT_INCLUDE": escape(str(paint_robot_model), {'"': "&quot;"}),
        "WALL_CENTER_X": f"{wall_x + 0.025:.9g}",
        "WALL_CENTER_Y": "0",
        "WALL_CENTER_Z": f"{target_height:.9g}",
        "WALL_LEG_CENTER_Z": f"{-(target_height + 0.30) / 2:.9g}",
        "WALL_LEG_HALF_HEIGHT": f"{(target_height - 0.30) / 2:.9g}",
        "WALL_FOOT_Z": f"{0.02 - target_height:.9g}",
        "WALL_HALF_WIDTH": f"{wall_half_width:.9g}",
        "WALL_POST_Y": f"{post_y:.9g}",
        "WALL_POST_NEG_Y": f"{-post_y:.9g}",
        "TARGET_X": f"{wall_x - 0.0015:.9g}",
        "TARGET_Y": f"{target_y:.9g}",
        "TARGET_Z": f"{target_height:.9g}",
        "STAND_X": f"{stand_x:.9g}",
        "STAND_Y": f"{stand_y:.9g}",
        "COLUMN_CENTER_Z": f"{stand_top / 2:.9g}",
        "COLUMN_HALF_HEIGHT": f"{stand_top / 2:.9g}",
        "CUP_OFFSET": f"{CUP_OFFSET:.9g}",
        "CUP_CENTER_Z": f"{stand_top + CUP_DEPTH / 2:.9g}",
        "TOOL_POS": " ".join(f"{value:.9g}" for value in tool_pos),
        "TOOL_QUAT": " ".join(f"{value:.9g}" for value in TOOL_QUAT),
    }
    content = TEMPLATE.read_text(encoding="utf-8")
    for key, value in substitutions.items():
        content = content.replace(f"@{key}@", value)
    if "@" in content.replace(str(paint_robot_model), ""):
        raise ValueError("Unfilled template placeholder in paint scene")
    scene_path.write_text(content, encoding="utf-8")
    return scene_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="Default: assets/mujoco/g1_runtime")
    parser.add_argument("--wall-x", type=float, default=0.60, help="Wall front surface x in metres")
    parser.add_argument("--wall-half-width", type=float, default=0.50, help="Panel half width along y in metres")
    parser.add_argument("--target-height", type=float, default=1.22, help="Panel/target centre z in metres")
    parser.add_argument("--target-y", type=float, default=-0.15, help="Target centre y in metres (right hand side is negative)")
    parser.add_argument("--stand-x", type=float, default=0.38, help="Pedestal/holder centre x in metres")
    parser.add_argument("--stand-y", type=float, default=-0.30, help="Pedestal/holder centre y in metres")
    parser.add_argument("--grip-height", type=float, default=0.92, help="Handle grip marker height in metres")
    args = parser.parse_args()
    try:
        path = generate_scene(**vars(args))
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    print(path)


if __name__ == "__main__":
    main()
