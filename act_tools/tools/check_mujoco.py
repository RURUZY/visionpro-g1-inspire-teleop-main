"""Offline dependency check: load/reset/step/render the existing G1 scene."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
scene = Path("/home/epcap/projects/visionpro-g1-inspire-teleop-main/assets/mujoco/g1_runtime/scene_43dof_inspire_hand_pnp_cube.xml")
model = mujoco.MjModel.from_xml_path(str(scene))
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
initial = data.qpos.copy()
for _ in range(100):
    mujoco.mj_step(model, data)
assert np.isfinite(data.qpos).all(), "Non-finite simulation state"
warnings = [int(w.number) for w in data.warning]
assert not any(warnings), warnings
mujoco.mj_resetData(model, data)
mujoco.mj_forward(model, data)
np.testing.assert_array_equal(data.qpos, initial)
camera = "head_camera" if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera") >= 0 else -1
with mujoco.Renderer(model, height=240, width=320) as renderer:
    renderer.update_scene(data, camera=camera)
    frame = renderer.render()
assert frame.shape == (240, 320, 3) and frame.std() > 1
Image.fromarray(frame).save(args.output / "g1-scene.png")
result = {
    "scene": str(scene), "mujoco": mujoco.__version__,
    "nq": model.nq, "nv": model.nv, "actuators": model.nu,
    "steps": 100, "reset": "passed", "render": list(frame.shape),
    "camera": camera, "warnings": warnings,
    "scope": "Offline dependency check only; no control, teleoperation or task success validation.",
}
(args.output / "mujoco-check.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
