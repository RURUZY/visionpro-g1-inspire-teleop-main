"""Create and reload a tiny synthetic LeRobot v3 video dataset locally."""
import argparse
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
root = Path(tempfile.mkdtemp(prefix="synthetic-lerobot-", dir=args.output)) / "dataset"
dataset = LeRobotDataset.create(
    repo_id="local/dependency_check", root=root, fps=20,
    robot_type="synthetic_dependency_check",
    features={
        "observation.images.front": {"dtype": "video", "shape": (64, 64, 3), "names": ["height", "width", "channels"]},
        "observation.state": {"dtype": "float32", "shape": (13,), "names": [f"test_state_{i}" for i in range(13)]},
        "action": {"dtype": "float32", "shape": (13,), "names": [f"test_action_{i}" for i in range(13)]},
    },
    video_backend="pyav", vcodec="h264", encoder_threads=2,
)
for i in range(10):
    dataset.add_frame({
        "observation.images.front": np.full((64, 64, 3), i * 20, dtype=np.uint8),
        "observation.state": np.full(13, i / 10, dtype=np.float32),
        "action": np.full(13, (i + 1) / 10, dtype=np.float32),
        "task": "Synthetic dependency check; not robot demonstration data",
    })
dataset.save_episode()
dataset.finalize()
loaded = LeRobotDataset(repo_id="local/dependency_check", root=root, video_backend="pyav")
assert len(loaded) == 10 and loaded.num_episodes == 1
for i in [0, 5, 9]:
    sample = loaded[i]
    assert sample["observation.images.front"].shape == (3, 64, 64)
    np.testing.assert_allclose(sample["observation.state"].numpy(), i / 10, atol=1e-6)
    np.testing.assert_allclose(sample["action"].numpy(), (i + 1) / 10, atol=1e-6)
    np.testing.assert_allclose(sample["observation.images.front"].mean().item(), i * 20 / 255, atol=0.02)
result = {
    "dataset_root": str(root), "format": loaded.meta.info["codebase_version"],
    "frames": len(loaded), "episodes": loaded.num_episodes,
    "fps": loaded.fps, "video_backend": "pyav", "codec": "h264",
    "roundtrip": "passed", "scope": "Synthetic data only; not training demonstrations.",
}
(args.output / "dataset-check.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
