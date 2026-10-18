"""Check ACT training/inference and local data I/O with synthetic samples."""
import argparse
import json
import os
from importlib.metadata import version
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/g1-act-matplotlib")

import av
import h5py
import numpy as np
import torch
import torchvision
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(2)
torch.manual_seed(0)

# 13 is a synthetic interface dimension, not a verified robot action mapping.
config = ACTConfig(
    device="cpu", push_to_hub=False,
    input_features={
        "observation.images.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(13,)),
    },
    output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(13,))},
    pretrained_backbone_weights=None,
    chunk_size=20, n_action_steps=5,
    dim_model=128, n_heads=4, dim_feedforward=512,
    n_encoder_layers=2, n_decoder_layers=1, n_vae_encoder_layers=2,
)
policy = ACTPolicy(config)
policy.train()
batch = {
    "observation.images.front": torch.rand(2, 3, 64, 64),
    "observation.state": torch.rand(2, 13),
    "action": torch.rand(2, 20, 13),
    "action_is_pad": torch.zeros(2, 20, dtype=torch.bool),
}
optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-5)
loss, metrics = policy(batch)
assert torch.isfinite(loss)
loss.backward()
gradients = [p.grad for p in policy.parameters() if p.grad is not None]
assert gradients and all(torch.isfinite(g).all() for g in gradients)
assert any(torch.count_nonzero(g).item() for g in gradients)
optimizer.step()
policy.eval()
policy.reset()
with torch.inference_mode():
    observations = {k: v[:1] for k, v in batch.items() if k.startswith("observation.")}
    action = policy.select_action(observations)
assert action.shape == (1, 13) and torch.isfinite(action).all()

data = np.arange(130, dtype=np.float32).reshape(10, 13)
with h5py.File(args.output / "synthetic-data.h5", "w") as handle:
    handle.create_dataset("action", data=data)
with h5py.File(args.output / "synthetic-data.h5", "r") as handle:
    np.testing.assert_array_equal(handle["action"][:], data)

video_path = args.output / "synthetic-video.mp4"
with av.open(str(video_path), "w") as container:
    stream = container.add_stream("libx264", rate=20)
    stream.width = stream.height = 64
    stream.pix_fmt = "yuv420p"
    for index in range(10):
        frame = av.VideoFrame.from_ndarray(np.full((64, 64, 3), index * 20, dtype=np.uint8), format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
with av.open(str(video_path)) as container:
    assert len(list(container.decode(video=0))) == 10

result = {
    "versions": {p: version(p) for p in ["torch", "torchvision", "lerobot", "numpy", "mujoco", "av", "h5py", "datasets"]},
    "device": "cpu", "torch_cuda_build": torch.version.cuda,
    "act_loss": float(loss.detach()), "act_backward": "passed",
    "act_action_shape": list(action.shape), "hdf5_roundtrip": "passed",
    "h264_roundtrip_frames": 10,
    "scope": "Synthetic dependency test; not a trained policy or verified robot action mapping.",
}
(args.output / "act-check.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
