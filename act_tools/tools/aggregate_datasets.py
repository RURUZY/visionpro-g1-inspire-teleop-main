#!/usr/bin/env python3
"""Merge LeRobot datasets with the same features into one (episodes, videos, stats).

  python tools/aggregate_datasets.py OUT_ROOT SRC_ROOT [SRC_ROOT ...]
"""
import sys
from pathlib import Path

from lerobot.datasets.aggregate import aggregate_datasets

REPO_ID = 'local/g1_inspire_wipe_sim'

if __name__ == '__main__':
    out, srcs = Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]]
    if out.exists():
        raise SystemExit(f'{out} already exists')
    aggregate_datasets([REPO_ID] * len(srcs), REPO_ID, roots=srcs, aggr_root=out)
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    meta = LeRobotDatasetMetadata(REPO_ID, root=out)
    print(f'merged {len(srcs)} datasets -> {out}: {meta.total_episodes} episodes, {meta.total_frames} frames')
