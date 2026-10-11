#!/usr/bin/env python3
"""Live grasp viewer, recording replay, or headless diagnostic screenshot."""
import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--endpoint", default="tcp://127.0.0.1:5558")
    p.add_argument("--replay", type=Path)
    p.add_argument("--snapshot", type=Path, help="Save a diagnostic PNG without opening a GUI (requires --replay)")
    p.add_argument("--video", type=Path, help="Export sampled diagnostic replay to MP4 (requires --replay)")
    p.add_argument("--frame", type=int, default=-1)
    p.add_argument("--side", choices=("left", "right"), default="right")
    args = p.parse_args()
    if args.snapshot or args.video:
        if not args.replay: p.error("--snapshot/--video requires --replay")
        os.environ.setdefault("MUJOCO_GL", "egl")
    from g1_teleop.sim.grasp_viewer import GraspDisplay, read_samples, run_desktop
    if args.snapshot or args.video:
        rows = read_samples(args.replay / "samples.jsonl")
        if not rows:
            p.error("recording has no complete samples")
        if args.snapshot and not -len(rows) <= args.frame < len(rows):
            p.error(f"--frame must be between {-len(rows)} and {len(rows)-1}")
        display = GraspDisplay(args.replay / "model.mjb")
        try:
            if args.snapshot:
                index = args.frame if args.frame >= 0 else len(rows) + args.frame
                display.history.extend(r for r in rows[max(0, index-299):index]
                                       if r["episode"] == rows[index]["episode"])
                display.episode = rows[index]["episode"]
                display.dashboard(rows[args.frame], args.side).save(args.snapshot)
                print(args.snapshot)
            if args.video:
                import cv2
                import numpy as np
                writer = cv2.VideoWriter(str(args.video), cv2.VideoWriter_fourcc(*"mp4v"), 5, (1280, 800))
                if not writer.isOpened(): raise RuntimeError("MP4 encoder unavailable")
                try:
                    from itertools import groupby
                    for _, group in groupby(rows, key=lambda r: r["episode"]):
                        episode = list(group)
                        i = 0
                        for t in np.arange(episode[0]["sim_time"], episode[-1]["sim_time"] + .2, .2):
                            while i + 1 < len(episode) and episode[i+1]["sim_time"] <= t + 1e-9:
                                i += 1
                            writer.write(cv2.cvtColor(np.asarray(display.dashboard(episode[i], args.side)), cv2.COLOR_RGB2BGR))
                finally:
                    writer.release()
                print(args.video)
        finally:
            display.close()
    else:
        run_desktop(args.endpoint, args.replay)


if __name__ == "__main__":
    main()
