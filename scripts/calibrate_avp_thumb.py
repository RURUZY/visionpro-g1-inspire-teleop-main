#!/usr/bin/env python3
"""Interactive simulation thumb calibration. Never publishes robot commands."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from calibrate_avp_hand import _CalibArgs, VisionProStreamer
from g1_teleop.bridge.cli import default_hand_calibration_file
from g1_teleop.hand.mapping import InspireHandMapper
from g1_teleop.hand.tracking_guard import validate_hand_skeleton
from g1_teleop.hand.thumb_calibration import fit_thumb_calibration


def sample_pose(streamer, mapper, side, channel, seconds):
    seen = set()
    samples = []
    rejected = 0
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        tracking = streamer.get_latest()
        hand = getattr(tracking, side, None) if tracking is not None else None
        try:
            validate_hand_skeleton(hand)
            key = np.asarray(hand).tobytes()
            if key not in seen:
                seen.add(key)
                raw, _ = mapper.measure_raw(hand)
                if not np.isfinite(raw[channel]):
                    raise ValueError("invalid thumb metric")
                samples.append(float(raw[channel]))
        except (ValueError, TypeError, IndexError, AttributeError):
            rejected += 1
        time.sleep(1 / 60)
    print(f"采到 {len(samples)} 个不同的有效骨架；跳过 {rejected} 次失效数据。", flush=True)
    if len(samples) < 30:
        raise ValueError("有效更新不足：请让手保持在头显视野内，重新采集。")
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--avp-endpoint", required=True)
    parser.add_argument("--side", choices=("left", "right"), default="right")
    parser.add_argument("--sample-seconds", type=float, default=3.0)
    args = parser.parse_args()
    if args.sample_seconds < 1:
        parser.error("--sample-seconds must be at least 1")
    config = _CalibArgs()
    config.enable_inspire_hand_sim = True
    config.enable_inspire_hand_dds = False
    base_path = default_hand_calibration_file(args.side)
    mapper = InspireHandMapper.from_args(config, calibration_file=base_path)
    output = ROOT / "scripts" / f"visionpro_{args.side}_hand_calibration.json"
    session = ROOT / "recordings" / ("thumb_calibration_" + datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z"))
    session.mkdir(parents=True, exist_ok=False)
    samples = {}
    poses = [
        ("thumb_bend_open", "拇指伸直，其他手指自然放松。"),
        ("thumb_bend_close", "拇指弯到你抓东西时舒适的最大弯曲程度，保持可见。"),
        ("thumb_rot_open", "拇指朝手掌外侧展开，尽量保持拇指伸直。"),
        ("thumb_rot_close", "拇指从根部转向掌心、靠近小指根部；尽量不要只弯指尖。"),
    ]
    print(f"仅采集 {args.side} 拇指，保留原有四指范围。此程序不控制机器人。", flush=True)
    streamer = VisionProStreamer(ip=args.avp_endpoint, record=False, benchmark_quiet=True)
    try:
        for key, instruction in poses:
            while True:
                input(f"\n{instruction}\n准备好后按回车，5 秒后开始采集。")
                for n in range(5, 0, -1):
                    print(n, flush=True)
                    time.sleep(1)
                print("开始采集，请保持姿势……", flush=True)
                try:
                    samples[key] = sample_pose(streamer, mapper, args.side, key.rsplit("_", 1)[0], args.sample_seconds)
                    break
                except ValueError as exc:
                    print(exc, flush=True)
        (session / "samples.json").write_text(json.dumps(samples, indent=2))
        fitted, report = fit_thumb_calibration(mapper.calibration, samples)
        (session / "report.json").write_text(json.dumps(report, indent=2))
        (session / "previous_effective.json").write_text(json.dumps(mapper.calibration.channels, indent=2))
        if output.exists():
            shutil.copy2(output, session / "previous_file.json")
        temporary = output.with_suffix(".json.tmp")
        fitted.save(str(temporary))
        temporary.replace(output)
        print(f"\n已保存：{output}\n记录：{session}", flush=True)
        print("仅更新拇指输入范围；四指范围和输出力度未改。重启桥接后生效。", flush=True)
        print("先空手验证拇指方向和范围，再测试抓取；尚未验证真实抓取效果。", flush=True)
    finally:
        streamer.cleanup()


if __name__ == "__main__":
    main()
