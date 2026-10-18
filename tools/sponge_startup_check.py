#!/usr/bin/env python3
"""Read-only checks used by the sponge auto-key workflow."""
import argparse
import json
import os
from pathlib import Path
import sys
import time


def controller_error(proc=Path('/proc')):
    count = 0
    for path in proc.glob('[0-9]*/cmdline'):
        try:
            args = path.read_bytes().split(b'\0')
        except (OSError, PermissionError):
            continue
        if not args or args[0].split(b'/')[-1] != b'g1_deploy_onnx_ref':
            continue
        if len(args) < 2 or args[1] not in (b'lo', b'127.0.0.1'):
            return '发现非回环网卡控制器，不能接入海绵仿真'
        count += 1
    if count != 1:
        return '终端二的仿真控制器未运行或不唯一；请重新启动 control 并等待 Init Done'
    return ''


def status_error(status, mode, now=None):
    now = time.time() if now is None else now
    age = now-status.get('updated_unix', 0)
    if not 0 <= age <= 2:
        return '海绵仿真状态已过期，请重新启动终端一'
    if status.get('guard_version') != 1:
        return '终端一还在运行旧脚本，请重启 scene 以加载落地检查'
    if status.get('halted'):
        return '海绵仿真已冻结：'+status['halted']+'；请结束并重启三个终端'
    if mode == 'preflight' and not status.get('elastic_band'):
        return '当前已解除悬挂，不可重新自动校准；请先结束并重启三个终端'
    if mode == 'drop' and not status.get('ready_drop'):
        return status.get('ready_reason') or '等待落地条件满足'
    if mode == 'teleop' and not status.get('ready_teleop'):
        return status.get('ready_reason') or '等待按 9 落地并连续稳定至少 2 秒'
    return ''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['controller', 'preflight', 'drop', 'teleop'])
    parser.add_argument('--status', type=Path, required=True)
    parser.add_argument('--wait', type=float, default=0)
    args = parser.parse_args()
    deadline = time.monotonic()+args.wait
    while True:
        reason = controller_error()
        if reason:
            print('[sponge] '+reason, file=sys.stderr)
            return 1
        if args.mode == 'controller':
            return 0
        try:
            status = json.loads(args.status.read_text())
            reason = status_error(status, args.mode)
            if not reason:
                pid = status.get('pid', 0)
                if not isinstance(pid, int) or pid <= 0:
                    reason = '仿真进程编号无效'
                else:
                    os.kill(pid, 0)  # Existence check only, never sends a signal.
        except (OSError, ValueError, TypeError) as exc:
            reason = f'无法读取运行中的海绵仿真状态：{exc}'
        if not reason:
            return 0
        if time.monotonic() >= deadline:
            print('[sponge] '+reason, file=sys.stderr)
            return 1
        time.sleep(.25)


if __name__ == '__main__':
    sys.exit(main())
