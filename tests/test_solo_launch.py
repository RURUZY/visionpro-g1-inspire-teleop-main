"""Check graceful replacement and argument quoting without launching a bridge."""

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_previous_bridge_exits_before_reconnect_and_arguments_are_preserved(tmp_path):
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"exists": True, "dead": False}))
    events = tmp_path / "events.jsonl"
    log = tmp_path / "bridge.log"
    tmux = tmp_path / "tmux"
    tmux.write_text(f"#!{sys.executable}\n" + '''
import json, os, sys
from pathlib import Path
state = Path(os.environ['FAKE_TMUX_STATE'])
events = Path(os.environ['FAKE_TMUX_EVENTS'])
args = sys.argv[1:]
with events.open('a') as stream:
    stream.write(json.dumps(args) + '\\n')
s = json.loads(state.read_text())
cmd = args[0]
status = 0
if cmd == 'has-session':
    status = 0 if s['exists'] else 1
elif cmd == 'display-message':
    print(int(s['dead']))
elif cmd == 'kill-session':
    s['exists'] = False
elif cmd == 'new-session':
    s.update(exists=True, dead=False)
    Path(os.environ['LOG']).write_text('Press F (or c) for CALIB_FULL\\n')
elif cmd == 'send-keys':
    key = args[-1]
    if key == 'o':
        s['dead'] = True
    message = {'f': 'CALIB_FULL done', ']': 'ENGAGE:', 's': 'CALIB_SYNC done'}.get(key)
    if message:
        with Path(os.environ['LOG']).open('a') as stream:
            stream.write(message + '\\n')
state.write_text(json.dumps(s))
sys.exit(status)
''')
    tmux.chmod(0o755)
    extra = ["--print-debug", "--literal", 'spaces $(not-a-command) `literal`']
    mapping = tmp_path / "mapping.jsonl"
    env = dict(os.environ, PATH=f"{tmp_path}:{os.environ['PATH']}", REPO=str(ROOT),
               LOG=str(log), MAPPING_LOG=str(mapping), WAIT_PUTON="0", BALANCE_SEC="0",
               FAKE_TMUX_STATE=str(state), FAKE_TMUX_EVENTS=str(events))
    result = subprocess.run(["bash", str(ROOT / "solo_teleop.sh"), "192.168.2.5", *extra],
                            env=env, input="\n", capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
    calls = [json.loads(line) for line in events.read_text().splitlines()]
    stop = next(i for i, c in enumerate(calls) if c[0] == "send-keys" and c[-1] == "o")
    kill = next(i for i, c in enumerate(calls) if c[0] == "kill-session")
    launch = next(i for i, c in enumerate(calls) if c[0] == "new-session")
    assert stop < kill < launch
    command = shlex.split(calls[launch][-1])
    assert command == ["bash", str(ROOT / "tools/solo_bridge_session.sh"), str(ROOT),
                       str(log), "tele", "192.168.2.5", "--mapping-log", str(mapping), *extra]
