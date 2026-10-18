"""Exercise auto reconnect with fake tmux/process inspection, no live sessions."""
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import pytest

ROOT = Path(os.environ.get('TELEOP_TEST_REPO', Path(__file__).resolve().parents[1]))
LAUNCHER = Path(os.environ.get('PROJECT_LAUNCHER_UNDER_TEST', ROOT / 'run_project.sh'))


def run_auto(tmp_path, *, exists=True, dead=False, interface='lo', controller=True):
    repo = tmp_path / 'repo with spaces'
    (repo / 'bin').mkdir(parents=True)
    for source, target in ((LAUNCHER, repo / 'run_project.sh'),
                           (ROOT / 'solo_teleop.sh', repo / 'solo_teleop.sh'),
                           (ROOT / 'bin/_common.sh', repo / 'bin/_common.sh')):
        shutil.copy2(source, target)
    (repo / '.env').write_text('SONIC_NET_IF=eno1\n')
    commands = tmp_path / 'commands'
    commands.mkdir()
    state, events, log = (tmp_path / name for name in ('state.json', 'events.jsonl', 'bridge.log'))
    state.write_text(json.dumps(dict(exists=exists, dead=dead)))
    tmux = commands / 'tmux'
    tmux.write_text(f'#!{sys.executable}\n' + '''
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
state = Path(os.environ['TEST_TMUX_STATE'])
with Path(os.environ['TEST_TMUX_EVENTS']).open('a') as f:
    f.write(json.dumps(args) + '\\n')
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
    if key == 'o': s['dead'] = True
    message = {'f':'CALIB_FULL done', ']':'ENGAGE:', 's':'CALIB_SYNC done'}.get(key)
    if message:
        with Path(os.environ['LOG']).open('a') as f: f.write(message + '\\n')
state.write_text(json.dumps(s))
sys.exit(status)
''')
    tmux.chmod(0o755)
    pgrep = commands / 'pgrep'
    pgrep.write_text('#!/bin/bash\n[[ "$TEST_CONTROLLER" == 1 ]] || exit 1\nprintf "%s\\n" "$TEST_INSPECT_PID"\n')
    pgrep.chmod(0o755)
    env = dict(os.environ, PATH=f'{commands}:{os.environ["PATH"]}',
               LOG=str(log), WAIT_PUTON='0', BALANCE_SEC='0',
               TEST_TMUX_STATE=str(state), TEST_TMUX_EVENTS=str(events),
               TEST_INSPECT_PID=str(os.getpid()), TEST_INTERFACE=interface,
               TEST_CONTROLLER=str(int(controller)))
    for key in ('SONIC_SIM_NET_IF', 'CYCLONEDDS_URI', 'G1_TELEOP_ROOT'):
        env.pop(key, None)
    # The PID is a readable local process; emulate only the decoded /proc argv.
    # No fake controller is launched, and the guard itself runs unchanged.
    harness = '''mapfile() { argv=(/test/g1_deploy_onnx_ref "$TEST_INTERFACE"); }
export -f mapfile
exec bash "$1" sim vr 192.168.2.5 --auto
'''
    result = subprocess.run(['bash', '-c', harness, 'test', str(repo / 'run_project.sh')],
                            env=env, input='\n', capture_output=True, text=True, timeout=10)
    calls = [json.loads(line) for line in events.read_text().splitlines()] if events.exists() else []
    return result, calls, repo


@pytest.mark.parametrize('exists,dead', [(True, False), (True, True), (False, False)])
def test_auto_reconnect_retains_original_calibration_order(tmp_path, exists, dead):
    result, calls, repo = run_auto(tmp_path, exists=exists, dead=dead)
    assert result.returncode == 0, result.stdout + result.stderr
    assert '自动校准：' in result.stdout
    assert '手动校准：' not in result.stdout
    launch = next(i for i, c in enumerate(calls) if c[0] == 'new-session')
    if exists:
        stop = next(i for i, c in enumerate(calls) if c[0] == 'send-keys' and c[-1] == 'o')
        kill = next(i for i, c in enumerate(calls) if c[0] == 'kill-session')
        assert stop < kill < launch
    keys = [c[-1] for c in calls[launch:] if c[0] == 'send-keys']
    assert keys == ['f', ']', 's', 't']
    command = shlex.split(calls[launch][-1])
    assert command[:3] == ['bash', str(repo / 'tools/solo_bridge_session.sh'), str(repo)]
    assert '--enable-inspire-hand-sim' in command
    assert '--disable-inspire-hand-dds' in command
    assert '--enable-inspire-hand-dds' not in command


@pytest.mark.parametrize('interface,controller', [('eno1', True), ('lo', False)])
def test_controller_check_precedes_any_session_changes(tmp_path, interface, controller):
    result, calls, _ = run_auto(tmp_path, interface=interface, controller=controller)
    assert result.returncode != 0
    assert not calls
