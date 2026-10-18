"""Run auto-key orchestration against fake tmux/checks, never a live session."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

TOOLS = Path(__file__).resolve().parents[1] / 'tools'


def run_auto(tmp_path, fail='', *, dead=False):
    repo = tmp_path/'repo with spaces'
    (repo/'tools').mkdir(parents=True)
    (repo/'venv/bin').mkdir(parents=True)
    (repo/'venv/bin/python').symlink_to(sys.executable)
    shutil.copy2(TOOLS/'sponge_solo_teleop.sh', repo/'tools/sponge_solo_teleop.sh')
    events = tmp_path/'events.jsonl'
    state = tmp_path/'state.json'
    state.write_text(json.dumps(dict(exists=True, dead=dead)))
    checker = repo/'tools/sponge_startup_check.py'
    checker.write_text('''import json, os, sys
from pathlib import Path
mode=sys.argv[1]
with Path(os.environ['EVENTS']).open('a') as f: f.write(json.dumps(['check', mode])+'\\n')
fail=os.environ.get('FAIL')
state=json.loads(Path(os.environ['STATE']).read_text())
# Model both the delayed stop caused by O and an unrelated later controller exit.
stop_sent=state.get('stop_sent', False)
delayed_exit=fail=='late_controller' and state.get('launched', False)
sys.exit(1 if mode==fail or (mode=='controller' and (stop_sent or delayed_exit)) else 0)
''')
    commands = tmp_path/'bin'; commands.mkdir()
    tmux = commands/'tmux'
    tmux.write_text(f'#!{sys.executable}\n'+'''import json, os, sys
from pathlib import Path
args=sys.argv[1:]
with Path(os.environ['EVENTS']).open('a') as f: f.write(json.dumps(args)+'\\n')
p=Path(os.environ['STATE']); s=json.loads(p.read_text()); status=0
if args[0]=='has-session': status=0 if s['exists'] else 1
elif args[0]=='display-message': print(int(s['dead']))
elif args[0]=='kill-session': s['exists']=False
elif args[0]=='new-session':
 s.update(exists=True, dead=False, launched=True)
 Path(os.environ['LOG']).write_text('Press F (or c) for CALIB_FULL\\n')
elif args[0]=='send-keys':
 key=args[-1]
 if key=='o': s.update(dead=True, stop_sent=True)
 if key=='C-c': s['dead']=True
 msg={'f':'CALIB_FULL done', ']':'ENGAGE:', 's':'CALIB_SYNC done'}.get(key)
 if msg:
  with Path(os.environ['LOG']).open('a') as f: f.write(msg+'\\n')
p.write_text(json.dumps(s)); sys.exit(status)
''')
    tmux.chmod(0o755)
    env = dict(os.environ, PATH=f'{commands}:{os.environ["PATH"]}',
        SONIC_VENV_SIM=str(repo/'venv'), EVENTS=str(events), STATE=str(state),
        LOG=str(tmp_path/'bridge.log'), WAIT_PUTON='0', BALANCE_SEC='0', FAIL=fail)
    result = subprocess.run(['bash', str(repo/'tools/sponge_solo_teleop.sh'), '192.168.2.5'],
        input='\n', text=True, capture_output=True, env=env, timeout=10)
    calls = [json.loads(s) for s in events.read_text().splitlines()]
    return result, calls


def test_auto_preserves_keys_and_checks_actual_readiness(tmp_path):
    result, calls = run_auto(tmp_path)
    assert result.returncode == 0, result.stderr+result.stdout
    assert calls[0] == ['check', 'preflight']
    keys = [c[-1] for c in calls if c[0]=='send-keys']
    assert keys == ['C-c','f',']','s','t']
    assert calls.index(['check', 'drop']) > next(i for i,c in enumerate(calls) if c[0]=='send-keys' and c[-1]==']')
    assert calls.index(['check', 'teleop']) < next(i for i,c in enumerate(calls) if c[0]=='send-keys' and c[-1]=='s')


@pytest.mark.parametrize('fail', ['preflight', 'controller', 'drop', 'teleop'])
def test_auto_aborts_before_unsafe_next_phase(tmp_path, fail):
    result, calls = run_auto(tmp_path, fail)
    assert result.returncode != 0
    keys = [c[-1] for c in calls if c[0]=='send-keys']
    assert 's' not in keys and 't' not in keys
    if fail == 'preflight': assert calls == [['check','preflight']]
    if fail == 'controller':
        assert keys == ['C-c']
        assert not any(c[0]=='new-session' for c in calls)


def test_dead_pane_is_replaced_without_sending_stop_or_interrupt(tmp_path):
    result, calls = run_auto(tmp_path, dead=True)
    assert result.returncode == 0, result.stderr+result.stdout
    assert [c[-1] for c in calls if c[0]=='send-keys'] == ['f',']','s','t']


def test_controller_exit_while_waiting_stops_before_calibration(tmp_path):
    result, calls = run_auto(tmp_path, 'late_controller')
    assert result.returncode != 0
    assert any(c[0]=='new-session' for c in calls)
    assert [c[-1] for c in calls if c[0]=='send-keys'] == ['C-c']
    assert '控制器已退出' in result.stdout
