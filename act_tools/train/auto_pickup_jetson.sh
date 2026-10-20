#!/usr/bin/env bash
# Unattended: wait for both pickup datasets, upload, train ACT on labpc (120k steps each, one after
# the other), evaluate on labpc while the next model trains (falls back to the Jetson if a labpc
# eval fails), then copy results and the 120k checkpoints back.
set -u
cd /home/epcap/projects/g1-act-tools
log() { echo "[$(date +%T)] $*"; }
ssh_ok() { ssh -o ConnectTimeout=15 -o ServerAliveInterval=30 labpc "$@"; }
WINENV='set HF_HUB_OFFLINE=1&& set MUJOCO_GL=glfw&& set OMP_NUM_THREADS=4&& set G1_ROBOT_XML=D:/g1act/assets/g1_29dof_with_inspire_hand.xml&&'
PY='D:\g1act\venv\Scripts\python.exe D:\g1act\code\eval\eval_policy.py'

for name in left200 left200_rand; do
  until [ -f pickup_teacher/runs/$name/summary.json ]; do sleep 60; done
  log "dataset $name ready: $(grep -o '"kept_episodes": [0-9]*' pickup_teacher/runs/$name/summary.json)"
  (cd pickup_teacher/runs/$name && rm -f ../pick_$name.zip && zip -qr ../pick_$name.zip dataset)
  ssh_ok "if not exist D:\\g1data\\pick_$name mkdir D:\\g1data\\pick_$name"
  scp -q pickup_teacher/runs/pick_$name.zip labpc:D:/g1data/pick_$name/ && \
  ssh_ok "powershell -NoProfile -Command \"Expand-Archive -Force D:\\g1data\\pick_$name\\pick_$name.zip D:\\g1data\\pick_$name\"" && log "uploaded $name"
done

train_start() {
  printf '{"Name":"%s","Steps":120000,"Save":20000,"Log":1000,"Batch":8,"Data":"D:\\\\g1data\\\\pick_%s\\\\dataset"}' act_pick_$1 $1 > runs/next_run.json
  scp -q runs/next_run.json labpc:D:/g1act/next_run.json
  ssh_ok "schtasks /Change /TN G1ACT_train /ENABLE && schtasks /Run /TN G1ACT_train && schtasks /Change /TN G1ACT_train /DISABLE" >/dev/null
  log "training act_pick_$1 started"; sleep 120
}
train_wait() {
  until ssh_ok "findstr /c:finished D:\\g1runs\\act_pick_$1.log" >/dev/null 2>&1; do sleep 120; done
  log "training act_pick_$1: $(ssh_ok "findstr /c:finished D:\\g1runs\\act_pick_$1.log")"
  mkdir -p runs/act_pick_$1/ckpt_120000
  scp -q -r "labpc:D:/g1runs/act_pick_$1/checkpoints/120000/pretrained_model" runs/act_pick_$1/ckpt_120000/ && log "copied act_pick_$1"
}
# ev <policy-tag> <condition>: tag is a model name (left200, left200_rand) or 'teacher'
ev() {
  local out=evalpick_$1_$2 pol extra="" jextra=""
  [ "$2" = rand ] && extra="--perturb light,color,camera,physics"
  if [ "$1" = teacher ]; then pol="--policy teacher"; jpol="$pol"
  else pol="--policy checkpoint --device cuda --checkpoint D:\\g1runs\\act_pick_$1\\checkpoints\\120000\\pretrained_model"
       jpol="--policy checkpoint --checkpoint runs/act_pick_$1/ckpt_120000/pretrained_model"; fi
  ssh_ok "$WINENV $PY $pol --variant pickup --side left --seed 2000 --trials 50 $extra --output D:\\g1runs\\$out > D:\\g1runs\\$out.log 2>&1"
  mkdir -p runs/$out
  if scp -q "labpc:D:/g1runs/$out/summary.json" "labpc:D:/g1runs/$out/trials.jsonl" runs/$out/ 2>/dev/null; then
    scp -q "labpc:D:/g1runs/$out.log" runs/ ; log "eval $out done on labpc"
  else
    log "eval $out failed on labpc, running on the Jetson"
    OMP_NUM_THREADS=4 .venv/bin/python eval/eval_policy.py $jpol --variant pickup --side left --seed 2000 \
      --trials 50 $extra --output runs/$out > runs/$out.log 2>&1; log "eval $out done on Jetson"
  fi
}

train_start left200; train_wait left200
train_start left200_rand
ev teacher nominal; ev left200 nominal; ev left200 rand
train_wait left200_rand
ev left200_rand nominal; ev left200_rand rand
log "ALL DONE"
