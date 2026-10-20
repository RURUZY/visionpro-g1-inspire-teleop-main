#!/usr/bin/env bash
# Unattended: wait for both pickup datasets, train ACT on labpc one after the other (120k steps),
# copy the 120k checkpoints back, evaluate each on nominal and randomized scenes (seed 2000, 50 trials).
set -u
cd /home/epcap/projects/g1-act-tools
log() { echo "[$(date +%T)] $*"; }
ssh_ok() { ssh -o ConnectTimeout=15 labpc "$@"; }
for name in left200 left200_rand; do
  until [ -f pickup_teacher/runs/$name/summary.json ]; do sleep 60; done
  log "dataset $name ready: $(grep -o '"kept_episodes": [0-9]*' pickup_teacher/runs/$name/summary.json)"
  (cd pickup_teacher/runs/$name && rm -f ../pick_$name.zip && zip -qr ../pick_$name.zip dataset)
  ssh_ok "if not exist D:\\g1data\\pick_$name mkdir D:\\g1data\\pick_$name"
  scp -q pickup_teacher/runs/pick_$name.zip labpc:D:/g1data/pick_$name/ && \
  ssh_ok "powershell -NoProfile -Command \"Expand-Archive -Force D:\\g1data\\pick_$name\\pick_$name.zip D:\\g1data\\pick_$name\"" && log "uploaded $name"
done
for name in left200 left200_rand; do
  run=act_pick_$name
  printf '{"Name":"%s","Steps":120000,"Save":20000,"Log":1000,"Batch":8,"Data":"D:\\\\g1data\\\\pick_%s\\\\dataset"}' $run $name > runs/next_run.json
  scp -q runs/next_run.json labpc:D:/g1act/next_run.json
  ssh_ok "schtasks /Change /TN G1ACT_train /ENABLE && schtasks /Run /TN G1ACT_train && schtasks /Change /TN G1ACT_train /DISABLE" >/dev/null
  log "training $run started"
  sleep 120
  until ssh_ok "findstr /c:finished D:\\g1runs\\$run.log" >/dev/null 2>&1; do sleep 120; done
  log "training $run: $(ssh_ok "findstr /c:finished D:\\g1runs\\$run.log")"
  mkdir -p runs/$run/ckpt_120000
  scp -q -r "labpc:D:/g1runs/$run/checkpoints/120000/pretrained_model" runs/$run/ckpt_120000/ && log "copied $run"
done
ev() {  # model condition
  local ck=runs/act_pick_$1/ckpt_120000/pretrained_model out=runs/evalpick_$1_$2 extra=""
  [ "$2" = rand ] && extra="--perturb light,color,camera,physics"
  OMP_NUM_THREADS=3 .venv/bin/python eval/eval_policy.py --policy checkpoint --checkpoint $ck --variant pickup \
    --side left --seed 2000 --trials 50 $extra --output $out > $out.log 2>&1
  log "eval $1 $2 done"
}
OMP_NUM_THREADS=3 .venv/bin/python eval/eval_policy.py --policy teacher --variant pickup --side left --seed 2000 \
  --trials 50 --output runs/evalpick_teacher_nominal > runs/evalpick_teacher_nominal.log 2>&1 &
ev left200 nominal & ev left200 rand & ev left200_rand nominal & wait
ev left200_rand rand
log "ALL DONE"
