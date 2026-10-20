# G1ACT_job (restart 2026-10-20): train ACT on the merged 1000 randomized pickup demos, then evaluate.
# The first attempt (2026-10-19) was terminated after 14k steps (exit 0xC000013A) with no checkpoint,
# so this run saves every 20k steps and uses 10 dataloader workers (video decoding was the bottleneck).
$ErrorActionPreference = "Continue"
$env:HF_HUB_OFFLINE = "1"; $env:HF_DATASETS_OFFLINE = "1"; $env:PYTHONUNBUFFERED = "1"
$env:MUJOCO_GL = "glfw"; $env:G1_ROBOT_XML = "D:/g1act/assets/g1_29dof_with_inspire_hand.xml"
$py = "D:\g1act\venv\Scripts\python.exe"; $log = "D:\g1runs\job_rand1000.log"
function Log($m) { "[$(Get-Date -Format s)] $m" | Out-File -Append -Encoding utf8 $log }
Log "restart: training"
& D:\g1act\train_act.ps1 -Name act_pick_rand1000 -Steps 200000 -Save 20000 -Log 1000 -Batch 8 -Workers 10 -Data "D:\g1data\pick_rand1000\dataset"
Log "training: $(Get-Content D:\g1runs\act_pick_rand1000.log -Tail 1)"
$env:OMP_NUM_THREADS = "4"
$evals = @()
foreach ($ck in "120000", "200000") {
  foreach ($cond in "nominal", "rand") {
    $name = "evalpick_rand1000_$($ck)_$cond"
    $extra = if ($cond -eq "rand") { "--perturb light,color,camera,physics" } else { "" }
    $evals += Start-Process -FilePath $py -PassThru -WindowStyle Hidden `
      -RedirectStandardOutput "D:\g1runs\$name.log" -RedirectStandardError "D:\g1runs\$name.err" `
      -ArgumentList "D:\g1act\code\eval\eval_policy.py --policy checkpoint --device cuda --checkpoint D:\g1runs\act_pick_rand1000\checkpoints\$ck\pretrained_model --variant pickup --side left --seed 2000 --trials 50 $extra --output D:\g1runs\$name"
  }
}
$evals | Wait-Process
Log "JOB DONE"
