# G1ACT_job: 800 more randomized pickup demos (8 parallel seeds) -> merge with the existing 200 ->
# train ACT 200k steps -> evaluate 120k and 200k checkpoints on nominal and randomized scenes.
$ErrorActionPreference = "Continue"
$env:HF_HUB_OFFLINE = "1"; $env:HF_DATASETS_OFFLINE = "1"; $env:PYTHONUNBUFFERED = "1"
$env:MUJOCO_GL = "glfw"; $env:G1_ROBOT_XML = "D:/g1act/assets/g1_29dof_with_inspire_hand.xml"; $env:OMP_NUM_THREADS = "2"
$py = "D:\g1act\venv\Scripts\python.exe"; $log = "D:\g1runs\job_rand1000.log"
function Log($m) { "[$(Get-Date -Format T)] $m" | Out-File -Append -Encoding utf8 $log }
"start $(Get-Date -Format s)" | Out-File -Encoding utf8 $log

$procs = @()
foreach ($s in 101..108) {
  $out = "D:\g1data\pickr_s$s"
  if (Test-Path $out) { Remove-Item -Recurse -Force $out }
  $procs += Start-Process -FilePath $py -PassThru -WindowStyle Hidden `
    -RedirectStandardOutput "D:\g1runs\gen_s$s.out" -RedirectStandardError "D:\g1runs\gen_s$s.err" `
    -ArgumentList "D:\g1act\code\pickup_teacher\pickup_teacher.py --episodes 100 --side left --seed $s --randomize light,color,camera,physics --output $out"
}
Log "generators started: $($procs.Count)"
$procs | Wait-Process
foreach ($s in 101..108) { Log "seed $s summary: $(Test-Path D:\g1data\pickr_s$s\summary.json)" }

$srcs = @("D:\g1data\pick_left200_rand\dataset") + (101..108 | ForEach-Object { "D:\g1data\pickr_s$_\dataset" })
$merged = "D:\g1data\pick_rand1000\dataset"
foreach ($try in 1..3) {  # one test merge once failed with a spurious hub lookup; retry
  if (Test-Path "D:\g1data\pick_rand1000") { Remove-Item -Recurse -Force "D:\g1data\pick_rand1000" }
  & $py D:\g1act\code\tools\aggregate_datasets.py $merged @srcs 2>&1 | ForEach-Object { "$_" } | Out-File -Append -Encoding utf8 $log
  if ($LASTEXITCODE -eq 0) { break }
  Log "merge attempt $try failed"
}
Log "merge done (exit $LASTEXITCODE)"

& D:\g1act\train_act.ps1 -Name act_pick_rand1000 -Steps 200000 -Save 40000 -Log 1000 -Batch 8 -Data $merged
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
