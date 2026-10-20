# G1ACT_job: language-conditioned pickup-and-wipe data (both regions, paraphrased instructions,
# randomization), 8 parallel seeds x 250 demos -> merge -> train LC-ACT 300k steps.
$ErrorActionPreference = "Continue"
$env:HF_HUB_OFFLINE = "1"; $env:HF_DATASETS_OFFLINE = "1"; $env:PYTHONUNBUFFERED = "1"
$env:MUJOCO_GL = "glfw"; $env:G1_ROBOT_XML = "D:/g1act/assets/g1_29dof_with_inspire_hand.xml"; $env:OMP_NUM_THREADS = "2"
$py = "D:\g1act\venv\Scripts\python.exe"; $log = "D:\g1runs\job_lc2000.log"
function Log($m) { "[$(Get-Date -Format s)] $m" | Out-File -Append -Encoding utf8 $log }
"start $(Get-Date -Format s)" | Out-File -Encoding utf8 $log
$procs = @()
foreach ($s in 301..308) {
  $out = "D:\g1data\lc_s$s"
  if (Test-Path $out) { Remove-Item -Recurse -Force $out }
  $procs += Start-Process -FilePath $py -PassThru -WindowStyle Hidden `
    -RedirectStandardOutput "D:\g1runs\lcgen_s$s.out" -RedirectStandardError "D:\g1runs\lcgen_s$s.err" `
    -ArgumentList "D:\g1act\code\pickup_teacher\pickup_teacher.py --episodes 250 --side both --seed $s --language --randomize light,color,camera,physics --output $out"
}
Log "generators started: $($procs.Count)"
$procs | Wait-Process
foreach ($s in 301..308) { Log "seed $s summary: $(Test-Path D:\g1data\lc_s$s\summary.json)" }
$srcs = 301..308 | ForEach-Object { "D:\g1data\lc_s$_\dataset" }
$merged = "D:\g1data\lc2000\dataset"
foreach ($try in 1..3) {
  if (Test-Path "D:\g1data\lc2000") { Remove-Item -Recurse -Force "D:\g1data\lc2000" }
  & $py D:\g1act\code\tools\aggregate_datasets.py $merged @srcs 2>&1 | ForEach-Object { "$_" } | Out-File -Append -Encoding utf8 $log
  if ($LASTEXITCODE -eq 0) { break }
  Log "merge attempt $try failed"
}
Log "merge done (exit $LASTEXITCODE)"
& D:\g1act\train_act.ps1 -Name act_lc2000 -Steps 300000 -Save 50000 -Log 1000 -Batch 8 -Workers 10 -Data $merged
Log "training: $(Get-Content D:\g1runs\act_lc2000.log -Tail 1)"
Log "JOB DONE"
