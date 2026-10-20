# Train ACT on the G1 wiping dataset. Usage: train_act.ps1 -Name act_smoke -Steps 50
# If D:\g1act\next_run.json exists (e.g. {"Name":"act_x","Steps":60000,"Data":"D:\\g1data\\x\\dataset"}), its values override the
# arguments and the file is renamed to next_run.used.json, so the scheduled task G1ACT_train can
# launch new runs without changing its command line (which would ask for the account password).
param([string]$Name = "act_left200", [int]$Steps = 20000, [int]$Save = 5000, [int]$Log = 200, [int]$Batch = 8,
      [string]$Data = "D:\g1data\left200\dataset")
$next = "D:\g1act\next_run.json"
if (Test-Path $next) {
  $cfg = Get-Content -Raw $next | ConvertFrom-Json
  foreach ($p in $cfg.PSObject.Properties) { Set-Variable -Name $p.Name -Value $p.Value }
  Move-Item -Force $next "D:\g1act\next_run.used.json"
}
$ErrorActionPreference = "Continue"; $env:HF_HUB_OFFLINE = "1"; $env:HF_DATASETS_OFFLINE = "1"; $env:PYTHONUNBUFFERED = "1"
$out = "D:\g1runs\$Name"
if (Test-Path $out) { Remove-Item -Recurse -Force $out }   # lerobot-train refuses an existing output dir
$logFile = "D:\g1runs\$Name.log"
"started $(Get-Date -Format s)" | Out-File -Encoding utf8 $logFile
& D:\g1act\venv\Scripts\lerobot-train.exe `
  --dataset.repo_id=local/g1_inspire_wipe_sim --dataset.root=$Data `
  --dataset.video_backend=pyav --policy.type=act --policy.device=cuda `
  --policy.chunk_size=20 --policy.n_action_steps=5 --policy.push_to_hub=false `
  --batch_size=$Batch --steps=$Steps --save_freq=$Save --log_freq=$Log --num_workers=4 `
  --wandb.enable=false --output_dir=$out 2>&1 | ForEach-Object { "$_" } | Out-File -Append -Encoding utf8 $logFile
"finished $(Get-Date -Format s) exit=$LASTEXITCODE" | Out-File -Append -Encoding utf8 $logFile
