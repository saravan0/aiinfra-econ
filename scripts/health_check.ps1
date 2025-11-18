# scripts\health_check.ps1
param(
  [string]$config = "config/model.yml"
)

$ErrorActionPreference = "Stop"

Write-Host "=== START health_check ==="

Write-Host "`n1) Run training pipeline..."
python -u -m src.model.train --config $config 2>&1 | Tee-Object -FilePath logs/train_run.log

Write-Host "`n2) Run robustness pipeline..."
python -u -m src.model.robustness --config $config 2>&1 | Tee-Object -FilePath logs/robust_run.log

Write-Host "`n3) Quick artifact checks..."
$expected = @(
  "models\ols_model.joblib",
  "models\fe_model.joblib",
  "models\elasticnet_cv.joblib",
  "reports\model_table.csv",
  "reports\model_plots.png",
  "reports\model_artifacts_manifest.json",
  "reports\robustness_vif.csv",
  "reports\robustness_card.md"
)

$missing = @()
foreach ($f in $expected) {
  if (-not (Test-Path $f)) { $missing += $f }
}

if ($missing.Count -gt 0) {
  Write-Host "ERROR: missing artifact(s):" -ForegroundColor Red
  $missing | ForEach-Object { Write-Host " - $_" }
  exit 2
}
Write-Host "Artifacts OK." -ForegroundColor Green

Write-Host "`n4) Check logs for ERROR|Traceback|Exception (train & robust)"
Select-String -Path logs\train_run.log,logs\robust_run.log -Pattern "ERROR","Traceback","Exception" -SimpleMatch | ForEach-Object {
  Write-Host $_.Line -ForegroundColor Yellow
}

Write-Host "`n5) Quick VIF sanity"
Import-Csv reports\robustness_vif.csv | Sort-Object {[double]$_."vif"} -Descending | Select-Object -First 10 | Format-Table

Write-Host "`n6) Show model_table top rows"
Import-Csv reports\model_table.csv | Select-Object -First 10 | Format-Table

Write-Host "`n7) Show fe_summary head"
if (Test-Path reports\fe_summary.txt) { Get-Content reports\fe_summary.txt -TotalCount 40 } else { Write-Host "no fe_summary.txt" }

Write-Host "`n=== END health_check ==="
