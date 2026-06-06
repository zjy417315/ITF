[CmdletBinding()]
param(
    [string]$PythonExe = "python",
    [string]$DataRoot = "",
    [string]$OutputJson = "outputs\main_results_summary.json",
    [string]$Device = "",
    [int]$MaxRaws = 1024
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

if (-not $DataRoot) {
    $DataRoot = Join-Path $repoRoot "DATA"
}

$datasetRoot = Join-Path $DataRoot "dataset"
$resultsRoot = Join-Path $DataRoot "results"
$checkpointRoot = Join-Path $DataRoot "checkpoints"
$cacheRoot = Join-Path $DataRoot "caches_optional"

$required = @(
    (Join-Path $datasetRoot "dataset_meta.json"),
    (Join-Path $datasetRoot "rgb_web_jpg"),
    (Join-Path $datasetRoot "raw"),
    (Join-Path $checkpointRoot "stage1_joint_best.pt"),
    (Join-Path $checkpointRoot "stage3_authcode_last.pt"),
    (Join-Path $resultsRoot "stage3_eval\last_default.json")
)

foreach ($path in $required) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Missing required artifact path: $path. See DATA.md for the expected layout."
    }
}

$env:VTRACE_DATA_ROOT = $datasetRoot
$env:VTRACE_EXPERIMENT_ROOT = $resultsRoot

$args = @(
    "-m", "src.tools.evaluate_main_results_four_conditions",
    "--stage3_eval_json", (Join-Path $resultsRoot "stage3_eval\last_default.json"),
    "--stage3_checkpoint", (Join-Path $checkpointRoot "stage3_authcode_last.pt"),
    "--joint_config_json", "configs\joint_score_eval_live32_weighted_competitive.json",
    "--stage1_checkpoint", (Join-Path $checkpointRoot "stage1_joint_best.pt"),
    "--prototype_cache_dir", (Join-Path $cacheRoot "stage3_prototype_cache_anchor12_joint512_live"),
    "--teacher_cache_dir", (Join-Path $cacheRoot "stage3_teacher_cache_joint512_live"),
    "--meta_path", (Join-Path $datasetRoot "dataset_meta.json"),
    "--rgb_dir", (Join-Path $datasetRoot "rgb_web_jpg"),
    "--source", "live_isp",
    "--max_raws", "$MaxRaws",
    "--table_schema", "primary",
    "--output_json", $OutputJson
)

if ($Device) {
    $args += @("--device", $Device)
}

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $OutputJson) | Out-Null
Write-Host "[reproduce] Running main four-condition evaluation"
& $PythonExe @args
