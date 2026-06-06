[CmdletBinding()]
param(
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

Write-Host "[smoke] Python version"
& $PythonExe --version

Write-Host "[smoke] Import core modules"
& $PythonExe -c "import src.itf, src.isp.simple_isp, src.topo, src.models.visual_encoder; print('imports ok')"

Write-Host "[smoke] Run compact synthetic demo"
& $PythonExe compact_demo\run_demo.py --verify

Write-Host "[smoke] Run non-integration tests"
& $PythonExe -m pytest -q -m "not integration"
