# Reproduce

## Smoke Test

```powershell
.\scripts\smoke_test.ps1
```

This runs imports, the compact synthetic demo, and non-integration tests.

## Compact Demo

```powershell
python compact_demo\run_demo.py --verify
```

The demo verifies enrollment, RGB-side verification, `r(I,c_s)` scoring, and
accept/reject output on synthetic data.

## Full Main Evaluation

Prepare `DATA/` as described in `DATA.md`, then run:

```powershell
.\scripts\reproduce_main_results.ps1
```

Main settings:

```text
seed = 42
val_ratio = 0.15
max_raws = 1024
source = live_isp
```

The script writes:

```text
outputs/main_results_summary.json
```

