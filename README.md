# Anonymous Reproducibility Package

This is a pre-submission reproducibility package for S&P review. It is kept
small on purpose: code, dependencies, a compact synthetic demo, core tests,
configuration, and scripts for full reproduction when the data bundle is
available.

Code license: Apache-2.0.

Full data and full model weights are not included in this Git repository. They
follow the original dataset/model redistribution terms.

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
.\scripts\smoke_test.ps1
```

The compact demo checks the enrollment-verification flow only. It does not
reproduce the full numerical tables in the paper.

## Layout

```text
src/              core implementation
tests/            smoke and unit tests
scripts/          smoke and reproduction entrypoints
configs/          small reproducibility configs
compact_demo/     synthetic functionality demo
DATA.md           expected full-data layout
REPRODUCE.md      commands for full reproduction
```

