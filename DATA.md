# Data Layout

The full data bundle is not included in this anonymous Git repository.

Expected local layout:

```text
DATA/
  dataset/
    dataset_meta.json
    raw/
    rgb_web_jpg/
  checkpoints/
    stage1_joint_best.pt
    stage3_authcode_last.pt
  caches_optional/
    stage3_prototype_cache_anchor12_joint512_live/
    stage3_teacher_cache_joint512_live/
  results/
    stage3_eval/
      last_default.json
```

Alternatively set:

```powershell
$env:VTRACE_DATA_ROOT = "<path-to-dataset>"
$env:VTRACE_EXPERIMENT_ROOT = "<path-to-results>"
```

The compact synthetic demo under `compact_demo/` is only for functionality
checking.

