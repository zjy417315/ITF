import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "DATA" / "dataset"
DEFAULT_EXPERIMENT_ROOT = PROJECT_ROOT / "DATA" / "results"


def resolve_dataset_root() -> Path:
    env_override = os.environ.get("VTRACE_DATA_ROOT")
    if env_override:
        return Path(env_override)
    return DEFAULT_DATASET_ROOT


def resolve_meta_path() -> Path:
    return resolve_dataset_root() / "dataset_meta.json"


def resolve_stage_cache_dir() -> Path:
    return resolve_dataset_root() / "stage_cache"


def resolve_experiment_root() -> Path:
    env_override = os.environ.get("VTRACE_EXPERIMENT_ROOT")
    if env_override:
        return Path(env_override)
    return DEFAULT_EXPERIMENT_ROOT