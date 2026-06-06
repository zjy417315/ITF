import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.prototype_dataset import PrototypeDistillationDataset
from src.tools.build_stage3_prototype_cache import build_stage3_prototype_cache


def save_dummy_rgb(path: Path, value: int):
    rgb = np.full((16, 16, 3), value, dtype=np.uint8)
    Image.fromarray(rgb).save(path, format="JPEG")


def test_prototype_cache_uses_anchor_versions_and_dataset_filters_versions(tmp_path):
    teacher_cache_dir = tmp_path / "teacher_cache"
    teacher_cache_dir.mkdir()
    rgb_dir = tmp_path / "rgb"
    rgb_dir.mkdir()
    output_dir = tmp_path / "prototype_cache"

    meta = {}
    teacher_specs = {
        "rawA": {
            1: torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
            2: torch.tensor([[3.0, 0.0], [3.0, 0.0]]),
            3: torch.tensor([[9.0, 0.0], [9.0, 0.0]]),
        },
        "rawB": {
            1: torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
            2: torch.tensor([[0.0, 3.0], [0.0, 3.0]]),
        },
    }

    for raw_anchor, version_map in teacher_specs.items():
        for version, teacher_seq in version_map.items():
            version_id = f"{raw_anchor}_v{version}"
            rgb_path = rgb_dir / f"{version_id}.jpg"
            save_dummy_rgb(rgb_path, value=version * 40)
            meta[version_id] = {
                "raw_anchor": raw_anchor,
                "version": version,
                "paths": {"jpg": str(rgb_path)},
            }
            torch.save(
                {
                    "raw_anchor": raw_anchor,
                    "version_id": version_id,
                    "teacher_seq": teacher_seq.float(),
                },
                teacher_cache_dir / f"{version_id}.pt",
            )

    meta_path = tmp_path / "dataset_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    build_stage3_prototype_cache(
        teacher_cache_dir=teacher_cache_dir,
        output_dir=output_dir,
        meta_path=meta_path,
        prototype_versions=[1, 2],
        overwrite=True,
    )

    raw_a_pack = torch.load(output_dir / "rawA.pt", map_location="cpu")
    expected_proto_seq = torch.tensor([[2.0, 0.0], [2.0, 0.0]])
    expected_proto_vec = torch.tensor([1.0, 0.0])

    assert raw_a_pack["source_versions"] == [1, 2]
    assert raw_a_pack["version_ids"] == ["rawA_v1", "rawA_v2"]
    assert torch.allclose(raw_a_pack["prototype_seq"], expected_proto_seq)
    assert torch.allclose(raw_a_pack["prototype_vec"], expected_proto_vec)

    dataset = PrototypeDistillationDataset(
        prototype_cache_dir=str(output_dir),
        meta_path=str(meta_path),
        rgb_dir=str(rgb_dir),
        include_versions=[1, 3],
    )

    version_ids = [dataset[idx]["sample_id"] for idx in range(len(dataset))]
    assert version_ids == ["rawA_v1", "rawA_v3", "rawB_v1"]
    assert dataset.prototype_dim == 2
    assert torch.allclose(dataset.get_prototype_tensor("rawA"), expected_proto_vec)
