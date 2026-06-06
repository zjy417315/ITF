import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.code_matching_dataset import CodeMatchingDataset
from src.models.prototype_verifier import PrototypeVerifier
from src.train.train_stage3_code import compute_claim_logits, summarize_verification_scores, transform_code_targets


def save_dummy_rgb(path: Path, value: int):
    rgb = np.full((16, 16, 3), value, dtype=np.uint8)
    Image.fromarray(rgb).save(path, format="JPEG")


def test_code_matching_dataset_filters_versions_and_exposes_codes(tmp_path):
    rgb_dir = tmp_path / "rgb"
    rgb_dir.mkdir()
    prototype_cache_dir = tmp_path / "prototype_cache"
    prototype_cache_dir.mkdir()

    meta = {}
    prototype_index = {}
    for raw_anchor, vec in {
        "rawA": torch.tensor([1.0, 0.0]),
        "rawB": torch.tensor([0.0, 1.0]),
    }.items():
        pack_path = prototype_cache_dir / f"{raw_anchor}.pt"
        torch.save(
            {
                "raw_anchor": raw_anchor,
                "prototype_vec": vec.float(),
                "prototype_seq": vec.float().unsqueeze(0),
            },
            pack_path,
        )
        prototype_index[raw_anchor] = pack_path.name

    with open(prototype_cache_dir / "prototype_index.json", "w", encoding="utf-8") as f:
        json.dump(prototype_index, f, indent=2)

    for raw_anchor in ["rawA", "rawB"]:
        for version in [1, 2, 3]:
            version_id = f"{raw_anchor}_v{version}"
            rgb_path = rgb_dir / f"{version_id}.jpg"
            save_dummy_rgb(rgb_path, value=version * 40)
            meta[version_id] = {
                "raw_anchor": raw_anchor,
                "version": version,
                "paths": {"jpg": str(rgb_path)},
            }

    meta_path = tmp_path / "dataset_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    dataset = CodeMatchingDataset(
        prototype_cache_dir=str(prototype_cache_dir),
        meta_path=str(meta_path),
        rgb_dir=str(rgb_dir),
        include_versions=[1, 3],
    )

    version_ids = [dataset[idx]["sample_id"] for idx in range(len(dataset))]
    assert version_ids == ["rawA_v1", "rawA_v3", "rawB_v1", "rawB_v3"]
    assert dataset.code_dim == 2
    assert torch.allclose(dataset.get_code_tensor("rawA"), torch.tensor([1.0, 0.0]))


def test_stage3_code_scores_prefer_correct_claim_and_metrics_are_reasonable():
    student_codes = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    claim_bank = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    logits, targets = compute_claim_logits(student_codes, ["rawA", "rawB"], claim_bank, ["rawA", "rawB", "rawC"])
    assert logits.argmax(dim=1).tolist() == [0, 1]
    assert targets.tolist() == [0, 1]

    pos_scores = logits.gather(1, targets.unsqueeze(1)).squeeze(1)
    negative_mask = torch.ones_like(logits, dtype=torch.bool)
    negative_mask.scatter_(1, targets.unsqueeze(1), False)
    neg_scores = logits[negative_mask].view(logits.shape[0], -1)
    hard_neg = neg_scores.max(dim=1).values

    metrics = summarize_verification_scores(pos_scores, hard_neg, neg_scores.reshape(-1))
    assert metrics["pairwise_auc"] > 0.99
    assert metrics["hard_auc"] > 0.99
    assert metrics["eer"] < 0.01


def test_stage3_code_pair_mlp_logits_match_bank_shape():
    torch.manual_seed(0)
    student_codes = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    claim_bank = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    verifier = PrototypeVerifier(d_model=2, hidden_dim=8, dropout=0.0)
    logits, targets = compute_claim_logits(
        student_codes,
        ["rawA", "rawB"],
        claim_bank,
        ["rawA", "rawB", "rawC"],
        verifier_head=verifier,
    )
    assert logits.shape == (2, 3)
    assert targets.tolist() == [0, 1]


def test_transform_code_targets_binary_sign_returns_unit_binary_vectors():
    codes = torch.tensor([[0.5, -0.2, 0.0]], dtype=torch.float32)
    transformed = transform_code_targets(codes, "binary_sign")
    expected = torch.tensor([[1.0, -1.0, 1.0]], dtype=torch.float32)
    expected = expected / expected.norm(dim=-1, keepdim=True)
    assert torch.allclose(transformed, expected)
