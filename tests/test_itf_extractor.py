from pathlib import Path
import sys

import pytest
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.itf import ImagingTraceFieldExtractor
from src.tools.data_roots import resolve_stage_cache_dir


class ConstantFeatureBackbone(nn.Module):
    def __init__(self, feature_map: torch.Tensor):
        super().__init__()
        self.register_buffer("feature_map", feature_map.float())

    def extract_map(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        return self.feature_map.expand(batch_size, -1, -1, -1).clone()


class MeanDrivenBackbone(nn.Module):
    def extract_map(self, x: torch.Tensor) -> torch.Tensor:
        base = x.mean(dim=1, keepdim=True)
        return torch.cat([base, 2.0 * base], dim=1)


def test_extract_stage_computes_l2_energy_and_zscore_normalization():
    feature_map = torch.tensor(
        [[
            [[3.0, 0.0], [0.0, 4.0]],
            [[4.0, 0.0], [0.0, 3.0]],
        ]]
    )
    backbone = ConstantFeatureBackbone(feature_map)
    extractor = ImagingTraceFieldExtractor(backbone=backbone, device="cpu")

    stage = torch.zeros(1, 2, 2)
    result = extractor.extract_stage(stage)

    expected_energy = torch.tensor([[5.0, 0.0], [0.0, 5.0]])
    expected_mean = expected_energy.mean()
    expected_std = expected_energy.std(unbiased=False)
    expected_itf = (expected_energy - expected_mean) / expected_std

    assert result.feature_map.shape == (1, 2, 2, 2)
    assert torch.allclose(result.energy_map.squeeze(0), expected_energy, atol=1e-6)
    assert torch.allclose(result.itf_map.squeeze(0), expected_itf, atol=1e-6)
    assert torch.allclose(result.norm_mean.reshape(()), expected_mean, atol=1e-6)
    assert torch.allclose(result.norm_std.reshape(()), expected_std, atol=1e-6)
    assert torch.allclose(result.itf_map.mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(result.itf_map.std(unbiased=False), torch.tensor(1.0), atol=1e-6)


def test_extract_stage_accepts_chw_bchw_or_hwc_input():
    backbone = MeanDrivenBackbone()
    extractor = ImagingTraceFieldExtractor(backbone=backbone, device="cpu")

    chw_stage = torch.arange(4, dtype=torch.float32).view(1, 2, 2)
    bchw_stage = chw_stage.unsqueeze(0)
    hwc_stage = chw_stage.permute(1, 2, 0)

    chw_result = extractor.extract_stage(chw_stage)
    bchw_result = extractor.extract_stage(bchw_stage)
    hwc_result = extractor.extract_stage(hwc_stage)

    assert torch.allclose(chw_result.energy_map, bchw_result.energy_map)
    assert torch.allclose(chw_result.itf_map, bchw_result.itf_map)
    assert torch.allclose(chw_result.energy_map, hwc_result.energy_map)
    assert torch.allclose(chw_result.itf_map, hwc_result.itf_map)


def test_extract_sequence_respects_stage_order_and_can_return_feature_maps():
    backbone = MeanDrivenBackbone()
    extractor = ImagingTraceFieldExtractor(
        backbone=backbone,
        device="cpu",
        stage_order=["s2", "s1", "s3"],
    )

    stage_dict = {
        "s1": torch.ones(1, 2, 2),
        "s2": torch.ones(1, 2, 2) * 2.0,
        "s3": torch.ones(1, 2, 2) * 3.0,
    }
    pack = extractor.extract_sequence(stage_dict, return_feature_maps=True)

    assert pack["stage_order"] == ["s2", "s1", "s3"]
    assert pack["energy_seq"].shape == (3, 2, 2)
    assert pack["itf_seq"].shape == (3, 2, 2)
    assert pack["norm_mean_seq"].shape == (3,)
    assert pack["norm_std_seq"].shape == (3,)
    assert pack["feature_seq"].shape == (3, 2, 2, 2)

    energy_means = pack["energy_seq"].mean(dim=(-2, -1))
    assert torch.all(energy_means[1:] != energy_means[:-1])


def test_extract_sequence_raises_for_missing_stage():
    backbone = MeanDrivenBackbone()
    extractor = ImagingTraceFieldExtractor(
        backbone=backbone,
        device="cpu",
        stage_order=["stage_a", "stage_b"],
    )

    with pytest.raises(KeyError):
        extractor.extract_sequence({"stage_a": torch.ones(1, 2, 2)})


@pytest.mark.integration
def test_real_checkpoint_and_stage_cache_smoke():
    checkpoint_path = Path("checkpoints/stage1_joint/stage1_joint_best.pt")
    stage_dir = resolve_stage_cache_dir()
    if not checkpoint_path.exists() or not stage_dir.exists():
        pytest.skip("Real checkpoint or stage cache is not available.")

    stage_files = sorted(stage_dir.glob("*.pt"))
    if not stage_files:
        pytest.skip("No cached stage files are available.")

    extractor = ImagingTraceFieldExtractor.from_checkpoint(str(checkpoint_path), device="cpu")
    pack = extractor.extract_from_stage_cache_file(str(stage_files[0]), return_feature_maps=False)

    assert pack["stage_order"] == ["stage_raw", "stage_demosaic", "stage_denoise", "stage_color", "rgb"]
    assert pack["energy_seq"].shape[0] == 5
    assert pack["itf_seq"].shape[0] == 5
    assert pack["energy_seq"].ndim == 3
    assert pack["itf_seq"].ndim == 3
    assert torch.isfinite(pack["itf_seq"]).all()
    assert torch.allclose(pack["itf_seq"].mean(dim=(-2, -1)), torch.zeros(5), atol=1e-4)
    assert torch.allclose(
        pack["itf_seq"].std(dim=(-2, -1), unbiased=False),
        torch.ones(5),
        atol=1e-4,
    )
