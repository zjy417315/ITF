from pathlib import Path
import sys

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.itf import ImagingTraceFieldExtractor
from src.topo import ITFTopologySummarizer
from src.tools.data_roots import resolve_stage_cache_dir
from src.tools.evaluate_topology import generic_sequence_distance


def test_topology_summarizer_outputs_fixed_resolution_and_finite_vectors():
    summarizer = ITFTopologySummarizer(pixel_size=0.5, birth_range=(-2.0, 2.0), pers_range=(0.0, 4.0))
    itf_seq = torch.stack(
        [
            torch.tensor(
                [
                    [-1.0, -0.5, -0.5, -1.0],
                    [-0.5, 1.0, 1.0, -0.5],
                    [-0.5, 1.0, 1.0, -0.5],
                    [-1.0, -0.5, -0.5, -1.0],
                ]
            ),
            torch.tensor(
                [
                    [0.0, 0.2, 0.1, 0.0],
                    [0.2, 0.8, 0.7, 0.1],
                    [0.1, 0.7, 0.9, 0.2],
                    [0.0, 0.1, 0.2, 0.0],
                ]
            ),
        ],
        dim=0,
    ).float()

    pack = summarizer.summarize_sequence(itf_seq, stage_order=["s1", "s2"])

    assert pack["stage_order"] == ["s1", "s2"]
    assert pack["topo_h0_seq"].shape[0] == 2
    assert pack["topo_h1_seq"].shape == pack["topo_h0_seq"].shape
    assert pack["topo_vec_seq"].shape[0] == 2
    assert pack["topo_vec_seq"].shape[1] == pack["topo_h0_seq"][0].numel() * 2
    assert torch.isfinite(pack["topo_h0_seq"]).all()
    assert torch.isfinite(pack["topo_h1_seq"]).all()
    assert torch.isfinite(pack["topo_vec_seq"]).all()
    assert torch.all(pack["diagram_count_h0"] >= 0)
    assert torch.all(pack["diagram_count_h1"] >= 0)


def test_topology_summarizer_handles_constant_fields_without_nan():
    summarizer = ITFTopologySummarizer(pixel_size=0.5, birth_range=(-1.0, 1.0), pers_range=(0.0, 2.0))
    itf_seq = torch.zeros(3, 8, 8)

    pack = summarizer.summarize_sequence(itf_seq)

    assert pack["topo_vec_seq"].shape[0] == 3
    assert torch.isfinite(pack["topo_vec_seq"]).all()
    assert torch.isfinite(pack["persistence_mass_h0"]).all()
    assert torch.isfinite(pack["persistence_mass_h1"]).all()


def test_generic_sequence_distance_supports_stage_weights():
    seq_a = torch.tensor(
        [
            [[0.0, 0.0], [0.0, 0.0]],
            [[1.0, 1.0], [1.0, 1.0]],
            [[3.0, 3.0], [3.0, 3.0]],
        ]
    )
    seq_b = torch.tensor(
        [
            [[1.0, 1.0], [1.0, 1.0]],
            [[2.0, 2.0], [2.0, 2.0]],
            [[3.0, 3.0], [3.0, 3.0]],
        ]
    )

    uniform_distance, stagewise = generic_sequence_distance(seq_a, seq_b, mode="l1")
    weighted_distance, weighted_stagewise = generic_sequence_distance(
        seq_a,
        seq_b,
        mode="l1",
        stage_weights=[0.1, 0.9, 0.0],
    )

    assert stagewise == weighted_stagewise
    assert pytest.approx(uniform_distance, abs=1e-6) == (1.0 + 1.0 + 0.0) / 3.0
    assert pytest.approx(weighted_distance, abs=1e-6) == 1.0


@pytest.mark.integration
def test_real_itf_to_topology_smoke():
    checkpoint_path = Path("checkpoints/stage1_joint/stage1_joint_best.pt")
    stage_dir = resolve_stage_cache_dir()
    if not checkpoint_path.exists() or not stage_dir.exists():
        pytest.skip("Real checkpoint or stage cache is not available.")

    stage_files = sorted(stage_dir.glob("*.pt"))
    if not stage_files:
        pytest.skip("No cached stage files are available.")

    extractor = ImagingTraceFieldExtractor.from_checkpoint(str(checkpoint_path), device="cpu")
    itf_pack = extractor.extract_from_stage_cache_file(str(stage_files[0]), return_feature_maps=False)
    summarizer = ITFTopologySummarizer()
    topo_pack = summarizer.summarize_sequence(itf_pack["itf_seq"], stage_order=itf_pack["stage_order"])

    assert topo_pack["stage_order"] == ["stage_raw", "stage_demosaic", "stage_denoise", "stage_color", "rgb"]
    assert topo_pack["topo_h0_seq"].shape[0] == 5
    assert topo_pack["topo_h1_seq"].shape[0] == 5
    assert topo_pack["topo_vec_seq"].shape[0] == 5
    assert torch.isfinite(topo_pack["topo_vec_seq"]).all()
