from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.tools.build_experiment_chapter_assets import build_main_results_table_tex
from src.train.train_stage3_code import summarize_verification_scores


def test_summarize_verification_scores_reports_tar_at_far_1e3():
    pos_scores = torch.tensor([0.9, 0.8, 0.7], dtype=torch.float32)
    hard_neg_scores = torch.tensor([0.3, 0.4, 0.5], dtype=torch.float32)
    all_neg_scores = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], dtype=torch.float32)

    metrics = summarize_verification_scores(pos_scores, hard_neg_scores, all_neg_scores)

    assert "tar_at_far_1e3" in metrics
    assert 0.0 <= metrics["tar_at_far_1e3"] <= metrics["tar_at_far_1e2"] <= 1.0


def test_build_main_results_table_tex_contains_all_settings_and_conditions():
    schema = {
        "Process": ["Joint", "ITF", "Topology"],
        "RGB Projection": ["Joint", "Geometric", "Topological"],
        "Active Verification": ["Final", "Main", "Protocol"],
    }
    summary = {
        "primary_table": {
            setting: {
                branch: {
                    condition: {
                        "auc": 0.9,
                        "eer": 0.1,
                        "tar_at_far_1e2": 0.8,
                        "tar_at_far_1e3": 0.7,
                    }
                    for condition in ["Ref", "Prop", "Shift-1", "Shift-2"]
                }
                for branch in branches
            }
            for setting, branches in schema.items()
        }
    }

    tex = build_main_results_table_tex(summary)

    assert "\\label{tab:main_results}" in tex
    assert "Process" in tex
    assert "RGB Projection" in tex
    assert "Active Verification" in tex
    assert "Shift-2" in tex
    assert "TAR@0.1\\%" in tex
