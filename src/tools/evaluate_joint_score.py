import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Sequence, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.itf import ImagingTraceFieldExtractor
from src.isp.simple_isp import ISPConfig, SimpleISP
from src.topo import ITFTopologySummarizer
from src.tools.data_roots import resolve_meta_path, resolve_stage_cache_dir
from src.tools.evaluate_topology import build_weighted_topology_sequence, generic_sequence_distance


def summarize_scalar(values: Sequence[float]) -> Dict[str, float]:
    values = list(values)
    if not values:
        return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": len(values),
        "mean": float(mean(values)),
        "std": float(pstdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
    }


def load_meta(meta_path: Path) -> Dict[str, Dict]:
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_raw_groups(meta: Dict[str, Dict], available_versions: set) -> Dict[str, Dict[int, str]]:
    groups: Dict[str, Dict[int, str]] = defaultdict(dict)
    for version_id, info in meta.items():
        if version_id not in available_versions:
            continue
        groups[info["raw_anchor"]][int(info["version"])] = version_id
    return groups


def select_raws(
    raw_groups: Dict[str, Dict[int, str]],
    required_versions: set,
    num_raws: int,
    seed: int,
) -> List[str]:
    candidates = [raw for raw, versions in raw_groups.items() if required_versions.issubset(versions.keys())]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[: min(num_raws, len(candidates))]


def fuse_scores(itf_score: float, topo_score: float, alpha: float, fusion_mode: str) -> float:
    alpha = float(alpha)
    if fusion_mode == "sum":
        return (1.0 - alpha) * itf_score + alpha * topo_score
    if fusion_mode == "max":
        return max((1.0 - alpha) * itf_score, alpha * topo_score)
    if fusion_mode == "product":
        eps = 1e-12
        return float((max(itf_score, eps) ** (1.0 - alpha)) * (max(topo_score, eps) ** alpha))
    if fusion_mode == "geo":
        return float(((1.0 - alpha) * (itf_score ** 2) + alpha * (topo_score ** 2)) ** 0.5)
    raise ValueError(f"Unsupported fusion mode: {fusion_mode}")


def normalize_score(
    raw_score: float,
    modality: str,
    normalization_mode: str,
    normalization_stats: Dict[str, Dict[str, float]],
) -> float:
    if normalization_mode == "none":
        return float(raw_score)

    stats = normalization_stats[modality]
    if normalization_mode == "cross":
        denom = max(stats["cross_mean"], 1e-12)
    elif normalization_mode == "margin":
        denom = max(stats["cross_mean"] - stats["anchor_mean"], 1e-12)
    else:
        raise ValueError(f"Unsupported normalization mode: {normalization_mode}")
    return float(raw_score / denom)


def compute_protocol_summary(anchor_scores: List[float], isp_scores: List[float], cross_scores: List[float]) -> Dict:
    triplet_hits = sum(int(a < i < x) for a, i, x in zip(anchor_scores, isp_scores, cross_scores))
    count = len(anchor_scores)
    return {
        "same_source_anchor_equivalence_v1_v2": {"sequence_distance": summarize_scalar(anchor_scores)},
        "same_source_isp_sensitivity_v1_vx": {"sequence_distance": summarize_scalar(isp_scores)},
        "cross_source_separation_v1_v1": {"sequence_distance": summarize_scalar(cross_scores)},
        "ordering_checks": {
            "ordered_triplet_accuracy": float(triplet_hits / count) if count else 0.0,
        },
        "derived_scores": {
            "cross_over_isp_ratio": float(mean(cross_scores) / max(mean(isp_scores), 1e-12)) if count else 0.0,
            "cross_over_anchor_ratio": float(mean(cross_scores) / max(mean(anchor_scores), 1e-12)) if count else 0.0,
        },
    }


def evaluate_joint_score(
    meta_path: Path,
    stage_cache_dir: Path,
    checkpoint_path: Path,
    protocol_versions: Sequence[int],
    num_raws: int,
    seed: int,
    device: str = None,
    source: str = "live_isp",
    crop_size: int = 512,
    feature_source: str = "layer4",
    scalarization: str = "l2",
    normalization: str = "zscore",
    topology_distance_mode: str = "l1",
    topology_normalize_vectors: bool = False,
    h0_weight: float = 1.3,
    h1_weight: float = 0.7,
    pixel_size: float = 0.25,
    birth_min: float = -3.0,
    birth_max: float = 3.0,
    pers_min: float = 0.0,
    pers_max: float = 6.0,
    kernel_sigma: float = 0.35,
    stage_weights: Sequence[float] = None,
    fusion_modes: Sequence[str] = ("sum", "max", "product", "geo"),
    normalization_modes: Sequence[str] = ("cross", "margin", "none"),
    alpha_grid: Sequence[float] = tuple(x / 20.0 for x in range(0, 21)),
):
    meta = load_meta(meta_path)
    if source == "live_isp":
        available_versions = set(meta.keys())
    else:
        available_versions = {path.stem for path in stage_cache_dir.glob("*.pt")}

    raw_groups = build_raw_groups(meta, available_versions)
    required_versions = {1, 2, 1}.union(set(protocol_versions))
    selected_raws = select_raws(raw_groups, required_versions=required_versions, num_raws=num_raws, seed=seed)

    if not selected_raws:
        raise RuntimeError("No RAW anchors with the requested version set are available.")
    if len(selected_raws) < 2:
        raise RuntimeError("At least two RAW anchors are required for joint-score evaluation.")

    extractor = ImagingTraceFieldExtractor.from_checkpoint(
        checkpoint_path=str(checkpoint_path),
        device=device,
        feature_source=feature_source,
        scalarization=scalarization,
        normalization=normalization,
    )
    summarizer = ITFTopologySummarizer(
        pixel_size=pixel_size,
        birth_range=(birth_min, birth_max),
        pers_range=(pers_min, pers_max),
        kernel_sigma=kernel_sigma,
    )
    isp = None
    if source == "live_isp":
        isp = SimpleISP(config=ISPConfig(center_crop=True, crop_size=crop_size))

    cached_itf: Dict[str, torch.Tensor] = {}
    cached_topology: Dict[str, torch.Tensor] = {}

    def load_itf(version_id: str) -> torch.Tensor:
        if version_id in cached_itf:
            return cached_itf[version_id]

        if source == "live_isp":
            info = meta[version_id]
            stages = isp.run(
                info["paths"]["raw"],
                config_override=info.get("isp_parameters", {}),
                degradation_override=info.get("degradation"),
            )
            pack = extractor.extract_sequence(stages, return_feature_maps=False)
        else:
            stage_file = stage_cache_dir / f"{version_id}.pt"
            pack = extractor.extract_from_stage_cache_file(str(stage_file), return_feature_maps=False)
        cached_itf[version_id] = pack["itf_seq"].float()
        return cached_itf[version_id]

    def load_topology(version_id: str) -> torch.Tensor:
        if version_id in cached_topology:
            return cached_topology[version_id]

        topo_pack = summarizer.summarize_sequence(load_itf(version_id))
        cached_topology[version_id] = build_weighted_topology_sequence(
            topo_pack=topo_pack,
            h0_weight=h0_weight,
            h1_weight=h1_weight,
        )
        return cached_topology[version_id]

    pair_records: Dict[int, List[Dict[str, float]]] = {version: [] for version in protocol_versions}
    for protocol_version in protocol_versions:
        for idx, raw_anchor in enumerate(selected_raws):
            versions = raw_groups[raw_anchor]
            ref_v1 = versions[1]
            benign_v2 = versions[2]
            shift_vx = versions[protocol_version]
            diff_raw_anchor = selected_raws[(idx + 1) % len(selected_raws)]
            cross_v1 = raw_groups[diff_raw_anchor][1]

            itf_anchor, _ = generic_sequence_distance(load_itf(ref_v1), load_itf(benign_v2), stage_weights=stage_weights)
            itf_isp, _ = generic_sequence_distance(load_itf(ref_v1), load_itf(shift_vx), stage_weights=stage_weights)
            itf_cross, _ = generic_sequence_distance(load_itf(ref_v1), load_itf(cross_v1), stage_weights=stage_weights)

            topo_anchor, _ = generic_sequence_distance(
                load_topology(ref_v1),
                load_topology(benign_v2),
                mode=topology_distance_mode,
                normalize_vectors=topology_normalize_vectors,
                stage_weights=stage_weights,
            )
            topo_isp, _ = generic_sequence_distance(
                load_topology(ref_v1),
                load_topology(shift_vx),
                mode=topology_distance_mode,
                normalize_vectors=topology_normalize_vectors,
                stage_weights=stage_weights,
            )
            topo_cross, _ = generic_sequence_distance(
                load_topology(ref_v1),
                load_topology(cross_v1),
                mode=topology_distance_mode,
                normalize_vectors=topology_normalize_vectors,
                stage_weights=stage_weights,
            )

            pair_records[protocol_version].append(
                {
                    "itf_anchor": itf_anchor,
                    "itf_isp": itf_isp,
                    "itf_cross": itf_cross,
                    "topo_anchor": topo_anchor,
                    "topo_isp": topo_isp,
                    "topo_cross": topo_cross,
                }
            )

    normalization_stats = {
        "itf": {
            "anchor_mean": float(mean(record["itf_anchor"] for version in protocol_versions for record in pair_records[version])),
            "isp_mean": float(mean(record["itf_isp"] for version in protocol_versions for record in pair_records[version])),
            "cross_mean": float(mean(record["itf_cross"] for version in protocol_versions for record in pair_records[version])),
        },
        "topology": {
            "anchor_mean": float(mean(record["topo_anchor"] for version in protocol_versions for record in pair_records[version])),
            "isp_mean": float(mean(record["topo_isp"] for version in protocol_versions for record in pair_records[version])),
            "cross_mean": float(mean(record["topo_cross"] for version in protocol_versions for record in pair_records[version])),
        },
    }

    baselines = {
        "itf": {},
        "topology": {},
    }
    for protocol_version in protocol_versions:
        records = pair_records[protocol_version]
        baselines["itf"][str(protocol_version)] = compute_protocol_summary(
            [r["itf_anchor"] for r in records],
            [r["itf_isp"] for r in records],
            [r["itf_cross"] for r in records],
        )
        baselines["topology"][str(protocol_version)] = compute_protocol_summary(
            [r["topo_anchor"] for r in records],
            [r["topo_isp"] for r in records],
            [r["topo_cross"] for r in records],
        )

    candidates = []
    for normalization_mode in normalization_modes:
        for fusion_mode in fusion_modes:
            for alpha in alpha_grid:
                protocol_results = {}
                acc_values = []
                ratio_values = []

                for protocol_version in protocol_versions:
                    anchor_scores = []
                    isp_scores = []
                    cross_scores = []
                    for record in pair_records[protocol_version]:
                        itf_anchor = normalize_score(record["itf_anchor"], "itf", normalization_mode, normalization_stats)
                        itf_isp = normalize_score(record["itf_isp"], "itf", normalization_mode, normalization_stats)
                        itf_cross = normalize_score(record["itf_cross"], "itf", normalization_mode, normalization_stats)
                        topo_anchor = normalize_score(record["topo_anchor"], "topology", normalization_mode, normalization_stats)
                        topo_isp = normalize_score(record["topo_isp"], "topology", normalization_mode, normalization_stats)
                        topo_cross = normalize_score(record["topo_cross"], "topology", normalization_mode, normalization_stats)

                        anchor_scores.append(fuse_scores(itf_anchor, topo_anchor, alpha=alpha, fusion_mode=fusion_mode))
                        isp_scores.append(fuse_scores(itf_isp, topo_isp, alpha=alpha, fusion_mode=fusion_mode))
                        cross_scores.append(fuse_scores(itf_cross, topo_cross, alpha=alpha, fusion_mode=fusion_mode))

                    summary = compute_protocol_summary(anchor_scores, isp_scores, cross_scores)
                    protocol_results[str(protocol_version)] = summary
                    acc_values.append(summary["ordering_checks"]["ordered_triplet_accuracy"])
                    ratio_values.append(summary["derived_scores"]["cross_over_isp_ratio"])

                candidates.append(
                    {
                        "normalization_mode": normalization_mode,
                        "fusion_mode": fusion_mode,
                        "alpha": float(alpha),
                        "protocol_results": protocol_results,
                        "score_tuple": [
                            float(mean(acc_values)),
                            float(min(acc_values)),
                            float(max(acc_values)),
                            float(mean(ratio_values)),
                        ],
                    }
                )

    candidates.sort(
        key=lambda item: (
            item["score_tuple"][0],
            item["score_tuple"][1],
            item["score_tuple"][3],
        ),
        reverse=True,
    )

    return {
        "config": {
            "meta_path": str(meta_path),
            "stage_cache_dir": str(stage_cache_dir),
            "checkpoint_path": str(checkpoint_path),
            "protocol_versions": list(protocol_versions),
            "num_raws": len(selected_raws),
            "seed": seed,
            "source": source,
            "crop_size": crop_size,
            "feature_source": feature_source,
            "scalarization": scalarization,
            "normalization": normalization,
            "topology_distance_mode": topology_distance_mode,
            "topology_normalize_vectors": topology_normalize_vectors,
            "h0_weight": h0_weight,
            "h1_weight": h1_weight,
            "pixel_size": pixel_size,
            "birth_min": birth_min,
            "birth_max": birth_max,
            "pers_min": pers_min,
            "pers_max": pers_max,
            "kernel_sigma": kernel_sigma,
            "stage_weights": list(stage_weights) if stage_weights is not None else None,
        },
        "normalization_stats": normalization_stats,
        "baselines": baselines,
        "best_candidate": candidates[0] if candidates else None,
        "top_candidates": candidates[:20],
    }


def print_summary(summary: Dict):
    print("=" * 70)
    print("Joint ITF + Topology Evaluation Summary")
    print(f"Samples: {summary['config']['num_raws']}")
    print(f"Protocols: {summary['config']['protocol_versions']}")
    best = summary["best_candidate"]
    if best is not None:
        print(
            f"Best fusion: norm={best['normalization_mode']}, "
            f"fusion={best['fusion_mode']}, alpha={best['alpha']:.2f}"
        )
        for protocol_version, protocol_result in best["protocol_results"].items():
            print(
                f"  v{protocol_version}: acc={protocol_result['ordering_checks']['ordered_triplet_accuracy']:.4f}, "
                f"cross/isp={protocol_result['derived_scores']['cross_over_isp_ratio']:.4f}"
            )
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Evaluate joint ITF + topology fusion scores")
    parser.add_argument(
        "--meta_path",
        type=str,
        default=str(resolve_meta_path()),
    )
    parser.add_argument(
        "--stage_cache_dir",
        type=str,
        default=str(resolve_stage_cache_dir()),
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=str(PROJECT_ROOT / "checkpoints" / "stage1_joint" / "stage1_joint_best.pt"),
    )
    parser.add_argument("--protocol_versions", type=int, nargs="*", default=[3, 4])
    parser.add_argument("--num_raws", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--source", type=str, default="live_isp", choices=["stage_cache", "live_isp"])
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument(
        "--feature_source",
        type=str,
        default="layer4",
        choices=["map_proj", "layer4", "layer3", "multiscale_l34"],
    )
    parser.add_argument(
        "--scalarization",
        type=str,
        default="l2",
        choices=["l2", "l1", "mean_abs", "max_abs"],
    )
    parser.add_argument(
        "--normalization",
        type=str,
        default="zscore",
        choices=["zscore", "robust_zscore", "minmax"],
    )
    parser.add_argument("--topology_distance_mode", type=str, default="l1", choices=["l1", "l2", "cosine"])
    parser.add_argument("--topology_normalize_vectors", action="store_true")
    parser.add_argument("--h0_weight", type=float, default=1.3)
    parser.add_argument("--h1_weight", type=float, default=0.7)
    parser.add_argument("--pixel_size", type=float, default=0.25)
    parser.add_argument("--birth_min", type=float, default=-3.0)
    parser.add_argument("--birth_max", type=float, default=3.0)
    parser.add_argument("--pers_min", type=float, default=0.0)
    parser.add_argument("--pers_max", type=float, default=6.0)
    parser.add_argument("--kernel_sigma", type=float, default=0.35)
    parser.add_argument(
        "--stage_weights",
        type=float,
        nargs="*",
        default=None,
        help="Optional per-stage aggregation weights, e.g. 0.5 0.5 0.75 2.0 1.25",
    )
    parser.add_argument("--fusion_modes", type=str, nargs="*", default=["sum", "max", "product", "geo"])
    parser.add_argument("--normalization_modes", type=str, nargs="*", default=["cross", "margin", "none"])
    parser.add_argument(
        "--alpha_grid",
        type=float,
        nargs="*",
        default=[x / 20.0 for x in range(0, 21)],
        help="Fusion weights to evaluate",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=str(PROJECT_ROOT / "data" / "joint_score_eval_live32.json"),
    )

    args = parser.parse_args()
    summary = evaluate_joint_score(
        meta_path=Path(args.meta_path),
        stage_cache_dir=Path(args.stage_cache_dir),
        checkpoint_path=Path(args.checkpoint_path),
        protocol_versions=args.protocol_versions,
        num_raws=args.num_raws,
        seed=args.seed,
        device=args.device,
        source=args.source,
        crop_size=args.crop_size,
        feature_source=args.feature_source,
        scalarization=args.scalarization,
        normalization=args.normalization,
        topology_distance_mode=args.topology_distance_mode,
        topology_normalize_vectors=args.topology_normalize_vectors,
        h0_weight=args.h0_weight,
        h1_weight=args.h1_weight,
        pixel_size=args.pixel_size,
        birth_min=args.birth_min,
        birth_max=args.birth_max,
        pers_min=args.pers_min,
        pers_max=args.pers_max,
        kernel_sigma=args.kernel_sigma,
        stage_weights=args.stage_weights,
        fusion_modes=args.fusion_modes,
        normalization_modes=args.normalization_modes,
        alpha_grid=args.alpha_grid,
    )

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print_summary(summary)
    print(f"Saved summary to: {output_json}")


if __name__ == "__main__":
    main()
