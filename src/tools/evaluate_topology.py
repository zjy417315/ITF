import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.itf import ImagingTraceFieldExtractor
from src.isp.simple_isp import ISPConfig, SimpleISP
from src.topo import ITFTopologySummarizer
from src.tools.data_roots import resolve_meta_path, resolve_stage_cache_dir


def summarize_scalar(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": len(values),
        "mean": float(mean(values)),
        "std": float(pstdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
    }


def summarize_stagewise(stage_values: List[List[float]]) -> List[Dict[str, float]]:
    if not stage_values:
        return []
    num_stages = len(stage_values[0])
    return [summarize_scalar([values[k] for values in stage_values]) for k in range(num_stages)]


def generic_sequence_distance(
    sequence_a: torch.Tensor,
    sequence_b: torch.Tensor,
    mode: str = "l2",
    normalize_vectors: bool = False,
    stage_weights: List[float] = None,
) -> Tuple[float, List[float]]:
    a = torch.as_tensor(sequence_a, dtype=torch.float32).reshape(sequence_a.shape[0], -1)
    b = torch.as_tensor(sequence_b, dtype=torch.float32).reshape(sequence_b.shape[0], -1)

    if normalize_vectors:
        a = torch.nn.functional.normalize(a, dim=1)
        b = torch.nn.functional.normalize(b, dim=1)

    if mode == "l2":
        stage_dist = (a - b).pow(2).mean(dim=1).sqrt()
    elif mode == "l1":
        stage_dist = (a - b).abs().mean(dim=1)
    elif mode == "cosine":
        stage_dist = 1.0 - torch.nn.functional.cosine_similarity(a, b, dim=1)
    else:
        raise ValueError(f"Unsupported distance mode: {mode}")

    if stage_weights is None:
        sequence_distance = stage_dist.mean()
    else:
        weights = torch.as_tensor(stage_weights, dtype=torch.float32)
        if weights.ndim != 1 or weights.numel() != stage_dist.numel():
            raise ValueError(
                f"Stage-weight length {weights.numel()} does not match stage count {stage_dist.numel()}."
            )
        weights = weights.clamp_min(0.0)
        if float(weights.sum().item()) <= 0.0:
            raise ValueError("Stage weights must contain at least one positive value.")
        weights = weights / weights.sum()
        sequence_distance = (stage_dist * weights).sum()

    return float(sequence_distance.item()), [float(x.item()) for x in stage_dist]


def build_weighted_topology_sequence(
    topo_pack: Dict[str, torch.Tensor],
    h0_weight: float,
    h1_weight: float,
) -> torch.Tensor:
    h0_seq = topo_pack["topo_h0_seq"].reshape(topo_pack["topo_h0_seq"].shape[0], -1).float() * float(h0_weight)
    h1_seq = topo_pack["topo_h1_seq"].reshape(topo_pack["topo_h1_seq"].shape[0], -1).float() * float(h1_weight)
    return torch.cat([h0_seq, h1_seq], dim=1)


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


def maybe_load_itf_from_cache(itf_cache_dir: Path, version_id: str):
    if itf_cache_dir is None:
        return None
    path = itf_cache_dir / f"{version_id}.pt"
    if not path.exists():
        return None
    pack = torch.load(path, map_location="cpu")
    return pack["itf_seq"].float(), pack.get("stage_order")


def maybe_load_topology_from_cache(topo_cache_dir: Path, version_id: str):
    if topo_cache_dir is None:
        return None
    path = topo_cache_dir / f"{version_id}.pt"
    if not path.exists():
        return None
    pack = torch.load(path, map_location="cpu")
    return pack["topo_vec_seq"].float(), pack.get("stage_order")


def _collect_pair_metrics(
    distances_anchor: List[float],
    stage_anchor: List[List[float]],
    distances_isp: List[float],
    stage_isp: List[List[float]],
    distances_cross: List[float],
    stage_cross: List[List[float]],
) -> Dict:
    return {
        "same_source_anchor_equivalence_v1_v2": {
            "sequence_distance": summarize_scalar(distances_anchor),
            "stagewise_distance": summarize_stagewise(stage_anchor),
        },
        "same_source_isp_sensitivity_v1_v3": {
            "sequence_distance": summarize_scalar(distances_isp),
            "stagewise_distance": summarize_stagewise(stage_isp),
        },
        "cross_source_separation_v1_v1": {
            "sequence_distance": summarize_scalar(distances_cross),
            "stagewise_distance": summarize_stagewise(stage_cross),
        },
    }


def evaluate_topology(
    meta_path: Path,
    stage_cache_dir: Path,
    checkpoint_path: Path,
    num_raws: int,
    seed: int,
    device: str = None,
    itf_cache_dir: Path = None,
    topo_cache_dir: Path = None,
    source: str = "stage_cache",
    crop_size: int = 512,
    feature_source: str = "layer4",
    scalarization: str = "l2",
    normalization: str = "zscore",
    reference_version: int = 1,
    benign_version: int = 2,
    isp_version: int = 3,
    cross_version: int = 1,
    topology_distance_mode: str = "l1",
    topology_normalize_vectors: bool = False,
    h0_weight: float = 1.3,
    h1_weight: float = 0.7,
    pixel_size: float = 0.25,
    birth_min: float = -3.0,
    birth_max: float = 3.0,
    pers_min: float = 0.0,
    pers_max: float = 6.0,
    kernel_sigma: float = 0.2,
    stage_weights: List[float] = None,
):
    meta = load_meta(meta_path)
    if source == "live_isp":
        available_versions = set(meta.keys())
    else:
        available_versions = {path.stem for path in stage_cache_dir.glob("*.pt")}
    raw_groups = build_raw_groups(meta, available_versions)
    required_versions = {reference_version, benign_version, isp_version, cross_version}
    selected_raws = select_raws(raw_groups, required_versions=required_versions, num_raws=num_raws, seed=seed)

    if not selected_raws:
        raise RuntimeError("No RAW anchors with the requested version set are available.")
    if len(selected_raws) < 2:
        raise RuntimeError("At least two RAW anchors are required for topology evaluation.")

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
    cached_stage_order: Dict[str, List[str]] = {}

    def load_itf(version_id: str) -> Tuple[torch.Tensor, List[str]]:
        if version_id in cached_itf:
            return cached_itf[version_id], cached_stage_order[version_id]

        itf_seq = None
        stage_order = None
        if source == "stage_cache":
            cached = maybe_load_itf_from_cache(itf_cache_dir, version_id)
            if cached is not None:
                itf_seq, stage_order = cached
        if itf_seq is None and source == "stage_cache":
            stage_file = stage_cache_dir / f"{version_id}.pt"
            pack = extractor.extract_from_stage_cache_file(str(stage_file), return_feature_maps=False)
            itf_seq = pack["itf_seq"].float()
            stage_order = pack.get("stage_order")
        elif itf_seq is None and source == "live_isp":
            info = meta[version_id]
            stages = isp.run(
                info["paths"]["raw"],
                config_override=info.get("isp_parameters", {}),
                degradation_override=info.get("degradation"),
            )
            pack = extractor.extract_sequence(stages, return_feature_maps=False)
            itf_seq = pack["itf_seq"].float()
            stage_order = pack.get("stage_order")

        cached_itf[version_id] = itf_seq
        cached_stage_order[version_id] = list(stage_order or extractor.stage_order)
        return cached_itf[version_id], cached_stage_order[version_id]

    def load_topology(version_id: str) -> Tuple[torch.Tensor, List[str]]:
        if version_id in cached_topology:
            return cached_topology[version_id], cached_stage_order[version_id]

        topo_seq = None
        stage_order = None
        cached = maybe_load_topology_from_cache(topo_cache_dir, version_id)
        if cached is not None:
            topo_seq, stage_order = cached
        if topo_seq is None:
            itf_seq, stage_order = load_itf(version_id)
            topo_pack = summarizer.summarize_sequence(itf_seq, stage_order=stage_order)
            topo_seq = build_weighted_topology_sequence(
                topo_pack=topo_pack,
                h0_weight=h0_weight,
                h1_weight=h1_weight,
            )
            stage_order = topo_pack.get("stage_order")

        cached_topology[version_id] = topo_seq
        cached_stage_order[version_id] = list(stage_order or extractor.stage_order)
        return cached_topology[version_id], cached_stage_order[version_id]

    itf_anchor_distances: List[float] = []
    itf_anchor_stage_distances: List[List[float]] = []
    itf_isp_distances: List[float] = []
    itf_isp_stage_distances: List[List[float]] = []
    itf_cross_distances: List[float] = []
    itf_cross_stage_distances: List[List[float]] = []

    topo_anchor_distances: List[float] = []
    topo_anchor_stage_distances: List[List[float]] = []
    topo_isp_distances: List[float] = []
    topo_isp_stage_distances: List[List[float]] = []
    topo_cross_distances: List[float] = []
    topo_cross_stage_distances: List[List[float]] = []

    itf_ordered_triplet_hits = 0
    topo_ordered_triplet_hits = 0

    for idx, raw_anchor in enumerate(selected_raws):
        versions = raw_groups[raw_anchor]
        v1 = versions[reference_version]
        v2 = versions[benign_version]
        v3 = versions[isp_version]

        diff_raw_anchor = selected_raws[(idx + 1) % len(selected_raws)]
        diff_v1 = raw_groups[diff_raw_anchor][cross_version]

        itf_v1, _ = load_itf(v1)
        itf_v2, _ = load_itf(v2)
        itf_v3, _ = load_itf(v3)
        itf_diff, _ = load_itf(diff_v1)

        topo_v1, _ = load_topology(v1)
        topo_v2, _ = load_topology(v2)
        topo_v3, _ = load_topology(v3)
        topo_diff, _ = load_topology(diff_v1)

        itf_anchor_dist, itf_anchor_stage = generic_sequence_distance(itf_v1, itf_v2, stage_weights=stage_weights)
        itf_isp_dist, itf_isp_stage = generic_sequence_distance(itf_v1, itf_v3, stage_weights=stage_weights)
        itf_cross_dist, itf_cross_stage = generic_sequence_distance(itf_v1, itf_diff, stage_weights=stage_weights)
        itf_anchor_distances.append(itf_anchor_dist)
        itf_anchor_stage_distances.append(itf_anchor_stage)
        itf_isp_distances.append(itf_isp_dist)
        itf_isp_stage_distances.append(itf_isp_stage)
        itf_cross_distances.append(itf_cross_dist)
        itf_cross_stage_distances.append(itf_cross_stage)
        itf_ordered_triplet_hits += int(itf_anchor_dist < itf_isp_dist < itf_cross_dist)

        topo_anchor_dist, topo_anchor_stage = generic_sequence_distance(
            topo_v1,
            topo_v2,
            mode=topology_distance_mode,
            normalize_vectors=topology_normalize_vectors,
            stage_weights=stage_weights,
        )
        topo_isp_dist, topo_isp_stage = generic_sequence_distance(
            topo_v1,
            topo_v3,
            mode=topology_distance_mode,
            normalize_vectors=topology_normalize_vectors,
            stage_weights=stage_weights,
        )
        topo_cross_dist, topo_cross_stage = generic_sequence_distance(
            topo_v1,
            topo_diff,
            mode=topology_distance_mode,
            normalize_vectors=topology_normalize_vectors,
            stage_weights=stage_weights,
        )
        topo_anchor_distances.append(topo_anchor_dist)
        topo_anchor_stage_distances.append(topo_anchor_stage)
        topo_isp_distances.append(topo_isp_dist)
        topo_isp_stage_distances.append(topo_isp_stage)
        topo_cross_distances.append(topo_cross_dist)
        topo_cross_stage_distances.append(topo_cross_stage)
        topo_ordered_triplet_hits += int(topo_anchor_dist < topo_isp_dist < topo_cross_dist)

    count = len(selected_raws)
    itf_metrics = _collect_pair_metrics(
        itf_anchor_distances,
        itf_anchor_stage_distances,
        itf_isp_distances,
        itf_isp_stage_distances,
        itf_cross_distances,
        itf_cross_stage_distances,
    )
    topo_metrics = _collect_pair_metrics(
        topo_anchor_distances,
        topo_anchor_stage_distances,
        topo_isp_distances,
        topo_isp_stage_distances,
        topo_cross_distances,
        topo_cross_stage_distances,
    )

    summary = {
        "config": {
            "meta_path": str(meta_path),
            "stage_cache_dir": str(stage_cache_dir),
            "checkpoint_path": str(checkpoint_path),
            "itf_cache_dir": str(itf_cache_dir) if itf_cache_dir is not None else None,
            "topo_cache_dir": str(topo_cache_dir) if topo_cache_dir is not None else None,
            "num_raws": count,
            "seed": seed,
            "source": source,
            "crop_size": crop_size,
            "feature_source": feature_source,
            "scalarization": scalarization,
            "normalization": normalization,
            "reference_version": reference_version,
            "benign_version": benign_version,
            "isp_version": isp_version,
            "cross_version": cross_version,
            "topology_variant": summarizer.describe_variant(),
            "topology_distance_mode": topology_distance_mode,
            "topology_normalize_vectors": topology_normalize_vectors,
            "h0_weight": h0_weight,
            "h1_weight": h1_weight,
            "stage_weights": list(stage_weights) if stage_weights is not None else None,
            "stage_order": extractor.stage_order,
        },
        "itf": {
            **itf_metrics,
            "ordering_checks": {
                "ordered_triplet_accuracy": float(itf_ordered_triplet_hits / count),
            },
            "derived_scores": {
                "cross_minus_isp_margin": float(mean(itf_cross_distances) - mean(itf_isp_distances)),
                "cross_over_isp_ratio": float(mean(itf_cross_distances) / max(mean(itf_isp_distances), 1e-8)),
            },
        },
        "topology": {
            **topo_metrics,
            "ordering_checks": {
                "ordered_triplet_accuracy": float(topo_ordered_triplet_hits / count),
            },
            "derived_scores": {
                "cross_minus_isp_margin": float(mean(topo_cross_distances) - mean(topo_isp_distances)),
                "cross_over_isp_ratio": float(mean(topo_cross_distances) / max(mean(topo_isp_distances), 1e-8)),
            },
        },
        "comparison": {
            "anchor_distance_ratio_topology_over_itf": float(
                mean(topo_anchor_distances) / max(mean(itf_anchor_distances), 1e-8)
            ),
            "isp_distance_ratio_topology_over_itf": float(
                mean(topo_isp_distances) / max(mean(itf_isp_distances), 1e-8)
            ),
            "cross_distance_ratio_topology_over_itf": float(
                mean(topo_cross_distances) / max(mean(itf_cross_distances), 1e-8)
            ),
            "cross_over_isp_gain": float(
                (mean(topo_cross_distances) / max(mean(topo_isp_distances), 1e-8))
                - (mean(itf_cross_distances) / max(mean(itf_isp_distances), 1e-8))
            ),
            "ordered_triplet_accuracy_gain": float(
                (topo_ordered_triplet_hits / count) - (itf_ordered_triplet_hits / count)
            ),
            "normalized_anchor_over_cross_itf": float(mean(itf_anchor_distances) / max(mean(itf_cross_distances), 1e-8)),
            "normalized_isp_over_cross_itf": float(mean(itf_isp_distances) / max(mean(itf_cross_distances), 1e-8)),
            "normalized_anchor_over_cross_topology": float(mean(topo_anchor_distances) / max(mean(topo_cross_distances), 1e-8)),
            "normalized_isp_over_cross_topology": float(mean(topo_isp_distances) / max(mean(topo_cross_distances), 1e-8)),
        },
    }
    return summary


def print_summary(summary: Dict):
    print("=" * 70)
    print("ITF vs Topology Evaluation Summary")
    print(f"Samples: {summary['config']['num_raws']}")
    print("- ITF")
    print(
        "  anchor / isp / cross: "
        f"{summary['itf']['same_source_anchor_equivalence_v1_v2']['sequence_distance']['mean']:.6f} / "
        f"{summary['itf']['same_source_isp_sensitivity_v1_v3']['sequence_distance']['mean']:.6f} / "
        f"{summary['itf']['cross_source_separation_v1_v1']['sequence_distance']['mean']:.6f}"
    )
    print(
        "  ordered triplet acc: "
        f"{summary['itf']['ordering_checks']['ordered_triplet_accuracy']:.4f}"
    )
    print(
        "  cross/isp ratio: "
        f"{summary['itf']['derived_scores']['cross_over_isp_ratio']:.4f}"
    )
    print("- Topology")
    print(
        "  anchor / isp / cross: "
        f"{summary['topology']['same_source_anchor_equivalence_v1_v2']['sequence_distance']['mean']:.6f} / "
        f"{summary['topology']['same_source_isp_sensitivity_v1_v3']['sequence_distance']['mean']:.6f} / "
        f"{summary['topology']['cross_source_separation_v1_v1']['sequence_distance']['mean']:.6f}"
    )
    print(
        "  ordered triplet acc: "
        f"{summary['topology']['ordering_checks']['ordered_triplet_accuracy']:.4f}"
    )
    print(
        "  cross/isp ratio: "
        f"{summary['topology']['derived_scores']['cross_over_isp_ratio']:.4f}"
    )
    print("- Comparison")
    print(
        "  topology cross/isp gain: "
        f"{summary['comparison']['cross_over_isp_gain']:.4f}"
    )
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Evaluate topology summaries built on top of ITF sequences")
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
    parser.add_argument(
        "--itf_cache_dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "itf_cache_centercrop"),
    )
    parser.add_argument(
        "--topo_cache_dir",
        type=str,
        default=None,
        help="Optional directory with precomputed topology cache files",
    )
    parser.add_argument("--num_raws", type=int, default=64)
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
    parser.add_argument("--reference_version", type=int, default=1)
    parser.add_argument("--benign_version", type=int, default=2)
    parser.add_argument("--isp_version", type=int, default=3)
    parser.add_argument("--cross_version", type=int, default=1)
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
    parser.add_argument("--topology_distance_mode", type=str, default="l1", choices=["l1", "l2", "cosine"])
    parser.add_argument("--topology_normalize_vectors", action="store_true")
    parser.add_argument("--h0_weight", type=float, default=1.3)
    parser.add_argument("--h1_weight", type=float, default=0.7)
    parser.add_argument(
        "--output_json",
        type=str,
        default=str(PROJECT_ROOT / "data" / "topology_evaluation_summary.json"),
    )

    args = parser.parse_args()
    itf_cache_dir = Path(args.itf_cache_dir) if args.itf_cache_dir else None
    if itf_cache_dir is not None and not itf_cache_dir.exists():
        itf_cache_dir = None
    topo_cache_dir = Path(args.topo_cache_dir) if args.topo_cache_dir else None
    if topo_cache_dir is not None and not topo_cache_dir.exists():
        topo_cache_dir = None

    summary = evaluate_topology(
        meta_path=Path(args.meta_path),
        stage_cache_dir=Path(args.stage_cache_dir),
        checkpoint_path=Path(args.checkpoint_path),
        num_raws=args.num_raws,
        seed=args.seed,
        device=args.device,
        itf_cache_dir=itf_cache_dir,
        topo_cache_dir=topo_cache_dir,
        source=args.source,
        crop_size=args.crop_size,
        feature_source=args.feature_source,
        scalarization=args.scalarization,
        normalization=args.normalization,
        reference_version=args.reference_version,
        benign_version=args.benign_version,
        isp_version=args.isp_version,
        cross_version=args.cross_version,
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
    )

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print_summary(summary)
    print(f"Saved summary to: {output_json}")


if __name__ == "__main__":
    main()
