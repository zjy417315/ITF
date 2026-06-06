import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.prototype_dataset import PrototypeDistillationDataset
from src.itf import ImagingTraceFieldExtractor
from src.isp.simple_isp import ISPConfig, SimpleISP
from src.models.auth_code_head import AuthCodeHead
from src.models.auth_codebook import AuthenticationCodebook
from src.models.claim_score_fusion import ClaimScoreFusionHead
from src.models.prototype_verifier import PrototypeVerifier
from src.models.student_code_head import StudentCodeHead
from src.models.student_token_head import StudentTokenHead
from src.models.teacher_tokenizer import TeacherTokenTokenizer
from src.models.visual_encoder import VisualEncoder
from src.topo import ITFTopologySummarizer
from src.tools.data_roots import (
    resolve_dataset_root,
    resolve_experiment_root,
    resolve_meta_path,
    resolve_stage_cache_dir,
)
from src.tools.ecc_codec import build_auth_codec
from src.tools.evaluate_joint_score import fuse_scores, normalize_score
from src.tools.evaluate_topology import build_weighted_topology_sequence, generic_sequence_distance
from src.tools.stage3_official_preset import (
    OFFICIAL_STAGE3_EVAL_MAX_RAWS,
    OFFICIAL_STAGE3_PRESET_NAME,
    resolve_stage3_official_config,
)
from src.train.train_stage3_authcode import (
    build_claim_score_outputs,
    build_claim_verifier_repr,
    claim_score_fusion_input_dim,
    combine_score_matrices,
    compute_claim_scores,
    compute_claim_verifier_scores,
    compute_protocol_score_bundle,
    compute_token_match_matrix,
    forward_student_branches,
    soft_bits,
)
from src.train.train_stage3_code import summarize_verification_scores
from src.train.train_stage3_prototype import (
    build_group_split,
    build_transforms,
    load_state_dict_shape_safe,
    set_seed,
)


DEFAULT_OFFICIAL_EVAL_JSON = Path(
    r"<artifact-local-path-redacted>"
)
DEFAULT_JOINT_CONFIG_JSON = PROJECT_ROOT / "data" / "joint_score_eval_live32_weighted_competitive.json"
DEFAULT_OUTPUT_JSON = PROJECT_ROOT / "Paper" / "generated" / "main_results_summary.json"
CONDITION_MAP = {"Ref": 1, "Prop": 2, "Shift-1": 3, "Shift-2": 4}


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ordered_index_map(
    dataset: PrototypeDistillationDataset,
    indices: Iterable[int],
    raw_anchors: List[str],
    versions: Iterable[int],
) -> Dict[int, List[int]]:
    lookup = {}
    raw_set = set(raw_anchors)
    version_set = {int(v) for v in versions}
    for idx in indices:
        record = dataset.records[idx]
        raw_anchor = record["raw_anchor"]
        version = int(record["version"])
        if raw_anchor in raw_set and version in version_set:
            lookup[(raw_anchor, version)] = idx
    ordered = {}
    for version in version_set:
        ordered_indices = []
        for raw_anchor in raw_anchors:
            key = (raw_anchor, version)
            if key not in lookup:
                raise RuntimeError(f"Missing validation sample for raw_anchor={raw_anchor}, version={version}.")
            ordered_indices.append(lookup[key])
        ordered[version] = ordered_indices
    return ordered


def summarize_condition_scores(
    positive_scores: torch.Tensor,
    shared_negative_matrix: torch.Tensor,
) -> Dict[str, float]:
    negative_mask = ~torch.eye(shared_negative_matrix.shape[0], dtype=torch.bool, device=shared_negative_matrix.device)
    negative_scores = shared_negative_matrix[negative_mask].view(shared_negative_matrix.shape[0], -1)
    hard_negative_scores = negative_scores.max(dim=1).values if negative_scores.numel() else positive_scores.new_zeros(positive_scores.shape)
    metrics = summarize_verification_scores(
        positive_scores.detach().cpu(),
        hard_negative_scores.detach().cpu(),
        negative_scores.reshape(-1).detach().cpu(),
    )
    return {
        "auc": float(metrics["pairwise_auc"]),
        "eer": float(metrics["eer"]),
        "tar_at_far_1e2": float(metrics["tar_at_far_1e2"]),
        "tar_at_far_1e3": float(metrics["tar_at_far_1e3"]),
        "hard_auc": float(metrics["hard_auc"]),
    }


def load_itf_components(
    checkpoint_path: Path,
    feature_source: str,
    scalarization: str,
    normalization: str,
    device: str | None,
    pixel_size: float,
    birth_min: float,
    birth_max: float,
    pers_min: float,
    pers_max: float,
    kernel_sigma: float,
    h0_weight: float,
    h1_weight: float,
):
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
    return extractor, summarizer, h0_weight, h1_weight


def collect_raw_representations(
    meta: Dict[str, Dict],
    version_ids: List[str],
    source: str,
    stage_cache_dir: Path,
    extractor: ImagingTraceFieldExtractor,
    summarizer: ITFTopologySummarizer,
    h0_weight: float,
    h1_weight: float,
    crop_size: int,
):
    cached_itf: Dict[str, torch.Tensor] = {}
    cached_topology: Dict[str, torch.Tensor] = {}
    isp = None
    if source == "live_isp":
        isp = SimpleISP(config=ISPConfig(center_crop=True, crop_size=crop_size))

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

    for version_id in version_ids:
        load_topology(version_id)
    return cached_itf, cached_topology


def compute_raw_side_results(
    raw_anchor_order: List[str],
    version_ids_by_raw: Dict[int, List[str]],
    cached_itf: Dict[str, torch.Tensor],
    cached_topology: Dict[str, torch.Tensor],
    stage_weights,
    topology_distance_mode: str,
    topology_normalize_vectors: bool,
    normalization_mode: str,
    fusion_mode: str,
    alpha: float,
):
    ref_ids = version_ids_by_raw[1]
    cross_itf_distances = []
    cross_topo_distances = []
    for i, ref_id in enumerate(ref_ids):
        for j, cross_id in enumerate(ref_ids):
            if i == j:
                continue
            cross_itf_distances.append(
                generic_sequence_distance(cached_itf[ref_id], cached_itf[cross_id], stage_weights=stage_weights)[0]
            )
            cross_topo_distances.append(
                generic_sequence_distance(
                    cached_topology[ref_id],
                    cached_topology[cross_id],
                    mode=topology_distance_mode,
                    normalize_vectors=topology_normalize_vectors,
                    stage_weights=stage_weights,
                )[0]
            )

    anchor_itf_distances = []
    anchor_topo_distances = []
    shift_itf_distances = []
    shift_topo_distances = []
    for ref_id, prop_id in zip(ref_ids, version_ids_by_raw[2]):
        anchor_itf_distances.append(
            generic_sequence_distance(cached_itf[ref_id], cached_itf[prop_id], stage_weights=stage_weights)[0]
        )
        anchor_topo_distances.append(
            generic_sequence_distance(
                cached_topology[ref_id],
                cached_topology[prop_id],
                mode=topology_distance_mode,
                normalize_vectors=topology_normalize_vectors,
                stage_weights=stage_weights,
            )[0]
        )
    for version in (3, 4):
        for ref_id, shift_id in zip(ref_ids, version_ids_by_raw[version]):
            shift_itf_distances.append(
                generic_sequence_distance(cached_itf[ref_id], cached_itf[shift_id], stage_weights=stage_weights)[0]
            )
            shift_topo_distances.append(
                generic_sequence_distance(
                    cached_topology[ref_id],
                    cached_topology[shift_id],
                    mode=topology_distance_mode,
                    normalize_vectors=topology_normalize_vectors,
                    stage_weights=stage_weights,
                )[0]
            )

    normalization_stats = {
        "itf": {
            "anchor_mean": float(torch.tensor(anchor_itf_distances).mean().item()),
            "isp_mean": float(torch.tensor(shift_itf_distances).mean().item()),
            "cross_mean": float(torch.tensor(cross_itf_distances).mean().item()),
        },
        "topology": {
            "anchor_mean": float(torch.tensor(anchor_topo_distances).mean().item()),
            "isp_mean": float(torch.tensor(shift_topo_distances).mean().item()),
            "cross_mean": float(torch.tensor(cross_topo_distances).mean().item()),
        },
    }

    def build_distance_matrices(query_ids: List[str]):
        n = len(query_ids)
        itf_distance = torch.zeros((n, n), dtype=torch.float32)
        topo_distance = torch.zeros((n, n), dtype=torch.float32)
        joint_distance = torch.zeros((n, n), dtype=torch.float32)
        for i, query_id in enumerate(query_ids):
            for j, ref_id in enumerate(ref_ids):
                itf_dist = generic_sequence_distance(
                    cached_itf[query_id],
                    cached_itf[ref_id],
                    stage_weights=stage_weights,
                )[0]
                topo_dist = generic_sequence_distance(
                    cached_topology[query_id],
                    cached_topology[ref_id],
                    mode=topology_distance_mode,
                    normalize_vectors=topology_normalize_vectors,
                    stage_weights=stage_weights,
                )[0]
                itf_distance[i, j] = float(itf_dist)
                topo_distance[i, j] = float(topo_dist)
                joint_distance[i, j] = float(
                    fuse_scores(
                        normalize_score(itf_dist, "itf", normalization_mode, normalization_stats),
                        normalize_score(topo_dist, "topology", normalization_mode, normalization_stats),
                        alpha=alpha,
                        fusion_mode=fusion_mode,
                    )
                )
        return {
            "ITF-only": -itf_distance,
            "Topology-only": -topo_distance,
            "ITF+Topology": -joint_distance,
        }

    ref_negative_pool = build_distance_matrices(ref_ids)
    results = {}
    for condition_name, version in CONDITION_MAP.items():
        score_matrices = build_distance_matrices(version_ids_by_raw[version])
        results[condition_name] = {}
        for branch_name, score_matrix in score_matrices.items():
            results[condition_name][branch_name] = summarize_condition_scores(
                positive_scores=torch.diag(score_matrix),
                shared_negative_matrix=ref_negative_pool[branch_name],
            )
    return results, normalization_stats


def load_stage3_models(
    checkpoint_path: Path,
    prototype_cache_dir: Path,
    teacher_cache_dir: Path,
    meta_path: Path,
    rgb_dir: Path,
    seed: int,
    val_ratio: float,
    max_raws: int,
    device: torch.device,
):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("config", {})
    official_eval_preset, official_config_resolved = resolve_stage3_official_config(
        config=config,
        preset_name=OFFICIAL_STAGE3_PRESET_NAME,
        cli_overrides={},
    )
    bit_scale = float(official_config_resolved.get("bit_scale", config.get("bit_scale", 3.0)))
    verifier_weight = float(official_config_resolved.get("claim_verifier_weight", config.get("claim_verifier_weight", 0.5)))
    claim_sequence_score_weight = config.get("claim_sequence_score_weight")
    if claim_sequence_score_weight is None:
        claim_sequence_score_weight = config.get("sequence_score_weight", 0.0)
    claim_sequence_score_weight = float(claim_sequence_score_weight)
    protocol_score_mode = str(official_config_resolved.get("protocol_score_mode", config.get("protocol_score_mode", "none")))
    protocol_score_alpha = float(config.get("protocol_score_alpha", 0.0))
    official_claim_score_mode = str(
        official_config_resolved.get("official_claim_score_mode", config.get("official_claim_score_mode", "fusion_head"))
    )
    token_gate_penalty = float(config.get("token_gate_penalty", 1.0))
    token_residual_weight = float(config.get("token_residual_weight", 0.25))
    claim_main_score_norm_mode = str(
        official_config_resolved.get("claim_main_score_norm_mode", config.get("claim_main_score_norm_mode", "none"))
    )
    protocol_score_norm_mode = str(
        official_config_resolved.get("protocol_score_norm_mode", config.get("protocol_score_norm_mode", "none"))
    )
    use_sequence = bool(
        config.get("sequence_score_weight", 0.0) > 0.0
        or config.get("teacher_sequence_match_weight", 0.0) > 0.0
        or config.get("student_sequence_match_weight", 0.0) > 0.0
    )
    _, eval_transform = build_transforms(
        config.get("augmentation_preset", "center"),
        image_size=int(config.get("image_size", 224)),
        resize_size=int(config.get("resize_size", 320)),
    )
    dataset = PrototypeDistillationDataset(
        prototype_cache_dir=prototype_cache_dir,
        meta_path=meta_path,
        rgb_dir=rgb_dir,
        teacher_cache_dir=teacher_cache_dir,
        teacher_key=config.get("teacher_key", "teacher_joint_seq"),
        include_versions=[1, 2, 3, 4],
        transform=eval_transform,
    )
    _, val_indices, _, val_raws = build_group_split(
        dataset=dataset,
        val_ratio=val_ratio,
        seed=seed,
        max_raws=max_raws,
    )
    auth_codec = build_auth_codec(
        code_dim=int(config.get("code_dim", 32)),
        ecc_scheme=config.get("ecc_scheme", "identity"),
        ecc_repetition=int(config.get("ecc_repetition", 2)),
        payload_dim=config.get("payload_dim"),
    )
    token_protocol_enabled = str(config.get("auth_protocol_variant", "legacy_continuous")) == "token_residual"
    student_code_space = str(config.get("student_code_space", "payload"))
    student_output_dim = auth_codec.code_dim if student_code_space == "codeword" else auth_codec.payload_dim
    auth_codebook = None
    if int(config.get("teacher_codebook_size", 0)) > 0 or checkpoint.get("auth_codebook_state_dict") is not None:
        codebook_size = int(config.get("teacher_codebook_size", 0))
        if checkpoint.get("auth_codebook_state_dict") is not None and codebook_size <= 0:
            codebook_size = int(checkpoint["auth_codebook_state_dict"]["codes"].shape[0])
        auth_codebook = AuthenticationCodebook(
            code_dim=auth_codec.code_dim,
            num_codes=codebook_size,
            temperature=float(config.get("teacher_codebook_temperature", 12.0)),
            learnable=False,
        ).to(device)
        if checkpoint.get("auth_codebook_state_dict") is not None:
            auth_codebook.load_state_dict(checkpoint["auth_codebook_state_dict"])
        auth_codebook.eval()
    apply_codebook_to_scores = auth_codebook is not None and str(config.get("codebook_mode", "replace")) == "replace"

    def encode_teacher_logits(logits):
        if logits is None:
            return None
        return auth_codec.encode_logits(logits)

    def encode_student_logits(logits):
        if logits is None:
            return None
        if student_code_space == "codeword":
            return logits
        return auth_codec.encode_logits(logits)

    def quantize_global_logits(logits):
        if logits is None or auth_codebook is None:
            return logits, None
        quantized, indices, _, _ = auth_codebook.quantize(logits, straight_through=False)
        return quantized, indices

    teacher_head = AuthCodeHead(
        d_in=dataset.prototype_dim,
        code_dim=auth_codec.payload_dim,
        hidden_dim=int(config.get("teacher_hidden_dim", 256)),
        use_sequence=use_sequence,
        dropout=float(config.get("teacher_dropout", 0.0)),
    ).to(device)
    teacher_head.load_state_dict(checkpoint["teacher_head_state_dict"])
    teacher_head.eval()

    student = VisualEncoder(
        d_out=student_output_dim,
        backbone_type=config.get("backbone_type", "resnet18"),
        pretrained=False,
        input_mode=config.get("input_mode", "residual_only"),
        residual_scale=float(config.get("residual_scale", 1.75)),
        residual_kernel=int(config.get("residual_kernel", 9)),
        use_stage_sequence_head=use_sequence,
        local_crop_mode=config.get("local_crop_mode", "none"),
        local_crop_size=int(config.get("local_crop_size", 160)),
        local_patch_offset=int(config.get("local_patch_offset", 24)),
    ).to(device)
    student.load_state_dict(checkpoint["student_state_dict"])
    student.eval()

    student_code_head = None
    if bool(config.get("use_student_code_head", False) or checkpoint.get("student_code_head_state_dict") is not None):
        student_code_head = StudentCodeHead(
            d_in=student.feature_dim,
            code_dim=auth_codec.code_dim,
            hidden_dim=int(config.get("student_code_head_hidden_dim", 256)),
            dropout=float(config.get("student_code_head_dropout", 0.0)),
        ).to(device)
        if checkpoint.get("student_code_head_state_dict") is not None:
            student_code_head.load_state_dict(checkpoint["student_code_head_state_dict"])
        student_code_head.eval()

    student_token_head = None
    if token_protocol_enabled and checkpoint.get("student_token_head_state_dict") is not None:
        token_head_state = checkpoint["student_token_head_state_dict"]
        num_tokens = int(config.get("teacher_token_classes", token_head_state["classifier.weight"].shape[0]))
        student_token_head = StudentTokenHead(
            d_in=student.feature_dim,
            num_tokens=num_tokens,
            code_dim=int(config.get("code_dim", 32)),
            hidden_dim=int(config.get("student_token_head_hidden_dim", 256)),
            dropout=float(config.get("student_token_head_dropout", 0.0)),
        ).to(device)
        student_token_head.load_state_dict(token_head_state)
        student_token_head.eval()

    teacher_tokenizer = None
    if token_protocol_enabled and checkpoint.get("teacher_tokenizer_state_dict") is not None:
        token_state = checkpoint["teacher_tokenizer_state_dict"]
        teacher_tokenizer = TeacherTokenTokenizer(
            code_dim=auth_codec.code_dim,
            num_tokens=int(token_state["prototypes"].shape[0]),
            temperature=float(config.get("teacher_token_temperature", 12.0)),
        ).to(device)
        teacher_tokenizer.load_state_dict(token_state)
        teacher_tokenizer.eval()

    claim_verifier = None
    claim_verifier_enabled = bool(
        config.get("use_claim_verifier_head", False)
        or checkpoint.get("claim_verifier_state_dict") is not None
    )
    if claim_verifier_enabled:
        verifier_dim = build_claim_verifier_repr(
            torch.zeros(1, auth_codec.code_dim, device=device),
            torch.zeros(1, auth_codec.code_dim, device=device),
            str(config.get("claim_verifier_input_mode", "bits")),
        ).shape[1]
        claim_verifier = PrototypeVerifier(
            d_model=verifier_dim,
            hidden_dim=int(config.get("claim_verifier_hidden_dim", 128)),
            dropout=float(config.get("claim_verifier_dropout", 0.0)),
        ).to(device)
        if checkpoint.get("claim_verifier_state_dict") is not None:
            claim_verifier.load_state_dict(checkpoint["claim_verifier_state_dict"])
        claim_verifier.eval()

    claim_score_fusion_head = None
    if (
        config.get("use_claim_score_fusion_head", False)
        or checkpoint.get("claim_score_fusion_state_dict") is not None
    ):
        claim_score_fusion_head = ClaimScoreFusionHead(
            input_dim=claim_score_fusion_input_dim(
                config.get("claim_score_fusion_protocol_modes"),
                include_token_score=token_protocol_enabled,
                include_verifier_score=bool(config.get("claim_verifier_feature_to_fusion", False))
                and claim_verifier_enabled,
            ),
            hidden_dim=int(config.get("claim_score_fusion_hidden_dim", 32)),
            dropout=float(config.get("claim_score_fusion_dropout", 0.0)),
            mode=str(config.get("claim_score_fusion_mode", "direct") or "direct"),
            residual_scale=float(config.get("claim_score_fusion_residual_scale", 1.0)),
        ).to(device)
        if checkpoint.get("claim_score_fusion_state_dict") is not None:
            load_state_dict_shape_safe(
                claim_score_fusion_head,
                checkpoint["claim_score_fusion_state_dict"],
                module_name="ClaimScoreFusionHead",
            )
        claim_score_fusion_head.eval()

    return {
        "config": config,
        "official_eval_preset": official_eval_preset,
        "official_config_resolved": official_config_resolved,
        "dataset": dataset,
        "val_indices": val_indices,
        "val_raws": list(val_raws),
        "auth_codec": auth_codec,
        "auth_codebook": auth_codebook,
        "apply_codebook_to_scores": apply_codebook_to_scores,
        "teacher_head": teacher_head,
        "student": student,
        "student_code_head": student_code_head,
        "student_token_head": student_token_head,
        "teacher_tokenizer": teacher_tokenizer,
        "claim_verifier": claim_verifier,
        "claim_score_fusion_head": claim_score_fusion_head,
        "bit_scale": bit_scale,
        "verifier_weight": verifier_weight,
        "claim_sequence_score_weight": claim_sequence_score_weight,
        "protocol_score_mode": protocol_score_mode,
        "protocol_score_alpha": protocol_score_alpha,
        "official_claim_score_mode": official_claim_score_mode,
        "token_gate_penalty": token_gate_penalty,
        "token_residual_weight": token_residual_weight,
        "claim_main_score_norm_mode": claim_main_score_norm_mode,
        "protocol_score_norm_mode": protocol_score_norm_mode,
        "use_sequence": use_sequence,
        "encode_teacher_logits": encode_teacher_logits,
        "encode_student_logits": encode_student_logits,
        "quantize_global_logits": quantize_global_logits,
    }


def collect_stage3_outputs(
    dataset: PrototypeDistillationDataset,
    indices: List[int],
    bundle: Dict,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    need_teacher: bool,
    need_student: bool,
) -> Dict[str, torch.Tensor | List[str] | None]:
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False, num_workers=num_workers)
    outputs = {
        "raw_anchor": [],
        "sample_id": [],
        "version": [],
        "teacher_global_logits": [],
        "teacher_stage_logits": [],
        "teacher_protocol_logits": [],
        "teacher_code_indices": [],
        "teacher_token_indices": [],
        "student_global_logits": [],
        "student_stage_logits": [],
        "student_protocol_logits": [],
        "student_code_indices": [],
        "student_token_logits": [],
    }

    with torch.no_grad():
        for batch in loader:
            outputs["raw_anchor"].extend(list(batch["raw_anchor"]))
            outputs["sample_id"].extend(list(batch["sample_id"]))
            outputs["version"].extend(batch["version"].tolist())

            if need_teacher:
                teacher_out = bundle["teacher_head"](
                    batch["teacher_vec"].to(device),
                    batch["teacher_seq"].to(device),
                    return_sequence=bundle["use_sequence"],
                    return_logits=True,
                )
                teacher_encoded_logits = bundle["encode_teacher_logits"](teacher_out["global_logits"])
                teacher_code_logits, teacher_code_indices = bundle["quantize_global_logits"](teacher_encoded_logits)
                teacher_global_logits = (
                    teacher_code_logits
                    if bundle["apply_codebook_to_scores"] and teacher_code_logits is not None
                    else teacher_encoded_logits
                )
                teacher_stage_logits = bundle["encode_teacher_logits"](teacher_out.get("stage_logits"))
                outputs["teacher_global_logits"].append(teacher_global_logits.detach().cpu())
                outputs["teacher_stage_logits"].append(
                    teacher_stage_logits.detach().cpu() if teacher_stage_logits is not None else None
                )
                outputs["teacher_protocol_logits"].append(
                    (teacher_code_logits if teacher_code_logits is not None else teacher_global_logits).detach().cpu()
                )
                outputs["teacher_code_indices"].append(
                    teacher_code_indices.detach().cpu() if teacher_code_indices is not None else None
                )
                if bundle["teacher_tokenizer"] is not None:
                    teacher_token_indices, _ = bundle["teacher_tokenizer"].assign(teacher_encoded_logits)
                    outputs["teacher_token_indices"].append(teacher_token_indices.detach().cpu())
                else:
                    outputs["teacher_token_indices"].append(None)

            if need_student:
                student_forward = forward_student_branches(
                    student=bundle["student"],
                    rgb_image=batch["rgb_image"].to(device),
                    use_sequence=bundle["use_sequence"],
                    encode_student_logits=bundle["encode_student_logits"],
                    auth_codebook=bundle["auth_codebook"],
                    apply_codebook_to_scores=bundle["apply_codebook_to_scores"],
                    student_code_head=bundle["student_code_head"],
                    student_token_head=bundle["student_token_head"],
                )
                student_global_logits = student_forward["score_logits"]
                student_stage_logits = student_forward["stage_logits"]
                student_protocol_logits = student_forward["recovery_protocol_logits"]
                if student_protocol_logits is None:
                    student_protocol_logits = student_global_logits
                student_token_logits = student_forward["token_logits"]
                if bundle["teacher_tokenizer"] is not None:
                    base_query_token_logits = bundle["teacher_tokenizer"].logits(student_protocol_logits)
                    student_token_logits = (
                        base_query_token_logits if student_token_logits is None else base_query_token_logits + student_token_logits
                    )
                outputs["student_global_logits"].append(student_global_logits.detach().cpu())
                outputs["student_stage_logits"].append(
                    student_stage_logits.detach().cpu() if student_stage_logits is not None else None
                )
                outputs["student_protocol_logits"].append(student_protocol_logits.detach().cpu())
                outputs["student_code_indices"].append(
                    student_forward["recovery_code_indices"].detach().cpu()
                    if student_forward["recovery_code_indices"] is not None
                    else None
                )
                outputs["student_token_logits"].append(
                    student_token_logits.detach().cpu() if student_token_logits is not None else None
                )

    def finalize(entries):
        non_null = [entry for entry in entries if entry is not None]
        if not non_null:
            return None
        return torch.cat(non_null, dim=0)

    return {
        "raw_anchor": outputs["raw_anchor"],
        "sample_id": outputs["sample_id"],
        "version": torch.tensor(outputs["version"], dtype=torch.long),
        "teacher_global_logits": finalize(outputs["teacher_global_logits"]),
        "teacher_stage_logits": finalize(outputs["teacher_stage_logits"]),
        "teacher_protocol_logits": finalize(outputs["teacher_protocol_logits"]),
        "teacher_code_indices": finalize(outputs["teacher_code_indices"]),
        "teacher_token_indices": finalize(outputs["teacher_token_indices"]),
        "student_global_logits": finalize(outputs["student_global_logits"]),
        "student_stage_logits": finalize(outputs["student_stage_logits"]),
        "student_protocol_logits": finalize(outputs["student_protocol_logits"]),
        "student_code_indices": finalize(outputs["student_code_indices"]),
        "student_token_logits": finalize(outputs["student_token_logits"]),
    }


def compute_rgb_side_results(bundle: Dict, reference_bank: Dict, query_bank_by_version: Dict[int, Dict], device: torch.device):
    ref_bits = soft_bits(reference_bank["student_global_logits"].to(device), bundle["bit_scale"])
    ref_stage_logits = reference_bank["student_stage_logits"].to(device) if reference_bank["student_stage_logits"] is not None else None
    ref_protocol_logits = reference_bank["student_protocol_logits"].to(device)
    ref_code_indices = reference_bank["student_code_indices"].to(device) if reference_bank["student_code_indices"] is not None else None

    def build_score_set(query_bank: Dict):
        query_bits = soft_bits(query_bank["student_global_logits"].to(device), bundle["bit_scale"])
        query_stage_logits = query_bank["student_stage_logits"].to(device) if query_bank["student_stage_logits"] is not None else None
        query_protocol_logits = query_bank["student_protocol_logits"].to(device)
        query_code_indices = query_bank["student_code_indices"].to(device) if query_bank["student_code_indices"] is not None else None
        main_scores = compute_claim_scores(
            query_bits,
            ref_bits,
            claim_verifier=None,
            verifier_weight=0.0,
            query_stage_logits=query_stage_logits,
            reference_stage_logits=ref_stage_logits,
            bit_scale=bundle["bit_scale"],
            sequence_score_weight=bundle["claim_sequence_score_weight"],
        )
        protocol_scores, extra_protocol_scores = compute_protocol_score_bundle(
            query_protocol_logits=query_protocol_logits,
            reference_protocol_logits=ref_protocol_logits,
            auth_codec=bundle["auth_codec"],
            primary_mode=bundle["protocol_score_mode"],
            extra_modes=bundle["config"].get("claim_score_fusion_protocol_modes"),
            query_code_indices=query_code_indices,
            reference_code_indices=ref_code_indices,
        )
        joint_scores = build_claim_score_outputs(
            main_scores,
            protocol_scores,
            token_scores=None,
            extra_protocol_scores=extra_protocol_scores,
            claim_score_fusion_head=bundle["claim_score_fusion_head"],
            official_mode=bundle["official_claim_score_mode"],
            alpha=bundle["protocol_score_alpha"],
            main_normalization=bundle["claim_main_score_norm_mode"],
            auxiliary_normalization=bundle["protocol_score_norm_mode"],
            gate_penalty=bundle["token_gate_penalty"],
            residual_weight=bundle["token_residual_weight"],
            hard_gate_threshold=float(bundle["config"].get("token_hard_gate_threshold", 0.5)),
        )["official_scores"]
        return {
            "ITF-only": main_scores.detach().cpu(),
            "Topology-only": protocol_scores.detach().cpu(),
            "ITF+Topology": joint_scores.detach().cpu(),
        }

    ref_scores = build_score_set(query_bank_by_version[1])
    results = {}
    for condition_name, version in CONDITION_MAP.items():
        score_set = build_score_set(query_bank_by_version[version])
        results[condition_name] = {}
        for branch_name, score_matrix in score_set.items():
            results[condition_name][branch_name] = summarize_condition_scores(
                positive_scores=torch.diag(score_matrix),
                shared_negative_matrix=ref_scores[branch_name],
            )
    return results


def compute_rgb_projection_results(
    bundle: Dict,
    reference_bank: Dict,
    query_bank_by_version: Dict[int, Dict],
    device: torch.device,
):
    ref_teacher_global_logits = reference_bank["teacher_global_logits"].to(device)
    ref_teacher_bits = soft_bits(ref_teacher_global_logits, bundle["bit_scale"])
    ref_teacher_stage_logits = reference_bank["teacher_stage_logits"].to(device) if reference_bank["teacher_stage_logits"] is not None else None
    ref_teacher_protocol_logits = reference_bank["teacher_protocol_logits"].to(device)
    ref_teacher_code_indices = reference_bank["teacher_code_indices"].to(device) if reference_bank["teacher_code_indices"] is not None else None

    def build_score_set(query_bank: Dict):
        query_student_global_logits = query_bank["student_global_logits"].to(device)
        query_student_bits = soft_bits(query_student_global_logits, bundle["bit_scale"])
        query_student_stage_logits = query_bank["student_stage_logits"].to(device) if query_bank["student_stage_logits"] is not None else None
        query_student_protocol_logits = query_bank["student_protocol_logits"].to(device)
        query_student_code_indices = query_bank["student_code_indices"].to(device) if query_bank["student_code_indices"] is not None else None

        geometric_scores = compute_claim_scores(
            query_student_bits,
            ref_teacher_bits,
            claim_verifier=None,
            verifier_weight=0.0,
            query_stage_logits=query_student_stage_logits,
            reference_stage_logits=ref_teacher_stage_logits,
            bit_scale=bundle["bit_scale"],
            sequence_score_weight=bundle["claim_sequence_score_weight"],
        )
        topological_scores, extra_protocol_scores = compute_protocol_score_bundle(
            query_protocol_logits=query_student_protocol_logits,
            reference_protocol_logits=ref_teacher_protocol_logits,
            auth_codec=bundle["auth_codec"],
            primary_mode=bundle["protocol_score_mode"],
            extra_modes=bundle["config"].get("claim_score_fusion_protocol_modes"),
            query_code_indices=query_student_code_indices,
            reference_code_indices=ref_teacher_code_indices,
        )
        joint_scores = build_claim_score_outputs(
            geometric_scores,
            topological_scores,
            token_scores=None,
            extra_protocol_scores=extra_protocol_scores,
            claim_score_fusion_head=bundle["claim_score_fusion_head"],
            official_mode="fusion_head",
            alpha=bundle["protocol_score_alpha"],
            main_normalization=bundle["claim_main_score_norm_mode"],
            auxiliary_normalization=bundle["protocol_score_norm_mode"],
            gate_penalty=bundle["token_gate_penalty"],
            residual_weight=bundle["token_residual_weight"],
            hard_gate_threshold=float(bundle["config"].get("token_hard_gate_threshold", 0.5)),
        )["official_scores"]

        if joint_scores is None:
            joint_scores = combine_score_matrices(
                geometric_scores,
                topological_scores,
                alpha=0.5,
                main_normalization="none",
                auxiliary_normalization="none",
            )
        return {
            "Geometric": geometric_scores.detach().cpu(),
            "Topological": topological_scores.detach().cpu(),
            "Joint": joint_scores.detach().cpu(),
        }

    ref_scores = build_score_set(query_bank_by_version[1])
    results = {}
    for condition_name, version in CONDITION_MAP.items():
        score_set = build_score_set(query_bank_by_version[version])
        results[condition_name] = {}
        for branch_name, score_matrix in score_set.items():
            results[condition_name][branch_name] = summarize_condition_scores(
                positive_scores=torch.diag(score_matrix),
                shared_negative_matrix=ref_scores[branch_name],
            )
    return results


def compute_active_results(bundle: Dict, reference_bank: Dict, query_bank_by_version: Dict[int, Dict], device: torch.device):
    verifier_input_mode = str(bundle["config"].get("claim_verifier_input_mode", "bits"))
    ref_teacher_global_logits = reference_bank["teacher_global_logits"].to(device)
    ref_teacher_bits = soft_bits(ref_teacher_global_logits, bundle["bit_scale"])
    ref_teacher_stage_logits = reference_bank["teacher_stage_logits"].to(device) if reference_bank["teacher_stage_logits"] is not None else None
    ref_teacher_protocol_logits = reference_bank["teacher_protocol_logits"].to(device)
    ref_teacher_code_indices = reference_bank["teacher_code_indices"].to(device) if reference_bank["teacher_code_indices"] is not None else None
    ref_teacher_verifier_repr = build_claim_verifier_repr(ref_teacher_bits, ref_teacher_global_logits, verifier_input_mode)
    ref_teacher_token_indices = reference_bank["teacher_token_indices"].to(device) if reference_bank["teacher_token_indices"] is not None else None

    def build_score_set(query_bank: Dict):
        query_student_global_logits = query_bank["student_global_logits"].to(device)
        query_student_bits = soft_bits(query_student_global_logits, bundle["bit_scale"])
        query_student_stage_logits = query_bank["student_stage_logits"].to(device) if query_bank["student_stage_logits"] is not None else None
        query_student_protocol_logits = query_bank["student_protocol_logits"].to(device)
        query_student_code_indices = query_bank["student_code_indices"].to(device) if query_bank["student_code_indices"] is not None else None
        query_student_token_logits = query_bank["student_token_logits"].to(device) if query_bank["student_token_logits"] is not None else None
        query_student_verifier_repr = build_claim_verifier_repr(
            query_student_bits,
            query_student_global_logits,
            verifier_input_mode,
        )
        main_scores = compute_claim_scores(
            query_student_bits,
            ref_teacher_bits,
            bundle["claim_verifier"],
            verifier_weight=bundle["verifier_weight"],
            verifier_score_mode=str(bundle["config"].get("claim_verifier_score_mode", "add")),
            query_verifier_repr=query_student_verifier_repr,
            reference_verifier_repr=ref_teacher_verifier_repr,
            query_stage_logits=query_student_stage_logits,
            reference_stage_logits=ref_teacher_stage_logits,
            bit_scale=bundle["bit_scale"],
            sequence_score_weight=bundle["claim_sequence_score_weight"],
        )
        verifier_scores = compute_claim_verifier_scores(
            query_student_bits,
            ref_teacher_bits,
            bundle["claim_verifier"],
            verifier_weight=bundle["verifier_weight"],
            verifier_score_mode=str(bundle["config"].get("claim_verifier_score_mode", "add")),
            query_verifier_repr=query_student_verifier_repr,
            reference_verifier_repr=ref_teacher_verifier_repr,
        )
        protocol_scores, extra_protocol_scores = compute_protocol_score_bundle(
            query_protocol_logits=query_student_protocol_logits,
            reference_protocol_logits=ref_teacher_protocol_logits,
            auth_codec=bundle["auth_codec"],
            primary_mode=bundle["protocol_score_mode"],
            extra_modes=bundle["config"].get("claim_score_fusion_protocol_modes"),
            query_code_indices=query_student_code_indices,
            reference_code_indices=ref_teacher_code_indices,
        )
        if bool(bundle["config"].get("claim_verifier_feature_to_fusion", False)) and verifier_scores is not None:
            extra_protocol_scores.append(verifier_scores)
        token_scores = compute_token_match_matrix(query_student_token_logits, ref_teacher_token_indices)
        joint_scores = build_claim_score_outputs(
            main_scores,
            protocol_scores,
            token_scores=token_scores,
            extra_protocol_scores=extra_protocol_scores,
            claim_score_fusion_head=bundle["claim_score_fusion_head"],
            official_mode=bundle["official_claim_score_mode"],
            alpha=bundle["protocol_score_alpha"],
            main_normalization=bundle["claim_main_score_norm_mode"],
            auxiliary_normalization=bundle["protocol_score_norm_mode"],
            gate_penalty=bundle["token_gate_penalty"],
            residual_weight=bundle["token_residual_weight"],
            hard_gate_threshold=float(bundle["config"].get("token_hard_gate_threshold", 0.5)),
        )["official_scores"]
        return {
            "ITF-only": main_scores.detach().cpu(),
            "Topology-only": protocol_scores.detach().cpu(),
            "ITF+Topology": joint_scores.detach().cpu(),
        }

    ref_scores = build_score_set(query_bank_by_version[1])
    results = {}
    for condition_name, version in CONDITION_MAP.items():
        score_set = build_score_set(query_bank_by_version[version])
        results[condition_name] = {}
        for branch_name, score_matrix in score_set.items():
            results[condition_name][branch_name] = summarize_condition_scores(
                positive_scores=torch.diag(score_matrix),
                shared_negative_matrix=ref_scores[branch_name],
            )
    return results


def transpose_results(results_by_condition: Dict[str, Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    branches = ["ITF-only", "Topology-only", "ITF+Topology"]
    output = {branch: {} for branch in branches}
    for condition_name, branch_values in results_by_condition.items():
        for branch_name in branches:
            output[branch_name][condition_name] = branch_values[branch_name]
    return output


def transpose_named_results(
    results_by_condition: Dict[str, Dict[str, Dict[str, float]]],
    branch_order: List[str],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    output = {branch: {} for branch in branch_order}
    for condition_name, branch_values in results_by_condition.items():
        for branch_name in branch_order:
            output[branch_name][condition_name] = branch_values[branch_name]
    return output


def build_primary_table(
    raw_side_by_condition: Dict[str, Dict[str, Dict[str, float]]],
    rgb_projection_by_condition: Dict[str, Dict[str, Dict[str, float]]],
    active_by_condition: Dict[str, Dict[str, Dict[str, float]]],
) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    return {
        "Process": {
            "Joint": transpose_named_results(raw_side_by_condition, ["ITF+Topology"])["ITF+Topology"],
            "ITF": transpose_named_results(raw_side_by_condition, ["ITF-only"])["ITF-only"],
            "Topology": transpose_named_results(raw_side_by_condition, ["Topology-only"])["Topology-only"],
        },
        "RGB Projection": {
            "Joint": transpose_named_results(rgb_projection_by_condition, ["Joint"])["Joint"],
            "Geometric": transpose_named_results(rgb_projection_by_condition, ["Geometric"])["Geometric"],
            "Topological": transpose_named_results(rgb_projection_by_condition, ["Topological"])["Topological"],
        },
        "Active Verification": {
            "Final": transpose_named_results(active_by_condition, ["ITF+Topology"])["ITF+Topology"],
            "Main": transpose_named_results(active_by_condition, ["ITF-only"])["ITF-only"],
            "Protocol": transpose_named_results(active_by_condition, ["Topology-only"])["Topology-only"],
        },
    }


def evaluate_main_results(args) -> Dict:
    set_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    stage3_eval_json = load_json(Path(args.stage3_eval_json))
    stage3_checkpoint = Path(args.stage3_checkpoint or stage3_eval_json["checkpoint"])
    joint_config_json = load_json(Path(args.joint_config_json))

    bundle = load_stage3_models(
        checkpoint_path=stage3_checkpoint,
        prototype_cache_dir=Path(args.prototype_cache_dir),
        teacher_cache_dir=Path(args.teacher_cache_dir),
        meta_path=Path(args.meta_path),
        rgb_dir=Path(args.rgb_dir),
        seed=args.seed,
        val_ratio=args.val_ratio,
        max_raws=args.max_raws,
        device=device,
    )
    if args.max_raws != OFFICIAL_STAGE3_EVAL_MAX_RAWS:
        print(
            f"[warning] main-results table is intended for max_raws={OFFICIAL_STAGE3_EVAL_MAX_RAWS}, got {args.max_raws}.",
            flush=True,
        )

    raw_anchor_order = list(bundle["val_raws"])
    ordered_indices = ordered_index_map(bundle["dataset"], bundle["val_indices"], raw_anchor_order, CONDITION_MAP.values())
    version_ids_by_raw = {
        version: [bundle["dataset"].records[idx]["version_id"] for idx in ordered_indices[version]]
        for version in CONDITION_MAP.values()
    }

    extractor, summarizer, h0_weight, h1_weight = load_itf_components(
        checkpoint_path=Path(args.stage1_checkpoint),
        feature_source=str(joint_config_json["config"].get("feature_source", "layer4")),
        scalarization=str(joint_config_json["config"].get("scalarization", "l2")),
        normalization=str(joint_config_json["config"].get("normalization", "zscore")),
        device=args.device,
        pixel_size=float(joint_config_json["config"].get("pixel_size", 0.25)),
        birth_min=float(joint_config_json["config"].get("birth_min", -3.0)),
        birth_max=float(joint_config_json["config"].get("birth_max", 3.0)),
        pers_min=float(joint_config_json["config"].get("pers_min", 0.0)),
        pers_max=float(joint_config_json["config"].get("pers_max", 6.0)),
        kernel_sigma=float(joint_config_json["config"].get("kernel_sigma", 0.35)),
        h0_weight=float(joint_config_json["config"].get("h0_weight", 1.0)),
        h1_weight=float(joint_config_json["config"].get("h1_weight", 1.0)),
    )
    cached_itf, cached_topology = collect_raw_representations(
        meta=bundle["dataset"].meta,
        version_ids=[vid for version_ids in version_ids_by_raw.values() for vid in version_ids],
        source=args.source,
        stage_cache_dir=Path(args.stage_cache_dir),
        extractor=extractor,
        summarizer=summarizer,
        h0_weight=h0_weight,
        h1_weight=h1_weight,
        crop_size=args.crop_size,
    )
    raw_side_by_condition, raw_norm_stats = compute_raw_side_results(
        raw_anchor_order=raw_anchor_order,
        version_ids_by_raw=version_ids_by_raw,
        cached_itf=cached_itf,
        cached_topology=cached_topology,
        stage_weights=joint_config_json["config"].get("stage_weights"),
        topology_distance_mode=str(joint_config_json["config"].get("topology_distance_mode", "l1")),
        topology_normalize_vectors=bool(joint_config_json["config"].get("topology_normalize_vectors", False)),
        normalization_mode=str(joint_config_json["best_candidate"]["normalization_mode"]),
        fusion_mode=str(joint_config_json["best_candidate"]["fusion_mode"]),
        alpha=float(joint_config_json["best_candidate"]["alpha"]),
    )

    reference_bank = collect_stage3_outputs(
        dataset=bundle["dataset"],
        indices=ordered_indices[1],
        bundle=bundle,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        need_teacher=True,
        need_student=True,
    )
    query_bank_by_version = {
        version: collect_stage3_outputs(
            dataset=bundle["dataset"],
            indices=ordered_indices[version],
            bundle=bundle,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            need_teacher=False,
            need_student=True,
        )
        for version in CONDITION_MAP.values()
    }

    rgb_side_by_condition = compute_rgb_side_results(bundle, reference_bank, query_bank_by_version, device)
    rgb_projection_by_condition = compute_rgb_projection_results(bundle, reference_bank, query_bank_by_version, device)
    active_by_condition = compute_active_results(bundle, reference_bank, query_bank_by_version, device)

    legacy_table = {
        "RAW-side": transpose_results(raw_side_by_condition),
        "RGB-side": transpose_results(rgb_side_by_condition),
        "Active": transpose_results(active_by_condition),
    }
    primary_table = build_primary_table(
        raw_side_by_condition=raw_side_by_condition,
        rgb_projection_by_condition=rgb_projection_by_condition,
        active_by_condition=active_by_condition,
    )

    return {
        "config": {
            "seed": args.seed,
            "val_ratio": args.val_ratio,
            "max_raws": args.max_raws,
            "source": args.source,
            "stage1_checkpoint": str(args.stage1_checkpoint),
            "stage3_checkpoint": str(stage3_checkpoint),
            "stage3_eval_json": str(args.stage3_eval_json),
            "joint_config_json": str(args.joint_config_json),
            "shared_negative_pool": "ref_v1_cross_source",
            "condition_mapping": CONDITION_MAP,
            "rgb_side_mode": "student_only",
            "rgb_projection_mode": "student_query_vs_teacher_reference_surrogate",
            "rgb_projection_joint_mode": "fusion_head_no_verifier_no_token",
            "active_reference": "teacher_v1_credential_bank",
            "official_eval_preset": bundle["official_eval_preset"],
            "official_config_resolved": bundle["official_config_resolved"],
            "table_schema": args.table_schema,
            "val_raws": len(raw_anchor_order),
        },
        "joint_reference_config": {
            "config": joint_config_json["config"],
            "best_candidate": joint_config_json["best_candidate"],
            "normalization_stats": raw_norm_stats,
        },
        "legacy_table": legacy_table,
        "primary_table": primary_table,
        "table": primary_table if args.table_schema == "primary" else legacy_table,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate four-condition main results across RAW/RGB/Active settings.")
    parser.add_argument("--stage3_eval_json", type=str, default=str(DEFAULT_OFFICIAL_EVAL_JSON))
    parser.add_argument("--stage3_checkpoint", type=str, default=None)
    parser.add_argument("--joint_config_json", type=str, default=str(DEFAULT_JOINT_CONFIG_JSON))
    parser.add_argument("--stage1_checkpoint", type=str, default=str(PROJECT_ROOT / "checkpoints" / "stage1_joint" / "stage1_joint_best.pt"))
    parser.add_argument("--prototype_cache_dir", type=str, default=str(resolve_experiment_root() / "stage3_prototype_cache_anchor12_joint512_live"))
    parser.add_argument("--teacher_cache_dir", type=str, default=str(resolve_experiment_root() / "stage3_teacher_cache_joint512_live"))
    parser.add_argument("--meta_path", type=str, default=str(resolve_meta_path()))
    parser.add_argument("--rgb_dir", type=str, default=str(resolve_dataset_root() / "rgb_web_jpg"))
    parser.add_argument("--stage_cache_dir", type=str, default=str(resolve_stage_cache_dir()))
    parser.add_argument("--source", type=str, default="stage_cache", choices=["stage_cache", "live_isp"])
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--max_raws", type=int, default=1024)
    parser.add_argument("--table_schema", type=str, default="legacy", choices=["legacy", "primary"])
    parser.add_argument("--output_json", type=str, default=str(DEFAULT_OUTPUT_JSON))
    return parser.parse_args()


def main():
    args = parse_args()
    summary = evaluate_main_results(args)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[saved] {output_path}")


if __name__ == "__main__":
    main()
