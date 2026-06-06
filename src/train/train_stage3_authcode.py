import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.prototype_dataset import PrototypeDistillationDataset
from src.models.auth_codebook import AuthenticationCodebook
from src.models.auth_code_head import AuthCodeHead
from src.models.claim_score_fusion import ClaimScoreFusionHead
from src.models.prototype_verifier import PrototypeVerifier
from src.models.student_code_head import StudentCodeHead
from src.models.student_token_head import StudentTokenHead
from src.models.teacher_tokenizer import TeacherTokenTokenizer
from src.models.visual_encoder import VisualEncoder
from src.train.train_stage3_code import (
    compute_hard_margin_loss,
    compute_pair_logistic_loss,
    compute_uniformity_loss,
    resolve_selection_value,
    summarize_verification_scores,
)
from src.train.train_stage3_prototype import (
    RawGroupBatchSampler,
    accuracy_for_versions,
    build_group_split,
    build_transforms,
    load_state_dict_shape_safe,
    set_seed,
)
from src.train.train_stage3_prototype import load_projection_checkpoint
from src.tools.data_roots import resolve_dataset_root, resolve_experiment_root, resolve_meta_path
from src.tools.ecc_codec import build_auth_codec
from src.tools.stage3_official_preset import (
    OFFICIAL_STAGE3_EVAL_MAX_RAWS,
    OFFICIAL_STAGE3_PRESET_NAME,
    apply_stage3_official_config_to_args,
)


def soft_bits(logits: torch.Tensor, scale: float) -> torch.Tensor:
    return torch.tanh(float(scale) * logits)


def claim_verifier_dim(code_dim: int, input_mode: str) -> int:
    mode = str(input_mode)
    if mode == "bits_logits":
        return int(code_dim) * 2
    return int(code_dim)


def build_claim_verifier_repr(
    bits: torch.Tensor,
    logits: torch.Tensor | None = None,
    input_mode: str = "bits",
) -> torch.Tensor:
    mode = str(input_mode)
    if mode == "bits":
        return bits
    if logits is None:
        raise ValueError(f"claim_verifier_input_mode={mode} requires logits.")
    norm_logits = F.normalize(logits, dim=-1)
    if mode == "logits":
        return norm_logits
    if mode == "bits_logits":
        return torch.cat([bits, norm_logits], dim=-1)
    raise ValueError(f"Unsupported claim_verifier_input_mode: {input_mode}")


def score_bank(query_bits: torch.Tensor, bank_bits: torch.Tensor) -> torch.Tensor:
    return torch.matmul(query_bits, bank_bits.T) / max(bank_bits.shape[-1], 1)


def score_stage_bank(query_stage_bits: torch.Tensor, bank_stage_bits: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bkd,skd->bsk", query_stage_bits, bank_stage_bits).mean(dim=-1) / max(bank_stage_bits.shape[-1], 1)


def score_stage_bank_cosine(query_stage_logits: torch.Tensor, bank_stage_logits: torch.Tensor) -> torch.Tensor:
    query_stage_logits = F.normalize(query_stage_logits, dim=-1)
    bank_stage_logits = F.normalize(bank_stage_logits, dim=-1)
    return torch.einsum("bkd,skd->bsk", query_stage_logits, bank_stage_logits).mean(dim=-1)


def build_bank_targets(raw_anchors, bank_raws, device: torch.device) -> torch.Tensor:
    raw_to_index = {raw_anchor: idx for idx, raw_anchor in enumerate(bank_raws)}
    return torch.tensor([raw_to_index[raw_anchor] for raw_anchor in raw_anchors], device=device)


def compute_sample_weights(
    versions: torch.Tensor,
    anchor_versions,
    shift_versions,
    anchor_weight: float,
    shift_weight: float,
) -> torch.Tensor:
    weights = torch.ones_like(versions, dtype=torch.float32)
    for version in anchor_versions:
        weights[versions == int(version)] = float(anchor_weight)
    for version in shift_versions:
        weights[versions == int(version)] = float(shift_weight)
    return weights


def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights = weights.to(values.device, dtype=values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1e-6)


def maybe_expand_repetition_state_dict(
    state_dict: dict,
    module_state_dict: dict,
    repetition: int,
) -> dict:
    if repetition <= 1:
        return state_dict
    expanded = dict(state_dict)
    for key, target_tensor in module_state_dict.items():
        source_tensor = expanded.get(key)
        if source_tensor is None:
            continue
        if not isinstance(source_tensor, torch.Tensor):
            continue
        if source_tensor.shape == target_tensor.shape:
            continue
        if source_tensor.ndim != target_tensor.ndim or source_tensor.ndim == 0:
            continue
        if source_tensor.shape[1:] != target_tensor.shape[1:]:
            continue
        if source_tensor.shape[0] * repetition != target_tensor.shape[0]:
            continue
        expanded[key] = source_tensor.repeat_interleave(repetition, dim=0)
    return expanded


@torch.no_grad()
def initialize_codebook_from_teacher_samples(
    auth_codebook: AuthenticationCodebook,
    dataset: PrototypeDistillationDataset,
    indices,
    teacher_head: AuthCodeHead,
    device: torch.device,
    auth_codec=None,
    batch_size: int = 256,
):
    sample_logits = []
    entries = []

    def flush_batch():
        nonlocal entries
        if not entries:
            return
        teacher_vec = torch.stack([entry["teacher_vec"] for entry in entries], dim=0).to(device)
        teacher_seq = torch.stack([entry["teacher_seq"] for entry in entries], dim=0).to(device)
        teacher_out = teacher_head(
            teacher_vec,
            teacher_seq,
            return_sequence=False,
            return_logits=True,
        )
        teacher_logits = teacher_out["global_logits"]
        if auth_codec is not None:
            teacher_logits = auth_codec.encode_logits(teacher_logits)
        sample_logits.append(teacher_logits.detach())
        entries = []

    for idx in indices:
        record = dataset.records[idx]
        raw_anchor = record["raw_anchor"]
        version_id = record["version_id"]
        entries.append(
            {
                "teacher_vec": dataset.teacher_vec_bank.get(version_id, dataset.prototype_bank[raw_anchor]).clone().float(),
                "teacher_seq": dataset.teacher_seq_bank.get(version_id, dataset.prototype_bank[raw_anchor].unsqueeze(0)).clone().float(),
            }
        )
        if len(entries) >= int(batch_size):
            flush_batch()
    flush_batch()
    if not sample_logits:
        raise RuntimeError("Failed to collect teacher logits for codebook initialization.")
    auth_codebook.initialize_from_samples(torch.cat(sample_logits, dim=0))


@torch.no_grad()
def collect_teacher_tokenizer_samples(
    dataset: PrototypeDistillationDataset,
    indices,
    teacher_head: AuthCodeHead,
    device: torch.device,
    auth_codec=None,
    batch_size: int = 256,
):
    sample_logits = []
    sample_raws = []
    sample_ids = []
    entries = []

    def flush_batch():
        nonlocal entries
        if not entries:
            return
        teacher_vec = torch.stack([entry["teacher_vec"] for entry in entries], dim=0).to(device)
        teacher_seq = torch.stack([entry["teacher_seq"] for entry in entries], dim=0).to(device)
        teacher_out = teacher_head(
            teacher_vec,
            teacher_seq,
            return_sequence=False,
            return_logits=True,
        )
        teacher_logits = teacher_out["global_logits"]
        if auth_codec is not None:
            teacher_logits = auth_codec.encode_logits(teacher_logits)
        sample_logits.append(teacher_logits.detach().cpu())
        entries = []

    for idx in indices:
        record = dataset.records[idx]
        raw_anchor = record["raw_anchor"]
        version_id = record["version_id"]
        entries.append(
            {
                "teacher_vec": dataset.teacher_vec_bank.get(version_id, dataset.prototype_bank[raw_anchor]).clone().float(),
                "teacher_seq": dataset.teacher_seq_bank.get(version_id, dataset.prototype_bank[raw_anchor].unsqueeze(0)).clone().float(),
            }
        )
        sample_raws.append(raw_anchor)
        sample_ids.append(record.get("sample_id", raw_anchor))
        if len(entries) >= int(batch_size):
            flush_batch()
    flush_batch()
    if not sample_logits:
        raise RuntimeError("Failed to collect teacher logits for tokenizer initialization.")
    return {
        "logits": torch.cat(sample_logits, dim=0),
        "raw_anchors": sample_raws,
        "sample_ids": sample_ids,
    }


def compute_token_match_matrix(
    query_token_logits: torch.Tensor | None,
    reference_token_indices: torch.Tensor | None,
) -> torch.Tensor | None:
    if query_token_logits is None or reference_token_indices is None:
        return None
    if query_token_logits.ndim != 2:
        raise ValueError("query_token_logits must have shape [B, K].")
    query_token_probs = torch.softmax(query_token_logits, dim=-1)
    return query_token_probs[:, reference_token_indices.reshape(-1)]


def compute_token_prototype_alignment_loss(
    query_token_logits: torch.Tensor | None,
    tokenizer: TeacherTokenTokenizer | None,
    target_indices: torch.Tensor | None,
    sample_weights: torch.Tensor,
) -> torch.Tensor:
    if query_token_logits is None or tokenizer is None or target_indices is None:
        return sample_weights.new_tensor(0.0)
    query_token_probs = torch.softmax(query_token_logits, dim=-1)
    pred_prototypes = query_token_probs @ tokenizer.normalized_prototypes()
    target_prototypes = tokenizer.lookup(target_indices)
    cosine = F.cosine_similarity(
        F.normalize(pred_prototypes, dim=-1),
        F.normalize(target_prototypes.detach(), dim=-1),
        dim=-1,
    )
    return weighted_mean(1.0 - cosine, sample_weights)


def compute_deterministic_gated_scores(
    main_scores: torch.Tensor,
    token_match_scores: torch.Tensor | None,
    residual_scores: torch.Tensor | None = None,
    gate_penalty: float = 1.0,
    residual_weight: float = 0.25,
) -> torch.Tensor:
    if token_match_scores is None:
        return combine_score_matrices(main_scores, residual_scores, alpha=residual_weight)
    gated_scores = main_scores - float(gate_penalty) * (1.0 - token_match_scores)
    if residual_scores is not None and float(residual_weight) != 0.0:
        gated_scores = gated_scores + float(residual_weight) * residual_scores * token_match_scores
    return gated_scores


def compute_hard_token_gated_scores(
    main_scores: torch.Tensor,
    token_match_scores: torch.Tensor | None,
    residual_scores: torch.Tensor | None = None,
    gate_penalty: float = 1.0,
    residual_weight: float = 0.25,
    threshold: float = 0.5,
) -> torch.Tensor:
    if token_match_scores is None:
        return compute_deterministic_gated_scores(
            main_scores,
            token_match_scores=None,
            residual_scores=residual_scores,
            gate_penalty=gate_penalty,
            residual_weight=residual_weight,
        )
    gate_mask = (token_match_scores >= float(threshold)).to(main_scores.dtype)
    hard_scores = main_scores - float(gate_penalty) * (1.0 - gate_mask)
    if residual_scores is not None and float(residual_weight) != 0.0:
        hard_scores = hard_scores + float(residual_weight) * residual_scores * gate_mask
    return hard_scores


def build_claim_score_outputs(
    main_scores: torch.Tensor,
    protocol_scores: torch.Tensor | None = None,
    token_scores: torch.Tensor | None = None,
    extra_protocol_scores: list[torch.Tensor] | None = None,
    claim_score_fusion_head: ClaimScoreFusionHead = None,
    official_mode: str = "deterministic_gate",
    alpha: float = 0.0,
    main_normalization: str = "none",
    auxiliary_normalization: str = "none",
    gate_penalty: float = 1.0,
    residual_weight: float = 0.25,
    hard_gate_threshold: float = 0.5,
):
    fusion_scores = fuse_claim_score_matrices(
        main_scores,
        protocol_scores,
        extra_protocol_scores=extra_protocol_scores,
        claim_score_fusion_head=claim_score_fusion_head,
        alpha=alpha,
        main_normalization=main_normalization,
        auxiliary_normalization=auxiliary_normalization,
        token_scores=token_scores,
    )
    gated_scores = compute_deterministic_gated_scores(
        main_scores,
        token_scores,
        residual_scores=protocol_scores,
        gate_penalty=gate_penalty,
        residual_weight=residual_weight,
    )
    hard_gated_scores = compute_hard_token_gated_scores(
        main_scores,
        token_scores,
        residual_scores=protocol_scores,
        gate_penalty=gate_penalty,
        residual_weight=residual_weight,
        threshold=hard_gate_threshold,
    )
    official_mode = str(official_mode or "deterministic_gate")
    if official_mode == "fusion_head":
        official_scores = fusion_scores
    elif official_mode == "deterministic_gate":
        official_scores = gated_scores
    else:
        raise ValueError(f"Unsupported official claim score mode: {official_mode}")
    return {
        "official_scores": official_scores,
        "fusion_scores": fusion_scores,
        "gated_scores": gated_scores,
        "hard_gated_scores": hard_gated_scores,
    }


def select_teacher_tokenizer(
    sample_logits: torch.Tensor,
    raw_anchors,
    sample_ids,
    candidate_classes,
    device: torch.device,
    temperature: float = 12.0,
):
    candidate_classes = [int(value) for value in candidate_classes if int(value) > 1]
    if not candidate_classes:
        raise ValueError("candidate_classes must contain at least one token count greater than 1.")

    best = None
    for num_tokens in candidate_classes:
        tokenizer = TeacherTokenTokenizer(
            code_dim=sample_logits.shape[-1],
            num_tokens=num_tokens,
            temperature=temperature,
        ).to(device)
        tokenizer.initialize_from_samples(sample_logits.to(device))
        token_indices, token_logits = tokenizer.assign(sample_logits.to(device))
        token_scores = compute_token_match_matrix(token_logits, token_indices)
        positive_mask = build_same_sample_mask(sample_ids, device=device)
        negative_mask = ~build_same_raw_mask(raw_anchors, device=device)
        pos, hard_neg, neg = summarize_masked_scores(token_scores, positive_mask, negative_mask=negative_mask)
        metrics = summarize_verification_scores(pos.detach().cpu(), hard_neg.detach().cpu(), neg.detach().cpu())
        candidate = {
            "num_tokens": num_tokens,
            "tokenizer": tokenizer,
            "metrics": metrics,
        }
        if best is None:
            best = candidate
            continue
        current = (
            float(candidate["metrics"]["eer"]),
            -float(candidate["metrics"]["pairwise_auc"]),
            int(candidate["num_tokens"]),
        )
        previous = (
            float(best["metrics"]["eer"]),
            -float(best["metrics"]["pairwise_auc"]),
            int(best["num_tokens"]),
        )
        if current < previous:
            best = candidate
    return best


def compute_bit_match_loss(pred_logits: torch.Tensor, target_logits: torch.Tensor, sample_weights: torch.Tensor) -> torch.Tensor:
    target_bits = torch.sign(target_logits.detach())
    target_bits[target_bits == 0] = 1.0
    sample_loss = F.softplus(-target_bits * pred_logits).mean(dim=1)
    return weighted_mean(sample_loss, sample_weights)


def compute_stage_bit_match_loss(pred_stage_logits: torch.Tensor, target_stage_logits: torch.Tensor, sample_weights: torch.Tensor) -> torch.Tensor:
    if pred_stage_logits is None or target_stage_logits is None:
        return sample_weights.new_tensor(0.0)
    target_bits = torch.sign(target_stage_logits.detach())
    target_bits[target_bits == 0] = 1.0
    sample_loss = F.softplus(-target_bits * pred_stage_logits).mean(dim=(1, 2))
    return weighted_mean(sample_loss, sample_weights)


def compute_soft_alignment_loss(pred_bits: torch.Tensor, target_bits: torch.Tensor, sample_weights: torch.Tensor) -> torch.Tensor:
    cosine = F.cosine_similarity(F.normalize(pred_bits, dim=-1), F.normalize(target_bits, dim=-1), dim=-1)
    return weighted_mean(1.0 - cosine, sample_weights)


def compute_logit_recovery_loss(pred_logits: torch.Tensor, target_logits: torch.Tensor, sample_weights: torch.Tensor) -> torch.Tensor:
    if pred_logits is None or target_logits is None:
        return sample_weights.new_tensor(0.0)
    sample_loss = F.smooth_l1_loss(pred_logits, target_logits.detach(), reduction="none").mean(dim=1)
    return weighted_mean(sample_loss, sample_weights)


def compute_stage_logit_recovery_loss(
    pred_stage_logits: torch.Tensor,
    target_stage_logits: torch.Tensor,
    sample_weights: torch.Tensor,
) -> torch.Tensor:
    if pred_stage_logits is None or target_stage_logits is None:
        return sample_weights.new_tensor(0.0)
    sample_loss = F.smooth_l1_loss(pred_stage_logits, target_stage_logits.detach(), reduction="none").mean(dim=(1, 2))
    return weighted_mean(sample_loss, sample_weights)


def compute_claim_scores(
    query_bits: torch.Tensor,
    reference_bits: torch.Tensor,
    claim_verifier: PrototypeVerifier = None,
    verifier_weight: float = 1.0,
    verifier_score_mode: str = "add",
    query_verifier_repr: torch.Tensor = None,
    reference_verifier_repr: torch.Tensor = None,
    query_stage_logits: torch.Tensor = None,
    reference_stage_logits: torch.Tensor = None,
    reference_stage_bits: torch.Tensor = None,
    bit_scale: float = 3.0,
    sequence_score_weight: float = 0.0,
) -> torch.Tensor:
    return compute_claim_score_components(
        query_bits=query_bits,
        reference_bits=reference_bits,
        claim_verifier=claim_verifier,
        verifier_weight=verifier_weight,
        verifier_score_mode=verifier_score_mode,
        query_verifier_repr=query_verifier_repr,
        reference_verifier_repr=reference_verifier_repr,
        query_stage_logits=query_stage_logits,
        reference_stage_logits=reference_stage_logits,
        reference_stage_bits=reference_stage_bits,
        bit_scale=bit_scale,
        sequence_score_weight=sequence_score_weight,
    )["claim_scores"]


def compute_claim_verifier_scores(
    query_bits: torch.Tensor,
    reference_bits: torch.Tensor,
    claim_verifier: PrototypeVerifier = None,
    verifier_weight: float = 1.0,
    verifier_score_mode: str = "add",
    query_verifier_repr: torch.Tensor = None,
    reference_verifier_repr: torch.Tensor = None,
) -> torch.Tensor | None:
    if claim_verifier is None or float(verifier_weight) == 0.0:
        return None
    verifier_query = query_verifier_repr if query_verifier_repr is not None else query_bits
    verifier_reference = reference_verifier_repr if reference_verifier_repr is not None else reference_bits
    verifier_scores = claim_verifier.score_bank(verifier_query, verifier_reference)
    verifier_score_mode = str(verifier_score_mode or "add")
    if verifier_score_mode in {"tanh_add", "feature_only"}:
        verifier_scores = torch.tanh(verifier_scores)
    elif verifier_score_mode != "add":
        raise ValueError(f"Unsupported claim verifier score mode: {verifier_score_mode}")
    return float(verifier_weight) * verifier_scores


def compute_claim_score_components(
    query_bits: torch.Tensor,
    reference_bits: torch.Tensor,
    claim_verifier: PrototypeVerifier = None,
    verifier_weight: float = 1.0,
    verifier_score_mode: str = "add",
    query_verifier_repr: torch.Tensor = None,
    reference_verifier_repr: torch.Tensor = None,
    query_stage_logits: torch.Tensor = None,
    reference_stage_logits: torch.Tensor = None,
    reference_stage_bits: torch.Tensor = None,
    bit_scale: float = 3.0,
    sequence_score_weight: float = 0.0,
) -> dict[str, torch.Tensor | None]:
    similarity_scores = score_bank(query_bits, reference_bits)
    verifier_scores = compute_claim_verifier_scores(
        query_bits=query_bits,
        reference_bits=reference_bits,
        claim_verifier=claim_verifier,
        verifier_weight=verifier_weight,
        verifier_score_mode=verifier_score_mode,
        query_verifier_repr=query_verifier_repr,
        reference_verifier_repr=reference_verifier_repr,
    )
    claim_scores = similarity_scores
    if verifier_scores is not None and str(verifier_score_mode or "add") in {"add", "tanh_add"}:
        claim_scores = claim_scores + verifier_scores
    if query_stage_logits is not None and float(sequence_score_weight) > 0.0:
        seq_scores = None
        if reference_stage_logits is not None:
            seq_scores = score_stage_bank_cosine(query_stage_logits, reference_stage_logits)
        elif reference_stage_bits is not None:
            seq_scores = score_stage_bank(soft_bits(query_stage_logits, bit_scale), reference_stage_bits)
        if seq_scores is None:
            return {
                "claim_scores": claim_scores,
                "verifier_scores": verifier_scores,
                "similarity_scores": similarity_scores,
            }
        seq_w = min(max(float(sequence_score_weight), 0.0), 1.0)
        claim_scores = (1.0 - seq_w) * claim_scores + seq_w * seq_scores
    return {
        "claim_scores": claim_scores,
        "verifier_scores": verifier_scores,
        "similarity_scores": similarity_scores,
    }


def summarize_target_scores(logits: torch.Tensor, targets: torch.Tensor):
    pos_scores = logits.gather(1, targets.unsqueeze(1)).squeeze(1)
    negative_mask = torch.ones_like(logits, dtype=torch.bool)
    negative_mask.scatter_(1, targets.unsqueeze(1), False)
    neg_scores = logits[negative_mask].view(logits.shape[0], -1)
    if neg_scores.shape[1] > 0:
        hard_neg_scores = neg_scores.max(dim=1).values
    else:
        hard_neg_scores = logits.new_zeros((logits.shape[0],))
    return pos_scores, hard_neg_scores, neg_scores.reshape(-1)


def compute_optional_claim_pair_hard_losses(
    score_matrix: torch.Tensor | None,
    claim_reference_mode: str,
    sample_weights: torch.Tensor,
    positive_margin: float,
    negative_margin: float,
    scale: float,
    topk: int,
    margin: float,
    claim_targets: torch.Tensor | None = None,
    claim_positive_mask: torch.Tensor | None = None,
    claim_negative_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if score_matrix is None:
        zero = sample_weights.new_tensor(0.0)
        return zero, zero
    claim_reference_mode = str(claim_reference_mode)
    if claim_reference_mode == "anchor_bank":
        return (
            compute_pair_logistic_loss(
                score_matrix,
                claim_targets,
                sample_weights,
                positive_margin=positive_margin,
                negative_margin=negative_margin,
                scale=scale,
                topk=topk,
            ),
            compute_hard_margin_loss(
                score_matrix,
                claim_targets,
                sample_weights,
                margin=margin,
            ),
        )
    if claim_reference_mode == "same_image":
        return (
            compute_masked_pair_logistic_loss(
                score_matrix,
                claim_positive_mask,
                sample_weights,
                negative_mask=claim_negative_mask,
                positive_margin=positive_margin,
                negative_margin=negative_margin,
                scale=scale,
                topk=topk,
            ),
            compute_masked_hard_margin_loss(
                score_matrix,
                claim_positive_mask,
                sample_weights,
                negative_mask=claim_negative_mask,
                margin=margin,
            ),
        )
    return (
        compute_masked_pair_logistic_loss(
            score_matrix,
            claim_positive_mask,
            sample_weights,
            positive_margin=positive_margin,
            negative_margin=negative_margin,
            scale=scale,
            topk=topk,
        ),
        compute_masked_hard_margin_loss(
            score_matrix,
            claim_positive_mask,
            sample_weights,
            margin=margin,
        ),
    )


def aggregate_claim_reference(
    bit_list,
    logit_list,
    bit_scale: float,
    claim_bank_mode: str = "mean_bits",
) -> torch.Tensor:
    if not bit_list:
        raise RuntimeError("aggregate_claim_reference requires at least one reference sample.")

    claim_bank_mode = str(claim_bank_mode)
    if claim_bank_mode == "mean_bits":
        return torch.stack(bit_list, dim=0).mean(dim=0)

    if not logit_list:
        raise RuntimeError(f"claim_bank_mode={claim_bank_mode} requires teacher logits.")

    mean_logits = torch.stack(logit_list, dim=0).mean(dim=0)
    if claim_bank_mode == "mean_logits_tanh":
        return soft_bits(mean_logits, bit_scale)
    if claim_bank_mode == "sign_mean_logits":
        signed = torch.sign(mean_logits)
        signed[signed == 0] = 1.0
        return signed
    raise ValueError(f"Unsupported claim_bank_mode: {claim_bank_mode}")


def build_same_raw_mask(raw_anchors, device: torch.device) -> torch.Tensor:
    raw_list = list(raw_anchors)
    count = len(raw_list)
    mask = torch.zeros((count, count), dtype=torch.bool, device=device)
    for i, raw_anchor in enumerate(raw_list):
        for j, candidate in enumerate(raw_list):
            if candidate == raw_anchor:
                mask[i, j] = True
    return mask


def build_same_sample_mask(sample_ids, device: torch.device) -> torch.Tensor:
    sample_list = list(sample_ids)
    count = len(sample_list)
    mask = torch.zeros((count, count), dtype=torch.bool, device=device)
    for i, sample_id in enumerate(sample_list):
        for j, candidate in enumerate(sample_list):
            if candidate == sample_id:
                mask[i, j] = True
    return mask


def resolve_student_match_targets(
    canonical_logits: torch.Tensor,
    sample_logits: torch.Tensor,
    canonical_stage_logits: torch.Tensor = None,
    sample_stage_logits: torch.Tensor = None,
    bit_scale: float = 1.0,
    target_mode: str = "canonical",
    blend_alpha: float = 0.5,
):
    target_mode = str(target_mode)
    alpha = min(max(float(blend_alpha), 0.0), 1.0)

    if target_mode == "canonical":
        target_logits = canonical_logits.detach() if canonical_logits is not None else None
        target_stage_logits = canonical_stage_logits.detach() if canonical_stage_logits is not None else None
    elif target_mode == "sample":
        target_logits = sample_logits.detach() if sample_logits is not None else None
        target_stage_logits = sample_stage_logits.detach() if sample_stage_logits is not None else None
    elif target_mode == "blend":
        if canonical_logits is None or sample_logits is None:
            raise ValueError("blend target_mode requires both canonical and sample logits.")
        target_logits = torch.lerp(canonical_logits.detach(), sample_logits.detach(), alpha)
        if canonical_stage_logits is None and sample_stage_logits is None:
            target_stage_logits = None
        elif canonical_stage_logits is None:
            target_stage_logits = sample_stage_logits.detach()
        elif sample_stage_logits is None:
            target_stage_logits = canonical_stage_logits.detach()
        else:
            target_stage_logits = torch.lerp(canonical_stage_logits.detach(), sample_stage_logits.detach(), alpha)
    else:
        raise ValueError(f"Unsupported student target mode: {target_mode}")

    target_bits = soft_bits(target_logits, bit_scale) if target_logits is not None else None
    return target_logits, target_stage_logits, target_bits


def compute_masked_pair_logistic_loss(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
    sample_weights: torch.Tensor,
    negative_mask: torch.Tensor = None,
    positive_margin: float = 0.60,
    negative_margin: float = 0.10,
    scale: float = 12.0,
    topk: int = 8,
) -> torch.Tensor:
    pos_scores = []
    neg_losses = []
    for idx in range(logits.shape[0]):
        pos = logits[idx][positive_mask[idx]]
        neg = logits[idx][negative_mask[idx] if negative_mask is not None else ~positive_mask[idx]]
        if pos.numel() == 0:
            pos = logits[idx, idx].unsqueeze(0)
        pos_scores.append(pos.max())
        if neg.numel() == 0:
            neg_losses.append(logits.new_tensor(0.0))
            continue
        if topk is not None and topk > 0:
            neg = torch.topk(neg, k=min(int(topk), neg.numel())).values
        neg_losses.append(F.softplus(float(scale) * (neg - float(negative_margin))).mean())
    pos_scores = torch.stack(pos_scores)
    neg_losses = torch.stack(neg_losses)
    pos_loss = F.softplus(-float(scale) * (pos_scores - float(positive_margin)))
    sample_losses = pos_loss + neg_losses
    return weighted_mean(sample_losses, sample_weights)


def compute_masked_hard_margin_loss(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
    sample_weights: torch.Tensor,
    negative_mask: torch.Tensor = None,
    margin: float = 0.10,
) -> torch.Tensor:
    pos_scores = []
    hard_neg_scores = []
    for idx in range(logits.shape[0]):
        pos = logits[idx][positive_mask[idx]]
        neg = logits[idx][negative_mask[idx] if negative_mask is not None else ~positive_mask[idx]]
        if pos.numel() == 0:
            pos = logits[idx, idx].unsqueeze(0)
        pos_scores.append(pos.max())
        hard_neg_scores.append(neg.max() if neg.numel() else logits.new_tensor(0.0))
    pos_scores = torch.stack(pos_scores)
    hard_neg_scores = torch.stack(hard_neg_scores)
    return weighted_mean(F.relu(float(margin) + hard_neg_scores - pos_scores), sample_weights)


def compute_pair_bce_calibration_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    sample_weights: torch.Tensor,
    topk: int = 8,
) -> torch.Tensor:
    pos_scores = logits.gather(1, targets.unsqueeze(1)).squeeze(1)
    negative_mask = torch.ones_like(logits, dtype=torch.bool)
    negative_mask.scatter_(1, targets.unsqueeze(1), False)
    neg_losses = []
    for idx in range(logits.shape[0]):
        neg = logits[idx][negative_mask[idx]]
        if neg.numel() == 0:
            neg_losses.append(logits.new_tensor(0.0))
            continue
        if topk is not None and topk > 0:
            neg = torch.topk(neg, k=min(int(topk), neg.numel())).values
        neg_losses.append(F.softplus(neg).mean())
    neg_losses = torch.stack(neg_losses)
    pos_loss = F.softplus(-pos_scores)
    return weighted_mean(pos_loss + neg_losses, sample_weights)


def compute_masked_pair_bce_calibration_loss(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
    sample_weights: torch.Tensor,
    negative_mask: torch.Tensor = None,
    topk: int = 8,
) -> torch.Tensor:
    pos_losses = []
    neg_losses = []
    for idx in range(logits.shape[0]):
        pos = logits[idx][positive_mask[idx]]
        neg = logits[idx][negative_mask[idx] if negative_mask is not None else ~positive_mask[idx]]
        if pos.numel() == 0:
            pos = logits[idx, idx].unsqueeze(0)
        pos_losses.append(F.softplus(-pos.max()))
        if neg.numel() == 0:
            neg_losses.append(logits.new_tensor(0.0))
            continue
        if topk is not None and topk > 0:
            neg = torch.topk(neg, k=min(int(topk), neg.numel())).values
        neg_losses.append(F.softplus(neg).mean())
    return weighted_mean(torch.stack(pos_losses) + torch.stack(neg_losses), sample_weights)


def compute_hard_pair_bce_calibration_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    sample_weights: torch.Tensor,
) -> torch.Tensor:
    pos_scores = logits.gather(1, targets.unsqueeze(1)).squeeze(1)
    negative_mask = torch.ones_like(logits, dtype=torch.bool)
    negative_mask.scatter_(1, targets.unsqueeze(1), False)
    hard_neg_scores = []
    for idx in range(logits.shape[0]):
        neg = logits[idx][negative_mask[idx]]
        hard_neg_scores.append(neg.max() if neg.numel() else logits.new_tensor(0.0))
    hard_neg_scores = torch.stack(hard_neg_scores)
    return weighted_mean(F.softplus(hard_neg_scores - pos_scores), sample_weights)


def compute_masked_hard_pair_bce_calibration_loss(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
    sample_weights: torch.Tensor,
    negative_mask: torch.Tensor = None,
) -> torch.Tensor:
    pos_scores = []
    hard_neg_scores = []
    for idx in range(logits.shape[0]):
        pos = logits[idx][positive_mask[idx]]
        neg = logits[idx][negative_mask[idx] if negative_mask is not None else ~positive_mask[idx]]
        if pos.numel() == 0:
            pos = logits[idx, idx].unsqueeze(0)
        pos_scores.append(pos.max())
        hard_neg_scores.append(neg.max() if neg.numel() else logits.new_tensor(0.0))
    pos_scores = torch.stack(pos_scores)
    hard_neg_scores = torch.stack(hard_neg_scores)
    return weighted_mean(F.softplus(hard_neg_scores - pos_scores), sample_weights)


def compute_optional_claim_calibration_losses(
    score_matrix: torch.Tensor | None,
    claim_reference_mode: str,
    sample_weights: torch.Tensor,
    topk: int,
    claim_targets: torch.Tensor | None = None,
    claim_positive_mask: torch.Tensor | None = None,
    claim_negative_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if score_matrix is None:
        zero = sample_weights.new_tensor(0.0)
        return zero, zero
    claim_reference_mode = str(claim_reference_mode)
    if claim_reference_mode == "anchor_bank":
        return (
            compute_pair_bce_calibration_loss(
                score_matrix,
                claim_targets,
                sample_weights,
                topk=topk,
            ),
            compute_hard_pair_bce_calibration_loss(
                score_matrix,
                claim_targets,
                sample_weights,
            ),
        )
    return (
        compute_masked_pair_bce_calibration_loss(
            score_matrix,
            claim_positive_mask,
            sample_weights,
            negative_mask=claim_negative_mask if claim_reference_mode == "same_image" else None,
            topk=topk,
        ),
        compute_masked_hard_pair_bce_calibration_loss(
            score_matrix,
            claim_positive_mask,
            sample_weights,
            negative_mask=claim_negative_mask if claim_reference_mode == "same_image" else None,
        ),
    )


def estimate_balanced_batch_threshold(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
    positive_quantile: float = 0.10,
    negative_quantile: float = 0.90,
) -> torch.Tensor:
    if pos_scores.numel() == 0 or neg_scores.numel() == 0:
        return pos_scores.new_tensor(0.0)
    positive_quantile = min(max(float(positive_quantile), 0.0), 1.0)
    negative_quantile = min(max(float(negative_quantile), 0.0), 1.0)
    pos_detached = pos_scores.detach().float()
    neg_detached = neg_scores.detach().float()
    pos_anchor = torch.quantile(pos_detached, positive_quantile)
    neg_anchor = torch.quantile(neg_detached, negative_quantile)
    return (0.5 * (pos_anchor + neg_anchor)).to(pos_scores.dtype)


def compute_operating_point_proxy_loss(
    pos_scores: torch.Tensor,
    hard_neg_scores: torch.Tensor,
    all_neg_scores: torch.Tensor,
    sample_weights: torch.Tensor,
    positive_quantile: float = 0.10,
    negative_quantile: float = 0.90,
    margin: float = 0.02,
    scale: float = 12.0,
) -> torch.Tensor:
    if pos_scores.numel() == 0 or hard_neg_scores.numel() == 0 or all_neg_scores.numel() == 0:
        return sample_weights.new_tensor(0.0)
    threshold = estimate_balanced_batch_threshold(
        pos_scores,
        all_neg_scores,
        positive_quantile=positive_quantile,
        negative_quantile=negative_quantile,
    )
    pos_loss = F.softplus(float(scale) * (threshold + float(margin) - pos_scores))
    neg_loss = F.softplus(float(scale) * (hard_neg_scores - (threshold - float(margin))))
    return weighted_mean(pos_loss + neg_loss, sample_weights)


def compute_positive_tail_rescue_loss(
    pos_scores: torch.Tensor,
    all_neg_scores: torch.Tensor,
    sample_weights: torch.Tensor,
    positive_quantile: float = 0.10,
    negative_quantile: float = 0.90,
    margin: float = 0.01,
    scale: float = 12.0,
) -> torch.Tensor:
    if pos_scores.numel() == 0 or all_neg_scores.numel() == 0:
        return sample_weights.new_tensor(0.0)
    threshold = estimate_balanced_batch_threshold(
        pos_scores,
        all_neg_scores,
        positive_quantile=positive_quantile,
        negative_quantile=negative_quantile,
    )
    rescue_gap = threshold + float(margin) - pos_scores
    tail_focus = torch.sigmoid(float(scale) * rescue_gap)
    rescue_loss = F.softplus(float(scale) * rescue_gap)
    return weighted_mean(tail_focus * rescue_loss, sample_weights)


def compute_optional_claim_operating_point_loss(
    score_matrix: torch.Tensor | None,
    claim_reference_mode: str,
    sample_weights: torch.Tensor,
    positive_quantile: float = 0.10,
    negative_quantile: float = 0.90,
    margin: float = 0.02,
    scale: float = 12.0,
    claim_targets: torch.Tensor | None = None,
    claim_positive_mask: torch.Tensor | None = None,
    claim_negative_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if score_matrix is None:
        return sample_weights.new_tensor(0.0)
    claim_reference_mode = str(claim_reference_mode)
    if claim_reference_mode == "anchor_bank":
        pos_scores, hard_neg_scores, all_neg_scores = summarize_target_scores(score_matrix, claim_targets)
    else:
        pos_scores, hard_neg_scores, all_neg_scores = summarize_masked_scores(
            score_matrix,
            claim_positive_mask,
            negative_mask=claim_negative_mask if claim_reference_mode == "same_image" else None,
        )
    return compute_operating_point_proxy_loss(
        pos_scores,
        hard_neg_scores,
        all_neg_scores,
        sample_weights=sample_weights,
        positive_quantile=positive_quantile,
        negative_quantile=negative_quantile,
        margin=margin,
        scale=scale,
    )


def compute_optional_claim_positive_tail_rescue_loss(
    score_matrix: torch.Tensor | None,
    claim_reference_mode: str,
    sample_weights: torch.Tensor,
    positive_quantile: float = 0.10,
    negative_quantile: float = 0.90,
    margin: float = 0.01,
    scale: float = 12.0,
    claim_targets: torch.Tensor | None = None,
    claim_positive_mask: torch.Tensor | None = None,
    claim_negative_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if score_matrix is None:
        return sample_weights.new_tensor(0.0)
    claim_reference_mode = str(claim_reference_mode)
    if claim_reference_mode == "anchor_bank":
        pos_scores, _, all_neg_scores = summarize_target_scores(score_matrix, claim_targets)
    else:
        pos_scores, _, all_neg_scores = summarize_masked_scores(
            score_matrix,
            claim_positive_mask,
            negative_mask=claim_negative_mask if claim_reference_mode == "same_image" else None,
        )
    return compute_positive_tail_rescue_loss(
        pos_scores,
        all_neg_scores,
        sample_weights=sample_weights,
        positive_quantile=positive_quantile,
        negative_quantile=negative_quantile,
        margin=margin,
        scale=scale,
    )


def compute_optional_claim_hardcase_weights(
    score_matrix: torch.Tensor | None,
    claim_reference_mode: str,
    sample_weights: torch.Tensor,
    strength: float = 0.0,
    margin: float = 0.05,
    scale: float = 12.0,
    claim_targets: torch.Tensor | None = None,
    claim_positive_mask: torch.Tensor | None = None,
    claim_negative_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if score_matrix is None or float(strength) <= 0.0:
        return sample_weights
    claim_reference_mode = str(claim_reference_mode)
    if claim_reference_mode == "anchor_bank":
        pos_scores, hard_neg_scores, _ = summarize_target_scores(score_matrix, claim_targets)
    else:
        pos_scores, hard_neg_scores, _ = summarize_masked_scores(
            score_matrix,
            claim_positive_mask,
            negative_mask=claim_negative_mask if claim_reference_mode == "same_image" else None,
        )
    gap = pos_scores - hard_neg_scores
    hardness = torch.sigmoid(float(scale) * (float(margin) - gap))
    return sample_weights * (1.0 + float(strength) * hardness.to(sample_weights.dtype))


def summarize_masked_scores(logits: torch.Tensor, positive_mask: torch.Tensor, negative_mask: torch.Tensor = None):
    pos_scores = []
    hard_neg_scores = []
    all_neg_scores = []
    for idx in range(logits.shape[0]):
        pos = logits[idx][positive_mask[idx]]
        neg = logits[idx][negative_mask[idx] if negative_mask is not None else ~positive_mask[idx]]
        if pos.numel() == 0:
            pos = logits[idx, idx].unsqueeze(0)
        pos_scores.append(pos.max())
        if neg.numel():
            all_neg_scores.append(neg)
            hard_neg_scores.append(neg.max())
        else:
            hard_neg_scores.append(logits.new_tensor(0.0))
    pos_scores = torch.stack(pos_scores)
    hard_neg_scores = torch.stack(hard_neg_scores)
    if all_neg_scores:
        all_neg_scores = torch.cat(all_neg_scores, dim=0)
    else:
        all_neg_scores = logits.new_tensor([])
    return pos_scores, hard_neg_scores, all_neg_scores


def sweep_same_raw_claim_calibration(
    batch_logits_records,
    claim_verifier: PrototypeVerifier,
    device: torch.device,
    bit_scales,
    verifier_weights,
    claim_sequence_score_weight: float = 0.0,
    verifier_score_mode: str = "add",
    sequence_weights=None,
):
    if not batch_logits_records:
        return None

    if bit_scales is None or len(bit_scales) == 0:
        raise ValueError("bit_scales must contain at least one candidate.")
    bit_scales = [float(scale) for scale in bit_scales]
    if claim_verifier is None:
        verifier_weights = [0.0]
    else:
        if verifier_weights is None or len(verifier_weights) == 0:
            raise ValueError("verifier_weights must contain at least one candidate when claim_verifier is enabled.")
        verifier_weights = [float(weight) for weight in verifier_weights]
    if sequence_weights is None or len(sequence_weights) == 0:
        sequence_weights = [float(claim_sequence_score_weight)]
    else:
        sequence_weights = [float(weight) for weight in sequence_weights]

    best = None
    with torch.no_grad():
        for bit_scale in bit_scales:
            for verifier_weight in verifier_weights:
                for sequence_weight in sequence_weights:
                    pos_scores = []
                    hard_neg_scores = []
                    all_neg_scores = []
                    for record in batch_logits_records:
                        student_logits = record["student_logits"].to(device)
                        teacher_logits = record["teacher_logits"].to(device)
                        student_stage_logits = record.get("student_stage_logits")
                        teacher_stage_logits = record.get("teacher_stage_logits")
                        if student_stage_logits is not None:
                            student_stage_logits = student_stage_logits.to(device)
                        if teacher_stage_logits is not None:
                            teacher_stage_logits = teacher_stage_logits.to(device)
                        student_bits = soft_bits(student_logits, bit_scale)
                        teacher_bits = soft_bits(teacher_logits, bit_scale)
                        teacher_stage_bits = soft_bits(teacher_stage_logits, bit_scale) if teacher_stage_logits is not None else None
                        claim_scores = compute_claim_scores(
                            student_bits,
                            teacher_bits,
                            claim_verifier,
                            verifier_weight=verifier_weight,
                            verifier_score_mode=verifier_score_mode,
                            query_stage_logits=student_stage_logits,
                            reference_stage_logits=teacher_stage_logits,
                            reference_stage_bits=teacher_stage_bits,
                            bit_scale=bit_scale,
                            sequence_score_weight=sequence_weight,
                        )
                        positive_mask = build_same_raw_mask(record["raw_anchors"], device=device)
                        pos, hard_neg, neg = summarize_masked_scores(claim_scores, positive_mask)
                        pos_scores.append(pos.detach().cpu())
                        hard_neg_scores.append(hard_neg.detach().cpu())
                        all_neg_scores.append(neg.detach().cpu())
                    metrics = summarize_verification_scores(
                        torch.cat(pos_scores, dim=0),
                        torch.cat(hard_neg_scores, dim=0),
                        torch.cat(all_neg_scores, dim=0),
                    )
                    candidate = {
                        "bit_scale": bit_scale,
                        "verifier_weight": verifier_weight,
                        "sequence_weight": sequence_weight,
                        "metrics": metrics,
                    }
                    if best is None or candidate["metrics"]["tar_at_far_1e2"] > best["metrics"]["tar_at_far_1e2"]:
                        best = candidate
    return best


def compute_score_distill_loss(
    student_scores: torch.Tensor,
    teacher_scores: torch.Tensor,
    targets: torch.Tensor,
    sample_weights: torch.Tensor,
    temperature: float = 1.0,
    topk: int = 0,
) -> torch.Tensor:
    if student_scores.shape != teacher_scores.shape:
        raise ValueError("student_scores and teacher_scores must have the same shape.")

    temperature = max(float(temperature), 1e-6)
    if int(topk) > 0 and student_scores.shape[1] > int(topk) + 1:
        topk = int(topk)
        negative_mask = torch.ones_like(teacher_scores, dtype=torch.bool)
        negative_mask.scatter_(1, targets.unsqueeze(1), False)
        teacher_neg = teacher_scores.masked_fill(~negative_mask, float("-inf"))
        topk_neg_idx = torch.topk(teacher_neg, k=topk, dim=1).indices
        gather_index = torch.cat([targets.unsqueeze(1), topk_neg_idx], dim=1)
        student_scores = student_scores.gather(1, gather_index)
        teacher_scores = teacher_scores.gather(1, gather_index)

    teacher_prob = torch.softmax(teacher_scores.detach() / temperature, dim=1)
    student_log_prob = torch.log_softmax(student_scores / temperature, dim=1)
    sample_loss = F.kl_div(student_log_prob, teacher_prob, reduction="none").sum(dim=1)
    return weighted_mean(sample_loss * (temperature ** 2), sample_weights)


def compute_hard_score_distill_loss(
    student_scores: torch.Tensor,
    teacher_scores: torch.Tensor,
    targets: torch.Tensor,
    sample_weights: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    if student_scores.shape != teacher_scores.shape:
        raise ValueError("student_scores and teacher_scores must have the same shape.")

    temperature = max(float(temperature), 1e-6)
    positive_student = student_scores.gather(1, targets.unsqueeze(1))
    positive_teacher = teacher_scores.gather(1, targets.unsqueeze(1))

    negative_mask = torch.ones_like(teacher_scores, dtype=torch.bool)
    negative_mask.scatter_(1, targets.unsqueeze(1), False)
    teacher_neg = teacher_scores.masked_fill(~negative_mask, float("-inf"))
    teacher_hard_idx = teacher_neg.argmax(dim=1, keepdim=True)

    hard_student = student_scores.gather(1, teacher_hard_idx)
    hard_teacher = teacher_scores.gather(1, teacher_hard_idx)

    paired_student = torch.cat([positive_student, hard_student], dim=1)
    paired_teacher = torch.cat([positive_teacher, hard_teacher], dim=1)
    teacher_prob = torch.softmax(paired_teacher.detach() / temperature, dim=1)
    student_log_prob = torch.log_softmax(paired_student / temperature, dim=1)
    sample_loss = F.kl_div(student_log_prob, teacher_prob, reduction="none").sum(dim=1)
    return weighted_mean(sample_loss * (temperature ** 2), sample_weights)


def compute_total_authcode_loss(args, **loss_terms: torch.Tensor) -> torch.Tensor:
    weight_map = {
        "teacher_consistency_loss": "teacher_consistency_weight",
        "teacher_sequence_match_loss": "teacher_sequence_match_weight",
        "teacher_pair_loss": "teacher_pair_weight",
        "teacher_bank_pair_loss": "teacher_bank_pair_weight",
        "teacher_bank_hard_loss": "teacher_bank_hard_weight",
        "teacher_claim_pair_loss": "teacher_claim_pair_weight",
        "teacher_claim_hard_loss": "teacher_claim_hard_weight",
        "teacher_codebook_commit_loss": "teacher_codebook_commit_weight",
        "student_bit_loss": "student_bit_weight",
        "student_sequence_match_loss": "student_sequence_match_weight",
        "student_pair_loss": "student_pair_weight",
        "student_claim_bit_loss": "student_claim_bit_weight",
        "student_claim_pair_loss": "student_claim_pair_weight",
        "student_claim_bank_pair_loss": "student_claim_bank_pair_weight",
        "student_protocol_claim_pair_loss": "student_protocol_claim_pair_weight",
        "student_protocol_claim_bank_pair_loss": "student_protocol_claim_bank_pair_weight",
        "student_verifier_claim_pair_loss": "student_verifier_claim_pair_weight",
        "student_claim_calibration_pair_loss": "student_claim_calibration_pair_weight",
        "student_hard_loss": "student_hard_weight",
        "student_claim_hard_loss": "student_claim_hard_weight",
        "student_claim_bank_hard_loss": "student_claim_bank_hard_weight",
        "student_protocol_claim_hard_loss": "student_protocol_claim_hard_weight",
        "student_protocol_claim_bank_hard_loss": "student_protocol_claim_bank_hard_weight",
        "student_verifier_claim_hard_loss": "student_verifier_claim_hard_weight",
        "student_verifier_claim_bank_hard_loss": "student_verifier_claim_bank_hard_weight",
        "student_claim_calibration_hard_loss": "student_claim_calibration_hard_weight",
        "student_claim_eer_proxy_loss": "student_claim_eer_proxy_weight",
        "student_claim_positive_tail_loss": "student_claim_positive_tail_weight",
        "student_soft_align_loss": "student_soft_align_weight",
        "student_claim_align_loss": "student_claim_align_weight",
        "student_recovery_bit_loss": "student_recovery_bit_weight",
        "student_recovery_stage_loss": "student_recovery_stage_weight",
        "student_recovery_align_loss": "student_recovery_align_weight",
        "student_token_class_loss": "student_token_class_weight",
        "student_token_proto_loss": "student_token_proto_weight",
        "student_score_distill_loss": "student_score_distill_weight",
        "student_hard_score_distill_loss": "student_hard_score_distill_weight",
        "student_claim_score_distill_loss": "student_claim_score_distill_weight",
        "student_codebook_class_loss": "student_codebook_class_weight",
        "student_codebook_proto_loss": "student_codebook_proto_weight",
        "teacher_balance_loss": "teacher_balance_weight",
        "teacher_decorrelation_loss": "teacher_decorrelation_weight",
        "teacher_uniformity_loss": "teacher_uniformity_weight",
        "codebook_usage_loss": "codebook_usage_weight",
        "codebook_separation_loss": "codebook_separation_weight",
    }
    if not loss_terms:
        raise ValueError("loss_terms must not be empty.")
    base_tensor = next(iter(loss_terms.values()))
    total = base_tensor.new_tensor(0.0)
    for loss_name, weight_attr in weight_map.items():
        if loss_name not in loss_terms:
            continue
        total = total + float(getattr(args, weight_attr, 0.0)) * loss_terms[loss_name]
    return total


def compute_balance_loss(bank_bits: torch.Tensor) -> torch.Tensor:
    return bank_bits.mean(dim=0).pow(2).mean()


def compute_decorrelation_loss(bank_bits: torch.Tensor) -> torch.Tensor:
    if bank_bits.shape[0] < 2:
        return bank_bits.new_tensor(0.0)
    centered = bank_bits - bank_bits.mean(dim=0, keepdim=True)
    cov = centered.T @ centered / max(bank_bits.shape[0] - 1, 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    return off_diag.pow(2).mean()


def build_bank_tensors(dataset: PrototypeDistillationDataset, raw_anchors, device: torch.device):
    proto_vec = torch.stack([dataset.get_prototype_tensor(raw_anchor) for raw_anchor in raw_anchors], dim=0).to(device)
    proto_seq = torch.stack([dataset.get_prototype_sequence(raw_anchor) for raw_anchor in raw_anchors], dim=0).to(device)
    return proto_vec, proto_seq, list(raw_anchors)


def build_claim_reference_bank(
    dataset: PrototypeDistillationDataset,
    indices,
    reference_raws,
    teacher_head: AuthCodeHead,
    use_sequence_score: bool,
    bit_scale: float,
    device: torch.device,
    reference_versions=None,
    claim_bank_mode: str = "mean_bits",
    batch_size: int = 128,
    auth_codec=None,
    auth_codebook: AuthenticationCodebook = None,
    apply_codebook_to_scores: bool = True,
):
    reference_versions = None if reference_versions is None else {int(v) for v in reference_versions}
    raw_to_selected_bits = {raw_anchor: [] for raw_anchor in reference_raws}
    raw_to_all_bits = {raw_anchor: [] for raw_anchor in reference_raws}
    raw_to_selected_logits = {raw_anchor: [] for raw_anchor in reference_raws}
    raw_to_all_logits = {raw_anchor: [] for raw_anchor in reference_raws}
    entries = []

    def flush_batch():
        nonlocal entries
        if not entries:
            return
        teacher_vec = torch.stack([entry["teacher_vec"] for entry in entries], dim=0).to(device)
        teacher_seq = torch.stack([entry["teacher_seq"] for entry in entries], dim=0).to(device)
        with torch.no_grad():
            teacher_out = teacher_head(
                teacher_vec,
                teacher_seq,
                return_sequence=use_sequence_score,
                return_logits=True,
            )
            teacher_logits = teacher_out["global_logits"]
            if auth_codec is not None:
                teacher_logits = auth_codec.encode_logits(teacher_logits)
            if auth_codebook is not None and apply_codebook_to_scores:
                teacher_logits, _, _, _ = auth_codebook.quantize(teacher_logits, straight_through=False)
            teacher_logits = teacher_logits.detach().cpu()
            teacher_bits = soft_bits(teacher_logits, bit_scale).detach().cpu()
        for entry, teacher_bit, teacher_logit in zip(entries, teacher_bits, teacher_logits):
            raw_to_all_bits[entry["raw_anchor"]].append(teacher_bit)
            raw_to_all_logits[entry["raw_anchor"]].append(teacher_logit)
            if entry["include_as_reference"]:
                raw_to_selected_bits[entry["raw_anchor"]].append(teacher_bit)
                raw_to_selected_logits[entry["raw_anchor"]].append(teacher_logit)
        entries = []

    for idx in indices:
        record = dataset.records[idx]
        raw_anchor = record["raw_anchor"]
        if raw_anchor not in raw_to_all_bits:
            continue
        version_id = record["version_id"]
        teacher_vec = dataset.teacher_vec_bank.get(version_id, dataset.prototype_bank[raw_anchor]).clone().float()
        teacher_seq = dataset.teacher_seq_bank.get(version_id, dataset.prototype_bank[raw_anchor].unsqueeze(0)).clone().float()
        entries.append(
            {
                "raw_anchor": raw_anchor,
                "include_as_reference": reference_versions is None or int(record["version"]) in reference_versions,
                "teacher_vec": teacher_vec,
                "teacher_seq": teacher_seq,
            }
        )
        if len(entries) >= int(batch_size):
            flush_batch()
    flush_batch()

    claim_bank_bits = []
    for raw_anchor in reference_raws:
        bit_list = raw_to_selected_bits[raw_anchor] or raw_to_all_bits[raw_anchor]
        logit_list = raw_to_selected_logits[raw_anchor] or raw_to_all_logits[raw_anchor]
        if not bit_list:
            raise RuntimeError(f"No claim reference bits found for raw anchor: {raw_anchor}")
        claim_bank_bits.append(
            aggregate_claim_reference(
                bit_list=bit_list,
                logit_list=logit_list,
                bit_scale=bit_scale,
                claim_bank_mode=claim_bank_mode,
            )
        )
    return torch.stack(claim_bank_bits, dim=0).to(device), list(reference_raws)


def forward_teacher_bank(
    teacher_head: AuthCodeHead,
    prototype_vec_bank: torch.Tensor,
    prototype_seq_bank: torch.Tensor,
    bit_scale: float,
    use_sequence_score: bool,
    auth_codec=None,
    auth_codebook: AuthenticationCodebook = None,
    apply_codebook_to_scores: bool = True,
):
    bank_out = teacher_head(
        prototype_vec_bank,
        prototype_seq_bank,
        return_sequence=use_sequence_score,
        return_logits=True,
    )
    global_logits = auth_codec.encode_logits(bank_out["global_logits"]) if auth_codec is not None else bank_out["global_logits"]
    if auth_codebook is not None and apply_codebook_to_scores:
        global_logits, _, _, _ = auth_codebook.quantize(global_logits, straight_through=False)
    encoded_bank_out = {"global_logits": global_logits}
    if use_sequence_score and "stage_logits" in bank_out:
        encoded_bank_out["stage_logits"] = (
            auth_codec.encode_logits(bank_out["stage_logits"]) if auth_codec is not None else bank_out["stage_logits"]
        )
    bank_bits = soft_bits(encoded_bank_out["global_logits"], bit_scale)
    bank_stage_bits = None
    if use_sequence_score and "stage_logits" in encoded_bank_out:
        bank_stage_bits = soft_bits(encoded_bank_out["stage_logits"], bit_scale)
    return encoded_bank_out, bank_bits, bank_stage_bits


def compute_combined_scores(
    query_logits: torch.Tensor,
    bank_bits: torch.Tensor,
    query_stage_logits: torch.Tensor = None,
    bank_stage_bits: torch.Tensor = None,
    bit_scale: float = 3.0,
    sequence_score_weight: float = 0.0,
):
    query_bits = soft_bits(query_logits, bit_scale)
    logits = score_bank(query_bits, bank_bits)
    if query_stage_logits is not None and bank_stage_bits is not None and float(sequence_score_weight) > 0.0:
        seq_logits = score_stage_bank(soft_bits(query_stage_logits, bit_scale), bank_stage_bits)
        seq_w = min(max(float(sequence_score_weight), 0.0), 1.0)
        logits = (1.0 - seq_w) * logits + seq_w * seq_logits
    return logits, query_bits


def compute_protocol_score_matrix(
    query_protocol_logits: torch.Tensor | None,
    reference_protocol_logits: torch.Tensor | None,
    auth_codec,
    mode: str = "none",
    query_code_indices: torch.Tensor | None = None,
    reference_code_indices: torch.Tensor | None = None,
) -> torch.Tensor | None:
    mode = str(mode or "none").lower()
    if mode == "none":
        return None
    if query_protocol_logits is None or reference_protocol_logits is None or auth_codec is None:
        return None

    if mode == "code_cosine":
        query_protocol_logits = F.normalize(query_protocol_logits, dim=-1)
        reference_protocol_logits = F.normalize(reference_protocol_logits, dim=-1)
        return torch.matmul(query_protocol_logits, reference_protocol_logits.T)

    query_code_bits = auth_codec.hard_codeword_bits(query_protocol_logits)
    reference_code_bits = auth_codec.hard_codeword_bits(reference_protocol_logits)
    if mode == "codeword_agreement":
        return (query_code_bits.unsqueeze(1) == reference_code_bits.unsqueeze(0)).float().mean(dim=-1)

    query_payload_bits = auth_codec.hard_payload_bits_from_codeword(query_protocol_logits)
    reference_payload_bits = auth_codec.hard_payload_bits_from_codeword(reference_protocol_logits)
    if mode == "payload_agreement":
        return (query_payload_bits.unsqueeze(1) == reference_payload_bits.unsqueeze(0)).float().mean(dim=-1)
    if mode == "decode_success":
        return (query_payload_bits.unsqueeze(1) == reference_payload_bits.unsqueeze(0)).all(dim=-1).float()
    if mode == "index_match":
        if query_code_indices is None or reference_code_indices is None:
            return None
        return (query_code_indices.unsqueeze(1) == reference_code_indices.unsqueeze(0)).float()
    raise ValueError(f"Unsupported protocol_score_mode: {mode}")


def normalize_protocol_mode_list(protocol_modes) -> list[str]:
    if protocol_modes is None:
        return []
    normalized = []
    for mode in protocol_modes:
        mode = str(mode or "none").lower()
        if mode == "none":
            continue
        if mode not in normalized:
            normalized.append(mode)
    return normalized


def compute_protocol_score_bundle(
    query_protocol_logits: torch.Tensor | None,
    reference_protocol_logits: torch.Tensor | None,
    auth_codec,
    primary_mode: str = "none",
    extra_modes=None,
    query_code_indices: torch.Tensor | None = None,
    reference_code_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor | None, list[torch.Tensor]]:
    primary_mode = str(primary_mode or "none").lower()
    extra_modes = normalize_protocol_mode_list(extra_modes)
    ordered_modes = []
    if primary_mode != "none":
        ordered_modes.append(primary_mode)
    for mode in extra_modes:
        if mode not in ordered_modes:
            ordered_modes.append(mode)

    score_map = {}
    for mode in ordered_modes:
        score_map[mode] = compute_protocol_score_matrix(
            query_protocol_logits=query_protocol_logits,
            reference_protocol_logits=reference_protocol_logits,
            auth_codec=auth_codec,
            mode=mode,
            query_code_indices=query_code_indices,
            reference_code_indices=reference_code_indices,
        )
    primary_scores = score_map.get(primary_mode)
    extra_scores = [score_map.get(mode) for mode in extra_modes]
    return primary_scores, extra_scores


def combine_score_matrices(
    main_scores: torch.Tensor,
    auxiliary_scores: torch.Tensor | None = None,
    alpha: float = 0.0,
    main_normalization: str = "none",
    auxiliary_normalization: str = "none",
) -> torch.Tensor:
    def normalize(scores: torch.Tensor, mode: str) -> torch.Tensor:
        mode = str(mode)
        if mode == "none":
            return scores
        if mode == "row_center":
            return scores - scores.mean(dim=1, keepdim=True)
        if mode == "row_zscore":
            return (scores - scores.mean(dim=1, keepdim=True)) / scores.std(
                dim=1,
                keepdim=True,
                unbiased=False,
            ).clamp_min(1e-6)
        raise ValueError(f"Unsupported score normalization mode: {mode}")

    alpha = min(max(float(alpha), 0.0), 1.0)
    main_scores = normalize(main_scores, main_normalization)
    if auxiliary_scores is None or alpha <= 0.0:
        return main_scores
    auxiliary_scores = normalize(auxiliary_scores, auxiliary_normalization)
    return (1.0 - alpha) * main_scores + alpha * auxiliary_scores


def _row_rank_percentile(scores: torch.Tensor) -> torch.Tensor:
    ranks = torch.argsort(torch.argsort(scores, dim=1, descending=False), dim=1, descending=False).to(scores.dtype)
    denom = max(scores.shape[1] - 1, 1)
    return ranks / float(denom)


def _row_second_best(scores: torch.Tensor) -> torch.Tensor:
    top2 = torch.topk(scores, k=min(2, scores.shape[1]), dim=1).values
    if top2.shape[1] == 1:
        return top2[:, :1]
    return top2[:, 1:2]


def claim_score_fusion_input_dim(
    extra_protocol_modes=None,
    include_token_score: bool = False,
    include_verifier_score: bool = False,
) -> int:
    token_dim = 4 if include_token_score else 0
    verifier_dim = 4 if include_verifier_score else 0
    return 12 + token_dim + verifier_dim + 4 * len(normalize_protocol_mode_list(extra_protocol_modes))


def build_claim_score_fusion_features(
    main_scores: torch.Tensor,
    protocol_scores: torch.Tensor | None,
    extra_protocol_scores: list[torch.Tensor] | None = None,
    token_scores: torch.Tensor | None = None,
) -> torch.Tensor:
    if protocol_scores is None:
        protocol_scores = torch.zeros_like(main_scores)
    extra_protocol_scores = list(extra_protocol_scores or [])

    main_mean = main_scores.mean(dim=1, keepdim=True)
    protocol_mean = protocol_scores.mean(dim=1, keepdim=True)
    main_std = main_scores.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
    protocol_std = protocol_scores.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)

    main_z = (main_scores - main_mean) / main_std
    protocol_z = (protocol_scores - protocol_mean) / protocol_std
    main_rank = _row_rank_percentile(main_scores)
    protocol_rank = _row_rank_percentile(protocol_scores)
    main_gap = main_scores - _row_second_best(main_scores)
    protocol_gap = protocol_scores - _row_second_best(protocol_scores)

    feature_list = [
        main_scores,
        protocol_scores,
        main_z,
        protocol_z,
        main_rank,
        protocol_rank,
        main_gap,
        protocol_gap,
        main_z * protocol_z,
        main_rank * protocol_rank,
        main_gap * protocol_gap,
        main_scores - protocol_scores,
    ]
    if token_scores is not None:
        token_mean = token_scores.mean(dim=1, keepdim=True)
        token_std = token_scores.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
        token_z = (token_scores - token_mean) / token_std
        token_rank = _row_rank_percentile(token_scores)
        token_gap = token_scores - _row_second_best(token_scores)
        feature_list.extend([token_scores, token_z, token_rank, token_gap])
    for extra_scores in extra_protocol_scores:
        if extra_scores is None:
            extra_scores = torch.zeros_like(protocol_scores)
        extra_mean = extra_scores.mean(dim=1, keepdim=True)
        extra_std = extra_scores.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
        extra_z = (extra_scores - extra_mean) / extra_std
        extra_rank = _row_rank_percentile(extra_scores)
        extra_gap = extra_scores - _row_second_best(extra_scores)
        feature_list.extend([extra_scores, extra_z, extra_rank, extra_gap])
    return torch.stack(feature_list, dim=-1)


def fuse_claim_score_matrices(
    main_scores: torch.Tensor,
    protocol_scores: torch.Tensor | None = None,
    extra_protocol_scores: list[torch.Tensor] | None = None,
    claim_score_fusion_head: ClaimScoreFusionHead = None,
    token_scores: torch.Tensor | None = None,
    alpha: float = 0.0,
    main_normalization: str = "none",
    auxiliary_normalization: str = "none",
) -> torch.Tensor:
    base_scores = combine_score_matrices(
        main_scores,
        protocol_scores,
        alpha=alpha,
        main_normalization=main_normalization,
        auxiliary_normalization=auxiliary_normalization,
    )
    if claim_score_fusion_head is None or protocol_scores is None:
        return base_scores
    pair_features = build_claim_score_fusion_features(
        main_scores,
        protocol_scores,
        extra_protocol_scores=extra_protocol_scores,
        token_scores=token_scores,
    )
    pair_features_flat = pair_features.reshape(-1, pair_features.shape[-1])
    base_scores_flat = base_scores.reshape(-1)
    if isinstance(claim_score_fusion_head, ClaimScoreFusionHead):
        fused_scores = claim_score_fusion_head(pair_features_flat, base_scores=base_scores_flat)
    else:
        fused_scores = claim_score_fusion_head(pair_features_flat)
        if fused_scores.ndim > 1 and fused_scores.shape[-1] == 1:
            fused_scores = fused_scores.squeeze(-1)
    return fused_scores.reshape(main_scores.shape[0], main_scores.shape[1])


def forward_student_branches(
    student: VisualEncoder,
    rgb_image: torch.Tensor,
    use_sequence: bool,
    encode_student_logits,
    auth_codebook: AuthenticationCodebook = None,
    apply_codebook_to_scores: bool = False,
    student_code_head: StudentCodeHead = None,
    student_token_head: StudentTokenHead = None,
):
    student_out = student(
        rgb_image,
        return_sequence=use_sequence,
        return_logits=True,
        return_features=student_code_head is not None or student_token_head is not None,
    )
    score_encoded_logits = encode_student_logits(student_out["global_logits"])
    score_code_logits = None
    score_code_indices = None
    score_code_prototypes = None
    if auth_codebook is not None:
        score_code_logits, score_code_indices, _, score_code_prototypes = auth_codebook.quantize(
            score_encoded_logits,
            straight_through=True,
        )
        score_logits = score_code_logits if apply_codebook_to_scores else score_encoded_logits
    else:
        score_logits = score_encoded_logits

    if student_code_head is not None:
        recovery_out = student_code_head(
            student_out["feature_vec"],
            base_logits=score_encoded_logits,
            return_logits=True,
        )
        recovery_logits = recovery_out["global_logits"]
        recovery_code_logits = None
        recovery_code_indices = None
        recovery_code_prototypes = None
        if auth_codebook is not None:
            recovery_code_logits, recovery_code_indices, _, recovery_code_prototypes = auth_codebook.quantize(
                recovery_logits,
                straight_through=True,
            )
    else:
        recovery_logits = score_encoded_logits
        recovery_code_logits = score_code_logits
        recovery_code_indices = score_code_indices
        recovery_code_prototypes = score_code_prototypes

    recovery_protocol_logits = recovery_code_logits if recovery_code_logits is not None else recovery_logits
    stage_logits = encode_student_logits(student_out.get("stage_logits"))
    token_out = (
        student_token_head(
            student_out["feature_vec"],
            base_logits=recovery_logits,
        )
        if student_token_head is not None
        else {}
    )
    return {
        "score_encoded_logits": score_encoded_logits,
        "score_logits": score_logits,
        "stage_logits": stage_logits,
        "recovery_logits": recovery_logits,
        "recovery_protocol_logits": recovery_protocol_logits,
        "recovery_code_indices": recovery_code_indices,
        "recovery_code_prototypes": recovery_code_prototypes,
        "token_logits": token_out.get("token_logits"),
        "token_probs": token_out.get("token_probs"),
    }


INIT_CHECKPOINT_CONFIG_DEFAULTS = {
    "teacher_key": "teacher_joint_seq",
    "code_dim": 32,
    "payload_dim": None,
    "ecc_scheme": "identity",
    "ecc_repetition": 2,
    "student_code_space": "payload",
    "teacher_hidden_dim": 256,
    "teacher_dropout": 0.0,
    "teacher_codebook_size": 0,
    "teacher_codebook_temperature": 12.0,
    "codebook_mode": "replace",
    "use_claim_verifier_head": False,
    "claim_verifier_hidden_dim": 128,
    "claim_verifier_dropout": 0.0,
    "claim_verifier_input_mode": "bits",
    "claim_verifier_score_mode": "add",
    "claim_verifier_feature_to_fusion": False,
    "use_student_code_head": False,
    "student_code_head_hidden_dim": 256,
    "student_code_head_dropout": 0.0,
    "use_claim_score_fusion_head": False,
    "claim_score_fusion_hidden_dim": 32,
    "claim_score_fusion_dropout": 0.0,
    "claim_score_fusion_protocol_modes": None,
    "claim_score_fusion_mode": "direct",
    "claim_score_fusion_residual_scale": 1.0,
    "backbone_type": "resnet18",
    "input_mode": "residual_only",
    "residual_scale": 1.75,
    "residual_kernel": 9,
    "local_crop_mode": "none",
    "local_crop_size": 160,
    "local_patch_offset": 24,
    "image_size": 224,
    "resize_size": 320,
    "augmentation_preset": "center",
    "include_versions": [1, 2, 3, 4],
    "anchor_versions": [1, 2],
    "shift_versions": [3, 4],
    "sequence_score_weight": 0.2,
    "teacher_sequence_match_weight": 0.5,
    "student_sequence_match_weight": 0.5,
    "teacher_consistency_weight": 1.0,
    "teacher_pair_weight": 0.5,
    "teacher_bank_pair_weight": 0.5,
    "teacher_bank_hard_weight": 0.25,
    "teacher_claim_pair_weight": 0.0,
    "teacher_claim_hard_weight": 0.0,
    "teacher_codebook_commit_weight": 0.0,
    "teacher_positive_margin": 0.55,
    "teacher_negative_margin": 0.10,
    "teacher_hard_margin": 0.15,
    "student_bit_weight": 1.0,
    "student_pair_weight": 1.0,
    "student_target_mode": "canonical",
    "student_target_blend_alpha": 0.5,
    "student_claim_bit_weight": 0.0,
    "student_claim_pair_weight": 0.0,
    "student_claim_bank_pair_weight": 0.0,
    "student_protocol_claim_pair_weight": 0.0,
    "student_protocol_claim_bank_pair_weight": 0.0,
    "student_verifier_claim_pair_weight": 0.0,
    "student_claim_calibration_pair_weight": 0.0,
    "student_hard_weight": 0.0,
    "student_claim_hard_weight": 0.0,
    "student_claim_bank_hard_weight": 0.0,
    "student_protocol_claim_hard_weight": 0.0,
    "student_protocol_claim_bank_hard_weight": 0.0,
    "student_verifier_claim_hard_weight": 0.0,
    "student_verifier_claim_bank_hard_weight": 0.0,
    "student_claim_calibration_hard_weight": 0.0,
    "student_claim_eer_proxy_weight": 0.0,
    "student_claim_eer_proxy_positive_quantile": 0.10,
    "student_claim_eer_proxy_negative_quantile": 0.90,
    "student_claim_eer_proxy_margin": 0.02,
    "student_claim_eer_proxy_scale": 12.0,
    "student_claim_positive_tail_weight": 0.0,
    "student_claim_positive_tail_positive_quantile": 0.10,
    "student_claim_positive_tail_negative_quantile": 0.90,
    "student_claim_positive_tail_margin": 0.01,
    "student_claim_positive_tail_scale": 12.0,
    "student_claim_hardcase_reweight_strength": 0.0,
    "student_claim_hardcase_reweight_margin": 0.05,
    "student_claim_hardcase_reweight_scale": 12.0,
    "student_soft_align_weight": 0.5,
    "student_claim_align_weight": 0.0,
    "student_recovery_bit_weight": 0.0,
    "student_recovery_stage_weight": 0.0,
    "student_recovery_align_weight": 0.0,
    "student_token_class_weight": 1.0,
    "student_token_proto_weight": 0.5,
    "student_score_distill_weight": 0.0,
    "student_hard_score_distill_weight": 0.0,
    "student_claim_score_distill_weight": 0.0,
    "student_codebook_class_weight": 0.0,
    "student_codebook_proto_weight": 0.0,
    "student_positive_margin": 0.55,
    "student_negative_margin": 0.10,
    "student_hard_margin": 0.15,
    "claim_reference_mode": "same_raw",
    "claim_reference_versions": None,
    "claim_bank_mode": "mean_bits",
    "claim_anchor_aux_weight": 0.0,
    "teacher_balance_weight": 0.1,
    "teacher_decorrelation_weight": 0.05,
    "teacher_uniformity_weight": 0.05,
    "codebook_usage_weight": 0.0,
    "codebook_separation_weight": 0.0,
    "uniformity_temperature": 2.0,
    "protocol_score_mode": "none",
    "protocol_score_alpha": 0.0,
    "official_claim_score_mode": "deterministic_gate",
    "claim_main_score_norm_mode": "none",
    "protocol_score_norm_mode": "none",
    "bit_scale": 3.0,
    "use_joint_claim_loss": False,
}


def _arg_matches_default(current_value, default_value) -> bool:
    if isinstance(default_value, list):
        if current_value is None:
            return False
        return list(current_value) == list(default_value)
    return current_value == default_value


def inherit_init_checkpoint_config(args, checkpoint_data: dict | None) -> list[str]:
    if checkpoint_data is None:
        return []
    checkpoint_config = checkpoint_data.get("config") or {}
    inherited_keys = []
    for key, default_value in INIT_CHECKPOINT_CONFIG_DEFAULTS.items():
        if key not in checkpoint_config:
            continue
        current_value = getattr(args, key, None)
        if _arg_matches_default(current_value, default_value):
            setattr(args, key, checkpoint_config[key])
            inherited_keys.append(key)

    if (
        not bool(getattr(args, "use_student_code_head", False))
        and checkpoint_data.get("student_code_head_state_dict") is not None
        and "use_student_code_head" not in inherited_keys
    ):
        args.use_student_code_head = True
        inherited_keys.append("use_student_code_head")
    if (
        not bool(getattr(args, "use_claim_verifier_head", False))
        and checkpoint_data.get("claim_verifier_state_dict") is not None
        and "use_claim_verifier_head" not in inherited_keys
    ):
        args.use_claim_verifier_head = True
        inherited_keys.append("use_claim_verifier_head")
    if (
        not bool(getattr(args, "use_claim_score_fusion_head", False))
        and checkpoint_data.get("claim_score_fusion_state_dict") is not None
        and "use_claim_score_fusion_head" not in inherited_keys
    ):
        args.use_claim_score_fusion_head = True
        inherited_keys.append("use_claim_score_fusion_head")
    if (
        int(getattr(args, "teacher_codebook_size", 0)) <= 0
        and checkpoint_data.get("auth_codebook_state_dict") is not None
        and "teacher_codebook_size" not in inherited_keys
    ):
        args.teacher_codebook_size = int(checkpoint_data["auth_codebook_state_dict"]["codes"].shape[0])
        inherited_keys.append("teacher_codebook_size")
    return inherited_keys


def should_load_claim_score_fusion_state_dict(args, checkpoint_data: dict | None) -> tuple[bool, str | None]:
    if checkpoint_data is None:
        return False, "checkpoint is missing"
    if checkpoint_data.get("claim_score_fusion_state_dict") is None:
        return False, "checkpoint has no claim_score_fusion_state_dict"
    checkpoint_config = checkpoint_data.get("config") or {}
    checkpoint_mode = str(checkpoint_config.get("claim_score_fusion_mode", "direct") or "direct")
    current_mode = str(getattr(args, "claim_score_fusion_mode", "direct") or "direct")
    if checkpoint_mode != current_mode:
        return (
            False,
            f"fusion mode mismatch ({checkpoint_mode} -> {current_mode})",
        )
    return True, None


def train_stage3_authcode(args):
    set_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    pin_memory = device.type == "cuda"

    official_eval_preset, official_config_resolved = apply_stage3_official_config_to_args(args)
    if official_eval_preset is not None and int(args.max_raws) != OFFICIAL_STAGE3_EVAL_MAX_RAWS:
        print(
            f"[warning] official preset {official_eval_preset} is usually reported with "
            f"max_raws={OFFICIAL_STAGE3_EVAL_MAX_RAWS}, but current training uses max_raws={args.max_raws}.",
            flush=True,
        )

    init_checkpoint_data = None
    if args.init_checkpoint:
        init_checkpoint_data = torch.load(args.init_checkpoint, map_location="cpu")
        inherited_keys = inherit_init_checkpoint_config(args, init_checkpoint_data)
        if inherited_keys:
            print(
                "Inherited init-checkpoint config keys: "
                + ", ".join(sorted(set(inherited_keys))),
                flush=True,
            )

    init_student_checkpoint_data = None
    if args.init_student_checkpoint:
        init_student_checkpoint_data = torch.load(args.init_student_checkpoint, map_location="cpu")
        inherited_keys = inherit_init_checkpoint_config(args, init_student_checkpoint_data)
        if inherited_keys:
            print(
                "Inherited init-student-checkpoint config keys: "
                + ", ".join(sorted(set(inherited_keys))),
                flush=True,
            )

    train_transform, eval_transform = build_transforms(
        args.augmentation_preset,
        image_size=args.image_size,
        resize_size=args.resize_size,
    )

    full_dataset = PrototypeDistillationDataset(
        prototype_cache_dir=args.prototype_cache_dir,
        meta_path=args.meta_path,
        rgb_dir=args.rgb_dir,
        teacher_cache_dir=args.teacher_cache_dir,
        teacher_key=args.teacher_key,
        include_versions=args.include_versions,
        transform=eval_transform,
    )
    train_indices, val_indices, train_raws, val_raws = build_group_split(
        dataset=full_dataset,
        val_ratio=args.val_ratio,
        seed=args.seed,
        max_raws=args.max_raws,
    )

    train_dataset = PrototypeDistillationDataset(
        prototype_cache_dir=args.prototype_cache_dir,
        meta_path=args.meta_path,
        rgb_dir=args.rgb_dir,
        teacher_cache_dir=args.teacher_cache_dir,
        teacher_key=args.teacher_key,
        include_versions=args.include_versions,
        transform=train_transform,
    )
    val_dataset = PrototypeDistillationDataset(
        prototype_cache_dir=args.prototype_cache_dir,
        meta_path=args.meta_path,
        rgb_dir=args.rgb_dir,
        teacher_cache_dir=args.teacher_cache_dir,
        teacher_key=args.teacher_key,
        include_versions=args.include_versions,
        transform=eval_transform,
    )

    train_sampler = RawGroupBatchSampler(
        group_ids=train_dataset.group_ids,
        indices=train_indices,
        groups_per_batch=max(1, args.batch_size // 4),
        seed=args.seed,
        shuffle=True,
        drop_last=True,
    )
    val_sampler = RawGroupBatchSampler(
        group_ids=val_dataset.group_ids,
        indices=val_indices,
        groups_per_batch=max(1, args.val_batch_size // 4),
        seed=args.seed,
        shuffle=False,
        drop_last=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )

    prototype_dim = full_dataset.prototype_dim
    use_sequence = bool(
        args.sequence_score_weight > 0.0
        or args.teacher_sequence_match_weight > 0.0
        or args.student_sequence_match_weight > 0.0
    )
    claim_sequence_score_weight = (
        args.sequence_score_weight
        if args.claim_sequence_score_weight is None
        else float(args.claim_sequence_score_weight)
    )
    auth_codec = build_auth_codec(
        code_dim=args.code_dim,
        ecc_scheme=args.ecc_scheme,
        ecc_repetition=args.ecc_repetition,
        payload_dim=args.payload_dim,
    )
    token_protocol_enabled = str(args.auth_protocol_variant) == "token_residual"
    student_output_dim = args.code_dim if args.student_code_space == "codeword" else auth_codec.payload_dim
    auth_codebook = None
    teacher_tokenizer = None
    selected_teacher_token_info = None
    codebook_initialized = False
    if args.teacher_codebook_size > 0:
        auth_codebook = AuthenticationCodebook(
            code_dim=args.code_dim,
            num_codes=args.teacher_codebook_size,
            temperature=args.teacher_codebook_temperature,
            learnable=not args.freeze_codebook,
        ).to(device)
    apply_codebook_to_scores = auth_codebook is not None and args.codebook_mode == "replace"

    def encode_teacher_logits(logits: torch.Tensor | None) -> torch.Tensor | None:
        if logits is None:
            return None
        return auth_codec.encode_logits(logits)

    def encode_student_logits(logits: torch.Tensor | None) -> torch.Tensor | None:
        if logits is None:
            return None
        if args.student_code_space == "codeword":
            return logits
        return auth_codec.encode_logits(logits)

    teacher_head = AuthCodeHead(
        d_in=prototype_dim,
        code_dim=auth_codec.payload_dim,
        hidden_dim=args.teacher_hidden_dim,
        use_sequence=use_sequence,
        dropout=args.teacher_dropout,
    ).to(device)
    if args.teacher_init_projection_checkpoint:
        if auth_codec.payload_dim != args.code_dim:
            print("Skip projection warm-start because ECC changes payload dimension.", flush=True)
        else:
            mean_vec, projection = load_projection_checkpoint(args.teacher_init_projection_checkpoint)
            teacher_head.initialize_from_projection(mean_vec, projection)
    student = VisualEncoder(
        d_out=student_output_dim,
        backbone_type=args.backbone_type,
        pretrained=not args.no_pretrained,
        input_mode=args.input_mode,
        residual_scale=args.residual_scale,
        residual_kernel=args.residual_kernel,
        use_stage_sequence_head=use_sequence,
        local_crop_mode=args.local_crop_mode,
        local_crop_size=args.local_crop_size,
        local_patch_offset=args.local_patch_offset,
    ).to(device)
    student_code_head = None
    student_token_head = None
    if args.use_student_code_head:
        student_code_head = StudentCodeHead(
            d_in=student.feature_dim,
            code_dim=auth_codec.code_dim,
            hidden_dim=args.student_code_head_hidden_dim,
            dropout=args.student_code_head_dropout,
        ).to(device)
    claim_verifier = None
    if args.use_claim_verifier_head:
        claim_verifier = PrototypeVerifier(
            d_model=claim_verifier_dim(args.code_dim, args.claim_verifier_input_mode),
            hidden_dim=args.claim_verifier_hidden_dim,
            dropout=args.claim_verifier_dropout,
        ).to(device)
    claim_score_fusion_head = None
    if args.use_claim_score_fusion_head:
            claim_score_fusion_head = ClaimScoreFusionHead(
                input_dim=claim_score_fusion_input_dim(
                    args.claim_score_fusion_protocol_modes,
                    include_token_score=token_protocol_enabled,
                    include_verifier_score=args.claim_verifier_feature_to_fusion and args.use_claim_verifier_head,
                ),
                hidden_dim=args.claim_score_fusion_hidden_dim,
                dropout=args.claim_score_fusion_dropout,
            mode=args.claim_score_fusion_mode,
            residual_scale=args.claim_score_fusion_residual_scale,
        ).to(device)

    if args.claim_calibration_scales and claim_verifier is not None and not args.claim_calibration_verifier_weights:
        args.claim_calibration_verifier_weights = [float(args.claim_verifier_weight)]

    if args.init_checkpoint:
        checkpoint = init_checkpoint_data
        if "teacher_head_state_dict" in checkpoint:
            load_status = load_state_dict_shape_safe(teacher_head, checkpoint["teacher_head_state_dict"], module_name="Teacher")
            if load_status.missing_keys:
                print(f"Teacher missing keys on init load: {load_status.missing_keys}", flush=True)
            if load_status.unexpected_keys:
                print(f"Teacher unexpected keys on init load: {load_status.unexpected_keys}", flush=True)
        if "student_state_dict" in checkpoint:
            student_state_dict = checkpoint["student_state_dict"]
            if args.student_code_space == "codeword" and auth_codec.scheme == "repetition":
                student_state_dict = maybe_expand_repetition_state_dict(
                    student_state_dict,
                    student.state_dict(),
                    auth_codec.repetition,
                )
            load_status = load_state_dict_shape_safe(student, student_state_dict, module_name="Student")
            if load_status.missing_keys:
                print(f"Student missing keys on init load: {load_status.missing_keys}", flush=True)
            if load_status.unexpected_keys:
                print(f"Student unexpected keys on init load: {load_status.unexpected_keys}", flush=True)
        if student_code_head is not None and checkpoint.get("student_code_head_state_dict") is not None:
            load_status = load_state_dict_shape_safe(
                student_code_head,
                checkpoint["student_code_head_state_dict"],
                module_name="StudentCodeHead",
            )
            if load_status.missing_keys:
                print(f"StudentCodeHead missing keys on init load: {load_status.missing_keys}", flush=True)
            if load_status.unexpected_keys:
                print(f"StudentCodeHead unexpected keys on init load: {load_status.unexpected_keys}", flush=True)
        if (
            claim_verifier is not None
            and "claim_verifier_state_dict" in checkpoint
            and checkpoint["claim_verifier_state_dict"] is not None
        ):
            load_status = load_state_dict_shape_safe(
                claim_verifier,
                checkpoint["claim_verifier_state_dict"],
                module_name="ClaimVerifier",
            )
            if load_status.missing_keys:
                print(f"ClaimVerifier missing keys on init load: {load_status.missing_keys}", flush=True)
            if load_status.unexpected_keys:
                print(f"ClaimVerifier unexpected keys on init load: {load_status.unexpected_keys}", flush=True)
        if claim_score_fusion_head is not None:
            should_load_fusion, skip_reason = should_load_claim_score_fusion_state_dict(args, checkpoint)
            if should_load_fusion:
                load_status = load_state_dict_shape_safe(
                    claim_score_fusion_head,
                    checkpoint["claim_score_fusion_state_dict"],
                    module_name="ClaimScoreFusionHead",
                )
                if load_status.missing_keys:
                    print(f"ClaimScoreFusionHead missing keys on init load: {load_status.missing_keys}", flush=True)
                if load_status.unexpected_keys:
                    print(f"ClaimScoreFusionHead unexpected keys on init load: {load_status.unexpected_keys}", flush=True)
            elif checkpoint.get("claim_score_fusion_state_dict") is not None:
                print(f"Skipping ClaimScoreFusionHead init load: {skip_reason}.", flush=True)
        if auth_codebook is not None and checkpoint.get("auth_codebook_state_dict") is not None:
            load_status = load_state_dict_shape_safe(
                auth_codebook,
                checkpoint["auth_codebook_state_dict"],
                module_name="AuthCodebook",
            )
            codebook_initialized = True
            if load_status.missing_keys:
                print(f"AuthCodebook missing keys on init load: {load_status.missing_keys}", flush=True)
            if load_status.unexpected_keys:
                print(f"AuthCodebook unexpected keys on init load: {load_status.unexpected_keys}", flush=True)

    if args.init_student_checkpoint:
        checkpoint = init_student_checkpoint_data
        student_state_dict = checkpoint["student_state_dict"]
        if args.student_code_space == "codeword" and auth_codec.scheme == "repetition":
            student_state_dict = maybe_expand_repetition_state_dict(
                student_state_dict,
                student.state_dict(),
                auth_codec.repetition,
            )
        load_status = load_state_dict_shape_safe(student, student_state_dict, module_name="Student")
        if load_status.missing_keys:
            print(f"Student missing keys on init load: {load_status.missing_keys}", flush=True)
        if load_status.unexpected_keys:
            print(f"Student unexpected keys on init load: {load_status.unexpected_keys}", flush=True)
        if student_code_head is not None and checkpoint.get("student_code_head_state_dict") is not None:
            load_status = load_state_dict_shape_safe(
                student_code_head,
                checkpoint["student_code_head_state_dict"],
                module_name="StudentCodeHead",
            )
            if load_status.missing_keys:
                print(f"StudentCodeHead missing keys on init load: {load_status.missing_keys}", flush=True)
            if load_status.unexpected_keys:
                print(f"StudentCodeHead unexpected keys on init load: {load_status.unexpected_keys}", flush=True)
        if claim_score_fusion_head is not None:
            should_load_fusion, skip_reason = should_load_claim_score_fusion_state_dict(args, checkpoint)
            if should_load_fusion:
                load_status = load_state_dict_shape_safe(
                    claim_score_fusion_head,
                    checkpoint["claim_score_fusion_state_dict"],
                    module_name="ClaimScoreFusionHead",
                )
                if load_status.missing_keys:
                    print(f"ClaimScoreFusionHead missing keys on init load: {load_status.missing_keys}", flush=True)
                if load_status.unexpected_keys:
                    print(f"ClaimScoreFusionHead unexpected keys on init load: {load_status.unexpected_keys}", flush=True)
            elif checkpoint.get("claim_score_fusion_state_dict") is not None:
                print(f"Skipping ClaimScoreFusionHead init load: {skip_reason}.", flush=True)

    if auth_codebook is not None and args.init_codebook_checkpoint:
        checkpoint = torch.load(args.init_codebook_checkpoint, map_location="cpu")
        if checkpoint.get("auth_codebook_state_dict") is None:
            raise RuntimeError("init_codebook_checkpoint does not contain auth_codebook_state_dict.")
        load_status = load_state_dict_shape_safe(
            auth_codebook,
            checkpoint["auth_codebook_state_dict"],
            module_name="AuthCodebook",
        )
        codebook_initialized = True
        if load_status.missing_keys:
            print(f"AuthCodebook missing keys on init_codebook load: {load_status.missing_keys}", flush=True)
        if load_status.unexpected_keys:
            print(f"AuthCodebook unexpected keys on init_codebook load: {load_status.unexpected_keys}", flush=True)

    if auth_codebook is not None and not codebook_initialized and not args.skip_codebook_init:
        initialize_codebook_from_teacher_samples(
            auth_codebook=auth_codebook,
            dataset=train_dataset,
            indices=list(range(len(train_dataset))),
            teacher_head=teacher_head,
            device=device,
            auth_codec=auth_codec,
            batch_size=args.val_batch_size,
        )

    if token_protocol_enabled:
        teacher_token_classes = int(args.teacher_token_classes)
        teacher_token_state = None
        if init_checkpoint_data is not None:
            teacher_token_state = init_checkpoint_data.get("teacher_tokenizer_state_dict")
        if teacher_token_state is not None:
            teacher_token_classes = int(teacher_token_state["prototypes"].shape[0])
            args.teacher_token_classes = teacher_token_classes
            teacher_tokenizer = TeacherTokenTokenizer(
                code_dim=auth_codec.code_dim,
                num_tokens=teacher_token_classes,
                temperature=args.teacher_token_temperature,
            ).to(device)
            load_status = load_state_dict_shape_safe(
                teacher_tokenizer,
                teacher_token_state,
                module_name="TeacherTokenizer",
            )
            if load_status.missing_keys:
                print(f"TeacherTokenizer missing keys on init load: {load_status.missing_keys}", flush=True)
            if load_status.unexpected_keys:
                print(f"TeacherTokenizer unexpected keys on init load: {load_status.unexpected_keys}", flush=True)
        else:
            token_samples = collect_teacher_tokenizer_samples(
                dataset=train_dataset,
                indices=train_indices,
                teacher_head=teacher_head,
                device=device,
                auth_codec=auth_codec,
                batch_size=args.val_batch_size,
            )
            if teacher_token_classes <= 1:
                selected_teacher_token_info = select_teacher_tokenizer(
                    sample_logits=token_samples["logits"],
                    raw_anchors=token_samples["raw_anchors"],
                    sample_ids=token_samples["sample_ids"],
                    candidate_classes=[8, 16, 32, 64],
                    device=device,
                    temperature=args.teacher_token_temperature,
                )
                teacher_tokenizer = selected_teacher_token_info["tokenizer"]
                args.teacher_token_classes = int(selected_teacher_token_info["num_tokens"])
            else:
                teacher_tokenizer = TeacherTokenTokenizer(
                    code_dim=auth_codec.code_dim,
                    num_tokens=teacher_token_classes,
                    temperature=args.teacher_token_temperature,
                ).to(device)
                teacher_tokenizer.initialize_from_samples(token_samples["logits"].to(device))
        student_token_head = StudentTokenHead(
            d_in=student.feature_dim,
            num_tokens=int(args.teacher_token_classes),
            code_dim=auth_codec.code_dim,
            hidden_dim=args.student_token_head_hidden_dim,
            dropout=args.student_token_head_dropout,
        ).to(device)
        if init_checkpoint_data is not None and init_checkpoint_data.get("student_token_head_state_dict") is not None:
            load_status = load_state_dict_shape_safe(
                student_token_head,
                init_checkpoint_data["student_token_head_state_dict"],
                module_name="StudentTokenHead",
            )
            if load_status.missing_keys:
                print(f"StudentTokenHead missing keys on init load: {load_status.missing_keys}", flush=True)
            if load_status.unexpected_keys:
                print(f"StudentTokenHead unexpected keys on init load: {load_status.unexpected_keys}", flush=True)
        if init_student_checkpoint_data is not None and init_student_checkpoint_data.get("student_token_head_state_dict") is not None:
            load_status = load_state_dict_shape_safe(
                student_token_head,
                init_student_checkpoint_data["student_token_head_state_dict"],
                module_name="StudentTokenHead",
            )
            if load_status.missing_keys:
                print(f"StudentTokenHead missing keys on init_student load: {load_status.missing_keys}", flush=True)
            if load_status.unexpected_keys:
                print(f"StudentTokenHead unexpected keys on init_student load: {load_status.unexpected_keys}", flush=True)

    if args.freeze_teacher_head:
        for param in teacher_head.parameters():
            param.requires_grad = False
    if args.freeze_student:
        for param in student.parameters():
            param.requires_grad = False
    if student_code_head is not None and args.freeze_student_code_head:
        for param in student_code_head.parameters():
            param.requires_grad = False
    if student_token_head is not None and args.freeze_student_token_head:
        for param in student_token_head.parameters():
            param.requires_grad = False
    if claim_verifier is not None and args.freeze_claim_verifier:
        for param in claim_verifier.parameters():
            param.requires_grad = False
    if claim_score_fusion_head is not None and args.freeze_claim_score_fusion_head:
        for param in claim_score_fusion_head.parameters():
            param.requires_grad = False
    if auth_codebook is not None and args.freeze_codebook:
        for param in auth_codebook.parameters():
            param.requires_grad = False

    parameters = []
    if not args.freeze_teacher_head:
        parameters.extend(list(teacher_head.parameters()))
    if not args.freeze_student:
        parameters.extend(list(student.parameters()))
    if student_code_head is not None and not args.freeze_student_code_head:
        parameters.extend(list(student_code_head.parameters()))
    if student_token_head is not None and not args.freeze_student_token_head:
        parameters.extend(list(student_token_head.parameters()))
    if claim_score_fusion_head is not None and not args.freeze_claim_score_fusion_head:
        parameters.extend(list(claim_score_fusion_head.parameters()))
    if auth_codebook is not None and not args.freeze_codebook:
        parameters.extend(list(auth_codebook.parameters()))
    if claim_verifier is not None and not args.freeze_claim_verifier:
        parameters.extend(claim_verifier.parameters())
    if not parameters:
        raise ValueError("No trainable parameters remain. Disable freezing or enable the claim verifier.")

    optimizer = optim.AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_proto_vec_bank, train_proto_seq_bank, train_bank_raws = build_bank_tensors(train_dataset, train_raws, device)
    val_proto_vec_bank, val_proto_seq_bank, val_bank_raws = build_bank_tensors(val_dataset, val_raws, device)

    print("=" * 72, flush=True)
    print("Stage-3 Joint Authentication Code", flush=True)
    print(f"Device          : {device}", flush=True)
    print(f"Prototype cache : {args.prototype_cache_dir}", flush=True)
    print(f"Teacher cache   : {args.teacher_cache_dir}", flush=True)
    print(
        f"Code dim        : payload={auth_codec.payload_dim}, codeword={auth_codec.code_dim}, "
        f"student_out={student_output_dim} (space={args.student_code_space}, ecc={auth_codec.scheme}, rep={auth_codec.repetition})",
        flush=True,
    )
    print(f"Train raws      : {len(train_raws)}", flush=True)
    print(f"Val raws        : {len(val_raws)}", flush=True)
    print(f"Train samples   : {len(train_indices)}", flush=True)
    print(f"Val samples     : {len(val_indices)}", flush=True)
    print(f"Teacher key     : {args.teacher_key}", flush=True)
    print(f"Use sequence    : {use_sequence}", flush=True)
    print(f"Freeze teacher  : {args.freeze_teacher_head}", flush=True)
    print(f"Freeze student  : {args.freeze_student}", flush=True)
    print(
        f"Auth codebook   : {auth_codebook is not None} "
        f"(size={args.teacher_codebook_size}, temp={args.teacher_codebook_temperature}, "
        f"mode={args.codebook_mode}, freeze={args.freeze_codebook})",
        flush=True,
    )
    print(
        f"Claim verifier  : {claim_verifier is not None} "
        f"(hidden={args.claim_verifier_hidden_dim}, drop={args.claim_verifier_dropout}, "
        f"weight={args.claim_verifier_weight}, input={args.claim_verifier_input_mode}, "
        f"mode={args.claim_verifier_score_mode}, fusion_aux={args.claim_verifier_feature_to_fusion}, "
        f"freeze={args.freeze_claim_verifier})",
        flush=True,
    )
    print(
        f"Student code hd : {student_code_head is not None} "
        f"(hidden={args.student_code_head_hidden_dim}, drop={args.student_code_head_dropout}, "
        f"freeze={args.freeze_student_code_head})",
        flush=True,
    )
    print(
        f"Student token hd: {student_token_head is not None} "
        f"(variant={args.auth_protocol_variant}, classes={args.teacher_token_classes}, "
        f"hidden={args.student_token_head_hidden_dim}, drop={args.student_token_head_dropout}, "
        f"freeze={args.freeze_student_token_head})",
        flush=True,
    )
    if selected_teacher_token_info is not None:
        print(
            f"Teacher token sw: classes={selected_teacher_token_info['num_tokens']}, "
            f"eer={selected_teacher_token_info['metrics']['eer']:.4f}, "
            f"auc={selected_teacher_token_info['metrics']['pairwise_auc']:.4f}",
            flush=True,
        )
    print(
        f"Score fusion hd : {claim_score_fusion_head is not None} "
        f"(hidden={args.claim_score_fusion_hidden_dim}, drop={args.claim_score_fusion_dropout}, "
        f"input_dim={claim_score_fusion_input_dim(args.claim_score_fusion_protocol_modes, include_token_score=token_protocol_enabled, include_verifier_score=args.claim_verifier_feature_to_fusion and claim_verifier is not None)}, "
        f"extra_modes={normalize_protocol_mode_list(args.claim_score_fusion_protocol_modes)}, "
        f"mode={args.claim_score_fusion_mode}, residual_scale={args.claim_score_fusion_residual_scale}, "
        f"freeze={args.freeze_claim_score_fusion_head})",
        flush=True,
    )
    print(
        f"Claim reference : {args.claim_reference_mode} "
        f"(versions={args.claim_reference_versions if args.claim_reference_versions is not None else args.anchor_versions}, "
        f"bank_mode={args.claim_bank_mode}, aux_w={args.claim_anchor_aux_weight})",
        flush=True,
    )
    print(
        f"Protocol score  : mode={args.protocol_score_mode}, alpha={args.protocol_score_alpha}, "
        f"main_norm={args.claim_main_score_norm_mode}, proto_norm={args.protocol_score_norm_mode}, "
        f"official={args.official_claim_score_mode}, gate_penalty={args.token_gate_penalty}, "
        f"token_residual_w={args.token_residual_weight}, use_joint_loss={args.use_joint_claim_loss}",
        flush=True,
    )
    print(
        f"Official preset : {official_eval_preset} ({official_config_resolved})",
        flush=True,
    )
    print(f"Claim seq weight: {claim_sequence_score_weight}", flush=True)
    print(
        f"Student target  : {args.student_target_mode} "
        f"(blend_alpha={args.student_target_blend_alpha})",
        flush=True,
    )
    print(
        f"Loss weights    : teacher_cons={args.teacher_consistency_weight}, teacher_pair={args.teacher_pair_weight}, "
        f"teacher_claim_pair={args.teacher_claim_pair_weight}, teacher_claim_hard={args.teacher_claim_hard_weight}, "
        f"student_bit={args.student_bit_weight}, student_pair={args.student_pair_weight}, "
        f"student_claim_bank_pair={args.student_claim_bank_pair_weight}, student_hard={args.student_hard_weight}, "
        f"student_claim_bank_hard={args.student_claim_bank_hard_weight}, "
        f"student_verifier_claim_pair={args.student_verifier_claim_pair_weight}, "
        f"student_claim_calib_pair={args.student_claim_calibration_pair_weight}, "
        f"student_verifier_claim_hard={args.student_verifier_claim_hard_weight}, "
        f"student_verifier_claim_bank_hard={args.student_verifier_claim_bank_hard_weight}, "
        f"student_claim_calib_hard={args.student_claim_calibration_hard_weight}, "
        f"student_claim_eer_proxy={args.student_claim_eer_proxy_weight}, "
        f"student_claim_tail={args.student_claim_positive_tail_weight}, "
        f"student_claim_hardcase_rw={args.student_claim_hardcase_reweight_strength}, "
        f"student_proto_claim_pair={args.student_protocol_claim_pair_weight}, "
        f"student_proto_claim_bank_pair={args.student_protocol_claim_bank_pair_weight}, "
        f"student_proto_claim_hard={args.student_protocol_claim_hard_weight}, "
        f"student_proto_claim_bank_hard={args.student_protocol_claim_bank_hard_weight}, "
        f"student_score_distill={args.student_score_distill_weight}, "
        f"student_hard_score_distill={args.student_hard_score_distill_weight}, "
        f"student_soft={args.student_soft_align_weight}, recovery_bit={args.student_recovery_bit_weight}, "
        f"recovery_stage={args.student_recovery_stage_weight}, recovery_align={args.student_recovery_align_weight}, "
        f"teacher_codebook_commit={args.teacher_codebook_commit_weight}, "
        f"student_codebook_class={args.student_codebook_class_weight}, "
        f"student_codebook_proto={args.student_codebook_proto_weight}, "
        f"codebook_usage={args.codebook_usage_weight}, codebook_separation={args.codebook_separation_weight}, "
        f"teacher_balance={args.teacher_balance_weight}, "
        f"teacher_decor={args.teacher_decorrelation_weight}, teacher_uniform={args.teacher_uniformity_weight}",
        flush=True,
    )
    print(f"Bit scale       : {args.bit_scale}", flush=True)
    print(f"Selection metric: {args.selection_metric}", flush=True)
    if args.claim_calibration_scales:
        print(
            f"Claim calib     : scales={args.claim_calibration_scales}, "
            f"verifier_w={args.claim_calibration_verifier_weights if claim_verifier is not None else [0.0]}, "
            f"seq_w={args.claim_calibration_sequence_weights if args.claim_calibration_sequence_weights is not None else [claim_sequence_score_weight]}",
            flush=True,
        )
    print("=" * 72, flush=True)

    history = []
    best_metric = float("-inf")

    for epoch in range(args.epochs):
        if hasattr(train_loader.batch_sampler, "set_epoch"):
            train_loader.batch_sampler.set_epoch(epoch)

        teacher_head.train()
        if args.freeze_teacher_head:
            teacher_head.eval()
        student.train()
        if args.freeze_student:
            student.eval()
        if student_code_head is not None:
            student_code_head.train()
            if args.freeze_student_code_head:
                student_code_head.eval()
        if student_token_head is not None:
            student_token_head.train()
            if args.freeze_student_token_head:
                student_token_head.eval()
        if claim_score_fusion_head is not None:
            claim_score_fusion_head.train()
            if args.freeze_claim_score_fusion_head:
                claim_score_fusion_head.eval()
        if claim_verifier is not None:
            claim_verifier.train()
        if auth_codebook is not None and args.freeze_codebook:
            auth_codebook.eval()
        train_loss = 0.0
        train_teacher_pos_scores = []
        train_teacher_hard_neg_scores = []
        train_teacher_all_neg_scores = []
        train_student_pos_scores = []
        train_student_hard_neg_scores = []
        train_student_all_neg_scores = []
        train_recovery_codeword_ber = []
        train_recovery_payload_ber = []
        train_recovery_decode_success = []
        train_recovery_code_index_match = []
        train_student_claim_pos_scores = []
        train_student_claim_hard_neg_scores = []
        train_student_claim_all_neg_scores = []
        train_student_protocol_claim_pos_scores = []
        train_student_protocol_claim_hard_neg_scores = []
        train_student_protocol_claim_all_neg_scores = []
        train_student_claim_joint_pos_scores = []
        train_student_claim_joint_hard_neg_scores = []
        train_student_claim_joint_all_neg_scores = []
        train_student_claim_official_pos_scores = []
        train_student_claim_official_hard_neg_scores = []
        train_student_claim_official_all_neg_scores = []
        train_student_anchor_claim_pos_scores = []
        train_student_anchor_claim_hard_neg_scores = []
        train_student_anchor_claim_all_neg_scores = []
        train_claim_bank_bits = None
        train_claim_bank_raws = None
        use_anchor_reference = args.claim_reference_mode == "anchor_bank" or float(args.claim_anchor_aux_weight) > 0.0
        if use_anchor_reference:
            train_claim_bank_bits, train_claim_bank_raws = build_claim_reference_bank(
                dataset=train_dataset,
                indices=train_indices,
                reference_raws=train_raws,
                teacher_head=teacher_head,
                use_sequence_score=use_sequence,
                bit_scale=args.bit_scale,
                device=device,
                reference_versions=args.claim_reference_versions or args.anchor_versions,
                claim_bank_mode=args.claim_bank_mode,
                auth_codec=auth_codec,
                auth_codebook=auth_codebook,
                apply_codebook_to_scores=apply_codebook_to_scores,
            )

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Train]")
        for batch in pbar:
            rgb_image = batch["rgb_image"].to(device, non_blocking=True)
            teacher_vec = batch["teacher_vec"].to(device, non_blocking=True)
            teacher_seq = batch["teacher_seq"].to(device, non_blocking=True)
            raw_anchors = list(batch["raw_anchor"])
            versions = batch["version"].to(device, non_blocking=True)
            sample_weights = compute_sample_weights(
                versions,
                anchor_versions=args.anchor_versions,
                shift_versions=args.shift_versions,
                anchor_weight=args.anchor_weight,
                shift_weight=args.shift_weight,
            )

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                bank_out, bank_bits, bank_stage_bits = forward_teacher_bank(
                    teacher_head,
                    train_proto_vec_bank,
                    train_proto_seq_bank,
                    bit_scale=args.bit_scale,
                    use_sequence_score=use_sequence,
                    auth_codec=auth_codec,
                    auth_codebook=auth_codebook,
                    apply_codebook_to_scores=apply_codebook_to_scores,
                )
                bank_logits = bank_out["global_logits"]
                bank_stage_logits = bank_out.get("stage_logits")
                bank_token_indices = None
                bank_token_logits = None
                if teacher_tokenizer is not None:
                    bank_token_indices, bank_token_logits = teacher_tokenizer.assign(bank_logits.detach())
                targets = build_bank_targets(raw_anchors, train_bank_raws, device=device)

                teacher_sample_out = teacher_head(
                    teacher_vec,
                    teacher_seq,
                    return_sequence=use_sequence,
                    return_logits=True,
                )
                teacher_sample_encoded_logits = encode_teacher_logits(teacher_sample_out["global_logits"])
                teacher_code_indices = None
                teacher_code_prototypes = None
                if auth_codebook is not None:
                    teacher_quantized_logits, teacher_code_indices, _, teacher_code_prototypes = auth_codebook.quantize(
                        teacher_sample_encoded_logits,
                        straight_through=True,
                    )
                    teacher_sample_logits = teacher_quantized_logits if apply_codebook_to_scores else teacher_sample_encoded_logits
                else:
                    teacher_sample_logits = teacher_sample_encoded_logits
                teacher_sample_stage_logits = encode_teacher_logits(teacher_sample_out.get("stage_logits"))
                teacher_sample_stage_bits = (
                    soft_bits(teacher_sample_stage_logits, args.bit_scale)
                    if teacher_sample_stage_logits is not None
                    else None
                )
                teacher_token_indices = None
                teacher_token_logits = None
                if teacher_tokenizer is not None:
                    teacher_token_indices, teacher_token_logits = teacher_tokenizer.assign(
                        teacher_sample_encoded_logits.detach()
                    )
                teacher_sample_scores, teacher_sample_bits = compute_combined_scores(
                    teacher_sample_logits,
                    bank_bits,
                    query_stage_logits=teacher_sample_stage_logits,
                    bank_stage_bits=bank_stage_bits,
                    bit_scale=args.bit_scale,
                    sequence_score_weight=args.sequence_score_weight,
                )

                student_forward = forward_student_branches(
                    student=student,
                    rgb_image=rgb_image,
                    use_sequence=use_sequence,
                    encode_student_logits=encode_student_logits,
                    auth_codebook=auth_codebook,
                    apply_codebook_to_scores=apply_codebook_to_scores,
                    student_code_head=student_code_head,
                    student_token_head=student_token_head,
                )
                student_encoded_logits = student_forward["score_encoded_logits"]
                student_logits = student_forward["score_logits"]
                student_stage_logits = student_forward["stage_logits"]
                student_recovery_logits = student_forward["recovery_logits"]
                student_recovery_protocol_logits = student_forward["recovery_protocol_logits"]
                student_code_indices = student_forward["recovery_code_indices"]
                student_code_prototypes = student_forward["recovery_code_prototypes"]
                student_token_logits = student_forward["token_logits"]
                if teacher_tokenizer is not None:
                    base_student_token_logits = teacher_tokenizer.logits(student_recovery_logits)
                    if student_token_logits is None:
                        student_token_logits = base_student_token_logits
                    else:
                        student_token_logits = base_student_token_logits + student_token_logits
                student_scores, student_bits = compute_combined_scores(
                    student_logits,
                    bank_bits,
                    query_stage_logits=student_stage_logits,
                    bank_stage_bits=bank_stage_bits,
                    bit_scale=args.bit_scale,
                    sequence_score_weight=args.sequence_score_weight,
                )
                student_verifier_repr = build_claim_verifier_repr(
                    student_bits,
                    student_logits,
                    args.claim_verifier_input_mode,
                )
                teacher_sample_verifier_repr = build_claim_verifier_repr(
                    teacher_sample_bits,
                    teacher_sample_logits.detach(),
                    args.claim_verifier_input_mode,
                )
                bank_verifier_repr = build_claim_verifier_repr(
                    bank_bits.detach(),
                    bank_logits.detach(),
                    args.claim_verifier_input_mode,
                )

                sample_ids = list(batch["sample_id"])
                teacher_claim_protocol_logits = (
                    teacher_code_prototypes.detach() if teacher_code_prototypes is not None else teacher_sample_encoded_logits.detach()
                )
                teacher_claim_code_indices = teacher_code_indices.detach() if teacher_code_indices is not None else None
                if args.claim_reference_mode == "anchor_bank":
                    claim_targets = build_bank_targets(raw_anchors, train_claim_bank_raws, device=device)
                    claim_target_bits = train_claim_bank_bits[claim_targets].detach()
                    student_claim_scores = compute_claim_scores(
                        student_bits,
                        train_claim_bank_bits.detach(),
                        claim_verifier,
                        verifier_weight=args.claim_verifier_weight,
                        verifier_score_mode=args.claim_verifier_score_mode,
                    )
                    teacher_claim_scores = compute_claim_scores(
                        teacher_sample_bits,
                        train_claim_bank_bits.detach(),
                        claim_verifier,
                        verifier_weight=args.claim_verifier_weight,
                        verifier_score_mode=args.claim_verifier_score_mode,
                    )
                    student_claim_official_scores = student_claim_scores
                    teacher_claim_official_scores = teacher_claim_scores
                    claim_positive_mask = None
                    claim_negative_mask = None
                elif args.claim_reference_mode == "same_image":
                    student_claim_scores = compute_claim_scores(
                        student_bits,
                        teacher_sample_bits.detach(),
                        claim_verifier,
                        verifier_weight=args.claim_verifier_weight,
                        verifier_score_mode=args.claim_verifier_score_mode,
                        query_verifier_repr=student_verifier_repr,
                        reference_verifier_repr=teacher_sample_verifier_repr.detach(),
                        query_stage_logits=student_stage_logits,
                        reference_stage_logits=teacher_sample_stage_logits.detach() if teacher_sample_stage_logits is not None else None,
                        reference_stage_bits=teacher_sample_stage_bits.detach() if teacher_sample_stage_bits is not None else None,
                        bit_scale=args.bit_scale,
                        sequence_score_weight=claim_sequence_score_weight,
                    )
                    teacher_claim_scores = compute_claim_scores(
                        teacher_sample_bits,
                        teacher_sample_bits.detach(),
                        claim_verifier,
                        verifier_weight=args.claim_verifier_weight,
                        verifier_score_mode=args.claim_verifier_score_mode,
                        query_verifier_repr=teacher_sample_verifier_repr,
                        reference_verifier_repr=teacher_sample_verifier_repr.detach(),
                        query_stage_logits=teacher_sample_stage_logits,
                        reference_stage_logits=teacher_sample_stage_logits.detach() if teacher_sample_stage_logits is not None else None,
                        reference_stage_bits=teacher_sample_stage_bits.detach() if teacher_sample_stage_bits is not None else None,
                        bit_scale=args.bit_scale,
                        sequence_score_weight=claim_sequence_score_weight,
                    )
                    claim_targets = torch.arange(student_claim_scores.shape[0], device=device)
                    claim_target_bits = teacher_sample_bits.detach()
                    claim_positive_mask = build_same_sample_mask(sample_ids, device=device)
                    claim_negative_mask = ~build_same_raw_mask(raw_anchors, device=device)
                else:
                    student_claim_scores = compute_claim_scores(
                        student_bits,
                        teacher_sample_bits.detach(),
                        claim_verifier,
                        verifier_weight=args.claim_verifier_weight,
                        verifier_score_mode=args.claim_verifier_score_mode,
                        query_verifier_repr=student_verifier_repr,
                        reference_verifier_repr=teacher_sample_verifier_repr.detach(),
                        query_stage_logits=student_stage_logits,
                        reference_stage_logits=teacher_sample_stage_logits.detach() if teacher_sample_stage_logits is not None else None,
                        reference_stage_bits=teacher_sample_stage_bits.detach() if teacher_sample_stage_bits is not None else None,
                        bit_scale=args.bit_scale,
                        sequence_score_weight=claim_sequence_score_weight,
                    )
                    teacher_claim_scores = compute_claim_scores(
                        teacher_sample_bits,
                        teacher_sample_bits.detach(),
                        claim_verifier,
                        verifier_weight=args.claim_verifier_weight,
                        verifier_score_mode=args.claim_verifier_score_mode,
                        query_verifier_repr=teacher_sample_verifier_repr,
                        reference_verifier_repr=teacher_sample_verifier_repr.detach(),
                        query_stage_logits=teacher_sample_stage_logits,
                        reference_stage_logits=teacher_sample_stage_logits.detach() if teacher_sample_stage_logits is not None else None,
                        reference_stage_bits=teacher_sample_stage_bits.detach() if teacher_sample_stage_bits is not None else None,
                        bit_scale=args.bit_scale,
                        sequence_score_weight=claim_sequence_score_weight,
                    )
                    claim_targets = torch.arange(student_claim_scores.shape[0], device=device)
                    claim_target_bits = teacher_sample_bits.detach()
                    claim_positive_mask = build_same_raw_mask(raw_anchors, device=device)
                    claim_negative_mask = None
                if args.claim_reference_mode == "anchor_bank":
                    student_verifier_claim_scores = compute_claim_verifier_scores(
                        student_bits,
                        train_claim_bank_bits.detach(),
                        claim_verifier,
                        verifier_weight=args.claim_verifier_weight,
                        verifier_score_mode=args.claim_verifier_score_mode,
                    )
                    teacher_verifier_claim_scores = compute_claim_verifier_scores(
                        teacher_sample_bits,
                        train_claim_bank_bits.detach(),
                        claim_verifier,
                        verifier_weight=args.claim_verifier_weight,
                        verifier_score_mode=args.claim_verifier_score_mode,
                    )
                else:
                    student_verifier_claim_scores = compute_claim_verifier_scores(
                        student_bits,
                        teacher_sample_bits.detach(),
                        claim_verifier,
                        verifier_weight=args.claim_verifier_weight,
                        verifier_score_mode=args.claim_verifier_score_mode,
                        query_verifier_repr=student_verifier_repr,
                        reference_verifier_repr=teacher_sample_verifier_repr.detach(),
                    )
                    teacher_verifier_claim_scores = compute_claim_verifier_scores(
                        teacher_sample_bits,
                        teacher_sample_bits.detach(),
                        claim_verifier,
                        verifier_weight=args.claim_verifier_weight,
                        verifier_score_mode=args.claim_verifier_score_mode,
                        query_verifier_repr=teacher_sample_verifier_repr,
                        reference_verifier_repr=teacher_sample_verifier_repr.detach(),
                    )
                student_token_claim_scores = None
                teacher_token_claim_scores = None
                if teacher_tokenizer is not None and args.claim_reference_mode != "anchor_bank":
                    student_token_claim_scores = compute_token_match_matrix(
                        student_token_logits,
                        teacher_token_indices.detach() if teacher_token_indices is not None else None,
                    )
                    teacher_token_claim_scores = compute_token_match_matrix(
                        teacher_token_logits,
                        teacher_token_indices.detach() if teacher_token_indices is not None else None,
                    )
                student_protocol_claim_scores = None
                teacher_protocol_claim_scores = None
                student_extra_protocol_claim_scores = []
                teacher_extra_protocol_claim_scores = []
                if args.claim_reference_mode != "anchor_bank":
                    student_protocol_claim_scores, student_extra_protocol_claim_scores = compute_protocol_score_bundle(
                        query_protocol_logits=student_recovery_protocol_logits,
                        reference_protocol_logits=teacher_claim_protocol_logits,
                        auth_codec=auth_codec,
                        primary_mode=args.protocol_score_mode,
                        extra_modes=args.claim_score_fusion_protocol_modes,
                        query_code_indices=student_code_indices,
                        reference_code_indices=teacher_claim_code_indices,
                    )
                    teacher_protocol_claim_scores, teacher_extra_protocol_claim_scores = compute_protocol_score_bundle(
                        query_protocol_logits=teacher_claim_protocol_logits,
                        reference_protocol_logits=teacher_claim_protocol_logits,
                        auth_codec=auth_codec,
                        primary_mode=args.protocol_score_mode,
                        extra_modes=args.claim_score_fusion_protocol_modes,
                        query_code_indices=teacher_claim_code_indices,
                        reference_code_indices=teacher_claim_code_indices,
                    )
                if args.claim_verifier_feature_to_fusion and student_verifier_claim_scores is not None:
                    student_extra_protocol_claim_scores.append(student_verifier_claim_scores)
                    teacher_extra_protocol_claim_scores.append(teacher_verifier_claim_scores)
                student_claim_score_outputs = build_claim_score_outputs(
                    student_claim_scores,
                    student_protocol_claim_scores,
                    token_scores=student_token_claim_scores,
                    extra_protocol_scores=student_extra_protocol_claim_scores,
                    claim_score_fusion_head=claim_score_fusion_head,
                    official_mode=args.official_claim_score_mode,
                    alpha=args.protocol_score_alpha,
                    main_normalization=args.claim_main_score_norm_mode,
                    auxiliary_normalization=args.protocol_score_norm_mode,
                    gate_penalty=args.token_gate_penalty,
                    residual_weight=args.token_residual_weight,
                    hard_gate_threshold=args.token_hard_gate_threshold,
                )
                teacher_claim_score_outputs = build_claim_score_outputs(
                    teacher_claim_scores,
                    teacher_protocol_claim_scores,
                    token_scores=teacher_token_claim_scores,
                    extra_protocol_scores=teacher_extra_protocol_claim_scores,
                    claim_score_fusion_head=claim_score_fusion_head,
                    official_mode=args.official_claim_score_mode,
                    alpha=args.protocol_score_alpha,
                    main_normalization=args.claim_main_score_norm_mode,
                    auxiliary_normalization=args.protocol_score_norm_mode,
                    gate_penalty=args.token_gate_penalty,
                    residual_weight=args.token_residual_weight,
                    hard_gate_threshold=args.token_hard_gate_threshold,
                )
                student_claim_joint_scores = student_claim_score_outputs["fusion_scores"]
                teacher_claim_joint_scores = teacher_claim_score_outputs["fusion_scores"]
                student_claim_official_scores = student_claim_score_outputs["official_scores"]
                teacher_claim_official_scores = teacher_claim_score_outputs["official_scores"]
                claim_loss_weights = compute_optional_claim_hardcase_weights(
                    student_claim_official_scores,
                    claim_reference_mode=args.claim_reference_mode,
                    sample_weights=sample_weights,
                    strength=args.student_claim_hardcase_reweight_strength,
                    margin=args.student_claim_hardcase_reweight_margin,
                    scale=args.student_claim_hardcase_reweight_scale,
                    claim_targets=claim_targets,
                    claim_positive_mask=claim_positive_mask,
                    claim_negative_mask=claim_negative_mask,
                )
                use_joint_claim_scores = args.use_joint_claim_loss or teacher_tokenizer is not None or claim_score_fusion_head is not None
                student_claim_loss_scores = student_claim_official_scores if use_joint_claim_scores else student_claim_scores
                teacher_claim_loss_scores = teacher_claim_official_scores if use_joint_claim_scores else teacher_claim_scores

                teacher_target_logits = bank_logits[targets]
                teacher_target_stage_logits = bank_stage_logits[targets] if bank_stage_logits is not None else None
                teacher_target_bits = bank_bits[targets]
                teacher_recovery_target_logits = (
                    teacher_code_prototypes.detach() if teacher_code_prototypes is not None else teacher_sample_encoded_logits.detach()
                )
                teacher_recovery_target_bits = soft_bits(teacher_recovery_target_logits, args.bit_scale)
                student_target_logits, student_target_stage_logits, student_target_bits = resolve_student_match_targets(
                    canonical_logits=teacher_target_logits,
                    sample_logits=teacher_sample_logits,
                    canonical_stage_logits=teacher_target_stage_logits,
                    sample_stage_logits=teacher_sample_stage_logits,
                    bit_scale=args.bit_scale,
                    target_mode=args.student_target_mode,
                    blend_alpha=args.student_target_blend_alpha,
                )

                teacher_consistency_loss = compute_bit_match_loss(teacher_sample_logits, teacher_target_logits, sample_weights)
                teacher_sequence_match_loss = compute_stage_bit_match_loss(
                    teacher_sample_stage_logits,
                    teacher_target_stage_logits,
                    sample_weights,
                )
                teacher_pair_loss = compute_pair_logistic_loss(
                    teacher_sample_scores,
                    targets,
                    sample_weights,
                    positive_margin=args.teacher_positive_margin,
                    negative_margin=args.teacher_negative_margin,
                    scale=args.pair_logit_scale,
                    topk=args.pair_logistic_topk,
                )

                student_bit_loss = compute_bit_match_loss(student_logits, student_target_logits, sample_weights)
                if args.claim_reference_mode == "anchor_bank":
                    student_claim_bit_loss = compute_soft_alignment_loss(student_bits, claim_target_bits, sample_weights)
                else:
                    student_claim_bit_loss = compute_bit_match_loss(student_logits, teacher_sample_logits, sample_weights)
                student_sequence_match_loss = compute_stage_bit_match_loss(
                    student_stage_logits,
                    student_target_stage_logits,
                    sample_weights,
                )
                student_pair_loss = compute_pair_logistic_loss(
                    student_scores,
                    targets,
                    sample_weights,
                    positive_margin=args.student_positive_margin,
                    negative_margin=args.student_negative_margin,
                    scale=args.pair_logit_scale,
                    topk=args.pair_logistic_topk,
                )
                student_hard_loss = compute_hard_margin_loss(
                    student_scores,
                    targets,
                    sample_weights,
                    margin=args.student_hard_margin,
                )
                student_claim_bank_pair_loss = sample_weights.new_tensor(0.0)
                student_claim_bank_hard_loss = sample_weights.new_tensor(0.0)
                student_protocol_claim_bank_pair_loss = sample_weights.new_tensor(0.0)
                student_protocol_claim_bank_hard_loss = sample_weights.new_tensor(0.0)
                student_verifier_claim_pair_loss = sample_weights.new_tensor(0.0)
                student_verifier_claim_hard_loss = sample_weights.new_tensor(0.0)
                student_verifier_claim_bank_hard_loss = sample_weights.new_tensor(0.0)
                if (
                    float(args.student_claim_bank_pair_weight) > 0.0
                    or float(args.student_claim_bank_hard_weight) > 0.0
                    or float(args.student_protocol_claim_bank_pair_weight) > 0.0
                    or float(args.student_protocol_claim_bank_hard_weight) > 0.0
                    or float(args.student_verifier_claim_bank_hard_weight) > 0.0
                ):
                    student_claim_bank_scores = compute_claim_scores(
                        student_bits,
                        bank_bits.detach(),
                        claim_verifier,
                        verifier_weight=args.claim_verifier_weight,
                        verifier_score_mode=args.claim_verifier_score_mode,
                        query_verifier_repr=student_verifier_repr,
                        reference_verifier_repr=bank_verifier_repr.detach(),
                        query_stage_logits=student_stage_logits,
                        reference_stage_logits=bank_stage_logits.detach() if bank_stage_logits is not None else None,
                        reference_stage_bits=bank_stage_bits.detach() if bank_stage_bits is not None else None,
                        bit_scale=args.bit_scale,
                        sequence_score_weight=claim_sequence_score_weight,
                    )
                    student_verifier_claim_bank_scores = compute_claim_verifier_scores(
                        student_bits,
                        bank_bits.detach(),
                        claim_verifier,
                        verifier_weight=args.claim_verifier_weight,
                        verifier_score_mode=args.claim_verifier_score_mode,
                        query_verifier_repr=student_verifier_repr,
                        reference_verifier_repr=bank_verifier_repr.detach(),
                    )
                    student_claim_bank_protocol_scores, student_claim_bank_extra_protocol_scores = compute_protocol_score_bundle(
                        query_protocol_logits=student_recovery_protocol_logits,
                        reference_protocol_logits=bank_logits.detach(),
                        auth_codec=auth_codec,
                        primary_mode=args.protocol_score_mode,
                        extra_modes=args.claim_score_fusion_protocol_modes,
                    )
                    student_claim_bank_token_scores = compute_token_match_matrix(
                        student_token_logits,
                        bank_token_indices.detach() if bank_token_indices is not None else None,
                    )
                    if args.claim_verifier_feature_to_fusion and student_verifier_claim_bank_scores is not None:
                        student_claim_bank_extra_protocol_scores.append(student_verifier_claim_bank_scores)
                    student_claim_bank_score_outputs = build_claim_score_outputs(
                        student_claim_bank_scores,
                        student_claim_bank_protocol_scores,
                        token_scores=student_claim_bank_token_scores,
                        extra_protocol_scores=student_claim_bank_extra_protocol_scores,
                        claim_score_fusion_head=claim_score_fusion_head,
                        official_mode=args.official_claim_score_mode,
                        alpha=args.protocol_score_alpha,
                        main_normalization=args.claim_main_score_norm_mode,
                        auxiliary_normalization=args.protocol_score_norm_mode,
                        gate_penalty=args.token_gate_penalty,
                        residual_weight=args.token_residual_weight,
                        hard_gate_threshold=args.token_hard_gate_threshold,
                    )
                    student_claim_bank_loss_scores = (
                        student_claim_bank_score_outputs["official_scores"] if use_joint_claim_scores else student_claim_bank_scores
                    )
                    student_claim_bank_pair_loss = compute_pair_logistic_loss(
                        student_claim_bank_loss_scores,
                        targets,
                        sample_weights,
                        positive_margin=args.student_positive_margin,
                        negative_margin=args.student_negative_margin,
                        scale=args.pair_logit_scale,
                        topk=args.pair_logistic_topk,
                    )
                    student_claim_bank_hard_loss = compute_hard_margin_loss(
                        student_claim_bank_loss_scores,
                        targets,
                        sample_weights,
                        margin=args.student_hard_margin,
                    )
                    if student_claim_bank_protocol_scores is not None:
                        student_protocol_claim_bank_pair_loss = compute_pair_logistic_loss(
                            student_claim_bank_protocol_scores,
                            targets,
                            sample_weights,
                            positive_margin=args.student_positive_margin,
                            negative_margin=args.student_negative_margin,
                            scale=args.pair_logit_scale,
                            topk=args.pair_logistic_topk,
                        )
                        student_protocol_claim_bank_hard_loss = compute_hard_margin_loss(
                            student_claim_bank_protocol_scores,
                            targets,
                            sample_weights,
                            margin=args.student_hard_margin,
                        )
                    if student_verifier_claim_bank_scores is not None:
                        student_verifier_claim_bank_hard_loss = compute_hard_margin_loss(
                            student_verifier_claim_bank_scores,
                            targets,
                            sample_weights,
                            margin=args.student_hard_margin,
                        )
                if args.claim_reference_mode == "anchor_bank":
                    teacher_claim_pair_loss = compute_pair_logistic_loss(
                        teacher_claim_loss_scores,
                        claim_targets,
                        sample_weights,
                        positive_margin=args.teacher_positive_margin,
                        negative_margin=args.teacher_negative_margin,
                        scale=args.pair_logit_scale,
                        topk=args.pair_logistic_topk,
                    )
                    teacher_claim_hard_loss = compute_hard_margin_loss(
                        teacher_claim_loss_scores,
                        claim_targets,
                        sample_weights,
                        margin=args.teacher_hard_margin,
                    )
                    student_claim_pair_loss = compute_pair_logistic_loss(
                        student_claim_loss_scores,
                        claim_targets,
                        claim_loss_weights,
                        positive_margin=args.student_positive_margin,
                        negative_margin=args.student_negative_margin,
                        scale=args.pair_logit_scale,
                        topk=args.pair_logistic_topk,
                    )
                    student_claim_hard_loss = compute_hard_margin_loss(
                        student_claim_loss_scores,
                        claim_targets,
                        claim_loss_weights,
                        margin=args.student_hard_margin,
                    )
                    if student_protocol_claim_scores is not None:
                        student_protocol_claim_pair_loss = compute_pair_logistic_loss(
                            student_protocol_claim_scores,
                            claim_targets,
                            claim_loss_weights,
                            positive_margin=args.student_positive_margin,
                            negative_margin=args.student_negative_margin,
                            scale=args.pair_logit_scale,
                            topk=args.pair_logistic_topk,
                        )
                        student_protocol_claim_hard_loss = compute_hard_margin_loss(
                            student_protocol_claim_scores,
                            claim_targets,
                            claim_loss_weights,
                            margin=args.student_hard_margin,
                        )
                    else:
                        student_protocol_claim_pair_loss = sample_weights.new_tensor(0.0)
                        student_protocol_claim_hard_loss = sample_weights.new_tensor(0.0)
                elif args.claim_reference_mode == "same_image":
                    teacher_claim_pair_loss = compute_masked_pair_logistic_loss(
                        teacher_claim_loss_scores,
                        claim_positive_mask,
                        sample_weights,
                        negative_mask=claim_negative_mask,
                        positive_margin=args.teacher_positive_margin,
                        negative_margin=args.teacher_negative_margin,
                        scale=args.pair_logit_scale,
                        topk=args.pair_logistic_topk,
                    )
                    teacher_claim_hard_loss = compute_masked_hard_margin_loss(
                        teacher_claim_loss_scores,
                        claim_positive_mask,
                        sample_weights,
                        negative_mask=claim_negative_mask,
                        margin=args.teacher_hard_margin,
                    )
                    student_claim_pair_loss = compute_masked_pair_logistic_loss(
                        student_claim_loss_scores,
                        claim_positive_mask,
                        claim_loss_weights,
                        negative_mask=claim_negative_mask,
                        positive_margin=args.student_positive_margin,
                        negative_margin=args.student_negative_margin,
                        scale=args.pair_logit_scale,
                        topk=args.pair_logistic_topk,
                    )
                    student_claim_hard_loss = compute_masked_hard_margin_loss(
                        student_claim_loss_scores,
                        claim_positive_mask,
                        claim_loss_weights,
                        negative_mask=claim_negative_mask,
                        margin=args.student_hard_margin,
                    )
                    if student_protocol_claim_scores is not None:
                        student_protocol_claim_pair_loss = compute_masked_pair_logistic_loss(
                            student_protocol_claim_scores,
                            claim_positive_mask,
                            claim_loss_weights,
                            negative_mask=claim_negative_mask,
                            positive_margin=args.student_positive_margin,
                            negative_margin=args.student_negative_margin,
                            scale=args.pair_logit_scale,
                            topk=args.pair_logistic_topk,
                        )
                        student_protocol_claim_hard_loss = compute_masked_hard_margin_loss(
                            student_protocol_claim_scores,
                            claim_positive_mask,
                            claim_loss_weights,
                            negative_mask=claim_negative_mask,
                            margin=args.student_hard_margin,
                        )
                    else:
                        student_protocol_claim_pair_loss = sample_weights.new_tensor(0.0)
                        student_protocol_claim_hard_loss = sample_weights.new_tensor(0.0)
                else:
                    teacher_claim_pair_loss = compute_masked_pair_logistic_loss(
                        teacher_claim_loss_scores,
                        claim_positive_mask,
                        sample_weights,
                        positive_margin=args.teacher_positive_margin,
                        negative_margin=args.teacher_negative_margin,
                        scale=args.pair_logit_scale,
                        topk=args.pair_logistic_topk,
                    )
                    teacher_claim_hard_loss = compute_masked_hard_margin_loss(
                        teacher_claim_loss_scores,
                        claim_positive_mask,
                        sample_weights,
                        margin=args.teacher_hard_margin,
                    )
                    student_claim_pair_loss = compute_masked_pair_logistic_loss(
                        student_claim_loss_scores,
                        claim_positive_mask,
                        claim_loss_weights,
                        positive_margin=args.student_positive_margin,
                        negative_margin=args.student_negative_margin,
                        scale=args.pair_logit_scale,
                        topk=args.pair_logistic_topk,
                    )
                    student_claim_hard_loss = compute_masked_hard_margin_loss(
                        student_claim_loss_scores,
                        claim_positive_mask,
                        claim_loss_weights,
                        margin=args.student_hard_margin,
                    )
                    if student_protocol_claim_scores is not None:
                        student_protocol_claim_pair_loss = compute_masked_pair_logistic_loss(
                            student_protocol_claim_scores,
                            claim_positive_mask,
                            claim_loss_weights,
                            positive_margin=args.student_positive_margin,
                            negative_margin=args.student_negative_margin,
                            scale=args.pair_logit_scale,
                            topk=args.pair_logistic_topk,
                        )
                        student_protocol_claim_hard_loss = compute_masked_hard_margin_loss(
                            student_protocol_claim_scores,
                            claim_positive_mask,
                            claim_loss_weights,
                            margin=args.student_hard_margin,
                        )
                    else:
                        student_protocol_claim_pair_loss = sample_weights.new_tensor(0.0)
                        student_protocol_claim_hard_loss = sample_weights.new_tensor(0.0)
                student_verifier_claim_pair_loss, student_verifier_claim_hard_loss = compute_optional_claim_pair_hard_losses(
                    student_verifier_claim_scores,
                    claim_reference_mode=args.claim_reference_mode,
                    sample_weights=claim_loss_weights,
                    positive_margin=args.student_positive_margin,
                    negative_margin=args.student_negative_margin,
                    scale=args.pair_logit_scale,
                    topk=args.pair_logistic_topk,
                    margin=args.student_hard_margin,
                    claim_targets=claim_targets,
                    claim_positive_mask=claim_positive_mask,
                    claim_negative_mask=claim_negative_mask,
                )
                student_claim_calibration_pair_loss = sample_weights.new_tensor(0.0)
                student_claim_calibration_hard_loss = sample_weights.new_tensor(0.0)
                student_claim_eer_proxy_loss = sample_weights.new_tensor(0.0)
                student_claim_positive_tail_loss = sample_weights.new_tensor(0.0)
                if (
                    float(args.student_claim_calibration_pair_weight) > 0.0
                    or float(args.student_claim_calibration_hard_weight) > 0.0
                ):
                    student_claim_calibration_pair_loss, student_claim_calibration_hard_loss = (
                        compute_optional_claim_calibration_losses(
                            student_claim_official_scores,
                            claim_reference_mode=args.claim_reference_mode,
                            sample_weights=claim_loss_weights,
                            topk=args.pair_logistic_topk,
                            claim_targets=claim_targets,
                            claim_positive_mask=claim_positive_mask,
                            claim_negative_mask=claim_negative_mask,
                        )
                    )
                if float(args.student_claim_eer_proxy_weight) > 0.0:
                    student_claim_eer_proxy_loss = compute_optional_claim_operating_point_loss(
                        student_claim_official_scores,
                        claim_reference_mode=args.claim_reference_mode,
                        sample_weights=claim_loss_weights,
                        positive_quantile=args.student_claim_eer_proxy_positive_quantile,
                        negative_quantile=args.student_claim_eer_proxy_negative_quantile,
                        margin=args.student_claim_eer_proxy_margin,
                        scale=args.student_claim_eer_proxy_scale,
                        claim_targets=claim_targets,
                        claim_positive_mask=claim_positive_mask,
                        claim_negative_mask=claim_negative_mask,
                    )
                if float(args.student_claim_positive_tail_weight) > 0.0:
                    student_claim_positive_tail_loss = compute_optional_claim_positive_tail_rescue_loss(
                        student_claim_official_scores,
                        claim_reference_mode=args.claim_reference_mode,
                        sample_weights=claim_loss_weights,
                        positive_quantile=args.student_claim_positive_tail_positive_quantile,
                        negative_quantile=args.student_claim_positive_tail_negative_quantile,
                        margin=args.student_claim_positive_tail_margin,
                        scale=args.student_claim_positive_tail_scale,
                        claim_targets=claim_targets,
                        claim_positive_mask=claim_positive_mask,
                        claim_negative_mask=claim_negative_mask,
                    )
                student_soft_align_loss = compute_soft_alignment_loss(
                    student_bits,
                    student_target_bits,
                    sample_weights,
                )
                student_recovery_bit_loss = compute_logit_recovery_loss(
                    student_recovery_logits,
                    teacher_recovery_target_logits,
                    sample_weights,
                )
                if student_code_head is not None:
                    student_recovery_stage_loss = sample_weights.new_tensor(0.0)
                else:
                    student_recovery_stage_loss = compute_stage_logit_recovery_loss(
                        student_stage_logits,
                        teacher_sample_stage_logits,
                        sample_weights,
                    )
                student_recovery_align_loss = compute_soft_alignment_loss(
                    soft_bits(student_recovery_logits, args.bit_scale),
                    teacher_recovery_target_bits,
                    sample_weights,
                )
                student_token_class_loss = sample_weights.new_tensor(0.0)
                student_token_proto_loss = sample_weights.new_tensor(0.0)
                if teacher_tokenizer is not None and student_token_logits is not None and teacher_token_indices is not None:
                    token_ce = F.cross_entropy(
                        student_token_logits,
                        teacher_token_indices.detach(),
                        reduction="none",
                    )
                    student_token_class_loss = weighted_mean(token_ce, sample_weights)
                    student_token_proto_loss = compute_token_prototype_alignment_loss(
                        student_token_logits,
                        teacher_tokenizer,
                        teacher_token_indices.detach(),
                        sample_weights,
                    )
                teacher_recovery_code_bits = auth_codec.hard_codeword_bits(teacher_recovery_target_logits)
                student_recovery_code_bits = auth_codec.hard_codeword_bits(student_recovery_protocol_logits)
                train_recovery_codeword_ber.append(
                    (student_recovery_code_bits != teacher_recovery_code_bits).float().mean(dim=1).detach().cpu()
                )
                teacher_recovery_payload_bits = auth_codec.hard_payload_bits_from_codeword(teacher_recovery_target_logits)
                student_recovery_payload_bits = auth_codec.hard_payload_bits_from_codeword(student_recovery_protocol_logits)
                train_recovery_payload_ber.append(
                    (student_recovery_payload_bits != teacher_recovery_payload_bits).float().mean(dim=1).detach().cpu()
                )
                train_recovery_decode_success.append(
                    (student_recovery_payload_bits == teacher_recovery_payload_bits).all(dim=1).float().detach().cpu()
                )
                if teacher_code_indices is not None and student_code_indices is not None:
                    train_recovery_code_index_match.append(
                        (student_code_indices == teacher_code_indices).float().detach().cpu()
                    )
                student_claim_align_loss = compute_soft_alignment_loss(
                    student_bits,
                    claim_target_bits,
                    sample_weights,
                )
                student_score_distill_loss = compute_score_distill_loss(
                    student_scores,
                    teacher_sample_scores,
                    targets,
                    sample_weights,
                    temperature=args.score_distill_temperature,
                    topk=args.score_distill_topk,
                )
                student_hard_score_distill_loss = compute_hard_score_distill_loss(
                    student_scores,
                    teacher_sample_scores,
                    targets,
                    sample_weights,
                    temperature=args.score_distill_temperature,
                )
                student_claim_score_distill_loss = compute_score_distill_loss(
                    student_claim_loss_scores,
                    teacher_claim_loss_scores,
                    claim_targets,
                    sample_weights,
                    temperature=args.score_distill_temperature,
                    topk=args.score_distill_topk,
                )
                teacher_codebook_commit_loss = sample_weights.new_tensor(0.0)
                student_codebook_class_loss = sample_weights.new_tensor(0.0)
                student_codebook_proto_loss = sample_weights.new_tensor(0.0)
                codebook_usage_loss = sample_weights.new_tensor(0.0)
                codebook_separation_loss = sample_weights.new_tensor(0.0)
                if auth_codebook is not None:
                    teacher_codebook_commit_loss = auth_codebook.commitment_loss(
                        teacher_sample_encoded_logits,
                        teacher_code_prototypes,
                        sample_weights,
                    )
                    student_codebook_class_loss = auth_codebook.classification_loss(
                        student_recovery_logits,
                        teacher_code_indices.detach(),
                        sample_weights,
                    )
                    student_codebook_proto_loss = auth_codebook.prototype_alignment_loss(
                        student_recovery_logits,
                        teacher_code_prototypes,
                        sample_weights,
                    )
                    bank_code_indices, _ = auth_codebook.assign(bank_logits.detach())
                    usage_indices = torch.cat(
                        [
                            teacher_code_indices.detach().reshape(-1),
                            bank_code_indices.reshape(-1),
                        ],
                        dim=0,
                    )
                    codebook_usage_loss = auth_codebook.usage_loss(usage_indices)
                    codebook_separation_loss = auth_codebook.separation_loss()
                if float(args.claim_anchor_aux_weight) > 0.0 and args.claim_reference_mode != "anchor_bank":
                    anchor_claim_targets = build_bank_targets(raw_anchors, train_claim_bank_raws, device=device)
                    anchor_claim_target_bits = train_claim_bank_bits[anchor_claim_targets].detach()
                    student_anchor_claim_scores = compute_claim_scores(
                        student_bits,
                        train_claim_bank_bits.detach(),
                        claim_verifier,
                        verifier_weight=args.claim_verifier_weight,
                        verifier_score_mode=args.claim_verifier_score_mode,
                    )
                    teacher_anchor_claim_scores = compute_claim_scores(
                        teacher_sample_bits,
                        train_claim_bank_bits.detach(),
                        claim_verifier,
                        verifier_weight=args.claim_verifier_weight,
                        verifier_score_mode=args.claim_verifier_score_mode,
                    )
                    anchor_pos, anchor_hard_neg, anchor_neg = summarize_target_scores(
                        student_anchor_claim_scores,
                        anchor_claim_targets,
                    )
                    train_student_anchor_claim_pos_scores.append(anchor_pos.detach().cpu())
                    train_student_anchor_claim_hard_neg_scores.append(anchor_hard_neg.detach().cpu())
                    train_student_anchor_claim_all_neg_scores.append(anchor_neg.detach().cpu())
                    aux_w = float(args.claim_anchor_aux_weight)
                    student_claim_pair_loss = student_claim_pair_loss + aux_w * compute_pair_logistic_loss(
                        student_anchor_claim_scores,
                        anchor_claim_targets,
                        sample_weights,
                        positive_margin=args.student_positive_margin,
                        negative_margin=args.student_negative_margin,
                        scale=args.pair_logit_scale,
                        topk=args.pair_logistic_topk,
                    )
                    student_claim_hard_loss = student_claim_hard_loss + aux_w * compute_hard_margin_loss(
                        student_anchor_claim_scores,
                        anchor_claim_targets,
                        sample_weights,
                        margin=args.student_hard_margin,
                    )
                    student_claim_align_loss = student_claim_align_loss + aux_w * compute_soft_alignment_loss(
                        student_bits,
                        anchor_claim_target_bits,
                        sample_weights,
                    )
                    student_claim_score_distill_loss = student_claim_score_distill_loss + aux_w * compute_score_distill_loss(
                        student_anchor_claim_scores,
                        teacher_anchor_claim_scores,
                        anchor_claim_targets,
                        sample_weights,
                        temperature=args.score_distill_temperature,
                        topk=args.score_distill_topk,
                    )

                teacher_bank_scores = score_bank(bank_bits, bank_bits)
                bank_targets = torch.arange(bank_logits.shape[0], device=device)
                bank_weights = torch.ones(bank_logits.shape[0], device=device, dtype=torch.float32)
                teacher_bank_pair_loss = compute_pair_logistic_loss(
                    teacher_bank_scores,
                    bank_targets,
                    bank_weights,
                    positive_margin=args.teacher_positive_margin,
                    negative_margin=args.teacher_negative_margin,
                    scale=args.pair_logit_scale,
                    topk=args.pair_logistic_topk,
                )
                teacher_bank_hard_loss = compute_hard_margin_loss(
                    teacher_bank_scores,
                    bank_targets,
                    bank_weights,
                    margin=args.teacher_hard_margin,
                )
                teacher_balance_loss = compute_balance_loss(bank_bits)
                teacher_decorrelation_loss = compute_decorrelation_loss(bank_bits)
                teacher_uniformity_loss = compute_uniformity_loss(F.normalize(bank_bits, dim=-1), temperature=args.uniformity_temperature)

                loss = compute_total_authcode_loss(
                    args,
                    teacher_consistency_loss=teacher_consistency_loss,
                    teacher_sequence_match_loss=teacher_sequence_match_loss,
                    teacher_pair_loss=teacher_pair_loss,
                    teacher_bank_pair_loss=teacher_bank_pair_loss,
                    teacher_bank_hard_loss=teacher_bank_hard_loss,
                    teacher_claim_pair_loss=teacher_claim_pair_loss,
                    teacher_claim_hard_loss=teacher_claim_hard_loss,
                    teacher_codebook_commit_loss=teacher_codebook_commit_loss,
                    student_bit_loss=student_bit_loss,
                    student_sequence_match_loss=student_sequence_match_loss,
                    student_pair_loss=student_pair_loss,
                    student_claim_bit_loss=student_claim_bit_loss,
                    student_claim_pair_loss=student_claim_pair_loss,
                    student_claim_bank_pair_loss=student_claim_bank_pair_loss,
                    student_protocol_claim_pair_loss=student_protocol_claim_pair_loss,
                    student_protocol_claim_bank_pair_loss=student_protocol_claim_bank_pair_loss,
                    student_verifier_claim_pair_loss=student_verifier_claim_pair_loss,
                    student_hard_loss=student_hard_loss,
                    student_claim_hard_loss=student_claim_hard_loss,
                    student_claim_bank_hard_loss=student_claim_bank_hard_loss,
                    student_protocol_claim_hard_loss=student_protocol_claim_hard_loss,
                    student_protocol_claim_bank_hard_loss=student_protocol_claim_bank_hard_loss,
                    student_verifier_claim_hard_loss=student_verifier_claim_hard_loss,
                    student_verifier_claim_bank_hard_loss=student_verifier_claim_bank_hard_loss,
                    student_claim_calibration_pair_loss=student_claim_calibration_pair_loss,
                    student_claim_calibration_hard_loss=student_claim_calibration_hard_loss,
                    student_claim_eer_proxy_loss=student_claim_eer_proxy_loss,
                    student_claim_positive_tail_loss=student_claim_positive_tail_loss,
                    student_soft_align_loss=student_soft_align_loss,
                    student_claim_align_loss=student_claim_align_loss,
                    student_recovery_bit_loss=student_recovery_bit_loss,
                    student_recovery_stage_loss=student_recovery_stage_loss,
                    student_recovery_align_loss=student_recovery_align_loss,
                    student_token_class_loss=student_token_class_loss,
                    student_token_proto_loss=student_token_proto_loss,
                    student_score_distill_loss=student_score_distill_loss,
                    student_hard_score_distill_loss=student_hard_score_distill_loss,
                    student_claim_score_distill_loss=student_claim_score_distill_loss,
                    student_codebook_class_loss=student_codebook_class_loss,
                    student_codebook_proto_loss=student_codebook_proto_loss,
                    teacher_balance_loss=teacher_balance_loss,
                    teacher_decorrelation_loss=teacher_decorrelation_loss,
                    teacher_uniformity_loss=teacher_uniformity_loss,
                    codebook_usage_loss=codebook_usage_loss,
                    codebook_separation_loss=codebook_separation_loss,
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += float(loss.item())

            teacher_pos, teacher_hard_neg, teacher_neg = summarize_target_scores(teacher_sample_scores, targets)
            train_teacher_pos_scores.append(teacher_pos.detach().cpu())
            train_teacher_hard_neg_scores.append(teacher_hard_neg.detach().cpu())
            train_teacher_all_neg_scores.append(teacher_neg.reshape(-1).detach().cpu())

            student_pos, student_hard_neg, student_neg = summarize_target_scores(student_scores, targets)
            train_student_pos_scores.append(student_pos.detach().cpu())
            train_student_hard_neg_scores.append(student_hard_neg.detach().cpu())
            train_student_all_neg_scores.append(student_neg.reshape(-1).detach().cpu())

            if args.claim_reference_mode == "anchor_bank":
                claim_pos, claim_hard_neg, claim_neg = summarize_target_scores(student_claim_scores, claim_targets)
                protocol_claim_pos, protocol_claim_hard_neg, protocol_claim_neg = (
                    summarize_target_scores(student_protocol_claim_scores, claim_targets)
                    if student_protocol_claim_scores is not None
                    else (None, None, None)
                )
                claim_joint_pos, claim_joint_hard_neg, claim_joint_neg = summarize_target_scores(
                    student_claim_joint_scores,
                    claim_targets,
                )
                claim_official_pos, claim_official_hard_neg, claim_official_neg = summarize_target_scores(
                    student_claim_official_scores,
                    claim_targets,
                )
            elif args.claim_reference_mode == "same_image":
                claim_pos, claim_hard_neg, claim_neg = summarize_masked_scores(
                    student_claim_scores,
                    claim_positive_mask,
                    negative_mask=claim_negative_mask,
                )
                protocol_claim_pos, protocol_claim_hard_neg, protocol_claim_neg = (
                    summarize_masked_scores(
                        student_protocol_claim_scores,
                        claim_positive_mask,
                        negative_mask=claim_negative_mask,
                    )
                    if student_protocol_claim_scores is not None
                    else (None, None, None)
                )
                claim_joint_pos, claim_joint_hard_neg, claim_joint_neg = summarize_masked_scores(
                    student_claim_joint_scores,
                    claim_positive_mask,
                    negative_mask=claim_negative_mask,
                )
                claim_official_pos, claim_official_hard_neg, claim_official_neg = summarize_masked_scores(
                    student_claim_official_scores,
                    claim_positive_mask,
                    negative_mask=claim_negative_mask,
                )
            else:
                claim_pos, claim_hard_neg, claim_neg = summarize_masked_scores(student_claim_scores, claim_positive_mask)
                protocol_claim_pos, protocol_claim_hard_neg, protocol_claim_neg = (
                    summarize_masked_scores(student_protocol_claim_scores, claim_positive_mask)
                    if student_protocol_claim_scores is not None
                    else (None, None, None)
                )
                claim_joint_pos, claim_joint_hard_neg, claim_joint_neg = summarize_masked_scores(
                    student_claim_joint_scores,
                    claim_positive_mask,
                )
                claim_official_pos, claim_official_hard_neg, claim_official_neg = summarize_masked_scores(
                    student_claim_official_scores,
                    claim_positive_mask,
                )
            train_student_claim_pos_scores.append(claim_pos.detach().cpu())
            train_student_claim_hard_neg_scores.append(claim_hard_neg.detach().cpu())
            train_student_claim_all_neg_scores.append(claim_neg.reshape(-1).detach().cpu())
            if protocol_claim_pos is not None:
                train_student_protocol_claim_pos_scores.append(protocol_claim_pos.detach().cpu())
                train_student_protocol_claim_hard_neg_scores.append(protocol_claim_hard_neg.detach().cpu())
                train_student_protocol_claim_all_neg_scores.append(protocol_claim_neg.reshape(-1).detach().cpu())
            train_student_claim_joint_pos_scores.append(claim_joint_pos.detach().cpu())
            train_student_claim_joint_hard_neg_scores.append(claim_joint_hard_neg.detach().cpu())
            train_student_claim_joint_all_neg_scores.append(claim_joint_neg.reshape(-1).detach().cpu())
            train_student_claim_official_pos_scores.append(claim_official_pos.detach().cpu())
            train_student_claim_official_hard_neg_scores.append(claim_official_hard_neg.detach().cpu())
            train_student_claim_official_all_neg_scores.append(claim_official_neg.reshape(-1).detach().cpu())

            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "StuPos": f"{student_pos.mean().item():.3f}", "TeaPos": f"{teacher_pos.mean().item():.3f}"})

        scheduler.step()

        teacher_head.eval()
        student.eval()
        if student_code_head is not None:
            student_code_head.eval()
        if student_token_head is not None:
            student_token_head.eval()
        if claim_score_fusion_head is not None:
            claim_score_fusion_head.eval()
        if claim_verifier is not None:
            claim_verifier.eval()
        if auth_codebook is not None:
            auth_codebook.eval()
        val_loss = 0.0
        val_teacher_pos_scores = []
        val_teacher_hard_neg_scores = []
        val_teacher_all_neg_scores = []
        val_student_pos_scores = []
        val_student_hard_neg_scores = []
        val_student_all_neg_scores = []
        val_recovery_codeword_ber = []
        val_recovery_payload_ber = []
        val_recovery_decode_success = []
        val_recovery_code_index_match = []
        val_student_claim_pos_scores = []
        val_student_claim_hard_neg_scores = []
        val_student_claim_all_neg_scores = []
        val_student_protocol_claim_pos_scores = []
        val_student_protocol_claim_hard_neg_scores = []
        val_student_protocol_claim_all_neg_scores = []
        val_student_claim_joint_pos_scores = []
        val_student_claim_joint_hard_neg_scores = []
        val_student_claim_joint_all_neg_scores = []
        val_student_claim_official_pos_scores = []
        val_student_claim_official_hard_neg_scores = []
        val_student_claim_official_all_neg_scores = []
        val_student_anchor_claim_pos_scores = []
        val_student_anchor_claim_hard_neg_scores = []
        val_student_anchor_claim_all_neg_scores = []
        val_student_anchor_acc = 0.0
        val_student_shift_acc = 0.0
        val_teacher_anchor_acc = 0.0
        val_teacher_shift_acc = 0.0
        val_same_raw_claim_logits = []
        val_claim_bank_bits = None
        val_claim_bank_raws = None

        with torch.no_grad():
            bank_out, bank_bits, bank_stage_bits = forward_teacher_bank(
                teacher_head,
                val_proto_vec_bank,
                val_proto_seq_bank,
                bit_scale=args.bit_scale,
                use_sequence_score=use_sequence,
                auth_codec=auth_codec,
                auth_codebook=auth_codebook,
                apply_codebook_to_scores=apply_codebook_to_scores,
            )
            bank_logits = bank_out["global_logits"]
            bank_stage_logits = bank_out.get("stage_logits")
            bank_token_indices = None
            bank_token_logits = None
            if teacher_tokenizer is not None:
                bank_token_indices, bank_token_logits = teacher_tokenizer.assign(bank_logits.detach())
            use_anchor_reference = args.claim_reference_mode == "anchor_bank" or float(args.claim_anchor_aux_weight) > 0.0
            if use_anchor_reference:
                val_claim_bank_bits, val_claim_bank_raws = build_claim_reference_bank(
                    dataset=val_dataset,
                    indices=val_indices,
                    reference_raws=val_raws,
                    teacher_head=teacher_head,
                    use_sequence_score=use_sequence,
                    bit_scale=args.bit_scale,
                    device=device,
                    reference_versions=args.claim_reference_versions or args.anchor_versions,
                    claim_bank_mode=args.claim_bank_mode,
                    auth_codec=auth_codec,
                    auth_codebook=auth_codebook,
                    apply_codebook_to_scores=apply_codebook_to_scores,
                )

            for batch in val_loader:
                rgb_image = batch["rgb_image"].to(device, non_blocking=True)
                teacher_vec = batch["teacher_vec"].to(device, non_blocking=True)
                teacher_seq = batch["teacher_seq"].to(device, non_blocking=True)
                raw_anchors = list(batch["raw_anchor"])
                versions = batch["version"].to(device, non_blocking=True)
                sample_weights = compute_sample_weights(
                    versions,
                    anchor_versions=args.anchor_versions,
                    shift_versions=args.shift_versions,
                    anchor_weight=args.anchor_weight,
                    shift_weight=args.shift_weight,
                )
                targets = build_bank_targets(raw_anchors, val_bank_raws, device=device)

                teacher_sample_out = teacher_head(
                    teacher_vec,
                    teacher_seq,
                    return_sequence=use_sequence,
                    return_logits=True,
                )
                teacher_sample_encoded_logits = encode_teacher_logits(teacher_sample_out["global_logits"])
                teacher_code_indices = None
                teacher_code_prototypes = None
                if auth_codebook is not None:
                    teacher_quantized_logits, teacher_code_indices, _, teacher_code_prototypes = auth_codebook.quantize(
                        teacher_sample_encoded_logits,
                        straight_through=True,
                    )
                    teacher_sample_logits = teacher_quantized_logits if apply_codebook_to_scores else teacher_sample_encoded_logits
                else:
                    teacher_sample_logits = teacher_sample_encoded_logits
                teacher_sample_stage_logits = encode_teacher_logits(teacher_sample_out.get("stage_logits"))
                teacher_sample_stage_bits = (
                    soft_bits(teacher_sample_stage_logits, args.bit_scale)
                    if teacher_sample_stage_logits is not None
                    else None
                )
                teacher_token_indices = None
                teacher_token_logits = None
                if teacher_tokenizer is not None:
                    teacher_token_indices, teacher_token_logits = teacher_tokenizer.assign(
                        teacher_sample_encoded_logits.detach()
                    )
                teacher_sample_scores, teacher_sample_bits = compute_combined_scores(
                    teacher_sample_logits,
                    bank_bits,
                    query_stage_logits=teacher_sample_stage_logits,
                    bank_stage_bits=bank_stage_bits,
                    bit_scale=args.bit_scale,
                    sequence_score_weight=args.sequence_score_weight,
                )

                student_forward = forward_student_branches(
                    student=student,
                    rgb_image=rgb_image,
                    use_sequence=use_sequence,
                    encode_student_logits=encode_student_logits,
                    auth_codebook=auth_codebook,
                    apply_codebook_to_scores=apply_codebook_to_scores,
                    student_code_head=student_code_head,
                    student_token_head=student_token_head,
                )
                student_encoded_logits = student_forward["score_encoded_logits"]
                student_logits = student_forward["score_logits"]
                student_stage_logits = student_forward["stage_logits"]
                student_recovery_logits = student_forward["recovery_logits"]
                student_recovery_protocol_logits = student_forward["recovery_protocol_logits"]
                student_code_indices = student_forward["recovery_code_indices"]
                student_code_prototypes = student_forward["recovery_code_prototypes"]
                student_token_logits = student_forward["token_logits"]
                if teacher_tokenizer is not None:
                    base_student_token_logits = teacher_tokenizer.logits(student_recovery_logits)
                    if student_token_logits is None:
                        student_token_logits = base_student_token_logits
                    else:
                        student_token_logits = base_student_token_logits + student_token_logits
                student_scores, student_bits = compute_combined_scores(
                    student_logits,
                    bank_bits,
                    query_stage_logits=student_stage_logits,
                    bank_stage_bits=bank_stage_bits,
                    bit_scale=args.bit_scale,
                    sequence_score_weight=args.sequence_score_weight,
                )
                student_verifier_repr = build_claim_verifier_repr(
                    student_bits,
                    student_logits,
                    args.claim_verifier_input_mode,
                )
                teacher_sample_verifier_repr = build_claim_verifier_repr(
                    teacher_sample_bits,
                    teacher_sample_logits.detach(),
                    args.claim_verifier_input_mode,
                )
                bank_verifier_repr = build_claim_verifier_repr(
                    bank_bits.detach(),
                    bank_logits.detach(),
                    args.claim_verifier_input_mode,
                )

                sample_ids = list(batch["sample_id"])
                teacher_claim_protocol_logits = (
                    teacher_code_prototypes.detach() if teacher_code_prototypes is not None else teacher_sample_encoded_logits.detach()
                )
                teacher_claim_code_indices = teacher_code_indices.detach() if teacher_code_indices is not None else None
                if args.claim_reference_mode == "anchor_bank":
                    claim_targets = build_bank_targets(raw_anchors, val_claim_bank_raws, device=device)
                    claim_target_bits = val_claim_bank_bits[claim_targets].detach()
                    student_claim_scores = compute_claim_scores(
                        student_bits,
                        val_claim_bank_bits.detach(),
                        claim_verifier,
                        verifier_weight=args.claim_verifier_weight,
                        verifier_score_mode=args.claim_verifier_score_mode,
                    )
                    teacher_claim_scores = compute_claim_scores(
                        teacher_sample_bits,
                        val_claim_bank_bits.detach(),
                        claim_verifier,
                        verifier_weight=args.claim_verifier_weight,
                        verifier_score_mode=args.claim_verifier_score_mode,
                    )
                    student_claim_official_scores = student_claim_scores
                    teacher_claim_official_scores = teacher_claim_scores
                    claim_positive_mask = None
                    claim_negative_mask = None
                elif args.claim_reference_mode == "same_image":
                    student_claim_scores = compute_claim_scores(
                        student_bits,
                        teacher_sample_bits.detach(),
                        claim_verifier,
                        verifier_weight=args.claim_verifier_weight,
                        verifier_score_mode=args.claim_verifier_score_mode,
                        query_verifier_repr=student_verifier_repr,
                        reference_verifier_repr=teacher_sample_verifier_repr.detach(),
                        query_stage_logits=student_stage_logits,
                        reference_stage_logits=teacher_sample_stage_logits.detach() if teacher_sample_stage_logits is not None else None,
                        reference_stage_bits=teacher_sample_stage_bits.detach() if teacher_sample_stage_bits is not None else None,
                        bit_scale=args.bit_scale,
                        sequence_score_weight=claim_sequence_score_weight,
                    )
                    teacher_claim_scores = compute_claim_scores(
                        teacher_sample_bits,
                        teacher_sample_bits.detach(),
                        claim_verifier,
                        verifier_weight=args.claim_verifier_weight,
                        verifier_score_mode=args.claim_verifier_score_mode,
                        query_verifier_repr=teacher_sample_verifier_repr,
                        reference_verifier_repr=teacher_sample_verifier_repr.detach(),
                        query_stage_logits=teacher_sample_stage_logits,
                        reference_stage_logits=teacher_sample_stage_logits.detach() if teacher_sample_stage_logits is not None else None,
                        reference_stage_bits=teacher_sample_stage_bits.detach() if teacher_sample_stage_bits is not None else None,
                        bit_scale=args.bit_scale,
                        sequence_score_weight=claim_sequence_score_weight,
                    )
                    claim_targets = torch.arange(student_claim_scores.shape[0], device=device)
                    claim_target_bits = teacher_sample_bits.detach()
                    claim_positive_mask = build_same_sample_mask(sample_ids, device=device)
                    claim_negative_mask = ~build_same_raw_mask(raw_anchors, device=device)
                else:
                    student_claim_scores = compute_claim_scores(
                        student_bits,
                        teacher_sample_bits.detach(),
                        claim_verifier,
                        verifier_weight=args.claim_verifier_weight,
                        verifier_score_mode=args.claim_verifier_score_mode,
                        query_verifier_repr=student_verifier_repr,
                        reference_verifier_repr=teacher_sample_verifier_repr.detach(),
                        query_stage_logits=student_stage_logits,
                        reference_stage_logits=teacher_sample_stage_logits.detach() if teacher_sample_stage_logits is not None else None,
                        reference_stage_bits=teacher_sample_stage_bits.detach() if teacher_sample_stage_bits is not None else None,
                        bit_scale=args.bit_scale,
                        sequence_score_weight=claim_sequence_score_weight,
                    )
                    teacher_claim_scores = compute_claim_scores(
                        teacher_sample_bits,
                        teacher_sample_bits.detach(),
                        claim_verifier,
                        verifier_weight=args.claim_verifier_weight,
                        verifier_score_mode=args.claim_verifier_score_mode,
                        query_verifier_repr=teacher_sample_verifier_repr,
                        reference_verifier_repr=teacher_sample_verifier_repr.detach(),
                        query_stage_logits=teacher_sample_stage_logits,
                        reference_stage_logits=teacher_sample_stage_logits.detach() if teacher_sample_stage_logits is not None else None,
                        reference_stage_bits=teacher_sample_stage_bits.detach() if teacher_sample_stage_bits is not None else None,
                        bit_scale=args.bit_scale,
                        sequence_score_weight=claim_sequence_score_weight,
                    )
                    claim_targets = torch.arange(student_claim_scores.shape[0], device=device)
                    claim_target_bits = teacher_sample_bits.detach()
                    claim_positive_mask = build_same_raw_mask(raw_anchors, device=device)
                    claim_negative_mask = None
                if args.claim_reference_mode == "anchor_bank":
                    student_verifier_claim_scores = compute_claim_verifier_scores(
                        student_bits,
                        val_claim_bank_bits.detach(),
                        claim_verifier,
                        verifier_weight=args.claim_verifier_weight,
                        verifier_score_mode=args.claim_verifier_score_mode,
                    )
                    teacher_verifier_claim_scores = compute_claim_verifier_scores(
                        teacher_sample_bits,
                        val_claim_bank_bits.detach(),
                        claim_verifier,
                        verifier_weight=args.claim_verifier_weight,
                        verifier_score_mode=args.claim_verifier_score_mode,
                    )
                else:
                    student_verifier_claim_scores = compute_claim_verifier_scores(
                        student_bits,
                        teacher_sample_bits.detach(),
                        claim_verifier,
                        verifier_weight=args.claim_verifier_weight,
                        verifier_score_mode=args.claim_verifier_score_mode,
                        query_verifier_repr=student_verifier_repr,
                        reference_verifier_repr=teacher_sample_verifier_repr.detach(),
                    )
                    teacher_verifier_claim_scores = compute_claim_verifier_scores(
                        teacher_sample_bits,
                        teacher_sample_bits.detach(),
                        claim_verifier,
                        verifier_weight=args.claim_verifier_weight,
                        verifier_score_mode=args.claim_verifier_score_mode,
                        query_verifier_repr=teacher_sample_verifier_repr,
                        reference_verifier_repr=teacher_sample_verifier_repr.detach(),
                    )
                student_token_claim_scores = None
                teacher_token_claim_scores = None
                if teacher_tokenizer is not None and args.claim_reference_mode != "anchor_bank":
                    student_token_claim_scores = compute_token_match_matrix(
                        student_token_logits,
                        teacher_token_indices.detach() if teacher_token_indices is not None else None,
                    )
                    teacher_token_claim_scores = compute_token_match_matrix(
                        teacher_token_logits,
                        teacher_token_indices.detach() if teacher_token_indices is not None else None,
                    )
                student_protocol_claim_scores = None
                teacher_protocol_claim_scores = None
                student_extra_protocol_claim_scores = []
                teacher_extra_protocol_claim_scores = []
                if args.claim_reference_mode != "anchor_bank":
                    student_protocol_claim_scores, student_extra_protocol_claim_scores = compute_protocol_score_bundle(
                        query_protocol_logits=student_recovery_protocol_logits,
                        reference_protocol_logits=teacher_claim_protocol_logits,
                        auth_codec=auth_codec,
                        primary_mode=args.protocol_score_mode,
                        extra_modes=args.claim_score_fusion_protocol_modes,
                        query_code_indices=student_code_indices,
                        reference_code_indices=teacher_claim_code_indices,
                    )
                    teacher_protocol_claim_scores, teacher_extra_protocol_claim_scores = compute_protocol_score_bundle(
                        query_protocol_logits=teacher_claim_protocol_logits,
                        reference_protocol_logits=teacher_claim_protocol_logits,
                        auth_codec=auth_codec,
                        primary_mode=args.protocol_score_mode,
                        extra_modes=args.claim_score_fusion_protocol_modes,
                        query_code_indices=teacher_claim_code_indices,
                        reference_code_indices=teacher_claim_code_indices,
                    )
                if args.claim_verifier_feature_to_fusion and student_verifier_claim_scores is not None:
                    student_extra_protocol_claim_scores.append(student_verifier_claim_scores)
                    teacher_extra_protocol_claim_scores.append(teacher_verifier_claim_scores)
                student_claim_score_outputs = build_claim_score_outputs(
                    student_claim_scores,
                    student_protocol_claim_scores,
                    token_scores=student_token_claim_scores,
                    extra_protocol_scores=student_extra_protocol_claim_scores,
                    claim_score_fusion_head=claim_score_fusion_head,
                    official_mode=args.official_claim_score_mode,
                    alpha=args.protocol_score_alpha,
                    main_normalization=args.claim_main_score_norm_mode,
                    auxiliary_normalization=args.protocol_score_norm_mode,
                    gate_penalty=args.token_gate_penalty,
                    residual_weight=args.token_residual_weight,
                    hard_gate_threshold=args.token_hard_gate_threshold,
                )
                teacher_claim_score_outputs = build_claim_score_outputs(
                    teacher_claim_scores,
                    teacher_protocol_claim_scores,
                    token_scores=teacher_token_claim_scores,
                    extra_protocol_scores=teacher_extra_protocol_claim_scores,
                    claim_score_fusion_head=claim_score_fusion_head,
                    official_mode=args.official_claim_score_mode,
                    alpha=args.protocol_score_alpha,
                    main_normalization=args.claim_main_score_norm_mode,
                    auxiliary_normalization=args.protocol_score_norm_mode,
                    gate_penalty=args.token_gate_penalty,
                    residual_weight=args.token_residual_weight,
                    hard_gate_threshold=args.token_hard_gate_threshold,
                )
                student_claim_joint_scores = student_claim_score_outputs["fusion_scores"]
                teacher_claim_joint_scores = teacher_claim_score_outputs["fusion_scores"]
                student_claim_official_scores = student_claim_score_outputs["official_scores"]
                teacher_claim_official_scores = teacher_claim_score_outputs["official_scores"]
                claim_loss_weights = compute_optional_claim_hardcase_weights(
                    student_claim_official_scores,
                    claim_reference_mode=args.claim_reference_mode,
                    sample_weights=sample_weights,
                    strength=args.student_claim_hardcase_reweight_strength,
                    margin=args.student_claim_hardcase_reweight_margin,
                    scale=args.student_claim_hardcase_reweight_scale,
                    claim_targets=claim_targets,
                    claim_positive_mask=claim_positive_mask,
                    claim_negative_mask=claim_negative_mask,
                )
                use_joint_claim_scores = args.use_joint_claim_loss or teacher_tokenizer is not None or claim_score_fusion_head is not None
                student_claim_loss_scores = student_claim_official_scores if use_joint_claim_scores else student_claim_scores
                teacher_claim_loss_scores = teacher_claim_official_scores if use_joint_claim_scores else teacher_claim_scores

                teacher_target_logits = bank_logits[targets]
                teacher_target_stage_logits = bank_stage_logits[targets] if bank_stage_logits is not None else None
                teacher_target_bits = bank_bits[targets]
                teacher_recovery_target_logits = (
                    teacher_code_prototypes.detach() if teacher_code_prototypes is not None else teacher_sample_encoded_logits.detach()
                )
                teacher_recovery_target_bits = soft_bits(teacher_recovery_target_logits, args.bit_scale)
                student_target_logits, student_target_stage_logits, student_target_bits = resolve_student_match_targets(
                    canonical_logits=teacher_target_logits,
                    sample_logits=teacher_sample_logits,
                    canonical_stage_logits=teacher_target_stage_logits,
                    sample_stage_logits=teacher_sample_stage_logits,
                    bit_scale=args.bit_scale,
                    target_mode=args.student_target_mode,
                    blend_alpha=args.student_target_blend_alpha,
                )

                teacher_consistency_loss = compute_bit_match_loss(teacher_sample_logits, teacher_target_logits, sample_weights)
                teacher_sequence_match_loss = compute_stage_bit_match_loss(teacher_sample_stage_logits, teacher_target_stage_logits, sample_weights)
                teacher_pair_loss = compute_pair_logistic_loss(
                    teacher_sample_scores,
                    targets,
                    sample_weights,
                    positive_margin=args.teacher_positive_margin,
                    negative_margin=args.teacher_negative_margin,
                    scale=args.pair_logit_scale,
                    topk=args.pair_logistic_topk,
                )
                student_bit_loss = compute_bit_match_loss(student_logits, student_target_logits, sample_weights)
                if args.claim_reference_mode == "anchor_bank":
                    student_claim_bit_loss = compute_soft_alignment_loss(student_bits, claim_target_bits, sample_weights)
                else:
                    student_claim_bit_loss = compute_bit_match_loss(student_logits, teacher_sample_logits, sample_weights)
                student_sequence_match_loss = compute_stage_bit_match_loss(student_stage_logits, student_target_stage_logits, sample_weights)
                student_pair_loss = compute_pair_logistic_loss(
                    student_scores,
                    targets,
                    sample_weights,
                    positive_margin=args.student_positive_margin,
                    negative_margin=args.student_negative_margin,
                    scale=args.pair_logit_scale,
                    topk=args.pair_logistic_topk,
                )
                student_hard_loss = compute_hard_margin_loss(
                    student_scores,
                    targets,
                    sample_weights,
                    margin=args.student_hard_margin,
                )
                student_claim_bank_pair_loss = sample_weights.new_tensor(0.0)
                student_claim_bank_hard_loss = sample_weights.new_tensor(0.0)
                student_protocol_claim_bank_pair_loss = sample_weights.new_tensor(0.0)
                student_protocol_claim_bank_hard_loss = sample_weights.new_tensor(0.0)
                student_verifier_claim_pair_loss = sample_weights.new_tensor(0.0)
                student_verifier_claim_hard_loss = sample_weights.new_tensor(0.0)
                student_verifier_claim_bank_hard_loss = sample_weights.new_tensor(0.0)
                if (
                    float(args.student_claim_bank_pair_weight) > 0.0
                    or float(args.student_claim_bank_hard_weight) > 0.0
                    or float(args.student_protocol_claim_bank_pair_weight) > 0.0
                    or float(args.student_protocol_claim_bank_hard_weight) > 0.0
                    or float(args.student_verifier_claim_bank_hard_weight) > 0.0
                ):
                    student_claim_bank_scores = compute_claim_scores(
                        student_bits,
                        bank_bits.detach(),
                        claim_verifier,
                        verifier_weight=args.claim_verifier_weight,
                        verifier_score_mode=args.claim_verifier_score_mode,
                        query_verifier_repr=student_verifier_repr,
                        reference_verifier_repr=bank_verifier_repr.detach(),
                        query_stage_logits=student_stage_logits,
                        reference_stage_logits=bank_stage_logits.detach() if bank_stage_logits is not None else None,
                        reference_stage_bits=bank_stage_bits.detach() if bank_stage_bits is not None else None,
                        bit_scale=args.bit_scale,
                        sequence_score_weight=claim_sequence_score_weight,
                    )
                    student_verifier_claim_bank_scores = compute_claim_verifier_scores(
                        student_bits,
                        bank_bits.detach(),
                        claim_verifier,
                        verifier_weight=args.claim_verifier_weight,
                        verifier_score_mode=args.claim_verifier_score_mode,
                        query_verifier_repr=student_verifier_repr,
                        reference_verifier_repr=bank_verifier_repr.detach(),
                    )
                    student_claim_bank_protocol_scores, student_claim_bank_extra_protocol_scores = compute_protocol_score_bundle(
                        query_protocol_logits=student_recovery_protocol_logits,
                        reference_protocol_logits=bank_logits.detach(),
                        auth_codec=auth_codec,
                        primary_mode=args.protocol_score_mode,
                        extra_modes=args.claim_score_fusion_protocol_modes,
                    )
                    student_claim_bank_token_scores = compute_token_match_matrix(
                        student_token_logits,
                        bank_token_indices.detach() if bank_token_indices is not None else None,
                    )
                    if args.claim_verifier_feature_to_fusion and student_verifier_claim_bank_scores is not None:
                        student_claim_bank_extra_protocol_scores.append(student_verifier_claim_bank_scores)
                    student_claim_bank_score_outputs = build_claim_score_outputs(
                        student_claim_bank_scores,
                        student_claim_bank_protocol_scores,
                        token_scores=student_claim_bank_token_scores,
                        extra_protocol_scores=student_claim_bank_extra_protocol_scores,
                        claim_score_fusion_head=claim_score_fusion_head,
                        official_mode=args.official_claim_score_mode,
                        alpha=args.protocol_score_alpha,
                        main_normalization=args.claim_main_score_norm_mode,
                        auxiliary_normalization=args.protocol_score_norm_mode,
                        gate_penalty=args.token_gate_penalty,
                        residual_weight=args.token_residual_weight,
                        hard_gate_threshold=args.token_hard_gate_threshold,
                    )
                    student_claim_bank_loss_scores = (
                        student_claim_bank_score_outputs["official_scores"] if use_joint_claim_scores else student_claim_bank_scores
                    )
                    student_claim_bank_pair_loss = compute_pair_logistic_loss(
                        student_claim_bank_loss_scores,
                        targets,
                        sample_weights,
                        positive_margin=args.student_positive_margin,
                        negative_margin=args.student_negative_margin,
                        scale=args.pair_logit_scale,
                        topk=args.pair_logistic_topk,
                    )
                    student_claim_bank_hard_loss = compute_hard_margin_loss(
                        student_claim_bank_loss_scores,
                        targets,
                        sample_weights,
                        margin=args.student_hard_margin,
                    )
                    if student_claim_bank_protocol_scores is not None:
                        student_protocol_claim_bank_pair_loss = compute_pair_logistic_loss(
                            student_claim_bank_protocol_scores,
                            targets,
                            sample_weights,
                            positive_margin=args.student_positive_margin,
                            negative_margin=args.student_negative_margin,
                            scale=args.pair_logit_scale,
                            topk=args.pair_logistic_topk,
                        )
                        student_protocol_claim_bank_hard_loss = compute_hard_margin_loss(
                            student_claim_bank_protocol_scores,
                            targets,
                            sample_weights,
                            margin=args.student_hard_margin,
                        )
                    if student_verifier_claim_bank_scores is not None:
                        student_verifier_claim_bank_hard_loss = compute_hard_margin_loss(
                            student_verifier_claim_bank_scores,
                            targets,
                            sample_weights,
                            margin=args.student_hard_margin,
                        )
                student_soft_align_loss = compute_soft_alignment_loss(student_bits, student_target_bits, sample_weights)
                student_recovery_bit_loss = compute_logit_recovery_loss(
                    student_recovery_logits,
                    teacher_recovery_target_logits,
                    sample_weights,
                )
                if student_code_head is not None:
                    student_recovery_stage_loss = sample_weights.new_tensor(0.0)
                else:
                    student_recovery_stage_loss = compute_stage_logit_recovery_loss(
                        student_stage_logits,
                        teacher_sample_stage_logits,
                        sample_weights,
                    )
                student_recovery_align_loss = compute_soft_alignment_loss(
                    soft_bits(student_recovery_logits, args.bit_scale),
                    teacher_recovery_target_bits,
                    sample_weights,
                )
                student_token_class_loss = sample_weights.new_tensor(0.0)
                student_token_proto_loss = sample_weights.new_tensor(0.0)
                if teacher_tokenizer is not None and student_token_logits is not None and teacher_token_indices is not None:
                    token_ce = F.cross_entropy(
                        student_token_logits,
                        teacher_token_indices.detach(),
                        reduction="none",
                    )
                    student_token_class_loss = weighted_mean(token_ce, sample_weights)
                    student_token_proto_loss = compute_token_prototype_alignment_loss(
                        student_token_logits,
                        teacher_tokenizer,
                        teacher_token_indices.detach(),
                        sample_weights,
                    )
                teacher_recovery_code_bits = auth_codec.hard_codeword_bits(teacher_recovery_target_logits)
                student_recovery_code_bits = auth_codec.hard_codeword_bits(student_recovery_protocol_logits)
                val_recovery_codeword_ber.append(
                    (student_recovery_code_bits != teacher_recovery_code_bits).float().mean(dim=1).detach().cpu()
                )
                teacher_recovery_payload_bits = auth_codec.hard_payload_bits_from_codeword(teacher_recovery_target_logits)
                student_recovery_payload_bits = auth_codec.hard_payload_bits_from_codeword(student_recovery_protocol_logits)
                val_recovery_payload_ber.append(
                    (student_recovery_payload_bits != teacher_recovery_payload_bits).float().mean(dim=1).detach().cpu()
                )
                val_recovery_decode_success.append(
                    (student_recovery_payload_bits == teacher_recovery_payload_bits).all(dim=1).float().detach().cpu()
                )
                if teacher_code_indices is not None and student_code_indices is not None:
                    val_recovery_code_index_match.append(
                        (student_code_indices == teacher_code_indices).float().detach().cpu()
                    )
                student_claim_align_loss = compute_soft_alignment_loss(student_bits, claim_target_bits, sample_weights)
                student_score_distill_loss = compute_score_distill_loss(
                    student_scores,
                    teacher_sample_scores,
                    targets,
                    sample_weights,
                    temperature=args.score_distill_temperature,
                    topk=args.score_distill_topk,
                )
                student_hard_score_distill_loss = compute_hard_score_distill_loss(
                    student_scores,
                    teacher_sample_scores,
                    targets,
                    sample_weights,
                    temperature=args.score_distill_temperature,
                )
                if args.claim_reference_mode == "anchor_bank":
                    teacher_claim_pair_loss = compute_pair_logistic_loss(
                        teacher_claim_loss_scores,
                        claim_targets,
                        sample_weights,
                        positive_margin=args.teacher_positive_margin,
                        negative_margin=args.teacher_negative_margin,
                        scale=args.pair_logit_scale,
                        topk=args.pair_logistic_topk,
                    )
                    teacher_claim_hard_loss = compute_hard_margin_loss(
                        teacher_claim_loss_scores,
                        claim_targets,
                        sample_weights,
                        margin=args.teacher_hard_margin,
                    )
                    student_claim_pair_loss = compute_pair_logistic_loss(
                        student_claim_loss_scores,
                        claim_targets,
                        claim_loss_weights,
                        positive_margin=args.student_positive_margin,
                        negative_margin=args.student_negative_margin,
                        scale=args.pair_logit_scale,
                        topk=args.pair_logistic_topk,
                    )
                    student_claim_hard_loss = compute_hard_margin_loss(
                        student_claim_loss_scores,
                        claim_targets,
                        claim_loss_weights,
                        margin=args.student_hard_margin,
                    )
                    if student_protocol_claim_scores is not None:
                        student_protocol_claim_pair_loss = compute_pair_logistic_loss(
                            student_protocol_claim_scores,
                            claim_targets,
                            claim_loss_weights,
                            positive_margin=args.student_positive_margin,
                            negative_margin=args.student_negative_margin,
                            scale=args.pair_logit_scale,
                            topk=args.pair_logistic_topk,
                        )
                        student_protocol_claim_hard_loss = compute_hard_margin_loss(
                            student_protocol_claim_scores,
                            claim_targets,
                            claim_loss_weights,
                            margin=args.student_hard_margin,
                        )
                    else:
                        student_protocol_claim_pair_loss = sample_weights.new_tensor(0.0)
                        student_protocol_claim_hard_loss = sample_weights.new_tensor(0.0)
                elif args.claim_reference_mode == "same_image":
                    teacher_claim_pair_loss = compute_masked_pair_logistic_loss(
                        teacher_claim_loss_scores,
                        claim_positive_mask,
                        sample_weights,
                        negative_mask=claim_negative_mask,
                        positive_margin=args.teacher_positive_margin,
                        negative_margin=args.teacher_negative_margin,
                        scale=args.pair_logit_scale,
                        topk=args.pair_logistic_topk,
                    )
                    teacher_claim_hard_loss = compute_masked_hard_margin_loss(
                        teacher_claim_loss_scores,
                        claim_positive_mask,
                        sample_weights,
                        negative_mask=claim_negative_mask,
                        margin=args.teacher_hard_margin,
                    )
                    student_claim_pair_loss = compute_masked_pair_logistic_loss(
                        student_claim_loss_scores,
                        claim_positive_mask,
                        claim_loss_weights,
                        negative_mask=claim_negative_mask,
                        positive_margin=args.student_positive_margin,
                        negative_margin=args.student_negative_margin,
                        scale=args.pair_logit_scale,
                        topk=args.pair_logistic_topk,
                    )
                    student_claim_hard_loss = compute_masked_hard_margin_loss(
                        student_claim_loss_scores,
                        claim_positive_mask,
                        claim_loss_weights,
                        negative_mask=claim_negative_mask,
                        margin=args.student_hard_margin,
                    )
                    if student_protocol_claim_scores is not None:
                        student_protocol_claim_pair_loss = compute_masked_pair_logistic_loss(
                            student_protocol_claim_scores,
                            claim_positive_mask,
                            claim_loss_weights,
                            negative_mask=claim_negative_mask,
                            positive_margin=args.student_positive_margin,
                            negative_margin=args.student_negative_margin,
                            scale=args.pair_logit_scale,
                            topk=args.pair_logistic_topk,
                        )
                        student_protocol_claim_hard_loss = compute_masked_hard_margin_loss(
                            student_protocol_claim_scores,
                            claim_positive_mask,
                            claim_loss_weights,
                            negative_mask=claim_negative_mask,
                            margin=args.student_hard_margin,
                        )
                    else:
                        student_protocol_claim_pair_loss = sample_weights.new_tensor(0.0)
                        student_protocol_claim_hard_loss = sample_weights.new_tensor(0.0)
                else:
                    teacher_claim_pair_loss = compute_masked_pair_logistic_loss(
                        teacher_claim_loss_scores,
                        claim_positive_mask,
                        sample_weights,
                        positive_margin=args.teacher_positive_margin,
                        negative_margin=args.teacher_negative_margin,
                        scale=args.pair_logit_scale,
                        topk=args.pair_logistic_topk,
                    )
                    teacher_claim_hard_loss = compute_masked_hard_margin_loss(
                        teacher_claim_loss_scores,
                        claim_positive_mask,
                        sample_weights,
                        margin=args.teacher_hard_margin,
                    )
                    student_claim_pair_loss = compute_masked_pair_logistic_loss(
                        student_claim_loss_scores,
                        claim_positive_mask,
                        claim_loss_weights,
                        positive_margin=args.student_positive_margin,
                        negative_margin=args.student_negative_margin,
                        scale=args.pair_logit_scale,
                        topk=args.pair_logistic_topk,
                    )
                    student_claim_hard_loss = compute_masked_hard_margin_loss(
                        student_claim_loss_scores,
                        claim_positive_mask,
                        claim_loss_weights,
                        margin=args.student_hard_margin,
                    )
                    if student_protocol_claim_scores is not None:
                        student_protocol_claim_pair_loss = compute_masked_pair_logistic_loss(
                            student_protocol_claim_scores,
                            claim_positive_mask,
                            claim_loss_weights,
                            positive_margin=args.student_positive_margin,
                            negative_margin=args.student_negative_margin,
                            scale=args.pair_logit_scale,
                            topk=args.pair_logistic_topk,
                        )
                        student_protocol_claim_hard_loss = compute_masked_hard_margin_loss(
                            student_protocol_claim_scores,
                            claim_positive_mask,
                            claim_loss_weights,
                            margin=args.student_hard_margin,
                        )
                    else:
                        student_protocol_claim_pair_loss = sample_weights.new_tensor(0.0)
                        student_protocol_claim_hard_loss = sample_weights.new_tensor(0.0)
                student_verifier_claim_pair_loss, student_verifier_claim_hard_loss = compute_optional_claim_pair_hard_losses(
                    student_verifier_claim_scores,
                    claim_reference_mode=args.claim_reference_mode,
                    sample_weights=claim_loss_weights,
                    positive_margin=args.student_positive_margin,
                    negative_margin=args.student_negative_margin,
                    scale=args.pair_logit_scale,
                    topk=args.pair_logistic_topk,
                    margin=args.student_hard_margin,
                    claim_targets=claim_targets,
                    claim_positive_mask=claim_positive_mask,
                    claim_negative_mask=claim_negative_mask,
                )
                student_claim_calibration_pair_loss = sample_weights.new_tensor(0.0)
                student_claim_calibration_hard_loss = sample_weights.new_tensor(0.0)
                student_claim_eer_proxy_loss = sample_weights.new_tensor(0.0)
                student_claim_positive_tail_loss = sample_weights.new_tensor(0.0)
                if (
                    float(args.student_claim_calibration_pair_weight) > 0.0
                    or float(args.student_claim_calibration_hard_weight) > 0.0
                ):
                    student_claim_calibration_pair_loss, student_claim_calibration_hard_loss = (
                        compute_optional_claim_calibration_losses(
                            student_claim_official_scores,
                            claim_reference_mode=args.claim_reference_mode,
                            sample_weights=claim_loss_weights,
                            topk=args.pair_logistic_topk,
                            claim_targets=claim_targets,
                            claim_positive_mask=claim_positive_mask,
                            claim_negative_mask=claim_negative_mask,
                        )
                    )
                if float(args.student_claim_eer_proxy_weight) > 0.0:
                    student_claim_eer_proxy_loss = compute_optional_claim_operating_point_loss(
                        student_claim_official_scores,
                        claim_reference_mode=args.claim_reference_mode,
                        sample_weights=claim_loss_weights,
                        positive_quantile=args.student_claim_eer_proxy_positive_quantile,
                        negative_quantile=args.student_claim_eer_proxy_negative_quantile,
                        margin=args.student_claim_eer_proxy_margin,
                        scale=args.student_claim_eer_proxy_scale,
                        claim_targets=claim_targets,
                        claim_positive_mask=claim_positive_mask,
                        claim_negative_mask=claim_negative_mask,
                    )
                if float(args.student_claim_positive_tail_weight) > 0.0:
                    student_claim_positive_tail_loss = compute_optional_claim_positive_tail_rescue_loss(
                        student_claim_official_scores,
                        claim_reference_mode=args.claim_reference_mode,
                        sample_weights=claim_loss_weights,
                        positive_quantile=args.student_claim_positive_tail_positive_quantile,
                        negative_quantile=args.student_claim_positive_tail_negative_quantile,
                        margin=args.student_claim_positive_tail_margin,
                        scale=args.student_claim_positive_tail_scale,
                        claim_targets=claim_targets,
                        claim_positive_mask=claim_positive_mask,
                        claim_negative_mask=claim_negative_mask,
                    )
                student_claim_score_distill_loss = compute_score_distill_loss(
                    student_claim_loss_scores,
                    teacher_claim_loss_scores,
                    claim_targets,
                    sample_weights,
                    temperature=args.score_distill_temperature,
                    topk=args.score_distill_topk,
                )
                teacher_codebook_commit_loss = sample_weights.new_tensor(0.0)
                student_codebook_class_loss = sample_weights.new_tensor(0.0)
                student_codebook_proto_loss = sample_weights.new_tensor(0.0)
                codebook_usage_loss = sample_weights.new_tensor(0.0)
                codebook_separation_loss = sample_weights.new_tensor(0.0)
                if auth_codebook is not None:
                    teacher_codebook_commit_loss = auth_codebook.commitment_loss(
                        teacher_sample_encoded_logits,
                        teacher_code_prototypes,
                        sample_weights,
                    )
                    student_codebook_class_loss = auth_codebook.classification_loss(
                        student_recovery_logits,
                        teacher_code_indices.detach(),
                        sample_weights,
                    )
                    student_codebook_proto_loss = auth_codebook.prototype_alignment_loss(
                        student_recovery_logits,
                        teacher_code_prototypes,
                        sample_weights,
                    )
                    bank_code_indices, _ = auth_codebook.assign(bank_logits.detach())
                    usage_indices = torch.cat(
                        [
                            teacher_code_indices.detach().reshape(-1),
                            bank_code_indices.reshape(-1),
                        ],
                        dim=0,
                    )
                    codebook_usage_loss = auth_codebook.usage_loss(usage_indices)
                    codebook_separation_loss = auth_codebook.separation_loss()
                if float(args.claim_anchor_aux_weight) > 0.0 and args.claim_reference_mode != "anchor_bank":
                    anchor_claim_targets = build_bank_targets(raw_anchors, val_claim_bank_raws, device=device)
                    anchor_claim_target_bits = val_claim_bank_bits[anchor_claim_targets].detach()
                    student_anchor_claim_scores = compute_claim_scores(
                        student_bits,
                        val_claim_bank_bits.detach(),
                        claim_verifier,
                        verifier_weight=args.claim_verifier_weight,
                        verifier_score_mode=args.claim_verifier_score_mode,
                    )
                    teacher_anchor_claim_scores = compute_claim_scores(
                        teacher_sample_bits,
                        val_claim_bank_bits.detach(),
                        claim_verifier,
                        verifier_weight=args.claim_verifier_weight,
                        verifier_score_mode=args.claim_verifier_score_mode,
                    )
                    anchor_pos, anchor_hard_neg, anchor_neg = summarize_target_scores(
                        student_anchor_claim_scores,
                        anchor_claim_targets,
                    )
                    val_student_anchor_claim_pos_scores.append(anchor_pos.cpu())
                    val_student_anchor_claim_hard_neg_scores.append(anchor_hard_neg.cpu())
                    val_student_anchor_claim_all_neg_scores.append(anchor_neg.cpu())
                    aux_w = float(args.claim_anchor_aux_weight)
                    student_claim_pair_loss = student_claim_pair_loss + aux_w * compute_pair_logistic_loss(
                        student_anchor_claim_scores,
                        anchor_claim_targets,
                        sample_weights,
                        positive_margin=args.student_positive_margin,
                        negative_margin=args.student_negative_margin,
                        scale=args.pair_logit_scale,
                        topk=args.pair_logistic_topk,
                    )
                    student_claim_hard_loss = student_claim_hard_loss + aux_w * compute_hard_margin_loss(
                        student_anchor_claim_scores,
                        anchor_claim_targets,
                        sample_weights,
                        margin=args.student_hard_margin,
                    )
                    student_claim_align_loss = student_claim_align_loss + aux_w * compute_soft_alignment_loss(
                        student_bits,
                        anchor_claim_target_bits,
                        sample_weights,
                    )
                    student_claim_score_distill_loss = student_claim_score_distill_loss + aux_w * compute_score_distill_loss(
                        student_anchor_claim_scores,
                        teacher_anchor_claim_scores,
                        anchor_claim_targets,
                        sample_weights,
                        temperature=args.score_distill_temperature,
                        topk=args.score_distill_topk,
                    )
                teacher_bank_scores = score_bank(bank_bits, bank_bits)
                bank_targets = torch.arange(bank_logits.shape[0], device=device)
                bank_weights = torch.ones(bank_logits.shape[0], device=device, dtype=torch.float32)
                teacher_bank_pair_loss = compute_pair_logistic_loss(
                    teacher_bank_scores,
                    bank_targets,
                    bank_weights,
                    positive_margin=args.teacher_positive_margin,
                    negative_margin=args.teacher_negative_margin,
                    scale=args.pair_logit_scale,
                    topk=args.pair_logistic_topk,
                )
                teacher_bank_hard_loss = compute_hard_margin_loss(
                    teacher_bank_scores,
                    bank_targets,
                    bank_weights,
                    margin=args.teacher_hard_margin,
                )
                teacher_balance_loss = compute_balance_loss(bank_bits)
                teacher_decorrelation_loss = compute_decorrelation_loss(bank_bits)
                teacher_uniformity_loss = compute_uniformity_loss(F.normalize(bank_bits, dim=-1), temperature=args.uniformity_temperature)

                loss = compute_total_authcode_loss(
                    args,
                    teacher_consistency_loss=teacher_consistency_loss,
                    teacher_sequence_match_loss=teacher_sequence_match_loss,
                    teacher_pair_loss=teacher_pair_loss,
                    teacher_bank_pair_loss=teacher_bank_pair_loss,
                    teacher_bank_hard_loss=teacher_bank_hard_loss,
                    teacher_claim_pair_loss=teacher_claim_pair_loss,
                    teacher_claim_hard_loss=teacher_claim_hard_loss,
                    teacher_codebook_commit_loss=teacher_codebook_commit_loss,
                    student_bit_loss=student_bit_loss,
                    student_sequence_match_loss=student_sequence_match_loss,
                    student_pair_loss=student_pair_loss,
                    student_claim_bit_loss=student_claim_bit_loss,
                    student_claim_pair_loss=student_claim_pair_loss,
                    student_claim_bank_pair_loss=student_claim_bank_pair_loss,
                    student_protocol_claim_pair_loss=student_protocol_claim_pair_loss,
                    student_protocol_claim_bank_pair_loss=student_protocol_claim_bank_pair_loss,
                    student_verifier_claim_pair_loss=student_verifier_claim_pair_loss,
                    student_hard_loss=student_hard_loss,
                    student_claim_hard_loss=student_claim_hard_loss,
                    student_claim_bank_hard_loss=student_claim_bank_hard_loss,
                    student_protocol_claim_hard_loss=student_protocol_claim_hard_loss,
                    student_protocol_claim_bank_hard_loss=student_protocol_claim_bank_hard_loss,
                    student_verifier_claim_hard_loss=student_verifier_claim_hard_loss,
                    student_verifier_claim_bank_hard_loss=student_verifier_claim_bank_hard_loss,
                    student_claim_calibration_pair_loss=student_claim_calibration_pair_loss,
                    student_claim_calibration_hard_loss=student_claim_calibration_hard_loss,
                    student_claim_eer_proxy_loss=student_claim_eer_proxy_loss,
                    student_claim_positive_tail_loss=student_claim_positive_tail_loss,
                    student_soft_align_loss=student_soft_align_loss,
                    student_claim_align_loss=student_claim_align_loss,
                    student_recovery_bit_loss=student_recovery_bit_loss,
                    student_recovery_stage_loss=student_recovery_stage_loss,
                    student_recovery_align_loss=student_recovery_align_loss,
                    student_token_class_loss=student_token_class_loss,
                    student_token_proto_loss=student_token_proto_loss,
                    student_score_distill_loss=student_score_distill_loss,
                    student_hard_score_distill_loss=student_hard_score_distill_loss,
                    student_claim_score_distill_loss=student_claim_score_distill_loss,
                    student_codebook_class_loss=student_codebook_class_loss,
                    student_codebook_proto_loss=student_codebook_proto_loss,
                    teacher_balance_loss=teacher_balance_loss,
                    teacher_decorrelation_loss=teacher_decorrelation_loss,
                    teacher_uniformity_loss=teacher_uniformity_loss,
                    codebook_usage_loss=codebook_usage_loss,
                    codebook_separation_loss=codebook_separation_loss,
                )
                val_loss += float(loss.item())

                teacher_pos, teacher_hard_neg, teacher_neg = summarize_target_scores(teacher_sample_scores, targets)
                val_teacher_pos_scores.append(teacher_pos.cpu())
                val_teacher_hard_neg_scores.append(teacher_hard_neg.cpu())
                val_teacher_all_neg_scores.append(teacher_neg.reshape(-1).cpu())
                teacher_correct = teacher_sample_scores.argmax(dim=1) == targets
                val_teacher_anchor_acc += accuracy_for_versions(teacher_correct, versions, args.anchor_versions)
                val_teacher_shift_acc += accuracy_for_versions(teacher_correct, versions, args.shift_versions)

                student_pos, student_hard_neg, student_neg = summarize_target_scores(student_scores, targets)
                val_student_pos_scores.append(student_pos.cpu())
                val_student_hard_neg_scores.append(student_hard_neg.cpu())
                val_student_all_neg_scores.append(student_neg.reshape(-1).cpu())
                student_correct = student_scores.argmax(dim=1) == targets
                val_student_anchor_acc += accuracy_for_versions(student_correct, versions, args.anchor_versions)
                val_student_shift_acc += accuracy_for_versions(student_correct, versions, args.shift_versions)

                if args.claim_reference_mode == "anchor_bank":
                    claim_pos, claim_hard_neg, claim_neg = summarize_target_scores(student_claim_scores, claim_targets)
                    protocol_claim_pos, protocol_claim_hard_neg, protocol_claim_neg = (
                        summarize_target_scores(student_protocol_claim_scores, claim_targets)
                        if student_protocol_claim_scores is not None
                        else (None, None, None)
                    )
                    claim_joint_pos, claim_joint_hard_neg, claim_joint_neg = summarize_target_scores(
                        student_claim_joint_scores,
                        claim_targets,
                    )
                    claim_official_pos, claim_official_hard_neg, claim_official_neg = summarize_target_scores(
                        student_claim_official_scores,
                        claim_targets,
                    )
                elif args.claim_reference_mode == "same_image":
                    claim_pos, claim_hard_neg, claim_neg = summarize_masked_scores(
                        student_claim_scores,
                        claim_positive_mask,
                        negative_mask=claim_negative_mask,
                    )
                    protocol_claim_pos, protocol_claim_hard_neg, protocol_claim_neg = (
                        summarize_masked_scores(
                            student_protocol_claim_scores,
                            claim_positive_mask,
                            negative_mask=claim_negative_mask,
                        )
                        if student_protocol_claim_scores is not None
                        else (None, None, None)
                    )
                    claim_joint_pos, claim_joint_hard_neg, claim_joint_neg = summarize_masked_scores(
                        student_claim_joint_scores,
                        claim_positive_mask,
                        negative_mask=claim_negative_mask,
                    )
                    claim_official_pos, claim_official_hard_neg, claim_official_neg = summarize_masked_scores(
                        student_claim_official_scores,
                        claim_positive_mask,
                        negative_mask=claim_negative_mask,
                    )
                else:
                    claim_pos, claim_hard_neg, claim_neg = summarize_masked_scores(student_claim_scores, claim_positive_mask)
                    protocol_claim_pos, protocol_claim_hard_neg, protocol_claim_neg = (
                        summarize_masked_scores(student_protocol_claim_scores, claim_positive_mask)
                        if student_protocol_claim_scores is not None
                        else (None, None, None)
                    )
                    claim_joint_pos, claim_joint_hard_neg, claim_joint_neg = summarize_masked_scores(
                        student_claim_joint_scores,
                        claim_positive_mask,
                    )
                    claim_official_pos, claim_official_hard_neg, claim_official_neg = summarize_masked_scores(
                        student_claim_official_scores,
                        claim_positive_mask,
                    )
                    if args.claim_calibration_scales:
                        val_same_raw_claim_logits.append(
                            {
                                "student_logits": student_logits.detach().cpu(),
                                "student_stage_logits": student_stage_logits.detach().cpu() if student_stage_logits is not None else None,
                                "teacher_logits": teacher_sample_logits.detach().cpu(),
                                "teacher_stage_logits": teacher_sample_stage_logits.detach().cpu() if teacher_sample_stage_logits is not None else None,
                                "raw_anchors": list(raw_anchors),
                            }
                        )
                val_student_claim_pos_scores.append(claim_pos.cpu())
                val_student_claim_hard_neg_scores.append(claim_hard_neg.cpu())
                val_student_claim_all_neg_scores.append(claim_neg.reshape(-1).cpu())
                if protocol_claim_pos is not None:
                    val_student_protocol_claim_pos_scores.append(protocol_claim_pos.cpu())
                    val_student_protocol_claim_hard_neg_scores.append(protocol_claim_hard_neg.cpu())
                    val_student_protocol_claim_all_neg_scores.append(protocol_claim_neg.reshape(-1).cpu())
                val_student_claim_joint_pos_scores.append(claim_joint_pos.cpu())
                val_student_claim_joint_hard_neg_scores.append(claim_joint_hard_neg.cpu())
                val_student_claim_joint_all_neg_scores.append(claim_joint_neg.reshape(-1).cpu())
                val_student_claim_official_pos_scores.append(claim_official_pos.cpu())
                val_student_claim_official_hard_neg_scores.append(claim_official_hard_neg.cpu())
                val_student_claim_official_all_neg_scores.append(claim_official_neg.reshape(-1).cpu())

        train_teacher_metrics = summarize_verification_scores(
            torch.cat(train_teacher_pos_scores, dim=0),
            torch.cat(train_teacher_hard_neg_scores, dim=0),
            torch.cat(train_teacher_all_neg_scores, dim=0),
        )
        train_student_metrics = summarize_verification_scores(
            torch.cat(train_student_pos_scores, dim=0),
            torch.cat(train_student_hard_neg_scores, dim=0),
            torch.cat(train_student_all_neg_scores, dim=0),
        )
        train_student_claim_metrics = summarize_verification_scores(
            torch.cat(train_student_claim_pos_scores, dim=0),
            torch.cat(train_student_claim_hard_neg_scores, dim=0),
            torch.cat(train_student_claim_all_neg_scores, dim=0),
        )
        train_student_protocol_claim_metrics = (
            summarize_verification_scores(
                torch.cat(train_student_protocol_claim_pos_scores, dim=0),
                torch.cat(train_student_protocol_claim_hard_neg_scores, dim=0),
                torch.cat(train_student_protocol_claim_all_neg_scores, dim=0),
            )
            if train_student_protocol_claim_pos_scores
            else None
        )
        train_student_claim_joint_metrics = summarize_verification_scores(
            torch.cat(train_student_claim_joint_pos_scores, dim=0),
            torch.cat(train_student_claim_joint_hard_neg_scores, dim=0),
            torch.cat(train_student_claim_joint_all_neg_scores, dim=0),
        )
        train_student_claim_official_metrics = summarize_verification_scores(
            torch.cat(train_student_claim_official_pos_scores, dim=0),
            torch.cat(train_student_claim_official_hard_neg_scores, dim=0),
            torch.cat(train_student_claim_official_all_neg_scores, dim=0),
        )
        val_teacher_metrics = summarize_verification_scores(
            torch.cat(val_teacher_pos_scores, dim=0),
            torch.cat(val_teacher_hard_neg_scores, dim=0),
            torch.cat(val_teacher_all_neg_scores, dim=0),
        )
        val_student_metrics = summarize_verification_scores(
            torch.cat(val_student_pos_scores, dim=0),
            torch.cat(val_student_hard_neg_scores, dim=0),
            torch.cat(val_student_all_neg_scores, dim=0),
        )
        val_student_claim_metrics = summarize_verification_scores(
            torch.cat(val_student_claim_pos_scores, dim=0),
            torch.cat(val_student_claim_hard_neg_scores, dim=0),
            torch.cat(val_student_claim_all_neg_scores, dim=0),
        )
        val_student_protocol_claim_metrics = (
            summarize_verification_scores(
                torch.cat(val_student_protocol_claim_pos_scores, dim=0),
                torch.cat(val_student_protocol_claim_hard_neg_scores, dim=0),
                torch.cat(val_student_protocol_claim_all_neg_scores, dim=0),
            )
            if val_student_protocol_claim_pos_scores
            else None
        )
        val_student_claim_joint_metrics = summarize_verification_scores(
            torch.cat(val_student_claim_joint_pos_scores, dim=0),
            torch.cat(val_student_claim_joint_hard_neg_scores, dim=0),
            torch.cat(val_student_claim_joint_all_neg_scores, dim=0),
        )
        val_student_claim_official_metrics = summarize_verification_scores(
            torch.cat(val_student_claim_official_pos_scores, dim=0),
            torch.cat(val_student_claim_official_hard_neg_scores, dim=0),
            torch.cat(val_student_claim_official_all_neg_scores, dim=0),
        )
        val_student_claim_calibrated = None
        if args.claim_reference_mode == "same_raw" and args.claim_calibration_scales:
            val_student_claim_calibrated = sweep_same_raw_claim_calibration(
                batch_logits_records=val_same_raw_claim_logits,
                claim_verifier=claim_verifier,
                device=device,
                bit_scales=args.claim_calibration_scales,
                verifier_weights=args.claim_calibration_verifier_weights,
                claim_sequence_score_weight=claim_sequence_score_weight,
                verifier_score_mode=args.claim_verifier_score_mode,
                sequence_weights=args.claim_calibration_sequence_weights,
            )
        train_student_anchor_claim_metrics = None
        val_student_anchor_claim_metrics = None
        if train_student_anchor_claim_pos_scores and val_student_anchor_claim_pos_scores:
            train_student_anchor_claim_metrics = summarize_verification_scores(
                torch.cat(train_student_anchor_claim_pos_scores, dim=0),
                torch.cat(train_student_anchor_claim_hard_neg_scores, dim=0),
                torch.cat(train_student_anchor_claim_all_neg_scores, dim=0),
            )
            val_student_anchor_claim_metrics = summarize_verification_scores(
                torch.cat(val_student_anchor_claim_pos_scores, dim=0),
                torch.cat(val_student_anchor_claim_hard_neg_scores, dim=0),
                torch.cat(val_student_anchor_claim_all_neg_scores, dim=0),
            )

        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": train_loss / max(len(train_loader), 1),
            "train_teacher_pairwise_auc": train_teacher_metrics["pairwise_auc"],
            "train_teacher_hard_auc": train_teacher_metrics["hard_auc"],
            "train_student_pairwise_auc": train_student_metrics["pairwise_auc"],
            "train_student_hard_auc": train_student_metrics["hard_auc"],
            "train_recovery_codeword_ber": torch.cat(train_recovery_codeword_ber, dim=0).mean().item(),
            "train_recovery_payload_ber": torch.cat(train_recovery_payload_ber, dim=0).mean().item(),
            "train_recovery_decode_success": torch.cat(train_recovery_decode_success, dim=0).mean().item(),
            "train_recovery_code_index_match": (
                torch.cat(train_recovery_code_index_match, dim=0).mean().item()
                if train_recovery_code_index_match
                else None
            ),
            "val_loss": val_loss / max(len(val_loader), 1),
            "val_teacher_pairwise_auc": val_teacher_metrics["pairwise_auc"],
            "val_teacher_hard_auc": val_teacher_metrics["hard_auc"],
            "val_teacher_eer": val_teacher_metrics["eer"],
            "val_student_pairwise_auc": val_student_metrics["pairwise_auc"],
            "val_student_hard_auc": val_student_metrics["hard_auc"],
            "val_student_eer": val_student_metrics["eer"],
            "val_student_tar_at_far_1e2": val_student_metrics["tar_at_far_1e2"],
            "val_recovery_codeword_ber": torch.cat(val_recovery_codeword_ber, dim=0).mean().item(),
            "val_recovery_payload_ber": torch.cat(val_recovery_payload_ber, dim=0).mean().item(),
            "val_recovery_decode_success": torch.cat(val_recovery_decode_success, dim=0).mean().item(),
            "val_recovery_code_index_match": (
                torch.cat(val_recovery_code_index_match, dim=0).mean().item()
                if val_recovery_code_index_match
                else None
            ),
            "train_student_claimed_pairwise_auc": train_student_claim_metrics["pairwise_auc"],
            "train_student_claimed_hard_auc": train_student_claim_metrics["hard_auc"],
            "train_student_claimed_eer": train_student_claim_metrics["eer"],
            "train_student_protocol_claimed_pairwise_auc": (
                train_student_protocol_claim_metrics["pairwise_auc"]
                if train_student_protocol_claim_metrics is not None
                else None
            ),
            "train_student_protocol_claimed_hard_auc": (
                train_student_protocol_claim_metrics["hard_auc"]
                if train_student_protocol_claim_metrics is not None
                else None
            ),
            "train_student_protocol_claimed_eer": (
                train_student_protocol_claim_metrics["eer"]
                if train_student_protocol_claim_metrics is not None
                else None
            ),
            "train_student_claimed_joint_pairwise_auc": train_student_claim_joint_metrics["pairwise_auc"],
            "train_student_claimed_joint_hard_auc": train_student_claim_joint_metrics["hard_auc"],
            "train_student_claimed_joint_eer": train_student_claim_joint_metrics["eer"],
            "train_student_claimed_official_pairwise_auc": train_student_claim_official_metrics["pairwise_auc"],
            "train_student_claimed_official_hard_auc": train_student_claim_official_metrics["hard_auc"],
            "train_student_claimed_official_eer": train_student_claim_official_metrics["eer"],
            "val_student_claimed_pairwise_auc": val_student_claim_metrics["pairwise_auc"],
            "val_student_claimed_hard_auc": val_student_claim_metrics["hard_auc"],
            "val_student_claimed_eer": val_student_claim_metrics["eer"],
            "val_student_claimed_tar_at_far_1e2": val_student_claim_metrics["tar_at_far_1e2"],
            "val_student_protocol_claimed_pairwise_auc": (
                val_student_protocol_claim_metrics["pairwise_auc"]
                if val_student_protocol_claim_metrics is not None
                else None
            ),
            "val_student_protocol_claimed_hard_auc": (
                val_student_protocol_claim_metrics["hard_auc"]
                if val_student_protocol_claim_metrics is not None
                else None
            ),
            "val_student_protocol_claimed_eer": (
                val_student_protocol_claim_metrics["eer"]
                if val_student_protocol_claim_metrics is not None
                else None
            ),
            "val_student_protocol_claimed_tar_at_far_1e2": (
                val_student_protocol_claim_metrics["tar_at_far_1e2"]
                if val_student_protocol_claim_metrics is not None
                else None
            ),
            "val_student_claimed_joint_pairwise_auc": val_student_claim_joint_metrics["pairwise_auc"],
            "val_student_claimed_joint_hard_auc": val_student_claim_joint_metrics["hard_auc"],
            "val_student_claimed_joint_eer": val_student_claim_joint_metrics["eer"],
            "val_student_claimed_joint_tar_at_far_1e2": val_student_claim_joint_metrics["tar_at_far_1e2"],
            "val_student_claimed_official_pairwise_auc": val_student_claim_official_metrics["pairwise_auc"],
            "val_student_claimed_official_hard_auc": val_student_claim_official_metrics["hard_auc"],
            "val_student_claimed_official_eer": val_student_claim_official_metrics["eer"],
            "val_student_claimed_official_tar_at_far_1e2": val_student_claim_official_metrics["tar_at_far_1e2"],
            "val_teacher_tar_at_far_1e2": val_teacher_metrics["tar_at_far_1e2"],
            "val_student_anchor_top1_acc": val_student_anchor_acc / max(len(val_loader), 1),
            "val_student_shift_top1_acc": val_student_shift_acc / max(len(val_loader), 1),
            "val_teacher_anchor_top1_acc": val_teacher_anchor_acc / max(len(val_loader), 1),
            "val_teacher_shift_top1_acc": val_teacher_shift_acc / max(len(val_loader), 1),
            "protocol_score_mode": args.protocol_score_mode,
            "protocol_score_alpha": args.protocol_score_alpha,
            "claim_main_score_norm_mode": args.claim_main_score_norm_mode,
            "protocol_score_norm_mode": args.protocol_score_norm_mode,
            "official_eval_preset": official_eval_preset,
            "official_config_resolved": official_config_resolved,
            "use_joint_claim_loss": args.use_joint_claim_loss,
            "freeze_claim_verifier": args.freeze_claim_verifier,
            "use_claim_score_fusion_head": args.use_claim_score_fusion_head,
            "claim_score_fusion_hidden_dim": args.claim_score_fusion_hidden_dim,
            "claim_score_fusion_dropout": args.claim_score_fusion_dropout,
            "claim_score_fusion_protocol_modes": normalize_protocol_mode_list(args.claim_score_fusion_protocol_modes),
            "claim_score_fusion_mode": args.claim_score_fusion_mode,
            "claim_score_fusion_residual_scale": args.claim_score_fusion_residual_scale,
        }
        if train_student_anchor_claim_metrics is not None and val_student_anchor_claim_metrics is not None:
            epoch_record.update(
                {
                    "train_student_anchor_claimed_pairwise_auc": train_student_anchor_claim_metrics["pairwise_auc"],
                    "train_student_anchor_claimed_hard_auc": train_student_anchor_claim_metrics["hard_auc"],
                    "train_student_anchor_claimed_eer": train_student_anchor_claim_metrics["eer"],
                    "val_student_anchor_claimed_pairwise_auc": val_student_anchor_claim_metrics["pairwise_auc"],
                    "val_student_anchor_claimed_hard_auc": val_student_anchor_claim_metrics["hard_auc"],
                    "val_student_anchor_claimed_eer": val_student_anchor_claim_metrics["eer"],
                    "val_student_anchor_claimed_tar_at_far_1e2": val_student_anchor_claim_metrics["tar_at_far_1e2"],
                }
            )
        if val_student_claim_calibrated is not None:
            epoch_record.update(
                {
                    "val_student_claimed_calibrated_pairwise_auc": val_student_claim_calibrated["metrics"]["pairwise_auc"],
                    "val_student_claimed_calibrated_hard_auc": val_student_claim_calibrated["metrics"]["hard_auc"],
                    "val_student_claimed_calibrated_eer": val_student_claim_calibrated["metrics"]["eer"],
                    "val_student_claimed_calibrated_tar_at_far_1e2": val_student_claim_calibrated["metrics"]["tar_at_far_1e2"],
                    "val_student_claimed_calibrated_bit_scale": val_student_claim_calibrated["bit_scale"],
                    "val_student_claimed_calibrated_verifier_weight": val_student_claim_calibrated["verifier_weight"],
                    "val_student_claimed_calibrated_sequence_weight": val_student_claim_calibrated["sequence_weight"],
                }
            )
        history.append(epoch_record)
        print(epoch_record, flush=True)

        selection_value = resolve_selection_value(epoch_record, args.selection_metric)
        if selection_value > best_metric:
            best_metric = selection_value
            torch.save(
                {
                    "epoch": epoch + 1,
                    "teacher_head_state_dict": teacher_head.state_dict(),
                    "student_state_dict": student.state_dict(),
                    "student_code_head_state_dict": (
                        student_code_head.state_dict() if student_code_head is not None else None
                    ),
                    "student_token_head_state_dict": (
                        student_token_head.state_dict() if student_token_head is not None else None
                    ),
                    "claim_score_fusion_state_dict": (
                        claim_score_fusion_head.state_dict() if claim_score_fusion_head is not None else None
                    ),
                    "auth_codebook_state_dict": auth_codebook.state_dict() if auth_codebook is not None else None,
                    "teacher_tokenizer_state_dict": (
                        teacher_tokenizer.state_dict() if teacher_tokenizer is not None else None
                    ),
                    "claim_verifier_state_dict": claim_verifier.state_dict() if claim_verifier is not None else None,
                    "config": vars(args),
                    "official_config_resolved": official_config_resolved,
                    "history": history,
                },
                save_dir / "stage3_authcode_best.pt",
            )

        with open(save_dir / "stage3_authcode_history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        torch.save(
            {
                "epoch": epoch + 1,
                "teacher_head_state_dict": teacher_head.state_dict(),
                "student_state_dict": student.state_dict(),
                "student_code_head_state_dict": (
                    student_code_head.state_dict() if student_code_head is not None else None
                ),
                "student_token_head_state_dict": (
                    student_token_head.state_dict() if student_token_head is not None else None
                ),
                "claim_score_fusion_state_dict": (
                    claim_score_fusion_head.state_dict() if claim_score_fusion_head is not None else None
                ),
                "auth_codebook_state_dict": auth_codebook.state_dict() if auth_codebook is not None else None,
                "teacher_tokenizer_state_dict": (
                    teacher_tokenizer.state_dict() if teacher_tokenizer is not None else None
                ),
                "claim_verifier_state_dict": claim_verifier.state_dict() if claim_verifier is not None else None,
                "config": vars(args),
                "official_config_resolved": official_config_resolved,
                "history": history,
            },
            save_dir / "stage3_authcode_last.pt",
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Train jointly learned authentication codes for stage-3")
    parser.add_argument("--prototype_cache_dir", type=str, default=str(resolve_experiment_root() / "stage3_prototype_cache_anchor12_joint512_live"))
    parser.add_argument("--teacher_cache_dir", type=str, default=str(resolve_experiment_root() / "stage3_teacher_cache_joint512_live"))
    parser.add_argument("--teacher_key", type=str, default="teacher_joint_seq")
    parser.add_argument("--meta_path", type=str, default=str(resolve_meta_path()))
    parser.add_argument("--rgb_dir", type=str, default=str(resolve_dataset_root() / "rgb_web_jpg"))
    parser.add_argument("--save_dir", type=str, default=str(resolve_experiment_root() / "stage3_authcode_checkpoints"))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--init_checkpoint", type=str, default=None)
    parser.add_argument("--init_student_checkpoint", type=str, default=None)
    parser.add_argument("--code_dim", type=int, default=32)
    parser.add_argument("--payload_dim", type=int, default=None)
    parser.add_argument("--ecc_scheme", type=str, default="identity", choices=["identity", "repetition"])
    parser.add_argument("--ecc_repetition", type=int, default=2)
    parser.add_argument("--student_code_space", type=str, default="payload", choices=["payload", "codeword"])
    parser.add_argument(
        "--auth_protocol_variant",
        type=str,
        default="legacy_continuous",
        choices=["legacy_continuous", "token_residual"],
    )
    parser.add_argument("--teacher_token_classes", type=int, default=0)
    parser.add_argument("--teacher_token_temperature", type=float, default=12.0)
    parser.add_argument("--teacher_codebook_size", type=int, default=0)
    parser.add_argument("--teacher_codebook_temperature", type=float, default=12.0)
    parser.add_argument("--codebook_mode", type=str, default="replace", choices=["replace", "auxiliary"])
    parser.add_argument("--init_codebook_checkpoint", type=str, default=None)
    parser.add_argument("--skip_codebook_init", action="store_true")
    parser.add_argument("--freeze_codebook", action="store_true")
    parser.add_argument("--teacher_hidden_dim", type=int, default=256)
    parser.add_argument("--teacher_dropout", type=float, default=0.0)
    parser.add_argument("--teacher_init_projection_checkpoint", type=str, default=None)
    parser.add_argument("--use_claim_verifier_head", action="store_true")
    parser.add_argument("--claim_verifier_hidden_dim", type=int, default=128)
    parser.add_argument("--claim_verifier_dropout", type=float, default=0.0)
    parser.add_argument("--claim_verifier_weight", type=float, default=0.5)
    parser.add_argument(
        "--claim_verifier_input_mode",
        type=str,
        default="bits",
        choices=["bits", "logits", "bits_logits"],
    )
    parser.add_argument(
        "--claim_verifier_score_mode",
        type=str,
        default="add",
        choices=["add", "tanh_add", "feature_only"],
    )
    parser.add_argument("--claim_verifier_feature_to_fusion", action="store_true")
    parser.add_argument("--freeze_claim_verifier", action="store_true")
    parser.add_argument("--claim_sequence_score_weight", type=float, default=None)
    parser.add_argument("--claim_calibration_scales", type=float, nargs="*", default=None)
    parser.add_argument("--claim_calibration_verifier_weights", type=float, nargs="*", default=None)
    parser.add_argument("--claim_calibration_sequence_weights", type=float, nargs="*", default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--val_batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--pair_logit_scale", type=float, default=12.0)
    parser.add_argument("--pair_logistic_topk", type=int, default=8)
    parser.add_argument("--bit_scale", type=float, default=3.0)
    parser.add_argument("--sequence_score_weight", type=float, default=0.2)
    parser.add_argument("--teacher_consistency_weight", type=float, default=1.0)
    parser.add_argument("--teacher_sequence_match_weight", type=float, default=0.5)
    parser.add_argument("--teacher_pair_weight", type=float, default=0.5)
    parser.add_argument("--teacher_bank_pair_weight", type=float, default=0.5)
    parser.add_argument("--teacher_bank_hard_weight", type=float, default=0.25)
    parser.add_argument("--teacher_claim_pair_weight", type=float, default=0.0)
    parser.add_argument("--teacher_claim_hard_weight", type=float, default=0.0)
    parser.add_argument("--teacher_codebook_commit_weight", type=float, default=0.0)
    parser.add_argument("--teacher_positive_margin", type=float, default=0.55)
    parser.add_argument("--teacher_negative_margin", type=float, default=0.10)
    parser.add_argument("--teacher_hard_margin", type=float, default=0.15)
    parser.add_argument("--student_bit_weight", type=float, default=1.0)
    parser.add_argument("--student_sequence_match_weight", type=float, default=0.5)
    parser.add_argument("--student_pair_weight", type=float, default=1.0)
    parser.add_argument("--student_target_mode", type=str, default="canonical", choices=["canonical", "sample", "blend"])
    parser.add_argument("--student_target_blend_alpha", type=float, default=0.5)
    parser.add_argument("--student_claim_bit_weight", type=float, default=0.0)
    parser.add_argument("--student_claim_pair_weight", type=float, default=0.0)
    parser.add_argument("--student_claim_bank_pair_weight", type=float, default=0.0)
    parser.add_argument("--student_protocol_claim_pair_weight", type=float, default=0.0)
    parser.add_argument("--student_protocol_claim_bank_pair_weight", type=float, default=0.0)
    parser.add_argument("--student_verifier_claim_pair_weight", type=float, default=0.0)
    parser.add_argument("--student_claim_calibration_pair_weight", type=float, default=0.0)
    parser.add_argument("--student_hard_weight", type=float, default=0.0)
    parser.add_argument("--student_claim_hard_weight", type=float, default=0.0)
    parser.add_argument("--student_claim_bank_hard_weight", type=float, default=0.0)
    parser.add_argument("--student_protocol_claim_hard_weight", type=float, default=0.0)
    parser.add_argument("--student_protocol_claim_bank_hard_weight", type=float, default=0.0)
    parser.add_argument("--student_verifier_claim_hard_weight", type=float, default=0.0)
    parser.add_argument("--student_verifier_claim_bank_hard_weight", type=float, default=0.0)
    parser.add_argument("--student_claim_calibration_hard_weight", type=float, default=0.0)
    parser.add_argument("--student_claim_eer_proxy_weight", type=float, default=0.0)
    parser.add_argument("--student_claim_eer_proxy_positive_quantile", type=float, default=0.10)
    parser.add_argument("--student_claim_eer_proxy_negative_quantile", type=float, default=0.90)
    parser.add_argument("--student_claim_eer_proxy_margin", type=float, default=0.02)
    parser.add_argument("--student_claim_eer_proxy_scale", type=float, default=12.0)
    parser.add_argument("--student_claim_positive_tail_weight", type=float, default=0.0)
    parser.add_argument("--student_claim_positive_tail_positive_quantile", type=float, default=0.10)
    parser.add_argument("--student_claim_positive_tail_negative_quantile", type=float, default=0.90)
    parser.add_argument("--student_claim_positive_tail_margin", type=float, default=0.01)
    parser.add_argument("--student_claim_positive_tail_scale", type=float, default=12.0)
    parser.add_argument("--student_claim_hardcase_reweight_strength", type=float, default=0.0)
    parser.add_argument("--student_claim_hardcase_reweight_margin", type=float, default=0.05)
    parser.add_argument("--student_claim_hardcase_reweight_scale", type=float, default=12.0)
    parser.add_argument("--student_soft_align_weight", type=float, default=0.5)
    parser.add_argument("--student_claim_align_weight", type=float, default=0.0)
    parser.add_argument("--student_recovery_bit_weight", type=float, default=0.0)
    parser.add_argument("--student_recovery_stage_weight", type=float, default=0.0)
    parser.add_argument("--student_recovery_align_weight", type=float, default=0.0)
    parser.add_argument("--student_token_class_weight", type=float, default=1.0)
    parser.add_argument("--student_token_proto_weight", type=float, default=0.5)
    parser.add_argument("--use_student_code_head", action="store_true")
    parser.add_argument("--student_code_head_hidden_dim", type=int, default=256)
    parser.add_argument("--student_code_head_dropout", type=float, default=0.0)
    parser.add_argument("--freeze_student_code_head", action="store_true")
    parser.add_argument("--student_token_head_hidden_dim", type=int, default=256)
    parser.add_argument("--student_token_head_dropout", type=float, default=0.0)
    parser.add_argument("--freeze_student_token_head", action="store_true")
    parser.add_argument("--use_claim_score_fusion_head", action="store_true")
    parser.add_argument("--claim_score_fusion_hidden_dim", type=int, default=32)
    parser.add_argument("--claim_score_fusion_dropout", type=float, default=0.0)
    parser.add_argument("--claim_score_fusion_protocol_modes", type=str, nargs="*", default=None)
    parser.add_argument("--claim_score_fusion_mode", type=str, default="direct", choices=["direct", "residual"])
    parser.add_argument("--claim_score_fusion_residual_scale", type=float, default=1.0)
    parser.add_argument("--freeze_claim_score_fusion_head", action="store_true")
    parser.add_argument("--student_score_distill_weight", type=float, default=0.0)
    parser.add_argument("--student_hard_score_distill_weight", type=float, default=0.0)
    parser.add_argument("--student_claim_score_distill_weight", type=float, default=0.0)
    parser.add_argument("--student_codebook_class_weight", type=float, default=0.0)
    parser.add_argument("--student_codebook_proto_weight", type=float, default=0.0)
    parser.add_argument("--student_positive_margin", type=float, default=0.55)
    parser.add_argument("--student_negative_margin", type=float, default=0.10)
    parser.add_argument("--student_hard_margin", type=float, default=0.15)
    parser.add_argument("--score_distill_temperature", type=float, default=0.25)
    parser.add_argument("--score_distill_topk", type=int, default=16)
    parser.add_argument(
        "--claim_reference_mode",
        type=str,
        default="same_raw",
        choices=["same_image", "same_raw", "anchor_bank"],
    )
    parser.add_argument("--claim_reference_versions", type=int, nargs="*", default=None)
    parser.add_argument(
        "--claim_bank_mode",
        type=str,
        default="mean_bits",
        choices=["mean_bits", "mean_logits_tanh", "sign_mean_logits"],
    )
    parser.add_argument("--claim_anchor_aux_weight", type=float, default=0.0)
    parser.add_argument("--teacher_balance_weight", type=float, default=0.1)
    parser.add_argument("--teacher_decorrelation_weight", type=float, default=0.05)
    parser.add_argument("--teacher_uniformity_weight", type=float, default=0.05)
    parser.add_argument("--codebook_usage_weight", type=float, default=0.0)
    parser.add_argument("--codebook_separation_weight", type=float, default=0.0)
    parser.add_argument("--uniformity_temperature", type=float, default=2.0)
    parser.add_argument("--backbone_type", type=str, default="resnet18", choices=["resnet18", "resnet50"])
    parser.add_argument("--input_mode", type=str, default="residual_only", choices=["rgb", "rgb_residual", "residual_only"])
    parser.add_argument("--residual_scale", type=float, default=1.75)
    parser.add_argument("--residual_kernel", type=int, default=9)
    parser.add_argument("--local_crop_mode", type=str, default="none", choices=["none", "center_patch5"])
    parser.add_argument("--local_crop_size", type=int, default=160)
    parser.add_argument("--local_patch_offset", type=int, default=24)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--resize_size", type=int, default=320)
    parser.add_argument("--augmentation_preset", type=str, default="center", choices=["strong", "mild", "none", "center", "center_multi"])
    parser.add_argument("--include_versions", type=int, nargs="*", default=[1, 2, 3, 4])
    parser.add_argument("--anchor_versions", type=int, nargs="*", default=[1, 2])
    parser.add_argument("--shift_versions", type=int, nargs="*", default=[3, 4])
    parser.add_argument("--anchor_weight", type=float, default=1.0)
    parser.add_argument("--shift_weight", type=float, default=1.0)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--max_raws", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--protocol_score_mode",
        type=str,
        default="none",
        choices=["none", "code_cosine", "codeword_agreement", "payload_agreement", "decode_success", "index_match"],
    )
    parser.add_argument("--protocol_score_alpha", type=float, default=0.0)
    parser.add_argument(
        "--official_claim_score_mode",
        type=str,
        default="deterministic_gate",
        choices=["deterministic_gate", "fusion_head"],
    )
    parser.add_argument("--token_gate_penalty", type=float, default=1.0)
    parser.add_argument("--token_residual_weight", type=float, default=0.25)
    parser.add_argument("--token_hard_gate_threshold", type=float, default=0.5)
    parser.add_argument(
        "--claim_main_score_norm_mode",
        type=str,
        default="none",
        choices=["none", "row_center", "row_zscore"],
    )
    parser.add_argument(
        "--protocol_score_norm_mode",
        type=str,
        default="none",
        choices=["none", "row_center", "row_zscore"],
    )
    parser.add_argument("--use_joint_claim_loss", action="store_true")
    parser.add_argument("--official_eval_preset", type=str, default=OFFICIAL_STAGE3_PRESET_NAME)
    parser.add_argument("--selection_metric", type=str, default="val_student_claimed_official_pairwise_auc")
    parser.add_argument("--freeze_teacher_head", action="store_true")
    parser.add_argument("--freeze_student", action="store_true")
    parser.add_argument("--no_pretrained", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train_stage3_authcode(parse_args())
