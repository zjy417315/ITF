import argparse
import json
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

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
from src.train.train_stage3_authcode import (
    aggregate_claim_reference,
    build_same_sample_mask,
    build_same_raw_mask,
    build_bank_targets,
    build_bank_tensors,
    build_claim_score_outputs,
    build_claim_verifier_repr,
    claim_score_fusion_input_dim,
    claim_verifier_dim,
    combine_score_matrices,
    compute_claim_scores,
    compute_claim_verifier_scores,
    compute_combined_scores,
    compute_protocol_score_bundle,
    compute_protocol_score_matrix,
    compute_token_match_matrix,
    fuse_claim_score_matrices,
    forward_student_branches,
    forward_teacher_bank,
    normalize_protocol_mode_list,
    summarize_masked_scores,
    soft_bits,
)
from src.train.train_stage3_code import summarize_verification_scores
from src.train.train_stage3_prototype import (
    accuracy_for_versions,
    build_group_split,
    build_transforms,
    load_state_dict_shape_safe,
    set_seed,
)
from src.tools.data_roots import resolve_dataset_root, resolve_experiment_root, resolve_meta_path
from src.tools.ecc_codec import build_auth_codec
from src.tools.stage3_official_preset import (
    OFFICIAL_STAGE3_EVAL_MAX_RAWS,
    OFFICIAL_STAGE3_PRESET_NAME,
    resolve_stage3_official_config,
)


def build_anchor_claim_bank(
    dataset: PrototypeDistillationDataset,
    indices,
    claim_bank_raws,
    teacher_head: AuthCodeHead,
    use_sequence: bool,
    bit_scale: float,
    anchor_versions,
    device: torch.device,
    claim_bank_mode: str = "mean_bits",
    auth_codec=None,
    auth_codebook: AuthenticationCodebook = None,
    apply_codebook_to_scores: bool = True,
) -> torch.Tensor:
    anchor_versions = {int(v) for v in anchor_versions}
    raw_to_anchor_bits = {raw_anchor: [] for raw_anchor in claim_bank_raws}
    raw_to_all_bits = {raw_anchor: [] for raw_anchor in claim_bank_raws}
    raw_to_anchor_logits = {raw_anchor: [] for raw_anchor in claim_bank_raws}
    raw_to_all_logits = {raw_anchor: [] for raw_anchor in claim_bank_raws}
    with torch.no_grad():
        for idx in indices:
            sample = dataset[idx]
            raw_anchor = sample["raw_anchor"]
            if raw_anchor not in raw_to_all_bits:
                continue
            teacher_out = teacher_head(
                sample["teacher_vec"].unsqueeze(0).to(device),
                sample["teacher_seq"].unsqueeze(0).to(device),
                return_sequence=use_sequence,
                return_logits=True,
            )
            teacher_logits = teacher_out["global_logits"]
            if auth_codec is not None:
                teacher_logits = auth_codec.encode_logits(teacher_logits)
            if auth_codebook is not None and apply_codebook_to_scores:
                teacher_logits, _, _, _ = auth_codebook.quantize(teacher_logits, straight_through=False)
            teacher_logits = teacher_logits.squeeze(0).detach().cpu()
            teacher_bits = soft_bits(teacher_logits.unsqueeze(0), bit_scale).squeeze(0).detach().cpu()
            raw_to_all_bits[raw_anchor].append(teacher_bits)
            raw_to_all_logits[raw_anchor].append(teacher_logits)
            if int(sample["version"]) in anchor_versions:
                raw_to_anchor_bits[raw_anchor].append(teacher_bits)
                raw_to_anchor_logits[raw_anchor].append(teacher_logits)

    claim_bank = []
    for raw_anchor in claim_bank_raws:
        bit_list = raw_to_anchor_bits[raw_anchor] or raw_to_all_bits[raw_anchor]
        logit_list = raw_to_anchor_logits[raw_anchor] or raw_to_all_logits[raw_anchor]
        claim_bank.append(
            aggregate_claim_reference(
                bit_list=bit_list,
                logit_list=logit_list,
                bit_scale=bit_scale,
                claim_bank_mode=claim_bank_mode,
            )
        )
    return torch.stack(claim_bank, dim=0).to(device)


def evaluate(args):
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint.get("config", {})
    set_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    official_eval_preset, official_config_resolved = resolve_stage3_official_config(
        config=config,
        preset_name=args.official_eval_preset,
        cli_overrides={
            "claim_reference_mode": args.claim_reference_mode,
            "official_claim_score_mode": args.override_official_claim_score_mode,
            "protocol_score_mode": args.protocol_score_mode,
            "bit_scale": args.override_bit_scale,
            "claim_verifier_weight": args.override_claim_verifier_weight,
            "claim_main_score_norm_mode": args.claim_main_score_norm_mode,
            "protocol_score_norm_mode": args.protocol_score_norm_mode,
        },
    )
    claim_reference_mode = str(
        official_config_resolved.get("claim_reference_mode", config.get("claim_reference_mode", "same_raw"))
    )
    bit_scale = float(official_config_resolved.get("bit_scale", config.get("bit_scale", 3.0)))
    verifier_weight = float(
        official_config_resolved.get("claim_verifier_weight", config.get("claim_verifier_weight", 0.5))
    )
    claim_sequence_score_weight_value = args.override_claim_sequence_score_weight
    if claim_sequence_score_weight_value is None:
        claim_sequence_score_weight_value = config.get("claim_sequence_score_weight")
    if claim_sequence_score_weight_value is None:
        claim_sequence_score_weight_value = config.get("sequence_score_weight", 0.0)
    claim_sequence_score_weight = float(claim_sequence_score_weight_value)
    protocol_score_mode = str(
        official_config_resolved.get("protocol_score_mode", config.get("protocol_score_mode", "none"))
    )
    protocol_score_alpha = float(
        args.protocol_score_alpha
        if args.protocol_score_alpha is not None
        else config.get("protocol_score_alpha", 0.0)
    )
    official_claim_score_mode = str(
        official_config_resolved.get("official_claim_score_mode", config.get("official_claim_score_mode", "deterministic_gate"))
    )
    token_gate_penalty = float(
        args.override_token_gate_penalty
        if args.override_token_gate_penalty is not None
        else config.get("token_gate_penalty", 1.0)
    )
    token_residual_weight = float(
        args.override_token_residual_weight
        if args.override_token_residual_weight is not None
        else config.get("token_residual_weight", 0.25)
    )
    claim_main_score_norm_mode = str(
        official_config_resolved.get("claim_main_score_norm_mode", config.get("claim_main_score_norm_mode", "none"))
    )
    protocol_score_norm_mode = str(
        official_config_resolved.get("protocol_score_norm_mode", config.get("protocol_score_norm_mode", "none"))
    )
    if official_eval_preset is not None and int(args.max_raws) != OFFICIAL_STAGE3_EVAL_MAX_RAWS:
        print(
            f"[warning] official preset {official_eval_preset} is usually reported with "
            f"max_raws={OFFICIAL_STAGE3_EVAL_MAX_RAWS}, but current evaluation uses max_raws={args.max_raws}.",
            flush=True,
        )

    _, eval_transform = build_transforms(
        args.eval_preset or config.get("augmentation_preset", "center"),
        image_size=int(config.get("image_size", 224)),
        resize_size=int(config.get("resize_size", 320)),
    )

    dataset = PrototypeDistillationDataset(
        prototype_cache_dir=args.prototype_cache_dir,
        meta_path=args.meta_path,
        rgb_dir=args.rgb_dir,
        teacher_cache_dir=args.teacher_cache_dir,
        teacher_key=args.teacher_key or config.get("teacher_key", "teacher_joint_seq"),
        include_versions=args.include_versions,
        transform=eval_transform,
    )
    train_indices, val_indices, train_raws, val_raws = build_group_split(
        dataset=dataset,
        val_ratio=args.val_ratio,
        seed=args.seed,
        max_raws=args.max_raws,
    )

    use_sequence = bool(
        config.get("sequence_score_weight", 0.0) > 0.0
        or config.get("teacher_sequence_match_weight", 0.0) > 0.0
        or config.get("student_sequence_match_weight", 0.0) > 0.0
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
        auth_codebook = AuthenticationCodebook(
            code_dim=auth_codec.code_dim,
            num_codes=int(config.get("teacher_codebook_size", 0) or checkpoint["auth_codebook_state_dict"]["codes"].shape[0]),
            temperature=float(config.get("teacher_codebook_temperature", 12.0)),
            learnable=False,
        ).to(device)
        if checkpoint.get("auth_codebook_state_dict") is not None:
            auth_codebook.load_state_dict(checkpoint["auth_codebook_state_dict"])
        auth_codebook.eval()
    codebook_mode = str(config.get("codebook_mode", "replace"))
    apply_codebook_to_scores = auth_codebook is not None and codebook_mode == "replace"

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

    student = None
    student_code_head = None
    student_token_head = None
    claim_verifier = None
    claim_score_fusion_head = None
    claim_verifier_enabled = bool(
        config.get("use_claim_verifier_head", False)
        or ("claim_verifier_state_dict" in checkpoint and checkpoint["claim_verifier_state_dict"] is not None)
    )
    if args.mode == "student":
        use_student_code_head = bool(
            config.get("use_student_code_head", False)
            or checkpoint.get("student_code_head_state_dict") is not None
        )
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
        if use_student_code_head:
            student_code_head = StudentCodeHead(
                d_in=student.feature_dim,
                code_dim=auth_codec.code_dim,
                hidden_dim=int(config.get("student_code_head_hidden_dim", 256)),
                dropout=float(config.get("student_code_head_dropout", 0.0)),
            ).to(device)
            if checkpoint.get("student_code_head_state_dict") is not None:
                student_code_head.load_state_dict(checkpoint["student_code_head_state_dict"])
            student_code_head.eval()
        if token_protocol_enabled and checkpoint.get("student_token_head_state_dict") is not None:
            token_head_state = checkpoint["student_token_head_state_dict"]
            fallback_num_tokens = None
            for key, value in token_head_state.items():
                if key.endswith("weight") and value.ndim == 2:
                    fallback_num_tokens = int(value.shape[0])
            if fallback_num_tokens is None:
                raise RuntimeError("Failed to infer student token head output dimension from checkpoint.")
            student_token_head = StudentTokenHead(
                d_in=student.feature_dim,
                num_tokens=int(config.get("teacher_token_classes", fallback_num_tokens)),
                code_dim=int(config.get("code_dim", 32)),
                hidden_dim=int(config.get("student_token_head_hidden_dim", 256)),
                dropout=float(config.get("student_token_head_dropout", 0.0)),
            ).to(device)
            student_token_head.load_state_dict(token_head_state)
            student_token_head.eval()
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
                load_status = load_state_dict_shape_safe(
                    claim_score_fusion_head,
                    checkpoint["claim_score_fusion_state_dict"],
                    module_name="ClaimScoreFusionHead",
                )
                if load_status.missing_keys:
                    print(f"ClaimScoreFusionHead missing keys on eval load: {load_status.missing_keys}", flush=True)
                if load_status.unexpected_keys:
                    print(f"ClaimScoreFusionHead unexpected keys on eval load: {load_status.unexpected_keys}", flush=True)
            claim_score_fusion_head.eval()
    if (
        config.get("use_claim_verifier_head", False)
        or ("claim_verifier_state_dict" in checkpoint and checkpoint["claim_verifier_state_dict"] is not None)
    ):
        claim_verifier = PrototypeVerifier(
            d_model=claim_verifier_dim(
                auth_codec.code_dim,
                config.get("claim_verifier_input_mode", "bits"),
            ),
            hidden_dim=int(config.get("claim_verifier_hidden_dim", 128)),
            dropout=float(config.get("claim_verifier_dropout", 0.0)),
        ).to(device)
        if checkpoint.get("claim_verifier_state_dict") is not None:
            load_status = load_state_dict_shape_safe(
                claim_verifier,
                checkpoint["claim_verifier_state_dict"],
                module_name="ClaimVerifier",
            )
            if load_status.missing_keys:
                print(f"ClaimVerifier missing keys on eval load: {load_status.missing_keys}", flush=True)
            if load_status.unexpected_keys:
                print(f"ClaimVerifier unexpected keys on eval load: {load_status.unexpected_keys}", flush=True)
        claim_verifier.eval()

    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    proto_vec_bank, proto_seq_bank, bank_raws = build_bank_tensors(dataset, val_raws, device)
    with torch.no_grad():
        bank_out, bank_bits, bank_stage_bits = forward_teacher_bank(
            teacher_head,
            proto_vec_bank,
            proto_seq_bank,
            bit_scale=bit_scale,
            use_sequence_score=use_sequence,
            auth_codec=auth_codec,
            auth_codebook=auth_codebook,
            apply_codebook_to_scores=apply_codebook_to_scores,
        )
    bank_logits = bank_out["global_logits"]
    bank_token_indices = None
    if teacher_tokenizer is not None:
        bank_token_indices, _ = teacher_tokenizer.assign(bank_logits.detach())
    claim_bank = None
    claim_bank_raws = list(val_raws)
    if claim_reference_mode == "anchor_bank":
        claim_versions = (
            args.claim_reference_versions
            if args.claim_reference_versions is not None
            else config.get("anchor_versions", [1, 2])
        )
        claim_bank = build_anchor_claim_bank(
            dataset=dataset,
            indices=val_indices,
            claim_bank_raws=claim_bank_raws,
            teacher_head=teacher_head,
            use_sequence=use_sequence,
            bit_scale=bit_scale,
            anchor_versions=claim_versions,
            device=device,
            claim_bank_mode=args.claim_bank_mode or config.get("claim_bank_mode", "mean_bits"),
            auth_codec=auth_codec,
            auth_codebook=auth_codebook,
            apply_codebook_to_scores=apply_codebook_to_scores,
        )

    pos_scores = []
    hard_neg_scores = []
    all_neg_scores = []
    claimed_pos_scores = []
    claimed_hard_neg_scores = []
    claimed_all_neg_scores = []
    claimed_main_pos_scores = []
    claimed_main_hard_neg_scores = []
    claimed_main_all_neg_scores = []
    claimed_protocol_pos_scores = []
    claimed_protocol_hard_neg_scores = []
    claimed_protocol_all_neg_scores = []
    claimed_verifier_pos_scores = []
    claimed_verifier_hard_neg_scores = []
    claimed_verifier_all_neg_scores = []
    claimed_token_pos_scores = []
    claimed_token_hard_neg_scores = []
    claimed_token_all_neg_scores = []
    claimed_gated_pos_scores = []
    claimed_gated_hard_neg_scores = []
    claimed_gated_all_neg_scores = []
    token_top1_scores = []
    recovery_codeword_ber = []
    recovery_payload_ber = []
    recovery_decode_success = []
    recovery_code_index_match = []
    anchor_acc_sum = 0.0
    shift_acc_sum = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            raw_anchors = list(batch["raw_anchor"])
            versions = batch["version"].to(device)
            targets = build_bank_targets(raw_anchors, bank_raws, device=device)
            teacher_batch_out = teacher_head(
                batch["teacher_vec"].to(device),
                batch["teacher_seq"].to(device),
                return_sequence=use_sequence,
                return_logits=True,
            )
            teacher_encoded_logits = encode_teacher_logits(teacher_batch_out["global_logits"])
            teacher_code_logits, teacher_code_indices = quantize_global_logits(teacher_encoded_logits)
            teacher_global_logits = teacher_code_logits if apply_codebook_to_scores and teacher_code_logits is not None else teacher_encoded_logits
            teacher_stage_logits = encode_teacher_logits(teacher_batch_out.get("stage_logits"))
            teacher_token_indices = None
            teacher_token_logits = None
            if teacher_tokenizer is not None:
                teacher_token_indices, teacher_token_logits = teacher_tokenizer.assign(teacher_encoded_logits)

            if args.mode == "teacher":
                query_global_logits = teacher_global_logits
                query_code_logits = teacher_code_logits if teacher_code_logits is not None else teacher_global_logits
                query_code_indices = teacher_code_indices
                query_stage_logits = teacher_stage_logits
                query_token_logits = teacher_tokenizer.logits(teacher_encoded_logits) if teacher_tokenizer is not None else None
            else:
                student_forward = forward_student_branches(
                    student=student,
                    rgb_image=batch["rgb_image"].to(device),
                    use_sequence=use_sequence,
                    encode_student_logits=encode_student_logits,
                    auth_codebook=auth_codebook,
                    apply_codebook_to_scores=apply_codebook_to_scores,
                    student_code_head=student_code_head,
                    student_token_head=student_token_head,
                )
                query_encoded_logits = student_forward["score_encoded_logits"]
                query_global_logits = student_forward["score_logits"]
                query_stage_logits = student_forward["stage_logits"]
                query_code_logits = student_forward["recovery_protocol_logits"]
                query_code_indices = student_forward["recovery_code_indices"]
                query_token_logits = student_forward["token_logits"]
                if teacher_tokenizer is not None:
                    base_query_token_logits = teacher_tokenizer.logits(query_code_logits)
                    if query_token_logits is None:
                        query_token_logits = base_query_token_logits
                    else:
                        query_token_logits = base_query_token_logits + query_token_logits

            scores, _ = compute_combined_scores(
                query_global_logits,
                bank_bits,
                query_stage_logits=query_stage_logits,
                bank_stage_bits=bank_stage_bits,
                bit_scale=bit_scale,
                sequence_score_weight=float(
                    args.override_sequence_score_weight
                    if args.override_sequence_score_weight is not None
                    else config.get("sequence_score_weight", 0.0)
                ),
            )

            pos = scores.gather(1, targets.unsqueeze(1)).squeeze(1)
            negative_mask = torch.ones_like(scores, dtype=torch.bool)
            negative_mask.scatter_(1, targets.unsqueeze(1), False)
            neg = scores[negative_mask].view(scores.shape[0], -1)
            hard_neg = neg.max(dim=1).values

            pos_scores.append(pos.cpu())
            hard_neg_scores.append(hard_neg.cpu())
            all_neg_scores.append(neg.reshape(-1).cpu())

            student_bits = soft_bits(query_global_logits, bit_scale)
            teacher_bits = soft_bits(teacher_global_logits, bit_scale)
            verifier_input_mode = str(config.get("claim_verifier_input_mode", "bits"))
            student_verifier_repr = build_claim_verifier_repr(student_bits, query_global_logits, verifier_input_mode)
            teacher_verifier_repr = build_claim_verifier_repr(teacher_bits, teacher_global_logits, verifier_input_mode)
            teacher_recovery_logits = teacher_code_logits if teacher_code_logits is not None else teacher_global_logits
            query_recovery_logits = query_code_logits if query_code_logits is not None else query_global_logits
            teacher_code_bits = auth_codec.hard_codeword_bits(teacher_recovery_logits)
            query_code_bits = auth_codec.hard_codeword_bits(query_recovery_logits)
            recovery_codeword_ber.append((query_code_bits != teacher_code_bits).float().mean(dim=1).cpu())
            teacher_payload_bits = auth_codec.hard_payload_bits_from_codeword(teacher_recovery_logits)
            query_payload_bits = auth_codec.hard_payload_bits_from_codeword(query_recovery_logits)
            recovery_payload_ber.append((query_payload_bits != teacher_payload_bits).float().mean(dim=1).cpu())
            recovery_decode_success.append((query_payload_bits == teacher_payload_bits).all(dim=1).float().cpu())
            if teacher_code_indices is not None and query_code_indices is not None:
                recovery_code_index_match.append((query_code_indices == teacher_code_indices).float().cpu())
            sample_ids = list(batch["sample_id"])
            claimed_token_scores = None
            claimed_protocol_scores = None
            claimed_verifier_scores = None
            claimed_extra_protocol_scores = []
            claimed_score_outputs = None
            if claim_reference_mode == "anchor_bank":
                claim_targets = build_bank_targets(raw_anchors, claim_bank_raws, device=device)
                claimed_main_scores = compute_claim_scores(
                    student_bits,
                    claim_bank,
                    claim_verifier,
                    verifier_weight=verifier_weight,
                    verifier_score_mode=str(config.get("claim_verifier_score_mode", "add")),
                    query_stage_logits=query_stage_logits,
                    reference_stage_logits=None,
                    reference_stage_bits=None,
                    bit_scale=bit_scale,
                    sequence_score_weight=claim_sequence_score_weight,
                )
                claimed_verifier_scores = compute_claim_verifier_scores(
                    student_bits,
                    claim_bank,
                    claim_verifier,
                    verifier_weight=verifier_weight,
                    verifier_score_mode=str(config.get("claim_verifier_score_mode", "add")),
                )
                claimed_scores = claimed_main_scores
                claimed_pos = claimed_scores.gather(1, claim_targets.unsqueeze(1)).squeeze(1)
                claim_negative_mask = torch.ones_like(claimed_scores, dtype=torch.bool)
                claim_negative_mask.scatter_(1, claim_targets.unsqueeze(1), False)
                claimed_neg = claimed_scores[claim_negative_mask].view(claimed_scores.shape[0], -1)
                claimed_hard_neg = claimed_neg.max(dim=1).values
                claimed_main_pos = claimed_main_scores.gather(1, claim_targets.unsqueeze(1)).squeeze(1)
                claimed_main_neg = claimed_main_scores[claim_negative_mask].view(claimed_main_scores.shape[0], -1)
                claimed_main_hard_neg = claimed_main_neg.max(dim=1).values
                if claimed_verifier_scores is not None:
                    verifier_pos = claimed_verifier_scores.gather(1, claim_targets.unsqueeze(1)).squeeze(1)
                    verifier_neg = claimed_verifier_scores[claim_negative_mask].view(claimed_verifier_scores.shape[0], -1)
                    verifier_hard_neg = verifier_neg.max(dim=1).values
                    claimed_verifier_pos_scores.append(verifier_pos.cpu())
                    claimed_verifier_hard_neg_scores.append(verifier_hard_neg.cpu())
                    claimed_verifier_all_neg_scores.append(verifier_neg.reshape(-1).cpu())
            elif claim_reference_mode == "same_image":
                teacher_stage_bits = (
                    soft_bits(teacher_stage_logits, bit_scale)
                    if teacher_stage_logits is not None
                    else None
                )
                claimed_main_scores = compute_claim_scores(
                    student_bits,
                    teacher_bits,
                    claim_verifier,
                    verifier_weight=verifier_weight,
                    verifier_score_mode=str(config.get("claim_verifier_score_mode", "add")),
                    query_verifier_repr=student_verifier_repr,
                    reference_verifier_repr=teacher_verifier_repr,
                    query_stage_logits=query_stage_logits,
                    reference_stage_logits=teacher_stage_logits,
                    reference_stage_bits=teacher_stage_bits,
                    bit_scale=bit_scale,
                    sequence_score_weight=claim_sequence_score_weight,
                )
                claimed_verifier_scores = compute_claim_verifier_scores(
                    student_bits,
                    teacher_bits,
                    claim_verifier,
                    verifier_weight=verifier_weight,
                    verifier_score_mode=str(config.get("claim_verifier_score_mode", "add")),
                    query_verifier_repr=student_verifier_repr,
                    reference_verifier_repr=teacher_verifier_repr,
                )
                claimed_protocol_scores, claimed_extra_protocol_scores = compute_protocol_score_bundle(
                    query_protocol_logits=query_recovery_logits,
                    reference_protocol_logits=teacher_recovery_logits,
                    auth_codec=auth_codec,
                    primary_mode=protocol_score_mode,
                    extra_modes=config.get("claim_score_fusion_protocol_modes"),
                    query_code_indices=query_code_indices,
                    reference_code_indices=teacher_code_indices,
                )
                if bool(config.get("claim_verifier_feature_to_fusion", False)) and claimed_verifier_scores is not None:
                    claimed_extra_protocol_scores.append(claimed_verifier_scores)
                claimed_token_scores = compute_token_match_matrix(
                    query_token_logits,
                    teacher_token_indices,
                )
                claimed_score_outputs = build_claim_score_outputs(
                    claimed_main_scores,
                    claimed_protocol_scores,
                    token_scores=claimed_token_scores,
                    extra_protocol_scores=claimed_extra_protocol_scores,
                    claim_score_fusion_head=claim_score_fusion_head,
                    official_mode=official_claim_score_mode,
                    alpha=protocol_score_alpha,
                    main_normalization=claim_main_score_norm_mode,
                    auxiliary_normalization=protocol_score_norm_mode,
                    gate_penalty=token_gate_penalty,
                    residual_weight=token_residual_weight,
                    hard_gate_threshold=float(config.get("token_hard_gate_threshold", 0.5)),
                )
                claimed_scores = claimed_score_outputs["official_scores"]
                claim_positive_mask = build_same_sample_mask(sample_ids, device=device)
                claim_negative_mask = ~build_same_raw_mask(raw_anchors, device=device)
                claimed_main_pos, claimed_main_hard_neg, claimed_main_neg = summarize_masked_scores(
                    claimed_main_scores,
                    claim_positive_mask,
                    negative_mask=claim_negative_mask,
                )
                if claimed_protocol_scores is not None:
                    protocol_pos, protocol_hard_neg, protocol_neg = summarize_masked_scores(
                        claimed_protocol_scores,
                        claim_positive_mask,
                        negative_mask=claim_negative_mask,
                    )
                    claimed_protocol_pos_scores.append(protocol_pos.cpu())
                    claimed_protocol_hard_neg_scores.append(protocol_hard_neg.cpu())
                    claimed_protocol_all_neg_scores.append(protocol_neg.reshape(-1).cpu())
                if claimed_verifier_scores is not None:
                    verifier_pos, verifier_hard_neg, verifier_neg = summarize_masked_scores(
                        claimed_verifier_scores,
                        claim_positive_mask,
                        negative_mask=claim_negative_mask,
                    )
                    claimed_verifier_pos_scores.append(verifier_pos.cpu())
                    claimed_verifier_hard_neg_scores.append(verifier_hard_neg.cpu())
                    claimed_verifier_all_neg_scores.append(verifier_neg.reshape(-1).cpu())
                if claimed_token_scores is not None:
                    token_pos, token_hard_neg, token_neg = summarize_masked_scores(
                        claimed_token_scores,
                        claim_positive_mask,
                        negative_mask=claim_negative_mask,
                    )
                    claimed_token_pos_scores.append(token_pos.cpu())
                    claimed_token_hard_neg_scores.append(token_hard_neg.cpu())
                    claimed_token_all_neg_scores.append(token_neg.reshape(-1).cpu())
                    gated_pos, gated_hard_neg, gated_neg = summarize_masked_scores(
                        claimed_score_outputs["gated_scores"],
                        claim_positive_mask,
                        negative_mask=claim_negative_mask,
                    )
                    claimed_gated_pos_scores.append(gated_pos.cpu())
                    claimed_gated_hard_neg_scores.append(gated_hard_neg.cpu())
                    claimed_gated_all_neg_scores.append(gated_neg.reshape(-1).cpu())
                claimed_pos, claimed_hard_neg, claimed_neg = summarize_masked_scores(
                    claimed_scores,
                    claim_positive_mask,
                    negative_mask=claim_negative_mask,
                )
            else:
                teacher_stage_bits = (
                    soft_bits(teacher_stage_logits, bit_scale)
                    if teacher_stage_logits is not None
                    else None
                )
                claimed_main_scores = compute_claim_scores(
                    student_bits,
                    teacher_bits,
                    claim_verifier,
                    verifier_weight=verifier_weight,
                    verifier_score_mode=str(config.get("claim_verifier_score_mode", "add")),
                    query_verifier_repr=student_verifier_repr,
                    reference_verifier_repr=teacher_verifier_repr,
                    query_stage_logits=query_stage_logits,
                    reference_stage_logits=teacher_stage_logits,
                    reference_stage_bits=teacher_stage_bits,
                    bit_scale=bit_scale,
                    sequence_score_weight=claim_sequence_score_weight,
                )
                claimed_verifier_scores = compute_claim_verifier_scores(
                    student_bits,
                    teacher_bits,
                    claim_verifier,
                    verifier_weight=verifier_weight,
                    verifier_score_mode=str(config.get("claim_verifier_score_mode", "add")),
                    query_verifier_repr=student_verifier_repr,
                    reference_verifier_repr=teacher_verifier_repr,
                )
                claimed_protocol_scores, claimed_extra_protocol_scores = compute_protocol_score_bundle(
                    query_protocol_logits=query_recovery_logits,
                    reference_protocol_logits=teacher_recovery_logits,
                    auth_codec=auth_codec,
                    primary_mode=protocol_score_mode,
                    extra_modes=config.get("claim_score_fusion_protocol_modes"),
                    query_code_indices=query_code_indices,
                    reference_code_indices=teacher_code_indices,
                )
                if bool(config.get("claim_verifier_feature_to_fusion", False)) and claimed_verifier_scores is not None:
                    claimed_extra_protocol_scores.append(claimed_verifier_scores)
                claimed_token_scores = compute_token_match_matrix(
                    query_token_logits,
                    teacher_token_indices,
                )
                claimed_score_outputs = build_claim_score_outputs(
                    claimed_main_scores,
                    claimed_protocol_scores,
                    token_scores=claimed_token_scores,
                    extra_protocol_scores=claimed_extra_protocol_scores,
                    claim_score_fusion_head=claim_score_fusion_head,
                    official_mode=official_claim_score_mode,
                    alpha=protocol_score_alpha,
                    main_normalization=claim_main_score_norm_mode,
                    auxiliary_normalization=protocol_score_norm_mode,
                    gate_penalty=token_gate_penalty,
                    residual_weight=token_residual_weight,
                    hard_gate_threshold=float(config.get("token_hard_gate_threshold", 0.5)),
                )
                claimed_scores = claimed_score_outputs["official_scores"]
                claim_positive_mask = build_same_raw_mask(raw_anchors, device=device)
                claimed_main_pos, claimed_main_hard_neg, claimed_main_neg = summarize_masked_scores(
                    claimed_main_scores,
                    claim_positive_mask,
                )
                if claimed_protocol_scores is not None:
                    protocol_pos, protocol_hard_neg, protocol_neg = summarize_masked_scores(
                        claimed_protocol_scores,
                        claim_positive_mask,
                    )
                    claimed_protocol_pos_scores.append(protocol_pos.cpu())
                    claimed_protocol_hard_neg_scores.append(protocol_hard_neg.cpu())
                    claimed_protocol_all_neg_scores.append(protocol_neg.reshape(-1).cpu())
                if claimed_verifier_scores is not None:
                    verifier_pos, verifier_hard_neg, verifier_neg = summarize_masked_scores(
                        claimed_verifier_scores,
                        claim_positive_mask,
                    )
                    claimed_verifier_pos_scores.append(verifier_pos.cpu())
                    claimed_verifier_hard_neg_scores.append(verifier_hard_neg.cpu())
                    claimed_verifier_all_neg_scores.append(verifier_neg.reshape(-1).cpu())
                if claimed_token_scores is not None:
                    token_pos, token_hard_neg, token_neg = summarize_masked_scores(
                        claimed_token_scores,
                        claim_positive_mask,
                    )
                    claimed_token_pos_scores.append(token_pos.cpu())
                    claimed_token_hard_neg_scores.append(token_hard_neg.cpu())
                    claimed_token_all_neg_scores.append(token_neg.reshape(-1).cpu())
                    gated_pos, gated_hard_neg, gated_neg = summarize_masked_scores(
                        claimed_score_outputs["gated_scores"],
                        claim_positive_mask,
                    )
                    claimed_gated_pos_scores.append(gated_pos.cpu())
                    claimed_gated_hard_neg_scores.append(gated_hard_neg.cpu())
                    claimed_gated_all_neg_scores.append(gated_neg.reshape(-1).cpu())
                claimed_pos, claimed_hard_neg, claimed_neg = summarize_masked_scores(claimed_scores, claim_positive_mask)
            claimed_main_pos_scores.append(claimed_main_pos.cpu())
            claimed_main_hard_neg_scores.append(claimed_main_hard_neg.cpu())
            claimed_main_all_neg_scores.append(claimed_main_neg.reshape(-1).cpu())
            claimed_pos_scores.append(claimed_pos.cpu())
            claimed_hard_neg_scores.append(claimed_hard_neg.cpu())
            claimed_all_neg_scores.append(claimed_neg.reshape(-1).cpu())
            if teacher_token_indices is not None and query_token_logits is not None:
                token_top1_scores.append(
                    (query_token_logits.argmax(dim=1) == teacher_token_indices).float().cpu()
                )

            correct = scores.argmax(dim=1) == targets
            anchor_acc = accuracy_for_versions(correct, versions, args.anchor_versions)
            shift_acc = accuracy_for_versions(correct, versions, args.shift_versions)
            anchor_acc_sum += 0.0 if math.isnan(anchor_acc) else anchor_acc
            shift_acc_sum += 0.0 if math.isnan(shift_acc) else shift_acc
            total_batches += 1

    pos_scores = torch.cat(pos_scores, dim=0)
    hard_neg_scores = torch.cat(hard_neg_scores, dim=0)
    all_neg_scores = torch.cat(all_neg_scores, dim=0)
    metrics = summarize_verification_scores(pos_scores, hard_neg_scores, all_neg_scores)
    if claimed_pos_scores:
        claimed_main_metrics = summarize_verification_scores(
            torch.cat(claimed_main_pos_scores, dim=0),
            torch.cat(claimed_main_hard_neg_scores, dim=0),
            torch.cat(claimed_main_all_neg_scores, dim=0),
        )
        metrics.update({f"claimed_main_{key}": value for key, value in claimed_main_metrics.items()})
        claimed_metrics = summarize_verification_scores(
            torch.cat(claimed_pos_scores, dim=0),
            torch.cat(claimed_hard_neg_scores, dim=0),
            torch.cat(claimed_all_neg_scores, dim=0),
        )
        metrics.update({f"claimed_{key}": value for key, value in claimed_metrics.items()})
        if claimed_protocol_pos_scores:
            claimed_protocol_metrics = summarize_verification_scores(
                torch.cat(claimed_protocol_pos_scores, dim=0),
                torch.cat(claimed_protocol_hard_neg_scores, dim=0),
                torch.cat(claimed_protocol_all_neg_scores, dim=0),
            )
            metrics.update({f"claimed_protocol_{key}": value for key, value in claimed_protocol_metrics.items()})
        if claimed_verifier_pos_scores:
            claimed_verifier_metrics = summarize_verification_scores(
                torch.cat(claimed_verifier_pos_scores, dim=0),
                torch.cat(claimed_verifier_hard_neg_scores, dim=0),
                torch.cat(claimed_verifier_all_neg_scores, dim=0),
            )
            metrics.update({f"claimed_verifier_{key}": value for key, value in claimed_verifier_metrics.items()})
        if claimed_token_pos_scores:
            claimed_token_metrics = summarize_verification_scores(
                torch.cat(claimed_token_pos_scores, dim=0),
                torch.cat(claimed_token_hard_neg_scores, dim=0),
                torch.cat(claimed_token_all_neg_scores, dim=0),
            )
            metrics.update({f"claimed_token_{key}": value for key, value in claimed_token_metrics.items()})
        if claimed_gated_pos_scores:
            claimed_gated_metrics = summarize_verification_scores(
                torch.cat(claimed_gated_pos_scores, dim=0),
                torch.cat(claimed_gated_hard_neg_scores, dim=0),
                torch.cat(claimed_gated_all_neg_scores, dim=0),
            )
            metrics.update({f"claimed_gated_{key}": value for key, value in claimed_gated_metrics.items()})
    metrics.update(
        {
            "mode": args.mode,
            "checkpoint": args.checkpoint,
            "teacher_key": args.teacher_key or config.get("teacher_key", "teacher_joint_seq"),
            "auth_protocol_variant": config.get("auth_protocol_variant", "legacy_continuous"),
            "code_dim": auth_codec.code_dim,
            "payload_dim": auth_codec.payload_dim,
            "ecc_scheme": auth_codec.scheme,
            "ecc_repetition": auth_codec.repetition,
            "student_code_space": student_code_space,
            "teacher_codebook_size": 0 if auth_codebook is None else auth_codebook.num_codes,
            "codebook_mode": codebook_mode,
            "use_student_code_head": student_code_head is not None,
            "use_student_token_head": student_token_head is not None,
            "use_claim_score_fusion_head": claim_score_fusion_head is not None,
            "bit_scale": bit_scale,
            "sequence_score_weight": float(
                args.override_sequence_score_weight
                if args.override_sequence_score_weight is not None
                else config.get("sequence_score_weight", 0.0)
            ),
            "claimed_score_mode": official_claim_score_mode,
            "claim_verifier_weight": verifier_weight,
            "claim_verifier_score_mode": config.get("claim_verifier_score_mode", "add"),
            "claim_verifier_feature_to_fusion": bool(config.get("claim_verifier_feature_to_fusion", False)),
            "claim_sequence_score_weight": claim_sequence_score_weight,
            "claim_reference_mode": claim_reference_mode,
            "claim_bank_mode": args.claim_bank_mode or config.get("claim_bank_mode", "mean_bits"),
            "protocol_score_mode": protocol_score_mode,
            "protocol_score_alpha": protocol_score_alpha,
            "token_gate_penalty": token_gate_penalty,
            "token_residual_weight": token_residual_weight,
            "official_eval_preset": official_eval_preset,
            "official_config_resolved": official_config_resolved,
            "official_expected_max_raws": OFFICIAL_STAGE3_EVAL_MAX_RAWS if official_eval_preset is not None else None,
            "token_top1_acc": (
                torch.cat(token_top1_scores, dim=0).mean().item() if token_top1_scores else None
            ),
            "claim_main_score_norm_mode": claim_main_score_norm_mode,
            "protocol_score_norm_mode": protocol_score_norm_mode,
            "claim_score_fusion_protocol_modes": normalize_protocol_mode_list(config.get("claim_score_fusion_protocol_modes")),
            "claim_score_fusion_mode": str(config.get("claim_score_fusion_mode", "direct") or "direct"),
            "claim_score_fusion_residual_scale": float(config.get("claim_score_fusion_residual_scale", 1.0)),
            "val_raws": len(val_raws),
            "val_samples": len(val_indices),
            "anchor_top1_acc": anchor_acc_sum / max(total_batches, 1),
            "shift_top1_acc": shift_acc_sum / max(total_batches, 1),
            "recovery_codeword_ber": torch.cat(recovery_codeword_ber, dim=0).mean().item(),
            "recovery_payload_ber": torch.cat(recovery_payload_ber, dim=0).mean().item(),
            "recovery_decode_success": torch.cat(recovery_decode_success, dim=0).mean().item(),
            "recovery_code_index_match": (
                torch.cat(recovery_code_index_match, dim=0).mean().item() if recovery_code_index_match else None
            ),
        }
    )

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate jointly learned authentication codes")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--mode", type=str, default="student", choices=["student", "teacher"])
    parser.add_argument("--prototype_cache_dir", type=str, default=str(resolve_experiment_root() / "stage3_prototype_cache_anchor12_joint512_live"))
    parser.add_argument("--teacher_cache_dir", type=str, default=str(resolve_experiment_root() / "stage3_teacher_cache_joint512_live"))
    parser.add_argument("--teacher_key", type=str, default=None)
    parser.add_argument("--meta_path", type=str, default=str(resolve_meta_path()))
    parser.add_argument("--rgb_dir", type=str, default=str(resolve_dataset_root() / "rgb_web_jpg"))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--max_raws", type=int, default=256)
    parser.add_argument("--include_versions", type=int, nargs="*", default=[1, 2, 3, 4])
    parser.add_argument("--anchor_versions", type=int, nargs="*", default=[1, 2])
    parser.add_argument("--shift_versions", type=int, nargs="*", default=[3, 4])
    parser.add_argument("--official_eval_preset", type=str, default=None)
    parser.add_argument("--eval_preset", type=str, default=None, choices=["strong", "mild", "none", "center", "center_multi"])
    parser.add_argument("--override_bit_scale", type=float, default=None)
    parser.add_argument("--override_sequence_score_weight", type=float, default=None)
    parser.add_argument("--override_claim_verifier_weight", type=float, default=None)
    parser.add_argument("--override_claim_sequence_score_weight", type=float, default=None)
    parser.add_argument("--override_official_claim_score_mode", type=str, default=None, choices=["deterministic_gate", "fusion_head"])
    parser.add_argument("--override_token_gate_penalty", type=float, default=None)
    parser.add_argument("--override_token_residual_weight", type=float, default=None)
    parser.add_argument(
        "--protocol_score_mode",
        type=str,
        default=None,
        choices=["none", "code_cosine", "codeword_agreement", "payload_agreement", "decode_success", "index_match"],
    )
    parser.add_argument("--protocol_score_alpha", type=float, default=None)
    parser.add_argument(
        "--claim_main_score_norm_mode",
        type=str,
        default=None,
        choices=["none", "row_center", "row_zscore"],
    )
    parser.add_argument(
        "--protocol_score_norm_mode",
        type=str,
        default=None,
        choices=["none", "row_center", "row_zscore"],
    )
    parser.add_argument(
        "--claim_reference_mode",
        type=str,
        default=None,
        choices=["same_image", "same_raw", "anchor_bank"],
    )
    parser.add_argument("--claim_reference_versions", type=int, nargs="*", default=None)
    parser.add_argument(
        "--claim_bank_mode",
        type=str,
        default=None,
        choices=["mean_bits", "mean_logits_tanh", "sign_mean_logits"],
    )
    parser.add_argument("--output_json", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
