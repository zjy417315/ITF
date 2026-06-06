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

from src.datasets.code_matching_dataset import CodeMatchingDataset
from src.models.prototype_verifier import PrototypeVerifier
from src.models.visual_encoder import VisualEncoder
from src.train.train_stage3_prototype import (
    RawGroupBatchSampler,
    accuracy_for_versions,
    binary_auc_from_scores,
    build_group_split,
    build_transforms,
    load_projection_checkpoint,
    load_state_dict_shape_safe,
    project_features,
    set_seed,
)
from src.tools.data_roots import resolve_dataset_root, resolve_experiment_root, resolve_meta_path


def fit_projection_from_train_codes(dataset: CodeMatchingDataset, train_raws, output_dim: int):
    if output_dim is None or output_dim <= 0 or output_dim >= dataset.code_dim:
        return None, None
    code_mat = torch.stack([dataset.get_code_tensor(raw_anchor) for raw_anchor in train_raws], dim=0).float()
    mean_vec = code_mat.mean(dim=0)
    centered = code_mat - mean_vec
    q = min(int(output_dim), centered.shape[0] - 1, centered.shape[1])
    if q <= 0:
        return None, None
    _, _, basis = torch.pca_lowrank(centered, q=q, center=False)
    return mean_vec, basis[:, :q]


def build_code_bank(dataset: CodeMatchingDataset, raw_anchors, projection_mean, projection_basis, device: torch.device):
    bank = project_features(
        torch.stack([dataset.get_code_tensor(raw_anchor) for raw_anchor in raw_anchors]).to(device),
        projection_mean,
        projection_basis,
    )
    seq_bank = None
    if hasattr(dataset, "get_code_sequence"):
        seq_bank = project_features(
            torch.stack([dataset.get_code_sequence(raw_anchor) for raw_anchor in raw_anchors]).to(device).reshape(len(raw_anchors), -1, dataset.code_dim),
            projection_mean,
            projection_basis,
        )
    return bank, seq_bank, list(raw_anchors)


def transform_code_targets(code_tensor: torch.Tensor, target_mode: str, normalize: bool = True) -> torch.Tensor:
    if target_mode == "continuous":
        return F.normalize(code_tensor, dim=-1) if normalize else code_tensor
    if target_mode == "binary_sign":
        binary = torch.sign(code_tensor)
        binary[binary == 0] = 1.0
        return F.normalize(binary, dim=-1) if normalize else binary
    raise ValueError(f"Unsupported code_target_mode: {target_mode}")


def compute_binary_claim_logits(
    student_code_logits: torch.Tensor,
    raw_anchors,
    claim_bank_bits: torch.Tensor,
    bank_raws,
    tanh_scale: float = 1.0,
):
    student_bits = torch.tanh(float(tanh_scale) * student_code_logits)
    logits = torch.matmul(student_bits, claim_bank_bits.T) / claim_bank_bits.shape[-1]
    raw_to_index = {raw_anchor: idx for idx, raw_anchor in enumerate(bank_raws)}
    targets = torch.tensor([raw_to_index[raw_anchor] for raw_anchor in raw_anchors], device=student_code_logits.device)
    return logits, targets


def compute_binary_sequence_claim_logits(
    student_code_seq_logits: torch.Tensor,
    claim_seq_bank_bits: torch.Tensor,
    tanh_scale: float = 1.0,
) -> torch.Tensor:
    student_bits = torch.tanh(float(tanh_scale) * student_code_seq_logits)
    return torch.einsum("bkd,skd->bsk", student_bits, claim_seq_bank_bits).mean(dim=-1) / claim_seq_bank_bits.shape[-1]


def build_batch_claim_bank(raw_anchors, claim_code: torch.Tensor, claim_code_seq: torch.Tensor = None):
    bank_raws = []
    bank_vecs = []
    bank_seqs = []
    for idx, raw_anchor in enumerate(raw_anchors):
        if raw_anchor in bank_raws:
            continue
        bank_raws.append(raw_anchor)
        bank_vecs.append(claim_code[idx])
        if claim_code_seq is not None:
            bank_seqs.append(claim_code_seq[idx])
    bank_vecs = torch.stack(bank_vecs, dim=0)
    bank_seqs = torch.stack(bank_seqs, dim=0) if bank_seqs else None
    return bank_vecs, bank_seqs, bank_raws


def compute_claim_logits(
    student_code: torch.Tensor,
    raw_anchors,
    claim_bank: torch.Tensor,
    bank_raws,
    verifier_head: PrototypeVerifier = None,
):
    student_code = F.normalize(student_code, dim=1)
    claim_bank = F.normalize(claim_bank, dim=1)
    if verifier_head is None:
        logits = torch.matmul(student_code, claim_bank.T)
    else:
        logits = verifier_head.score_bank(student_code, claim_bank)
    raw_to_index = {raw_anchor: idx for idx, raw_anchor in enumerate(bank_raws)}
    targets = torch.tensor([raw_to_index[raw_anchor] for raw_anchor in raw_anchors], device=student_code.device)
    return logits, targets


def compute_sequence_claim_logits(student_code_seq: torch.Tensor, claim_seq_bank: torch.Tensor) -> torch.Tensor:
    student_code_seq = F.normalize(student_code_seq, dim=-1)
    claim_seq_bank = F.normalize(claim_seq_bank, dim=-1)
    return torch.einsum("bkd,skd->bsk", student_code_seq, claim_seq_bank).mean(dim=-1)


def compute_combined_claim_logits(
    student_code: torch.Tensor,
    raw_anchors,
    claim_bank: torch.Tensor,
    bank_raws,
    student_code_seq: torch.Tensor = None,
    claim_seq_bank: torch.Tensor = None,
    sequence_score_weight: float = 0.0,
    verifier_head: PrototypeVerifier = None,
    code_target_mode: str = "continuous",
    student_code_logits: torch.Tensor = None,
    claim_bank_bits: torch.Tensor = None,
    student_code_seq_logits: torch.Tensor = None,
    claim_seq_bank_bits: torch.Tensor = None,
    bit_tanh_scale: float = 2.0,
):
    if code_target_mode == "binary_sign" and student_code_logits is not None and claim_bank_bits is not None and verifier_head is None:
        logits, targets = compute_binary_claim_logits(
            student_code_logits,
            raw_anchors,
            claim_bank_bits,
            bank_raws,
            tanh_scale=bit_tanh_scale,
        )
    else:
        logits, targets = compute_claim_logits(
            student_code,
            raw_anchors,
            claim_bank,
            bank_raws,
            verifier_head=verifier_head,
        )
    if float(sequence_score_weight) > 0.0:
        if code_target_mode == "binary_sign" and student_code_seq_logits is not None and claim_seq_bank_bits is not None:
            seq_logits = compute_binary_sequence_claim_logits(
                student_code_seq_logits,
                claim_seq_bank_bits,
                tanh_scale=bit_tanh_scale,
            )
        elif student_code_seq is not None and claim_seq_bank is not None:
            seq_logits = compute_sequence_claim_logits(student_code_seq, claim_seq_bank)
        else:
            seq_logits = None
        if seq_logits is not None:
            seq_w = min(max(float(sequence_score_weight), 0.0), 1.0)
            logits = (1.0 - seq_w) * logits + seq_w * seq_logits
    return logits, targets


def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights = weights.to(values.device, dtype=values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1e-6)


def compute_version_weights(
    versions,
    device: torch.device,
    anchor_versions,
    shift_versions,
    anchor_weight: float,
    shift_weight: float,
) -> torch.Tensor:
    version_tensor = torch.as_tensor(versions, device=device, dtype=torch.long)
    weights = torch.ones_like(version_tensor, dtype=torch.float32)
    for version in anchor_versions:
        weights[version_tensor == int(version)] = float(anchor_weight)
    for version in shift_versions:
        weights[version_tensor == int(version)] = float(shift_weight)
    return weights


def compute_alignment_loss(student_code: torch.Tensor, claim_code: torch.Tensor, sample_weights: torch.Tensor) -> torch.Tensor:
    cosine = F.cosine_similarity(student_code, claim_code, dim=1)
    return weighted_mean(1.0 - cosine, sample_weights)


def compute_sequence_alignment_loss(student_code_seq: torch.Tensor, claim_code_seq: torch.Tensor, sample_weights: torch.Tensor) -> torch.Tensor:
    if student_code_seq is None or claim_code_seq is None:
        return sample_weights.new_tensor(0.0)
    cosine = F.cosine_similarity(student_code_seq, claim_code_seq, dim=-1)
    sample_loss = 1.0 - cosine.mean(dim=1)
    return weighted_mean(sample_loss, sample_weights)


def compute_bit_logistic_loss(student_code_logits: torch.Tensor, claim_code_bits: torch.Tensor, sample_weights: torch.Tensor) -> torch.Tensor:
    if student_code_logits is None or claim_code_bits is None:
        return sample_weights.new_tensor(0.0)
    target = claim_code_bits.to(student_code_logits.device, dtype=student_code_logits.dtype)
    sample_loss = F.softplus(-target * student_code_logits).mean(dim=1)
    return weighted_mean(sample_loss, sample_weights)


def compute_sequence_bit_logistic_loss(student_code_seq_logits: torch.Tensor, claim_code_seq_bits: torch.Tensor, sample_weights: torch.Tensor) -> torch.Tensor:
    if student_code_seq_logits is None or claim_code_seq_bits is None:
        return sample_weights.new_tensor(0.0)
    target = claim_code_seq_bits.to(student_code_seq_logits.device, dtype=student_code_seq_logits.dtype)
    sample_loss = F.softplus(-target * student_code_seq_logits).mean(dim=(1, 2))
    return weighted_mean(sample_loss, sample_weights)


def compute_margin_verification_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    sample_weights: torch.Tensor,
    margin: float = 0.10,
    topk: int = 8,
) -> torch.Tensor:
    pos_scores = logits.gather(1, targets.unsqueeze(1)).squeeze(1)
    negative_mask = torch.ones_like(logits, dtype=torch.bool)
    negative_mask.scatter_(1, targets.unsqueeze(1), False)
    neg_scores = logits[negative_mask].view(logits.shape[0], -1)
    if neg_scores.shape[1] == 0:
        neg_loss = pos_scores.new_zeros(pos_scores.shape[0])
        pos_loss = F.softplus(-float(scale) * (pos_scores - float(positive_margin)))
        sample_losses = pos_loss + neg_loss
        return weighted_mean(sample_losses, sample_weights)
    if topk is not None and topk > 0:
        neg_scores = torch.topk(neg_scores, k=min(int(topk), neg_scores.shape[1]), dim=1).values
    pair_terms = F.softplus(margin + neg_scores - pos_scores.unsqueeze(1))
    sample_losses = pair_terms.mean(dim=1)
    return weighted_mean(sample_losses, sample_weights)


def compute_pair_logistic_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    sample_weights: torch.Tensor,
    positive_margin: float = 0.60,
    negative_margin: float = 0.10,
    scale: float = 12.0,
    topk: int = 8,
) -> torch.Tensor:
    pos_scores = logits.gather(1, targets.unsqueeze(1)).squeeze(1)
    negative_mask = torch.ones_like(logits, dtype=torch.bool)
    negative_mask.scatter_(1, targets.unsqueeze(1), False)
    neg_scores = logits[negative_mask].view(logits.shape[0], -1)
    if topk is not None and topk > 0:
        neg_scores = torch.topk(neg_scores, k=min(int(topk), neg_scores.shape[1]), dim=1).values

    pos_loss = F.softplus(-float(scale) * (pos_scores - float(positive_margin)))
    neg_loss = F.softplus(float(scale) * (neg_scores - float(negative_margin))).mean(dim=1)
    sample_losses = pos_loss + neg_loss
    return weighted_mean(sample_losses, sample_weights)


def compute_hard_margin_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    sample_weights: torch.Tensor,
    margin: float = 0.10,
) -> torch.Tensor:
    pos_scores, hard_neg_scores = extract_pos_and_hard_neg(logits, targets)
    return weighted_mean(F.relu(margin + hard_neg_scores - pos_scores), sample_weights)


def compute_uniformity_loss(codes: torch.Tensor, temperature: float = 2.0) -> torch.Tensor:
    if codes.shape[0] < 2:
        return codes.new_tensor(0.0)
    pdist = torch.pdist(F.normalize(codes, dim=1), p=2).pow(2)
    return torch.log(torch.exp(-float(temperature) * pdist).mean().clamp_min(1e-12))


def extract_pos_and_hard_neg(logits: torch.Tensor, targets: torch.Tensor):
    pos_scores = logits.gather(1, targets.unsqueeze(1)).squeeze(1)
    negative_mask = torch.ones_like(logits, dtype=torch.bool)
    negative_mask.scatter_(1, targets.unsqueeze(1), False)
    neg_scores = logits[negative_mask].view(logits.shape[0], -1)
    if neg_scores.shape[1] == 0:
        hard_neg_scores = pos_scores.new_zeros(pos_scores.shape[0])
    else:
        hard_neg_scores = neg_scores.max(dim=1).values
    return pos_scores, hard_neg_scores


def tar_at_far(pos_scores: torch.Tensor, neg_scores: torch.Tensor, far: float) -> float:
    if pos_scores.numel() == 0 or neg_scores.numel() == 0:
        return 0.0
    neg_sorted, _ = torch.sort(neg_scores)
    idx = int(max(0, min(len(neg_sorted) - 1, round((1.0 - far) * (len(neg_sorted) - 1)))))
    threshold = neg_sorted[idx]
    return float((pos_scores >= threshold).float().mean().item())


def compute_eer(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> float:
    if pos_scores.numel() == 0 or neg_scores.numel() == 0:
        return 1.0
    thresholds = torch.cat([pos_scores, neg_scores]).unique(sorted=True)
    best_gap = None
    best_eer = 1.0
    for threshold in thresholds:
        far = float((neg_scores >= threshold).float().mean().item())
        frr = float((pos_scores < threshold).float().mean().item())
        gap = abs(far - frr)
        if best_gap is None or gap < best_gap:
            best_gap = gap
            best_eer = 0.5 * (far + frr)
    return float(best_eer)


def summarize_verification_scores(pos_scores: torch.Tensor, hard_neg_scores: torch.Tensor, all_neg_scores: torch.Tensor):
    return {
        "pairwise_auc": binary_auc_from_scores(pos_scores, all_neg_scores),
        "hard_auc": binary_auc_from_scores(pos_scores, hard_neg_scores),
        "tar_at_far_1e2": tar_at_far(pos_scores, all_neg_scores, far=0.01),
        "tar_at_far_1e3": tar_at_far(pos_scores, all_neg_scores, far=0.001),
        "tar_at_far_5e2": tar_at_far(pos_scores, all_neg_scores, far=0.05),
        "eer": compute_eer(pos_scores, all_neg_scores),
        "positive_score_mean": float(pos_scores.mean().item()) if pos_scores.numel() else 0.0,
        "hard_negative_score_mean": float(hard_neg_scores.mean().item()) if hard_neg_scores.numel() else 0.0,
        "negative_score_mean": float(all_neg_scores.mean().item()) if all_neg_scores.numel() else 0.0,
        "hard_win_rate": float((pos_scores > hard_neg_scores).float().mean().item()) if pos_scores.numel() else 0.0,
    }


def resolve_selection_value(record: dict, metric_name: str) -> float:
    value = float(record.get(metric_name, float("-inf")))
    lower_is_better_tokens = ("loss", "eer")
    if any(token in metric_name.lower() for token in lower_is_better_tokens):
        return -value
    return value


def train_stage3_code(args):
    set_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    pin_memory = device.type == "cuda"

    train_transform, eval_transform = build_transforms(
        args.augmentation_preset,
        image_size=args.image_size,
        resize_size=args.resize_size,
    )

    full_dataset = CodeMatchingDataset(
        prototype_cache_dir=args.prototype_cache_dir,
        meta_path=args.meta_path,
        rgb_dir=args.rgb_dir,
        include_versions=args.include_versions,
        transform=eval_transform,
    )
    train_indices, val_indices, train_raws, val_raws = build_group_split(
        dataset=full_dataset,
        val_ratio=args.val_ratio,
        seed=args.seed,
        max_raws=args.max_raws,
    )

    train_dataset = CodeMatchingDataset(
        prototype_cache_dir=args.prototype_cache_dir,
        meta_path=args.meta_path,
        rgb_dir=args.rgb_dir,
        include_versions=args.include_versions,
        transform=train_transform,
    )
    val_dataset = CodeMatchingDataset(
        prototype_cache_dir=args.prototype_cache_dir,
        meta_path=args.meta_path,
        rgb_dir=args.rgb_dir,
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

    if args.projection_checkpoint:
        projection_mean, projection_basis = load_projection_checkpoint(args.projection_checkpoint)
    else:
        projection_mean, projection_basis = fit_projection_from_train_codes(
            dataset=full_dataset,
            train_raws=train_raws,
            output_dim=args.project_dim,
        )
    projected_dim = int(projection_basis.shape[1]) if projection_basis is not None else full_dataset.code_dim

    student = VisualEncoder(
        d_out=projected_dim,
        backbone_type=args.backbone_type,
        pretrained=not args.no_pretrained,
        input_mode=args.input_mode,
        residual_scale=args.residual_scale,
        residual_kernel=args.residual_kernel,
        use_stage_sequence_head=(args.stage_sequence_weight > 0.0 or args.sequence_score_weight > 0.0 or args.sequence_bit_logistic_weight > 0.0),
        local_crop_mode=args.local_crop_mode,
        local_crop_size=args.local_crop_size,
        local_patch_offset=args.local_patch_offset,
    ).to(device)
    verifier_head = None
    if args.verifier_mode == "pair_mlp":
        verifier_head = PrototypeVerifier(
            d_model=projected_dim,
            hidden_dim=args.verifier_hidden_dim,
            dropout=args.verifier_dropout,
        ).to(device)

    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location="cpu")
        load_status = load_state_dict_shape_safe(student, checkpoint["student_state_dict"], module_name="Student")
        if load_status.missing_keys:
            print(f"Student missing keys on init load: {load_status.missing_keys}", flush=True)
        if load_status.unexpected_keys:
            print(f"Student unexpected keys on init load: {load_status.unexpected_keys}", flush=True)
        if verifier_head is not None and "verifier_state_dict" in checkpoint:
            verifier_status = load_state_dict_shape_safe(
                verifier_head,
                checkpoint["verifier_state_dict"],
                module_name="Verifier",
            )
            if verifier_status.missing_keys:
                print(f"Verifier missing keys on init load: {verifier_status.missing_keys}", flush=True)
            if verifier_status.unexpected_keys:
                print(f"Verifier unexpected keys on init load: {verifier_status.unexpected_keys}", flush=True)

    parameters = list(student.parameters())
    if verifier_head is not None:
        parameters.extend(verifier_head.parameters())
    optimizer = optim.AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_bank, train_seq_bank, train_bank_raws = build_code_bank(train_dataset, train_raws, projection_mean, projection_basis, device)
    val_bank, val_seq_bank, val_bank_raws = build_code_bank(val_dataset, val_raws, projection_mean, projection_basis, device)
    train_bank = transform_code_targets(train_bank, args.code_target_mode)
    val_bank = transform_code_targets(val_bank, args.code_target_mode)
    if train_seq_bank is not None:
        train_seq_bank = transform_code_targets(train_seq_bank, args.code_target_mode)
    if val_seq_bank is not None:
        val_seq_bank = transform_code_targets(val_seq_bank, args.code_target_mode)
    train_bank_bits = transform_code_targets(train_bank, args.code_target_mode, normalize=False) if args.code_target_mode == "binary_sign" else None
    val_bank_bits = transform_code_targets(val_bank, args.code_target_mode, normalize=False) if args.code_target_mode == "binary_sign" else None
    train_seq_bank_bits = transform_code_targets(train_seq_bank, args.code_target_mode, normalize=False) if (args.code_target_mode == "binary_sign" and train_seq_bank is not None) else None
    val_seq_bank_bits = transform_code_targets(val_seq_bank, args.code_target_mode, normalize=False) if (args.code_target_mode == "binary_sign" and val_seq_bank is not None) else None

    print("=" * 72, flush=True)
    print("Stage-3 Code Matching", flush=True)
    print(f"Device          : {device}", flush=True)
    print(f"Prototype cache : {args.prototype_cache_dir}", flush=True)
    print(f"Projection ckpt : {args.projection_checkpoint or 'fit_from_train'}", flush=True)
    print(f"Init checkpoint : {args.init_checkpoint or 'none'}", flush=True)
    print(f"Code dim        : {full_dataset.code_dim}", flush=True)
    print(f"Projected dim   : {projected_dim}", flush=True)
    print(f"Train raws      : {len(train_raws)}", flush=True)
    print(f"Val raws        : {len(val_raws)}", flush=True)
    print(f"Train samples   : {len(train_indices)}", flush=True)
    print(f"Val samples     : {len(val_indices)}", flush=True)
    print(f"RGB versions    : {args.include_versions}", flush=True)
    print(f"Anchor vers     : {args.anchor_versions}", flush=True)
    print(f"Shift vers      : {args.shift_versions}", flush=True)
    print(f"Input mode      : {args.input_mode} (scale={args.residual_scale}, k={args.residual_kernel})", flush=True)
    print(f"Local crop      : {args.local_crop_mode} (size={args.local_crop_size}, offset={args.local_patch_offset})", flush=True)
    print(
        f"Loss weights    : align={args.alignment_weight}, seq={args.stage_sequence_weight}, ce={args.classification_weight}, "
        f"margin={args.margin_weight}, hard={args.hard_margin_weight}, pairlog={args.pair_logistic_weight}, "
        f"bit={args.bit_logistic_weight}, seqbit={args.sequence_bit_logistic_weight}, "
        f"uniform={args.uniformity_weight}",
        flush=True,
    )
    print(f"Code target mode : {args.code_target_mode} (bit_tanh_scale={args.bit_tanh_scale})", flush=True)
    print(f"Seq score weight: {args.sequence_score_weight}", flush=True)
    print(f"Train bank scope : {args.train_bank_scope}", flush=True)
    print(f"Selection metric: {args.selection_metric}", flush=True)
    print("=" * 72, flush=True)

    history = []
    best_metric = float("-inf")

    for epoch in range(args.epochs):
        if hasattr(train_loader.batch_sampler, "set_epoch"):
            train_loader.batch_sampler.set_epoch(epoch)

        student.train()
        if verifier_head is not None:
            verifier_head.train()
        train_loss = 0.0
        train_pos_scores = []
        train_hard_neg_scores = []
        train_all_neg_scores = []
        train_anchor_acc = 0.0
        train_shift_acc = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Train]")
        for batch in pbar:
            rgb_image = batch["rgb_image"].to(device, non_blocking=True)
            claim_code = project_features(
                batch["claim_code"].to(device, non_blocking=True),
                projection_mean,
                projection_basis,
            )
            claim_code_bits = transform_code_targets(claim_code, args.code_target_mode, normalize=False) if args.code_target_mode == "binary_sign" else None
            claim_code = transform_code_targets(claim_code, args.code_target_mode)
            claim_code_seq_raw = batch["claim_code_seq"].to(device, non_blocking=True)
            claim_code_seq = project_features(
                claim_code_seq_raw.reshape(-1, claim_code_seq_raw.shape[-1]),
                projection_mean,
                projection_basis,
            ).reshape(claim_code_seq_raw.shape[0], claim_code_seq_raw.shape[1], -1)
            claim_code_seq_bits = transform_code_targets(claim_code_seq, args.code_target_mode, normalize=False) if args.code_target_mode == "binary_sign" else None
            claim_code_seq = transform_code_targets(claim_code_seq, args.code_target_mode)
            raw_anchors = list(batch["raw_anchor"])
            versions = batch["version"].to(device, non_blocking=True)
            sample_weights = compute_version_weights(
                versions=versions,
                device=device,
                anchor_versions=args.anchor_versions,
                shift_versions=args.shift_versions,
                anchor_weight=args.anchor_weight,
                shift_weight=args.shift_weight,
            )

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                student_out = student(
                    rgb_image,
                    return_sequence=(args.stage_sequence_weight > 0.0 or args.sequence_score_weight > 0.0 or args.sequence_bit_logistic_weight > 0.0),
                    return_logits=(args.code_target_mode == "binary_sign" or args.bit_logistic_weight > 0.0 or args.sequence_bit_logistic_weight > 0.0),
                )
                if isinstance(student_out, dict):
                    student_code = student_out["global_repr"]
                    student_code_seq = student_out.get("stage_repr")
                    student_code_logits = student_out.get("global_logits")
                    student_code_seq_logits = student_out.get("stage_logits")
                else:
                    student_code = student_out
                    student_code_seq = None
                    student_code_logits = None
                    student_code_seq_logits = None
                if args.train_bank_scope == "batch":
                    train_step_bank, train_step_seq_bank, train_step_raws = build_batch_claim_bank(
                        raw_anchors,
                        claim_code,
                        claim_code_seq,
                    )
                    if args.code_target_mode == "binary_sign":
                        train_step_bank_bits, train_step_seq_bank_bits, _ = build_batch_claim_bank(
                            raw_anchors,
                            claim_code_bits,
                            claim_code_seq_bits,
                        )
                    else:
                        train_step_bank_bits, train_step_seq_bank_bits = None, None
                else:
                    train_step_bank, train_step_seq_bank, train_step_raws = train_bank, train_seq_bank, train_bank_raws
                    train_step_bank_bits, train_step_seq_bank_bits = train_bank_bits, train_seq_bank_bits
                logits, targets = compute_combined_claim_logits(
                    student_code,
                    raw_anchors,
                    train_step_bank,
                    train_step_raws,
                    student_code_seq=student_code_seq,
                    claim_seq_bank=train_step_seq_bank,
                    sequence_score_weight=args.sequence_score_weight,
                    verifier_head=verifier_head,
                    code_target_mode=args.code_target_mode,
                    student_code_logits=student_code_logits,
                    claim_bank_bits=train_step_bank_bits,
                    student_code_seq_logits=student_code_seq_logits,
                    claim_seq_bank_bits=train_step_seq_bank_bits,
                    bit_tanh_scale=args.bit_tanh_scale,
                )
                align_loss = compute_alignment_loss(student_code, claim_code, sample_weights)
                seq_align_loss = compute_sequence_alignment_loss(student_code_seq, claim_code_seq, sample_weights)
                bit_loss = compute_bit_logistic_loss(student_code_logits, claim_code_bits, sample_weights)
                seq_bit_loss = compute_sequence_bit_logistic_loss(student_code_seq_logits, claim_code_seq_bits, sample_weights)
                ce_terms = F.cross_entropy(logits / max(args.temperature, 1e-6), targets, reduction="none")
                cls_loss = weighted_mean(ce_terms, sample_weights)
                margin_loss = compute_margin_verification_loss(
                    logits,
                    targets,
                    sample_weights,
                    margin=args.margin,
                    topk=args.margin_topk,
                )
                hard_margin_loss = compute_hard_margin_loss(
                    logits,
                    targets,
                    sample_weights,
                    margin=args.hard_margin,
                )
                pair_logistic_loss = compute_pair_logistic_loss(
                    logits,
                    targets,
                    sample_weights,
                    positive_margin=args.pair_positive_margin,
                    negative_margin=args.pair_negative_margin,
                    scale=args.pair_logit_scale,
                    topk=args.pair_logistic_topk,
                )
                uniformity_loss = compute_uniformity_loss(student_code, temperature=args.uniformity_temperature)
                loss = (
                    args.alignment_weight * align_loss
                    + args.stage_sequence_weight * seq_align_loss
                    + args.classification_weight * cls_loss
                    + args.margin_weight * margin_loss
                    + args.hard_margin_weight * hard_margin_loss
                    + args.pair_logistic_weight * pair_logistic_loss
                    + args.bit_logistic_weight * bit_loss
                    + args.sequence_bit_logistic_weight * seq_bit_loss
                    + args.uniformity_weight * uniformity_loss
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            pos_scores, hard_neg_scores = extract_pos_and_hard_neg(logits, targets)
            train_loss += float(loss.item())
            train_pos_scores.append(pos_scores.detach().cpu())
            train_hard_neg_scores.append(hard_neg_scores.detach().cpu())
            for row_idx, target_idx in enumerate(targets.tolist()):
                row = logits[row_idx]
                mask = torch.ones_like(row, dtype=torch.bool)
                mask[target_idx] = False
                train_all_neg_scores.append(row[mask].detach().cpu())
            correct = logits.argmax(dim=1) == targets
            anchor_acc = accuracy_for_versions(correct, versions, args.anchor_versions)
            shift_acc = accuracy_for_versions(correct, versions, args.shift_versions)
            train_anchor_acc += 0.0 if torch.isnan(torch.tensor(anchor_acc)) else anchor_acc
            train_shift_acc += 0.0 if torch.isnan(torch.tensor(shift_acc)) else shift_acc
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Pos": f"{pos_scores.mean().item():.3f}", "HNeg": f"{hard_neg_scores.mean().item():.3f}"})

        scheduler.step()

        student.eval()
        if verifier_head is not None:
            verifier_head.eval()
        val_loss = 0.0
        val_pos_scores = []
        val_hard_neg_scores = []
        val_all_neg_scores = []
        val_anchor_acc = 0.0
        val_shift_acc = 0.0
        with torch.no_grad():
            for batch in val_loader:
                rgb_image = batch["rgb_image"].to(device, non_blocking=True)
                claim_code = project_features(
                    batch["claim_code"].to(device, non_blocking=True),
                    projection_mean,
                    projection_basis,
                )
                claim_code_bits = transform_code_targets(claim_code, args.code_target_mode, normalize=False) if args.code_target_mode == "binary_sign" else None
                claim_code = transform_code_targets(claim_code, args.code_target_mode)
                claim_code_seq_raw = batch["claim_code_seq"].to(device, non_blocking=True)
                claim_code_seq = project_features(
                    claim_code_seq_raw.reshape(-1, claim_code_seq_raw.shape[-1]),
                    projection_mean,
                    projection_basis,
                ).reshape(claim_code_seq_raw.shape[0], claim_code_seq_raw.shape[1], -1)
                claim_code_seq_bits = transform_code_targets(claim_code_seq, args.code_target_mode, normalize=False) if args.code_target_mode == "binary_sign" else None
                claim_code_seq = transform_code_targets(claim_code_seq, args.code_target_mode)
                raw_anchors = list(batch["raw_anchor"])
                versions = batch["version"].to(device, non_blocking=True)
                sample_weights = compute_version_weights(
                    versions=versions,
                    device=device,
                    anchor_versions=args.anchor_versions,
                    shift_versions=args.shift_versions,
                    anchor_weight=args.anchor_weight,
                    shift_weight=args.shift_weight,
                )

                student_out = student(
                    rgb_image,
                    return_sequence=(args.stage_sequence_weight > 0.0 or args.sequence_score_weight > 0.0 or args.sequence_bit_logistic_weight > 0.0),
                    return_logits=(args.code_target_mode == "binary_sign" or args.bit_logistic_weight > 0.0 or args.sequence_bit_logistic_weight > 0.0),
                )
                if isinstance(student_out, dict):
                    student_code = student_out["global_repr"]
                    student_code_seq = student_out.get("stage_repr")
                    student_code_logits = student_out.get("global_logits")
                    student_code_seq_logits = student_out.get("stage_logits")
                else:
                    student_code = student_out
                    student_code_seq = None
                    student_code_logits = None
                    student_code_seq_logits = None
                logits, targets = compute_combined_claim_logits(
                    student_code,
                    raw_anchors,
                    val_bank,
                    val_bank_raws,
                    student_code_seq=student_code_seq,
                    claim_seq_bank=val_seq_bank,
                    sequence_score_weight=args.sequence_score_weight,
                    verifier_head=verifier_head,
                    code_target_mode=args.code_target_mode,
                    student_code_logits=student_code_logits,
                    claim_bank_bits=val_bank_bits,
                    student_code_seq_logits=student_code_seq_logits,
                    claim_seq_bank_bits=val_seq_bank_bits,
                    bit_tanh_scale=args.bit_tanh_scale,
                )
                align_loss = compute_alignment_loss(student_code, claim_code, sample_weights)
                seq_align_loss = compute_sequence_alignment_loss(student_code_seq, claim_code_seq, sample_weights)
                bit_loss = compute_bit_logistic_loss(student_code_logits, claim_code_bits, sample_weights)
                seq_bit_loss = compute_sequence_bit_logistic_loss(student_code_seq_logits, claim_code_seq_bits, sample_weights)
                ce_terms = F.cross_entropy(logits / max(args.temperature, 1e-6), targets, reduction="none")
                cls_loss = weighted_mean(ce_terms, sample_weights)
                margin_loss = compute_margin_verification_loss(
                    logits,
                    targets,
                    sample_weights,
                    margin=args.margin,
                    topk=args.margin_topk,
                )
                hard_margin_loss = compute_hard_margin_loss(
                    logits,
                    targets,
                    sample_weights,
                    margin=args.hard_margin,
                )
                pair_logistic_loss = compute_pair_logistic_loss(
                    logits,
                    targets,
                    sample_weights,
                    positive_margin=args.pair_positive_margin,
                    negative_margin=args.pair_negative_margin,
                    scale=args.pair_logit_scale,
                    topk=args.pair_logistic_topk,
                )
                uniformity_loss = compute_uniformity_loss(student_code, temperature=args.uniformity_temperature)
                loss = (
                    args.alignment_weight * align_loss
                    + args.stage_sequence_weight * seq_align_loss
                    + args.classification_weight * cls_loss
                    + args.margin_weight * margin_loss
                    + args.hard_margin_weight * hard_margin_loss
                    + args.pair_logistic_weight * pair_logistic_loss
                    + args.bit_logistic_weight * bit_loss
                    + args.sequence_bit_logistic_weight * seq_bit_loss
                    + args.uniformity_weight * uniformity_loss
                )
                val_loss += float(loss.item())

                pos_scores, hard_neg_scores = extract_pos_and_hard_neg(logits, targets)
                val_pos_scores.append(pos_scores.cpu())
                val_hard_neg_scores.append(hard_neg_scores.cpu())
                for row_idx, target_idx in enumerate(targets.tolist()):
                    row = logits[row_idx]
                    mask = torch.ones_like(row, dtype=torch.bool)
                    mask[target_idx] = False
                    val_all_neg_scores.append(row[mask].cpu())
                correct = logits.argmax(dim=1) == targets
                anchor_acc = accuracy_for_versions(correct, versions, args.anchor_versions)
                shift_acc = accuracy_for_versions(correct, versions, args.shift_versions)
                val_anchor_acc += 0.0 if torch.isnan(torch.tensor(anchor_acc)) else anchor_acc
                val_shift_acc += 0.0 if torch.isnan(torch.tensor(shift_acc)) else shift_acc

        train_pos = torch.cat(train_pos_scores, dim=0) if train_pos_scores else torch.empty(0)
        train_hard_neg = torch.cat(train_hard_neg_scores, dim=0) if train_hard_neg_scores else torch.empty(0)
        train_all_neg = torch.cat(train_all_neg_scores, dim=0) if train_all_neg_scores else torch.empty(0)
        val_pos = torch.cat(val_pos_scores, dim=0) if val_pos_scores else torch.empty(0)
        val_hard_neg = torch.cat(val_hard_neg_scores, dim=0) if val_hard_neg_scores else torch.empty(0)
        val_all_neg = torch.cat(val_all_neg_scores, dim=0) if val_all_neg_scores else torch.empty(0)

        train_metrics = summarize_verification_scores(train_pos, train_hard_neg, train_all_neg)
        val_metrics = summarize_verification_scores(val_pos, val_hard_neg, val_all_neg)

        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": train_loss / max(len(train_loader), 1),
            "train_pairwise_auc": train_metrics["pairwise_auc"],
            "train_hard_auc": train_metrics["hard_auc"],
            "train_tar_at_far_1e2": train_metrics["tar_at_far_1e2"],
            "train_tar_at_far_5e2": train_metrics["tar_at_far_5e2"],
            "train_eer": train_metrics["eer"],
            "train_positive_score_mean": train_metrics["positive_score_mean"],
            "train_hard_negative_score_mean": train_metrics["hard_negative_score_mean"],
            "train_anchor_top1_acc": train_anchor_acc / max(len(train_loader), 1),
            "train_shift_top1_acc": train_shift_acc / max(len(train_loader), 1),
            "val_loss": val_loss / max(len(val_loader), 1),
            "val_pairwise_auc": val_metrics["pairwise_auc"],
            "val_hard_auc": val_metrics["hard_auc"],
            "val_tar_at_far_1e2": val_metrics["tar_at_far_1e2"],
            "val_tar_at_far_5e2": val_metrics["tar_at_far_5e2"],
            "val_eer": val_metrics["eer"],
            "val_positive_score_mean": val_metrics["positive_score_mean"],
            "val_hard_negative_score_mean": val_metrics["hard_negative_score_mean"],
            "val_hard_win_rate": val_metrics["hard_win_rate"],
            "val_anchor_top1_acc": val_anchor_acc / max(len(val_loader), 1),
            "val_shift_top1_acc": val_shift_acc / max(len(val_loader), 1),
        }
        history.append(epoch_record)
        print(epoch_record, flush=True)

        selection_value = resolve_selection_value(epoch_record, args.selection_metric)
        if selection_value > best_metric:
            best_metric = selection_value
            torch.save(
                {
                    "epoch": epoch + 1,
                    "student_state_dict": student.state_dict(),
                    "verifier_state_dict": verifier_head.state_dict() if verifier_head is not None else None,
                    "config": vars(args),
                    "history": history,
                    "projection_mean": projection_mean,
                    "projection_basis": projection_basis,
                },
                save_dir / "stage3_code_best.pt",
            )

        with open(save_dir / "stage3_code_history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Train stage-3 hyperspherical code matching")
    parser.add_argument("--prototype_cache_dir", type=str, default=str(resolve_experiment_root() / "stage3_prototype_cache_anchor12_joint512_live"))
    parser.add_argument("--meta_path", type=str, default=str(resolve_meta_path()))
    parser.add_argument("--rgb_dir", type=str, default=str(resolve_dataset_root() / "rgb_web_jpg"))
    parser.add_argument("--save_dir", type=str, default=str(resolve_experiment_root() / "stage3_code_checkpoints"))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--init_checkpoint", type=str, default=None)
    parser.add_argument("--projection_checkpoint", type=str, default=str(resolve_experiment_root() / "teacher_supervised_scale512" / "joint_d256_r0p001.pt"))
    parser.add_argument("--project_dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--val_batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--alignment_weight", type=float, default=1.0)
    parser.add_argument("--stage_sequence_weight", type=float, default=0.0)
    parser.add_argument("--sequence_score_weight", type=float, default=0.0)
    parser.add_argument("--classification_weight", type=float, default=0.1)
    parser.add_argument("--margin_weight", type=float, default=1.0)
    parser.add_argument("--hard_margin_weight", type=float, default=0.5)
    parser.add_argument("--pair_logistic_weight", type=float, default=0.0)
    parser.add_argument("--pair_positive_margin", type=float, default=0.60)
    parser.add_argument("--pair_negative_margin", type=float, default=0.10)
    parser.add_argument("--pair_logit_scale", type=float, default=12.0)
    parser.add_argument("--pair_logistic_topk", type=int, default=8)
    parser.add_argument("--bit_logistic_weight", type=float, default=0.0)
    parser.add_argument("--sequence_bit_logistic_weight", type=float, default=0.0)
    parser.add_argument("--bit_tanh_scale", type=float, default=2.0)
    parser.add_argument("--uniformity_weight", type=float, default=0.05)
    parser.add_argument("--uniformity_temperature", type=float, default=2.0)
    parser.add_argument("--margin", type=float, default=0.10)
    parser.add_argument("--hard_margin", type=float, default=0.15)
    parser.add_argument("--margin_topk", type=int, default=8)
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
    parser.add_argument("--max_raws", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--selection_metric", type=str, default="val_hard_auc")
    parser.add_argument("--train_bank_scope", type=str, default="full", choices=["full", "batch"])
    parser.add_argument("--code_target_mode", type=str, default="continuous", choices=["continuous", "binary_sign"])
    parser.add_argument("--verifier_mode", type=str, default="cosine", choices=["cosine", "pair_mlp"])
    parser.add_argument("--verifier_hidden_dim", type=int, default=256)
    parser.add_argument("--verifier_dropout", type=float, default=0.0)
    parser.add_argument("--no_pretrained", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train_stage3_code(parse_args())
