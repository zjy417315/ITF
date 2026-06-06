import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Sampler
from torchvision import transforms as T
from torchvision.transforms import functional as TF
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.prototype_dataset import PrototypeDistillationDataset
from src.models.prototype_verifier import PrototypeVerifier
from src.models.visual_encoder import VisualEncoder
from src.tools.data_roots import resolve_dataset_root, resolve_experiment_root, resolve_meta_path


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def build_group_split(dataset: PrototypeDistillationDataset, val_ratio: float, seed: int, max_raws: int = None):
    raw_to_indices = {}
    for idx, raw_anchor in enumerate(dataset.group_ids):
        raw_to_indices.setdefault(raw_anchor, []).append(idx)

    raw_anchors = sorted(raw_to_indices.keys())
    rng = random.Random(seed)
    rng.shuffle(raw_anchors)
    if max_raws is not None:
        raw_anchors = raw_anchors[: min(max_raws, len(raw_anchors))]

    val_count = max(1, int(round(len(raw_anchors) * val_ratio))) if len(raw_anchors) > 1 else 0
    val_raws = set(raw_anchors[:val_count])
    train_raws = set(raw_anchors[val_count:])
    if not train_raws and val_raws:
        moved = next(iter(val_raws))
        val_raws.remove(moved)
        train_raws.add(moved)

    train_indices = [idx for idx, raw_anchor in enumerate(dataset.group_ids) if raw_anchor in train_raws]
    val_indices = [idx for idx, raw_anchor in enumerate(dataset.group_ids) if raw_anchor in val_raws]
    return train_indices, val_indices, sorted(train_raws), sorted(val_raws)


class RawGroupBatchSampler(Sampler):
    def __init__(self, group_ids, indices, groups_per_batch: int, seed: int = 42, shuffle: bool = True, drop_last: bool = True):
        self.group_ids = group_ids
        self.indices = list(indices)
        self.groups_per_batch = max(1, int(groups_per_batch))
        self.seed = seed
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.epoch = 0

        raw_to_indices = {}
        for idx in self.indices:
            raw_to_indices.setdefault(self.group_ids[idx], []).append(idx)
        self.grouped_indices = [sorted(v) for _, v in sorted(raw_to_indices.items())]

    def __len__(self):
        total_groups = len(self.grouped_indices)
        if self.drop_last:
            return total_groups // self.groups_per_batch
        return (total_groups + self.groups_per_batch - 1) // self.groups_per_batch

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        grouped = [group[:] for group in self.grouped_indices]
        if self.shuffle:
            rng.shuffle(grouped)

        for start in range(0, len(grouped), self.groups_per_batch):
            chunk = grouped[start : start + self.groups_per_batch]
            if len(chunk) < self.groups_per_batch and self.drop_last:
                continue
            batch = []
            for group in chunk:
                if self.shuffle:
                    rng.shuffle(group)
                batch.extend(group)
            if self.shuffle:
                rng.shuffle(batch)
            yield batch


class CenterMultiCropTransform:
    def __init__(self, image_size: int, resize_size: int, crop_sizes=None):
        self.image_size = int(image_size)
        self.resize_size = int(resize_size)
        if crop_sizes is None:
            crop_sizes = [self.image_size + 32, self.image_size, max(self.image_size - 32, 128)]
        crop_sizes = [min(int(size), self.resize_size) for size in crop_sizes]
        self.crop_sizes = []
        for size in crop_sizes:
            if size not in self.crop_sizes:
                self.crop_sizes.append(size)
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, image):
        image = TF.resize(image, [self.resize_size, self.resize_size])
        views = []
        for crop_size in self.crop_sizes:
            crop = TF.center_crop(image, [crop_size, crop_size])
            if crop_size != self.image_size:
                crop = TF.resize(crop, [self.image_size, self.image_size])
            crop = TF.to_tensor(crop)
            crop = self.normalize(crop)
            views.append(crop)
        return torch.stack(views, dim=0)


def build_transforms(preset: str = "mild", image_size: int = 224, resize_size: int = None):
    image_size = int(image_size)
    resize_size = int(resize_size) if resize_size is not None else None
    if preset == "strong":
        train_transform = T.Compose(
            [
                T.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    elif preset == "mild":
        resize_size = resize_size or max(image_size + 32, 256)
        train_transform = T.Compose(
            [
                T.Resize(resize_size),
                T.RandomCrop(image_size),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    elif preset == "none":
        train_transform = T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        eval_transform = T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    elif preset == "center":
        resize_size = resize_size or max(image_size + 64, 320)
        train_transform = T.Compose(
            [
                T.Resize(resize_size),
                T.CenterCrop(image_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        eval_transform = T.Compose(
            [
                T.Resize(resize_size),
                T.CenterCrop(image_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    elif preset == "center_multi":
        resize_size = resize_size or max(image_size + 96, 320)
        train_transform = CenterMultiCropTransform(image_size=image_size, resize_size=resize_size)
        eval_transform = CenterMultiCropTransform(image_size=image_size, resize_size=resize_size)
    else:
        raise ValueError(f"Unsupported augmentation preset: {preset}")

    if preset in {"strong", "mild"}:
        eval_transform = T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    return train_transform, eval_transform


def compute_source_logits(student_repr: torch.Tensor, raw_anchors, prototype_bank: torch.Tensor, bank_raws):
    logits = torch.matmul(F.normalize(student_repr, dim=1), F.normalize(prototype_bank, dim=1).T)
    raw_to_index = {raw_anchor: idx for idx, raw_anchor in enumerate(bank_raws)}
    target_index = torch.tensor([raw_to_index[raw_anchor] for raw_anchor in raw_anchors], device=student_repr.device)
    return logits, target_index


def compute_patch_source_logits(
    patch_repr: torch.Tensor,
    prototype_bank: torch.Tensor,
    topk: int = 8,
    pooling: str = "topk_mean",
    temperature: float = 8.0,
) -> torch.Tensor:
    if patch_repr is None:
        raise ValueError("patch_repr is required for patch-based logit modes.")
    patch_repr = F.normalize(patch_repr, dim=-1)
    prototype_bank = F.normalize(prototype_bank, dim=-1)
    patch_scores = torch.einsum("bpd,sd->bps", patch_repr, prototype_bank)
    if pooling == "mean":
        return patch_scores.mean(dim=1)
    if pooling == "max":
        return patch_scores.max(dim=1).values
    if pooling == "logsumexp":
        temperature = max(float(temperature), 1e-6)
        return torch.logsumexp(patch_scores * temperature, dim=1) / temperature
    if pooling == "softmax_mean":
        temperature = max(float(temperature), 1e-6)
        weights = torch.softmax(patch_scores * temperature, dim=1)
        return (weights * patch_scores).sum(dim=1)
    if pooling == "topk_mean":
        topk = max(1, min(int(topk), patch_scores.shape[1]))
        return torch.topk(patch_scores, k=topk, dim=1).values.mean(dim=1)
    raise ValueError(f"Unsupported patch pooling: {pooling}")


def compute_source_logits_with_mode(
    student_repr: torch.Tensor,
    raw_anchors,
    prototype_bank: torch.Tensor,
    bank_raws,
    patch_repr: torch.Tensor = None,
    logit_mode: str = "global",
    patch_topk: int = 8,
    patch_logit_weight: float = 0.5,
    patch_pooling: str = "topk_mean",
    patch_pooling_temperature: float = 8.0,
):
    global_logits, target_index = compute_source_logits(student_repr, raw_anchors, prototype_bank, bank_raws)
    if logit_mode == "global":
        return global_logits, target_index

    patch_logits = compute_patch_source_logits(
        patch_repr=patch_repr,
        prototype_bank=prototype_bank,
        topk=patch_topk,
        pooling=patch_pooling,
        temperature=patch_pooling_temperature,
    )
    if logit_mode == "patch_topk_mean":
        return patch_logits, target_index
    if logit_mode == "global_patch_mean":
        patch_weight = min(max(float(patch_logit_weight), 0.0), 1.0)
        global_weight = 1.0 - patch_weight
        return global_weight * global_logits + patch_weight * patch_logits, target_index
    raise ValueError(f"Unsupported logit_mode: {logit_mode}")


def compute_student_bank_logits(
    student_repr: torch.Tensor,
    raw_anchors,
    prototype_bank: torch.Tensor,
    bank_raws,
    patch_repr: torch.Tensor = None,
    logit_mode: str = "global",
    patch_topk: int = 8,
    patch_logit_weight: float = 0.5,
    patch_pooling: str = "topk_mean",
    patch_pooling_temperature: float = 8.0,
    verifier=None,
    verifier_mode: str = "global_mlp",
    verifier_patch_topk: int = 4,
    verifier_patch_temperature: float = 8.0,
):
    base_logits, target_index = compute_source_logits_with_mode(
        student_repr=student_repr,
        raw_anchors=raw_anchors,
        prototype_bank=prototype_bank,
        bank_raws=bank_raws,
        patch_repr=patch_repr,
        logit_mode=logit_mode,
        patch_topk=patch_topk,
        patch_logit_weight=patch_logit_weight,
        patch_pooling=patch_pooling,
        patch_pooling_temperature=patch_pooling_temperature,
    )
    if verifier is None:
        return base_logits, target_index
    if verifier_mode == "global_mlp":
        return verifier.score_bank(student_repr, prototype_bank), target_index
    if verifier_mode == "stats_local":
        return verifier.score_stats_bank(
            student_repr=student_repr,
            patch_repr=patch_repr,
            prototype_bank=prototype_bank,
            topk=verifier_patch_topk,
            temperature=verifier_patch_temperature,
        ), target_index
    raise ValueError(f"Unsupported verifier_mode: {verifier_mode}")


def compute_source_top1_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    with torch.no_grad():
        return float((logits.argmax(dim=1) == targets).float().mean().item())


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


def compute_similarity_stats(logits: torch.Tensor, targets: torch.Tensor):
    positive_scores = logits.gather(1, targets.unsqueeze(1)).squeeze(1)
    neg_logits = logits.clone()
    neg_logits.scatter_(1, targets.unsqueeze(1), float("-inf"))
    hardest_negative_scores = neg_logits.max(dim=1).values
    correct = logits.argmax(dim=1) == targets
    return positive_scores, hardest_negative_scores, correct


def compute_pairwise_verification_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    sample_weights: torch.Tensor,
    margin: float = 0.0,
) -> torch.Tensor:
    positive_scores = logits.gather(1, targets.unsqueeze(1)).squeeze(1)
    negative_mask = torch.ones_like(logits, dtype=torch.bool)
    negative_mask.scatter_(1, targets.unsqueeze(1), False)
    negative_scores = logits[negative_mask].view(logits.shape[0], -1)
    pairwise_terms = F.softplus(margin + negative_scores - positive_scores.unsqueeze(1))
    sample_losses = pairwise_terms.mean(dim=1)
    return weighted_mean(sample_losses, sample_weights)


def compute_group_supervised_contrastive_loss(
    embeddings: torch.Tensor,
    raw_anchors,
    sample_weights: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    embeddings = F.normalize(embeddings, dim=1)
    logits = torch.matmul(embeddings, embeddings.T) / max(float(temperature), 1e-6)
    self_mask = torch.eye(logits.shape[0], device=logits.device, dtype=torch.bool)
    logits = logits.masked_fill(self_mask, float("-inf"))

    raw_labels = list(raw_anchors)
    positive_mask = torch.tensor(
        [[raw_labels[i] == raw_labels[j] for j in range(len(raw_labels))] for i in range(len(raw_labels))],
        device=logits.device,
        dtype=torch.bool,
    )
    positive_mask &= ~self_mask

    valid_mask = positive_mask.any(dim=1)
    if not valid_mask.any():
        return embeddings.new_tensor(0.0)

    log_denom = torch.logsumexp(logits, dim=1, keepdim=True)
    log_prob = logits - log_denom
    pos_counts = positive_mask.sum(dim=1).clamp_min(1)
    sample_losses = -(log_prob.masked_fill(~positive_mask, 0.0).sum(dim=1) / pos_counts)

    valid_losses = sample_losses[valid_mask]
    valid_weights = sample_weights[valid_mask]
    return weighted_mean(valid_losses, valid_weights)


def compute_teacher_logit_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    sample_weights: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    temperature = max(float(temperature), 1e-6)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    per_sample = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=1)
    return weighted_mean(per_sample * (temperature ** 2), sample_weights)


def compute_teacher_margin_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    sample_weights: torch.Tensor,
) -> torch.Tensor:
    student_pos = student_logits.gather(1, targets.unsqueeze(1)).squeeze(1)
    teacher_pos = teacher_logits.gather(1, targets.unsqueeze(1)).squeeze(1)

    negative_mask = torch.ones_like(student_logits, dtype=torch.bool)
    negative_mask.scatter_(1, targets.unsqueeze(1), False)
    student_neg = student_logits[negative_mask].view(student_logits.shape[0], -1)
    teacher_neg = teacher_logits[negative_mask].view(teacher_logits.shape[0], -1)

    student_margin = student_pos.unsqueeze(1) - student_neg
    teacher_margin = teacher_pos.unsqueeze(1) - teacher_neg
    sample_losses = F.smooth_l1_loss(student_margin, teacher_margin, reduction="none").mean(dim=1)
    return weighted_mean(sample_losses, sample_weights)


def compute_sequence_alignment_loss(
    student_seq: torch.Tensor,
    target_seq: torch.Tensor,
    sample_weights: torch.Tensor,
) -> torch.Tensor:
    if student_seq is None or target_seq is None:
        return sample_weights.new_tensor(0.0)
    cosine = F.cosine_similarity(student_seq, target_seq, dim=-1)
    sample_losses = 1.0 - cosine.mean(dim=1)
    return weighted_mean(sample_losses, sample_weights)


def binary_auc_from_scores(positive_scores: torch.Tensor, negative_scores: torch.Tensor) -> float:
    positive_scores = positive_scores.detach().float().cpu()
    negative_scores = negative_scores.detach().float().cpu()
    n_pos = int(positive_scores.numel())
    n_neg = int(negative_scores.numel())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    scores = torch.cat([positive_scores, negative_scores], dim=0)
    labels = torch.cat(
        [
            torch.ones(n_pos, dtype=torch.int64),
            torch.zeros(n_neg, dtype=torch.int64),
        ],
        dim=0,
    )
    order = torch.argsort(scores, stable=True)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, scores.numel() + 1, dtype=torch.float32)
    pos_ranks = ranks[labels == 1].sum().item()
    return float((pos_ranks - n_pos * (n_pos + 1) / 2.0) / max(n_pos * n_neg, 1))


def accuracy_for_versions(correct: torch.Tensor, versions: torch.Tensor, version_ids) -> float:
    mask = torch.zeros_like(versions, dtype=torch.bool)
    for version_id in version_ids:
        mask |= versions == int(version_id)
    if not mask.any():
        return float("nan")
    return float(correct[mask].float().mean().item())


def build_batch_prototype_bank(raw_anchors, prototype_vec: torch.Tensor):
    bank_raws = []
    bank_vecs = []
    for raw_anchor, vec in zip(raw_anchors, prototype_vec):
        if raw_anchor in bank_raws:
            continue
        bank_raws.append(raw_anchor)
        bank_vecs.append(vec)
    return torch.stack(bank_vecs, dim=0), bank_raws


def fit_projection_from_train_data(dataset: PrototypeDistillationDataset, train_indices, train_raws, output_dim: int):
    if output_dim is None or output_dim <= 0 or output_dim >= dataset.prototype_dim:
        return None, None

    feature_list = [dataset.get_prototype_tensor(raw_anchor) for raw_anchor in train_raws]
    for idx in train_indices:
        record = dataset.records[idx]
        version_id = record["version_id"]
        teacher_vec = dataset.teacher_vec_bank.get(version_id)
        if teacher_vec is not None:
            feature_list.append(teacher_vec)

    feature_mat = torch.stack(feature_list, dim=0).float()
    feature_mean = feature_mat.mean(dim=0)
    centered = feature_mat - feature_mean
    q = min(int(output_dim), centered.shape[0] - 1, centered.shape[1])
    if q <= 0:
        return None, None
    _, _, basis = torch.pca_lowrank(centered, q=q, center=False)
    return feature_mean, basis[:, :q]


def load_projection_checkpoint(projection_checkpoint: str):
    if not projection_checkpoint:
        return None, None
    pack = torch.load(projection_checkpoint, map_location="cpu")
    mean_vec = pack.get("mean_vec")
    projection = pack.get("projection")
    if mean_vec is None or projection is None:
        raise KeyError(
            f"Projection checkpoint {projection_checkpoint} must contain 'mean_vec' and 'projection'."
        )
    return mean_vec.float(), projection.float()


def project_features(features: torch.Tensor, projection_mean: torch.Tensor, projection_basis: torch.Tensor) -> torch.Tensor:
    if projection_mean is None or projection_basis is None:
        return F.normalize(features, dim=-1)
    centered = features - projection_mean.to(features.device, dtype=features.dtype)
    projected = centered @ projection_basis.to(features.device, dtype=features.dtype)
    return F.normalize(projected, dim=-1)


def load_state_dict_shape_safe(module: torch.nn.Module, state_dict: dict, module_name: str = "module"):
    current_state = module.state_dict()
    filtered_state = {}
    skipped = []
    for key, value in state_dict.items():
        if key not in current_state:
            skipped.append((key, "missing_in_target"))
            continue
        if current_state[key].shape != value.shape:
            skipped.append((key, f"shape_mismatch {tuple(value.shape)} != {tuple(current_state[key].shape)}"))
            continue
        filtered_state[key] = value
    load_status = module.load_state_dict(filtered_state, strict=False)
    if skipped:
        preview = ", ".join([f"{name} ({reason})" for name, reason in skipped[:8]])
        if len(skipped) > 8:
            preview += f", ... total={len(skipped)}"
        print(f"{module_name} skipped keys on shape-safe load: {preview}", flush=True)
    return load_status


def train_stage3_prototype(args):
    set_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    pin_memory = device.type == "cuda"

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

    if args.projection_checkpoint:
        projection_mean, projection_basis = load_projection_checkpoint(args.projection_checkpoint)
    else:
        projection_mean, projection_basis = fit_projection_from_train_data(
            dataset=full_dataset,
            train_indices=train_indices,
            train_raws=train_raws,
            output_dim=args.project_dim,
        )
    projected_dim = int(projection_basis.shape[1]) if projection_basis is not None else full_dataset.prototype_dim

    student = VisualEncoder(
        d_out=projected_dim,
        backbone_type=args.backbone_type,
        pretrained=not args.no_pretrained,
        input_mode=args.input_mode,
        residual_scale=args.residual_scale,
        residual_kernel=args.residual_kernel,
        use_stage_sequence_head=(args.stage_sequence_weight > 0.0 or args.prototype_sequence_weight > 0.0),
        local_crop_mode=args.local_crop_mode,
        local_crop_size=args.local_crop_size,
        local_patch_offset=args.local_patch_offset,
    ).to(device)
    verifier = PrototypeVerifier(
        d_model=projected_dim,
        hidden_dim=args.verifier_hidden_dim,
        dropout=args.verifier_dropout,
    ).to(device) if args.use_verifier_head else None
    if args.use_verifier_head and args.verifier_mode == "global_mlp" and args.logit_mode != "global":
        raise ValueError("verifier_mode=global_mlp only supports --logit_mode global.")
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location="cpu")
        load_status = load_state_dict_shape_safe(student, checkpoint["student_state_dict"], module_name="Student")
        if load_status.missing_keys:
            print(f"Student missing keys on init load: {load_status.missing_keys}", flush=True)
        if load_status.unexpected_keys:
            print(f"Student unexpected keys on init load: {load_status.unexpected_keys}", flush=True)
        if verifier is not None and "verifier_state_dict" in checkpoint:
            verifier_status = load_state_dict_shape_safe(verifier, checkpoint["verifier_state_dict"], module_name="Verifier")
            if verifier_status.missing_keys:
                print(f"Verifier missing keys on init load: {verifier_status.missing_keys}", flush=True)
            if verifier_status.unexpected_keys:
                print(f"Verifier unexpected keys on init load: {verifier_status.unexpected_keys}", flush=True)

    if args.freeze_student_encoder:
        for param in student.parameters():
            param.requires_grad = False

    optim_params = [param for param in student.parameters() if param.requires_grad]
    if verifier is not None:
        optim_params.extend(verifier.parameters())
    if not optim_params:
        raise ValueError("No trainable parameters found. Disable --freeze_student_encoder or enable verifier head.")
    optimizer = optim.AdamW(optim_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72, flush=True)
    print("Stage-3 Prototype Distillation", flush=True)
    print(f"Device          : {device}", flush=True)
    print(f"Prototype cache : {args.prototype_cache_dir}", flush=True)
    print(f"Teacher cache   : {args.teacher_cache_dir}", flush=True)
    print(f"Teacher key     : {args.teacher_key}", flush=True)
    print(f"Projection ckpt : {args.projection_checkpoint or 'fit_from_train'}", flush=True)
    print(f"Init checkpoint : {args.init_checkpoint or 'none'}", flush=True)
    print(f"Prototype dim   : {full_dataset.prototype_dim}", flush=True)
    print(f"Projected dim   : {projected_dim}", flush=True)
    print(f"Train raws      : {len(train_raws)}", flush=True)
    print(f"Val raws        : {len(val_raws)}", flush=True)
    print(f"Train samples   : {len(train_indices)}", flush=True)
    print(f"Val samples     : {len(val_indices)}", flush=True)
    print(f"RGB versions    : {args.include_versions}", flush=True)
    print(f"Anchor vers     : {args.anchor_versions}", flush=True)
    print(f"Shift vers      : {args.shift_versions}", flush=True)
    print(f"Cls scope       : {args.classification_scope}", flush=True)
    print(f"Pair scope      : {args.pairwise_scope}", flush=True)
    print(f"Teacher distill : {args.teacher_distill_weight} @ {args.teacher_distill_scope}", flush=True)
    print(f"Teacher margin  : {args.teacher_margin_distill_weight}", flush=True)
    print(f"Stage seq loss  : teacher={args.stage_sequence_weight}, proto={args.prototype_sequence_weight}", flush=True)
    print(f"Input mode      : {args.input_mode} (scale={args.residual_scale}, k={args.residual_kernel})", flush=True)
    print(f"Local crop      : {args.local_crop_mode} (size={args.local_crop_size}, offset={args.local_patch_offset})", flush=True)
    print(
        f"Logit mode      : {args.logit_mode} "
        f"(pool={args.patch_pooling}, topk={args.patch_topk}, temp={args.patch_pooling_temperature}, patch_w={args.patch_logit_weight})",
        flush=True,
    )
    print(f"Freeze encoder  : {args.freeze_student_encoder}", flush=True)
    print(f"Image size      : {args.image_size} (resize={args.resize_size})", flush=True)
    print(f"Verifier head   : {args.use_verifier_head} (hidden={args.verifier_hidden_dim}, drop={args.verifier_dropout})", flush=True)
    print(f"Verifier mode   : {args.verifier_mode} (topk={args.verifier_patch_topk}, temp={args.verifier_patch_temperature})", flush=True)
    print(f"Select metric   : {args.selection_metric}", flush=True)
    print(f"Group SupCon    : {args.group_contrastive_weight}", flush=True)
    print("=" * 72, flush=True)

    history = []
    best_metric = float("-inf")

    train_bank_raws = train_raws
    train_bank = project_features(
        torch.stack([train_dataset.get_prototype_tensor(raw) for raw in train_bank_raws]).to(device),
        projection_mean,
        projection_basis,
    )
    val_bank_raws = val_raws
    val_bank = project_features(
        torch.stack([val_dataset.get_prototype_tensor(raw) for raw in val_bank_raws]).to(device),
        projection_mean,
        projection_basis,
    )

    for epoch in range(args.epochs):
        if hasattr(train_loader.batch_sampler, "set_epoch"):
            train_loader.batch_sampler.set_epoch(epoch)

        student.train()
        if args.freeze_student_encoder:
            student.eval()
        if verifier is not None:
            verifier.train()
        train_loss = 0.0
        train_gallery_acc = 0.0
        train_verify_acc = 0.0
        train_margin = 0.0
        train_pairwise_loss = 0.0
        train_group_contrastive_loss = 0.0
        train_teacher_distill_loss = 0.0
        train_teacher_margin_distill_loss = 0.0
        train_stage_sequence_loss = 0.0
        train_prototype_sequence_loss = 0.0
        train_anchor_acc = 0.0
        train_shift_acc = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Train]")
        for batch in pbar:
            rgb_image = batch["rgb_image"].to(device, non_blocking=True)
            proto_vec = project_features(batch["prototype_vec"].to(device, non_blocking=True), projection_mean, projection_basis)
            teacher_vec = project_features(batch["teacher_vec"].to(device, non_blocking=True), projection_mean, projection_basis)
            proto_seq_raw = batch["prototype_seq"].to(device, non_blocking=True)
            teacher_seq_raw = batch["teacher_seq"].to(device, non_blocking=True)
            proto_seq = project_features(
                proto_seq_raw.reshape(-1, proto_seq_raw.shape[-1]),
                projection_mean,
                projection_basis,
            ).reshape(proto_seq_raw.shape[0], proto_seq_raw.shape[1], -1)
            teacher_seq = project_features(
                teacher_seq_raw.reshape(-1, teacher_seq_raw.shape[-1]),
                projection_mean,
                projection_basis,
            ).reshape(teacher_seq_raw.shape[0], teacher_seq_raw.shape[1], -1)
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
                    return_sequence=(args.stage_sequence_weight > 0.0 or args.prototype_sequence_weight > 0.0),
                    return_patch_tokens=(args.logit_mode != "global" or (verifier is not None and args.verifier_mode == "stats_local")),
                )
                if isinstance(student_out, dict):
                    student_repr = student_out["global_repr"]
                    student_stage_seq = student_out.get("stage_repr")
                    student_patch_repr = student_out.get("patch_repr")
                else:
                    student_repr = student_out
                    student_stage_seq = None
                    student_patch_repr = None
                if args.classification_scope == "batch":
                    cls_bank, cls_bank_raws = build_batch_prototype_bank(raw_anchors, proto_vec)
                else:
                    cls_bank, cls_bank_raws = train_bank, train_bank_raws
                logits, targets = compute_student_bank_logits(
                    student_repr=student_repr,
                    raw_anchors=raw_anchors,
                    prototype_bank=cls_bank,
                    bank_raws=cls_bank_raws,
                    patch_repr=student_patch_repr,
                    logit_mode=args.logit_mode,
                    patch_topk=args.patch_topk,
                    patch_logit_weight=args.patch_logit_weight,
                    patch_pooling=args.patch_pooling,
                    patch_pooling_temperature=args.patch_pooling_temperature,
                    verifier=verifier,
                    verifier_mode=args.verifier_mode,
                    verifier_patch_topk=args.verifier_patch_topk,
                    verifier_patch_temperature=args.verifier_patch_temperature,
                )
                if args.pairwise_scope == "batch":
                    pair_bank, pair_bank_raws = cls_bank, cls_bank_raws
                else:
                    pair_bank, pair_bank_raws = train_bank, train_bank_raws
                pair_logits, pair_targets = compute_student_bank_logits(
                    student_repr=student_repr,
                    raw_anchors=raw_anchors,
                    prototype_bank=pair_bank,
                    bank_raws=pair_bank_raws,
                    patch_repr=student_patch_repr,
                    logit_mode=args.logit_mode,
                    patch_topk=args.patch_topk,
                    patch_logit_weight=args.patch_logit_weight,
                    patch_pooling=args.patch_pooling,
                    patch_pooling_temperature=args.patch_pooling_temperature,
                    verifier=verifier,
                    verifier_mode=args.verifier_mode,
                    verifier_patch_topk=args.verifier_patch_topk,
                    verifier_patch_temperature=args.verifier_patch_temperature,
                )
                if args.teacher_distill_scope == "batch":
                    distill_bank, distill_bank_raws = cls_bank, cls_bank_raws
                else:
                    distill_bank, distill_bank_raws = train_bank, train_bank_raws
                student_distill_logits, student_distill_targets = compute_student_bank_logits(
                    student_repr=student_repr,
                    raw_anchors=raw_anchors,
                    prototype_bank=distill_bank,
                    bank_raws=distill_bank_raws,
                    patch_repr=student_patch_repr,
                    logit_mode=args.logit_mode,
                    patch_topk=args.patch_topk,
                    patch_logit_weight=args.patch_logit_weight,
                    patch_pooling=args.patch_pooling,
                    patch_pooling_temperature=args.patch_pooling_temperature,
                    verifier=verifier,
                    verifier_mode=args.verifier_mode,
                    verifier_patch_topk=args.verifier_patch_topk,
                    verifier_patch_temperature=args.verifier_patch_temperature,
                )
                teacher_distill_logits, _ = compute_source_logits(teacher_vec, raw_anchors, distill_bank, distill_bank_raws)
                positive_scores, hardest_negative_scores, correct = compute_similarity_stats(pair_logits, pair_targets)

                cos_terms = 1.0 - F.cosine_similarity(student_repr, proto_vec, dim=1)
                cos_loss = weighted_mean(cos_terms, sample_weights)
                teacher_terms = 1.0 - F.cosine_similarity(student_repr, teacher_vec, dim=1)
                teacher_loss = weighted_mean(teacher_terms, sample_weights)
                ce_terms = F.cross_entropy(logits / args.temperature, targets, reduction="none")
                cls_loss = weighted_mean(ce_terms, sample_weights)
                pairwise_loss = compute_pairwise_verification_loss(
                    pair_logits,
                    pair_targets,
                    sample_weights,
                    margin=args.pairwise_margin,
                )
                group_contrastive_loss = compute_group_supervised_contrastive_loss(
                    embeddings=student_repr,
                    raw_anchors=raw_anchors,
                    sample_weights=sample_weights,
                    temperature=args.group_contrastive_temperature,
                )
                teacher_distill_loss = compute_teacher_logit_distillation_loss(
                    student_logits=student_distill_logits,
                    teacher_logits=teacher_distill_logits,
                    sample_weights=sample_weights,
                    temperature=args.teacher_distill_temperature,
                )
                teacher_margin_distill_loss = compute_teacher_margin_distillation_loss(
                    student_logits=student_distill_logits,
                    teacher_logits=teacher_distill_logits,
                    targets=student_distill_targets,
                    sample_weights=sample_weights,
                )
                stage_sequence_loss = compute_sequence_alignment_loss(
                    student_seq=student_stage_seq,
                    target_seq=teacher_seq,
                    sample_weights=sample_weights,
                )
                prototype_sequence_loss = compute_sequence_alignment_loss(
                    student_seq=student_stage_seq,
                    target_seq=proto_seq,
                    sample_weights=sample_weights,
                )
                margin_terms = F.relu(args.margin + hardest_negative_scores - positive_scores)
                margin_loss = weighted_mean(margin_terms, sample_weights)
                loss = (
                    args.cosine_weight * cos_loss
                    + args.teacher_weight * teacher_loss
                    + args.classification_weight * cls_loss
                    + args.pairwise_weight * pairwise_loss
                    + args.group_contrastive_weight * group_contrastive_loss
                    + args.teacher_distill_weight * teacher_distill_loss
                    + args.teacher_margin_distill_weight * teacher_margin_distill_loss
                    + args.stage_sequence_weight * stage_sequence_loss
                    + args.prototype_sequence_weight * prototype_sequence_loss
                    + args.margin_weight * margin_loss
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_acc = float(correct.float().mean().item())
            verify_acc = float((positive_scores > hardest_negative_scores).float().mean().item())
            margin_mean = float((positive_scores - hardest_negative_scores).mean().item())
            anchor_acc = accuracy_for_versions(correct, versions, args.anchor_versions)
            shift_acc = accuracy_for_versions(correct, versions, args.shift_versions)
            train_loss += float(loss.item())
            train_gallery_acc += batch_acc
            train_verify_acc += verify_acc
            train_margin += margin_mean
            train_pairwise_loss += float(pairwise_loss.item())
            train_group_contrastive_loss += float(group_contrastive_loss.item())
            train_teacher_distill_loss += float(teacher_distill_loss.item())
            train_teacher_margin_distill_loss += float(teacher_margin_distill_loss.item())
            train_stage_sequence_loss += float(stage_sequence_loss.item())
            train_prototype_sequence_loss += float(prototype_sequence_loss.item())
            train_anchor_acc += 0.0 if np.isnan(anchor_acc) else anchor_acc
            train_shift_acc += 0.0 if np.isnan(shift_acc) else shift_acc
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "GAcc": f"{batch_acc * 100:.1f}%", "VAcc": f"{verify_acc * 100:.1f}%"})

        scheduler.step()

        student.eval()
        if verifier is not None:
            verifier.eval()
        val_loss = 0.0
        val_gallery_acc = 0.0
        val_batch_gallery_acc = 0.0
        val_verify_acc = 0.0
        val_margin = 0.0
        val_pairwise_loss = 0.0
        val_group_contrastive_loss = 0.0
        val_teacher_distill_loss = 0.0
        val_teacher_margin_distill_loss = 0.0
        val_stage_sequence_loss = 0.0
        val_prototype_sequence_loss = 0.0
        val_anchor_acc = 0.0
        val_shift_acc = 0.0
        val_pairwise_win_sum = 0.0
        val_pairwise_count = 0
        val_pos_scores = []
        val_neg_scores = []
        val_all_neg_scores = []
        with torch.no_grad():
            for batch in val_loader:
                rgb_image = batch["rgb_image"].to(device, non_blocking=True)
                proto_vec = project_features(batch["prototype_vec"].to(device, non_blocking=True), projection_mean, projection_basis)
                teacher_vec = project_features(batch["teacher_vec"].to(device, non_blocking=True), projection_mean, projection_basis)
                proto_seq_raw = batch["prototype_seq"].to(device, non_blocking=True)
                teacher_seq_raw = batch["teacher_seq"].to(device, non_blocking=True)
                proto_seq = project_features(
                    proto_seq_raw.reshape(-1, proto_seq_raw.shape[-1]),
                    projection_mean,
                    projection_basis,
                ).reshape(proto_seq_raw.shape[0], proto_seq_raw.shape[1], -1)
                teacher_seq = project_features(
                    teacher_seq_raw.reshape(-1, teacher_seq_raw.shape[-1]),
                    projection_mean,
                    projection_basis,
                ).reshape(teacher_seq_raw.shape[0], teacher_seq_raw.shape[1], -1)
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
                    return_sequence=(args.stage_sequence_weight > 0.0 or args.prototype_sequence_weight > 0.0),
                    return_patch_tokens=(args.logit_mode != "global" or (verifier is not None and args.verifier_mode == "stats_local")),
                )
                if isinstance(student_out, dict):
                    student_repr = student_out["global_repr"]
                    student_stage_seq = student_out.get("stage_repr")
                    student_patch_repr = student_out.get("patch_repr")
                else:
                    student_repr = student_out
                    student_stage_seq = None
                    student_patch_repr = None
                batch_bank, batch_bank_raws = build_batch_prototype_bank(raw_anchors, proto_vec)
                batch_logits, batch_targets = compute_student_bank_logits(
                    student_repr=student_repr,
                    raw_anchors=raw_anchors,
                    prototype_bank=batch_bank,
                    bank_raws=batch_bank_raws,
                    patch_repr=student_patch_repr,
                    logit_mode=args.logit_mode,
                    patch_topk=args.patch_topk,
                    patch_logit_weight=args.patch_logit_weight,
                    patch_pooling=args.patch_pooling,
                    patch_pooling_temperature=args.patch_pooling_temperature,
                    verifier=verifier,
                    verifier_mode=args.verifier_mode,
                    verifier_patch_topk=args.verifier_patch_topk,
                    verifier_patch_temperature=args.verifier_patch_temperature,
                )
                full_logits, full_targets = compute_student_bank_logits(
                    student_repr=student_repr,
                    raw_anchors=raw_anchors,
                    prototype_bank=val_bank,
                    bank_raws=val_bank_raws,
                    patch_repr=student_patch_repr,
                    logit_mode=args.logit_mode,
                    patch_topk=args.patch_topk,
                    patch_logit_weight=args.patch_logit_weight,
                    patch_pooling=args.patch_pooling,
                    patch_pooling_temperature=args.patch_pooling_temperature,
                    verifier=verifier,
                    verifier_mode=args.verifier_mode,
                    verifier_patch_topk=args.verifier_patch_topk,
                    verifier_patch_temperature=args.verifier_patch_temperature,
                )
                positive_scores, hardest_negative_scores, correct = compute_similarity_stats(full_logits, full_targets)

                cos_terms = 1.0 - F.cosine_similarity(student_repr, proto_vec, dim=1)
                cos_loss = weighted_mean(cos_terms, sample_weights)
                teacher_terms = 1.0 - F.cosine_similarity(student_repr, teacher_vec, dim=1)
                teacher_loss = weighted_mean(teacher_terms, sample_weights)
                if args.classification_scope == "batch":
                    cls_logits, cls_targets = batch_logits, batch_targets
                else:
                    cls_logits, cls_targets = full_logits, full_targets
                if args.pairwise_scope == "batch":
                    pair_logits, pair_targets = batch_logits, batch_targets
                else:
                    pair_logits, pair_targets = full_logits, full_targets
                if args.teacher_distill_scope == "batch":
                    distill_logits, distill_targets = batch_logits, batch_targets
                    teacher_distill_bank, teacher_distill_raws = build_batch_prototype_bank(raw_anchors, proto_vec)
                else:
                    distill_logits, distill_targets = full_logits, full_targets
                    teacher_distill_bank, teacher_distill_raws = val_bank, val_bank_raws
                teacher_distill_logits, _ = compute_source_logits(
                    teacher_vec,
                    raw_anchors,
                    teacher_distill_bank,
                    teacher_distill_raws,
                )
                ce_terms = F.cross_entropy(cls_logits / args.temperature, cls_targets, reduction="none")
                cls_loss = weighted_mean(ce_terms, sample_weights)
                pairwise_loss = compute_pairwise_verification_loss(
                    pair_logits,
                    pair_targets,
                    sample_weights,
                    margin=args.pairwise_margin,
                )
                group_contrastive_loss = compute_group_supervised_contrastive_loss(
                    embeddings=student_repr,
                    raw_anchors=raw_anchors,
                    sample_weights=sample_weights,
                    temperature=args.group_contrastive_temperature,
                )
                teacher_distill_loss = compute_teacher_logit_distillation_loss(
                    student_logits=distill_logits,
                    teacher_logits=teacher_distill_logits,
                    sample_weights=sample_weights,
                    temperature=args.teacher_distill_temperature,
                )
                teacher_margin_distill_loss = compute_teacher_margin_distillation_loss(
                    student_logits=distill_logits,
                    teacher_logits=teacher_distill_logits,
                    targets=distill_targets,
                    sample_weights=sample_weights,
                )
                stage_sequence_loss = compute_sequence_alignment_loss(
                    student_seq=student_stage_seq,
                    target_seq=teacher_seq,
                    sample_weights=sample_weights,
                )
                prototype_sequence_loss = compute_sequence_alignment_loss(
                    student_seq=student_stage_seq,
                    target_seq=proto_seq,
                    sample_weights=sample_weights,
                )
                margin_terms = F.relu(args.margin + hardest_negative_scores - positive_scores)
                margin_loss = weighted_mean(margin_terms, sample_weights)
                loss = (
                    args.cosine_weight * cos_loss
                    + args.teacher_weight * teacher_loss
                    + args.classification_weight * cls_loss
                    + args.pairwise_weight * pairwise_loss
                    + args.group_contrastive_weight * group_contrastive_loss
                    + args.teacher_distill_weight * teacher_distill_loss
                    + args.teacher_margin_distill_weight * teacher_margin_distill_loss
                    + args.stage_sequence_weight * stage_sequence_loss
                    + args.prototype_sequence_weight * prototype_sequence_loss
                    + args.margin_weight * margin_loss
                )

                val_loss += float(loss.item())
                val_pairwise_loss += float(pairwise_loss.item())
                val_group_contrastive_loss += float(group_contrastive_loss.item())
                val_teacher_distill_loss += float(teacher_distill_loss.item())
                val_teacher_margin_distill_loss += float(teacher_margin_distill_loss.item())
                val_stage_sequence_loss += float(stage_sequence_loss.item())
                val_prototype_sequence_loss += float(prototype_sequence_loss.item())
                val_gallery_acc += float(correct.float().mean().item())
                val_batch_gallery_acc += compute_source_top1_accuracy(batch_logits, batch_targets)
                val_verify_acc += float((positive_scores > hardest_negative_scores).float().mean().item())
                val_margin += float((positive_scores - hardest_negative_scores).mean().item())
                anchor_acc = accuracy_for_versions(correct, versions, args.anchor_versions)
                shift_acc = accuracy_for_versions(correct, versions, args.shift_versions)
                val_anchor_acc += 0.0 if np.isnan(anchor_acc) else anchor_acc
                val_shift_acc += 0.0 if np.isnan(shift_acc) else shift_acc
                val_pos_scores.append(positive_scores.cpu())
                val_neg_scores.append(hardest_negative_scores.cpu())
                for row_idx, target_idx in enumerate(full_targets.tolist()):
                    mask = torch.ones(full_logits.shape[1], device=full_logits.device, dtype=torch.bool)
                    mask[target_idx] = False
                    row_neg_scores = full_logits[row_idx][mask]
                    val_pairwise_win_sum += float((positive_scores[row_idx] > row_neg_scores).float().mean().item())
                    val_pairwise_count += 1
                    val_all_neg_scores.append(row_neg_scores.cpu())

        val_auc = binary_auc_from_scores(
            torch.cat(val_pos_scores, dim=0) if val_pos_scores else torch.empty(0),
            torch.cat(val_neg_scores, dim=0) if val_neg_scores else torch.empty(0),
        )
        val_pairwise_auc = binary_auc_from_scores(
            torch.cat(val_pos_scores, dim=0) if val_pos_scores else torch.empty(0),
            torch.cat(val_all_neg_scores, dim=0) if val_all_neg_scores else torch.empty(0),
        )

        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": train_loss / max(len(train_loader), 1),
            "train_gallery_acc": train_gallery_acc / max(len(train_loader), 1),
            "train_verify_acc": train_verify_acc / max(len(train_loader), 1),
            "train_margin": train_margin / max(len(train_loader), 1),
            "train_pairwise_loss": train_pairwise_loss / max(len(train_loader), 1),
            "train_group_contrastive_loss": train_group_contrastive_loss / max(len(train_loader), 1),
            "train_teacher_distill_loss": train_teacher_distill_loss / max(len(train_loader), 1),
            "train_teacher_margin_distill_loss": train_teacher_margin_distill_loss / max(len(train_loader), 1),
            "train_stage_sequence_loss": train_stage_sequence_loss / max(len(train_loader), 1),
            "train_prototype_sequence_loss": train_prototype_sequence_loss / max(len(train_loader), 1),
            "train_anchor_gallery_acc": train_anchor_acc / max(len(train_loader), 1),
            "train_shift_gallery_acc": train_shift_acc / max(len(train_loader), 1),
            "val_loss": val_loss / max(len(val_loader), 1),
            "val_pairwise_loss": val_pairwise_loss / max(len(val_loader), 1),
            "val_group_contrastive_loss": val_group_contrastive_loss / max(len(val_loader), 1),
            "val_teacher_distill_loss": val_teacher_distill_loss / max(len(val_loader), 1),
            "val_teacher_margin_distill_loss": val_teacher_margin_distill_loss / max(len(val_loader), 1),
            "val_stage_sequence_loss": val_stage_sequence_loss / max(len(val_loader), 1),
            "val_prototype_sequence_loss": val_prototype_sequence_loss / max(len(val_loader), 1),
            "val_gallery_acc": val_gallery_acc / max(len(val_loader), 1),
            "val_batch_gallery_acc": val_batch_gallery_acc / max(len(val_loader), 1),
            "val_verify_acc": val_verify_acc / max(len(val_loader), 1),
            "val_margin": val_margin / max(len(val_loader), 1),
            "val_anchor_gallery_acc": val_anchor_acc / max(len(val_loader), 1),
            "val_shift_gallery_acc": val_shift_acc / max(len(val_loader), 1),
            "val_verify_auc": val_auc,
            "val_pairwise_win_rate": val_pairwise_win_sum / max(val_pairwise_count, 1),
            "val_pairwise_auc": val_pairwise_auc,
        }
        history.append(epoch_record)
        print(epoch_record, flush=True)

        selection_metric = epoch_record.get(args.selection_metric, float("-inf"))
        if selection_metric > best_metric:
            best_metric = selection_metric
            checkpoint_payload = {
                "epoch": epoch + 1,
                "student_state_dict": student.state_dict(),
                "config": vars(args),
                "history": history,
                "projection_mean": projection_mean,
                "projection_basis": projection_basis,
            }
            if verifier is not None:
                checkpoint_payload["verifier_state_dict"] = verifier.state_dict()
            torch.save(checkpoint_payload, save_dir / "stage3_proto_best.pt")

        with open(save_dir / "stage3_proto_history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Train stage-3 source prototype distillation")
    parser.add_argument(
        "--prototype_cache_dir",
        type=str,
        default=str(resolve_experiment_root() / "stage3_prototype_cache_anchor12"),
    )
    parser.add_argument("--meta_path", type=str, default=str(resolve_meta_path()))
    parser.add_argument("--rgb_dir", type=str, default=str(resolve_dataset_root() / "rgb_web_jpg"))
    parser.add_argument("--teacher_cache_dir", type=str, default=str(resolve_experiment_root() / "stage3_teacher_cache_full"))
    parser.add_argument("--teacher_key", type=str, default="teacher_seq")
    parser.add_argument("--save_dir", type=str, default=str(resolve_experiment_root() / "stage3_proto_checkpoints"))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--init_checkpoint", type=str, default=None)
    parser.add_argument("--projection_checkpoint", type=str, default=None)
    parser.add_argument("--project_dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--val_batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--cosine_weight", type=float, default=1.0)
    parser.add_argument("--teacher_weight", type=float, default=1.0)
    parser.add_argument("--classification_weight", type=float, default=1.0)
    parser.add_argument("--pairwise_weight", type=float, default=0.0)
    parser.add_argument("--pairwise_margin", type=float, default=0.0)
    parser.add_argument("--group_contrastive_weight", type=float, default=0.0)
    parser.add_argument("--group_contrastive_temperature", type=float, default=0.07)
    parser.add_argument("--teacher_distill_weight", type=float, default=0.0)
    parser.add_argument("--teacher_distill_temperature", type=float, default=1.0)
    parser.add_argument("--teacher_margin_distill_weight", type=float, default=0.0)
    parser.add_argument("--stage_sequence_weight", type=float, default=0.0)
    parser.add_argument("--prototype_sequence_weight", type=float, default=0.0)
    parser.add_argument("--margin_weight", type=float, default=0.5)
    parser.add_argument("--margin", type=float, default=0.10)
    parser.add_argument("--use_verifier_head", action="store_true")
    parser.add_argument("--verifier_hidden_dim", type=int, default=256)
    parser.add_argument("--verifier_dropout", type=float, default=0.0)
    parser.add_argument("--verifier_mode", type=str, default="global_mlp", choices=["global_mlp", "stats_local"])
    parser.add_argument("--verifier_patch_topk", type=int, default=4)
    parser.add_argument("--verifier_patch_temperature", type=float, default=8.0)
    parser.add_argument("--backbone_type", type=str, default="resnet18", choices=["resnet18", "resnet50"])
    parser.add_argument("--input_mode", type=str, default="rgb", choices=["rgb", "rgb_residual", "residual_only"])
    parser.add_argument("--residual_scale", type=float, default=1.0)
    parser.add_argument("--residual_kernel", type=int, default=5)
    parser.add_argument("--local_crop_mode", type=str, default="none", choices=["none", "center_patch5"])
    parser.add_argument("--local_crop_size", type=int, default=160)
    parser.add_argument("--local_patch_offset", type=int, default=24)
    parser.add_argument(
        "--logit_mode",
        type=str,
        default="global",
        choices=["global", "patch_topk_mean", "global_patch_mean"],
    )
    parser.add_argument("--patch_topk", type=int, default=8)
    parser.add_argument("--patch_logit_weight", type=float, default=0.5)
    parser.add_argument(
        "--patch_pooling",
        type=str,
        default="topk_mean",
        choices=["topk_mean", "mean", "max", "logsumexp", "softmax_mean"],
    )
    parser.add_argument("--patch_pooling_temperature", type=float, default=8.0)
    parser.add_argument("--freeze_student_encoder", action="store_true")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--resize_size", type=int, default=None)
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--augmentation_preset", type=str, default="mild", choices=["strong", "mild", "none", "center", "center_multi"])
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_raws", type=int, default=None)
    parser.add_argument("--include_versions", type=int, nargs="*", default=[1, 2, 3, 4])
    parser.add_argument("--anchor_versions", type=int, nargs="*", default=[1, 2])
    parser.add_argument("--shift_versions", type=int, nargs="*", default=[3, 4])
    parser.add_argument("--anchor_weight", type=float, default=1.0)
    parser.add_argument("--shift_weight", type=float, default=1.5)
    parser.add_argument("--classification_scope", type=str, default="batch", choices=["batch", "full"])
    parser.add_argument("--pairwise_scope", type=str, default="full", choices=["batch", "full"])
    parser.add_argument("--teacher_distill_scope", type=str, default="full", choices=["batch", "full"])
    parser.add_argument(
        "--selection_metric",
        type=str,
        default="val_verify_auc",
        choices=[
            "val_verify_auc",
            "val_gallery_acc",
            "val_batch_gallery_acc",
            "val_verify_acc",
            "val_margin",
            "val_pairwise_win_rate",
            "val_pairwise_auc",
        ],
    )
    return parser.parse_args()


if __name__ == "__main__":
    train_stage3_prototype(parse_args())
