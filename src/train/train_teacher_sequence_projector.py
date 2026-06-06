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
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.teacher_sequence_dataset import TeacherSequenceDataset
from src.models.encoders import PathEncoder
from src.train.train_stage3_prototype import (
    accuracy_for_versions,
    binary_auc_from_scores,
    build_batch_prototype_bank,
    compute_pairwise_verification_loss,
    compute_similarity_stats,
    compute_source_logits,
    compute_version_weights,
    weighted_mean,
)
from src.tools.data_roots import resolve_experiment_root, resolve_meta_path


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def build_group_split(dataset: TeacherSequenceDataset, val_ratio: float, seed: int, max_raws: int = None):
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


def encode_bank(encoder: PathEncoder, dataset: TeacherSequenceDataset, raw_anchors, device: torch.device) -> torch.Tensor:
    if not raw_anchors:
        return torch.empty(0, encoder.proj[-1].out_features, device=device)
    seq = torch.stack([dataset.get_prototype_sequence(raw_anchor) for raw_anchor in raw_anchors], dim=0).to(device)
    return encoder(seq)


def train_teacher_sequence_projector(args):
    set_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    pin_memory = device.type == "cuda"

    dataset = TeacherSequenceDataset(
        prototype_cache_dir=args.prototype_cache_dir,
        teacher_cache_dir=args.teacher_cache_dir,
        meta_path=args.meta_path,
        include_versions=args.include_versions,
    )
    train_indices, val_indices, train_raws, val_raws = build_group_split(
        dataset=dataset,
        val_ratio=args.val_ratio,
        seed=args.seed,
        max_raws=args.max_raws,
    )

    train_sampler = RawGroupBatchSampler(
        group_ids=dataset.group_ids,
        indices=train_indices,
        groups_per_batch=max(1, args.batch_size // 4),
        seed=args.seed,
        shuffle=True,
        drop_last=True,
    )
    val_sampler = RawGroupBatchSampler(
        group_ids=dataset.group_ids,
        indices=val_indices,
        groups_per_batch=max(1, args.val_batch_size // 4),
        seed=args.seed,
        shuffle=False,
        drop_last=False,
    )

    train_loader = DataLoader(dataset, batch_sampler=train_sampler, num_workers=0, pin_memory=pin_memory)
    val_loader = DataLoader(dataset, batch_sampler=val_sampler, num_workers=0, pin_memory=pin_memory)

    encoder = PathEncoder(d_f=dataset.sequence_dim, d_z=args.embed_dim, num_stages=dataset.stage_count).to(device)
    optimizer = optim.AdamW(encoder.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72, flush=True)
    print("Teacher Sequence Projector", flush=True)
    print(f"Device          : {device}", flush=True)
    print(f"Prototype cache : {args.prototype_cache_dir}", flush=True)
    print(f"Teacher cache   : {args.teacher_cache_dir}", flush=True)
    print(f"Sequence dim    : {dataset.sequence_dim}", flush=True)
    print(f"Embed dim       : {args.embed_dim}", flush=True)
    print(f"Train raws      : {len(train_raws)}", flush=True)
    print(f"Val raws        : {len(val_raws)}", flush=True)
    print(f"Train samples   : {len(train_indices)}", flush=True)
    print(f"Val samples     : {len(val_indices)}", flush=True)
    print("=" * 72, flush=True)

    history = []
    best_metric = float("-inf")

    for epoch in range(args.epochs):
        if hasattr(train_loader.batch_sampler, "set_epoch"):
            train_loader.batch_sampler.set_epoch(epoch)

        encoder.train()
        train_loss = 0.0
        train_full_acc = 0.0
        train_pairwise_acc = 0.0
        train_pairwise_loss_sum = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [TeacherTrain]")
        for batch in pbar:
            teacher_seq = batch["teacher_seq"].to(device, non_blocking=True)
            prototype_seq = batch["prototype_seq"].to(device, non_blocking=True)
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
                teacher_emb = encoder(teacher_seq)
                prototype_emb = encoder(prototype_seq)

                if args.classification_scope == "batch":
                    cls_bank, cls_bank_raws = build_batch_prototype_bank(raw_anchors, prototype_emb)
                else:
                    cls_bank = encode_bank(encoder, dataset, train_raws, device)
                    cls_bank_raws = train_raws

                logits, targets = compute_source_logits(teacher_emb, raw_anchors, cls_bank, cls_bank_raws)
                positive_scores, hardest_negative_scores, correct = compute_similarity_stats(logits, targets)

                align_terms = 1.0 - F.cosine_similarity(teacher_emb, prototype_emb, dim=1)
                align_loss = weighted_mean(align_terms, sample_weights)
                cls_terms = F.cross_entropy(logits / args.temperature, targets, reduction="none")
                cls_loss = weighted_mean(cls_terms, sample_weights)
                pairwise_loss = compute_pairwise_verification_loss(logits, targets, sample_weights, margin=args.pairwise_margin)
                margin_terms = F.relu(args.margin + hardest_negative_scores - positive_scores)
                margin_loss = weighted_mean(margin_terms, sample_weights)
                loss = (
                    args.align_weight * align_loss
                    + args.classification_weight * cls_loss
                    + args.pairwise_weight * pairwise_loss
                    + args.margin_weight * margin_loss
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += float(loss.item())
            train_full_acc += float(correct.float().mean().item())
            train_pairwise_acc += float((positive_scores > hardest_negative_scores).float().mean().item())
            train_pairwise_loss_sum += float(pairwise_loss.item())
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Acc": f"{float(correct.float().mean().item()) * 100:.1f}%"})

        scheduler.step()

        encoder.eval()
        with torch.no_grad():
            train_bank = encode_bank(encoder, dataset, train_raws, device)
            val_bank = encode_bank(encoder, dataset, val_raws, device)

        val_loss = 0.0
        val_full_acc = 0.0
        val_batch_acc = 0.0
        val_pairwise_acc = 0.0
        val_pairwise_loss_sum = 0.0
        val_pairwise_win_sum = 0.0
        val_pairwise_count = 0
        val_anchor_acc = 0.0
        val_shift_acc = 0.0
        val_pos_scores = []
        val_hard_neg_scores = []
        val_all_neg_scores = []

        encoder.eval()
        with torch.no_grad():
            for batch in val_loader:
                teacher_seq = batch["teacher_seq"].to(device, non_blocking=True)
                prototype_seq = batch["prototype_seq"].to(device, non_blocking=True)
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

                teacher_emb = encoder(teacher_seq)
                prototype_emb = encoder(prototype_seq)

                batch_logits, batch_targets = compute_source_logits(
                    teacher_emb,
                    raw_anchors,
                    *build_batch_prototype_bank(raw_anchors, prototype_emb),
                )
                full_logits, full_targets = compute_source_logits(teacher_emb, raw_anchors, val_bank, val_raws)
                positive_scores, hardest_negative_scores, correct = compute_similarity_stats(full_logits, full_targets)

                align_terms = 1.0 - F.cosine_similarity(teacher_emb, prototype_emb, dim=1)
                align_loss = weighted_mean(align_terms, sample_weights)
                if args.classification_scope == "batch":
                    cls_logits, cls_targets = batch_logits, batch_targets
                else:
                    cls_logits, cls_targets = full_logits, full_targets
                cls_terms = F.cross_entropy(cls_logits / args.temperature, cls_targets, reduction="none")
                cls_loss = weighted_mean(cls_terms, sample_weights)
                pairwise_loss = compute_pairwise_verification_loss(full_logits, full_targets, sample_weights, margin=args.pairwise_margin)
                margin_terms = F.relu(args.margin + hardest_negative_scores - positive_scores)
                margin_loss = weighted_mean(margin_terms, sample_weights)
                loss = (
                    args.align_weight * align_loss
                    + args.classification_weight * cls_loss
                    + args.pairwise_weight * pairwise_loss
                    + args.margin_weight * margin_loss
                )

                val_loss += float(loss.item())
                val_full_acc += float(correct.float().mean().item())
                val_batch_acc += compute_source_top1_accuracy(batch_logits, batch_targets)
                val_pairwise_acc += float((positive_scores > hardest_negative_scores).float().mean().item())
                val_pairwise_loss_sum += float(pairwise_loss.item())
                anchor_acc = accuracy_for_versions(correct, versions, args.anchor_versions)
                shift_acc = accuracy_for_versions(correct, versions, args.shift_versions)
                val_anchor_acc += 0.0 if np.isnan(anchor_acc) else anchor_acc
                val_shift_acc += 0.0 if np.isnan(shift_acc) else shift_acc
                val_pos_scores.append(positive_scores.cpu())
                val_hard_neg_scores.append(hardest_negative_scores.cpu())

                for row_idx, target_idx in enumerate(full_targets.tolist()):
                    mask = torch.ones(full_logits.shape[1], device=full_logits.device, dtype=torch.bool)
                    mask[target_idx] = False
                    row_neg = full_logits[row_idx][mask]
                    val_all_neg_scores.append(row_neg.cpu())
                    val_pairwise_win_sum += float((positive_scores[row_idx] > row_neg).float().mean().item())
                    val_pairwise_count += 1

        pos_scores = torch.cat(val_pos_scores, dim=0) if val_pos_scores else torch.empty(0)
        hard_neg_scores = torch.cat(val_hard_neg_scores, dim=0) if val_hard_neg_scores else torch.empty(0)
        all_neg_scores = torch.cat(val_all_neg_scores, dim=0) if val_all_neg_scores else torch.empty(0)

        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": train_loss / max(len(train_loader), 1),
            "train_full_acc": train_full_acc / max(len(train_loader), 1),
            "train_pairwise_acc": train_pairwise_acc / max(len(train_loader), 1),
            "train_pairwise_loss": train_pairwise_loss_sum / max(len(train_loader), 1),
            "val_loss": val_loss / max(len(val_loader), 1),
            "val_full_acc": val_full_acc / max(len(val_loader), 1),
            "val_batch_acc": val_batch_acc / max(len(val_loader), 1),
            "val_pairwise_acc": val_pairwise_acc / max(len(val_loader), 1),
            "val_pairwise_loss": val_pairwise_loss_sum / max(len(val_loader), 1),
            "val_anchor_acc": val_anchor_acc / max(len(val_loader), 1),
            "val_shift_acc": val_shift_acc / max(len(val_loader), 1),
            "val_hard_auc": binary_auc_from_scores(pos_scores, hard_neg_scores),
            "val_pairwise_auc": binary_auc_from_scores(pos_scores, all_neg_scores),
            "val_pairwise_win_rate": val_pairwise_win_sum / max(val_pairwise_count, 1),
        }
        history.append(epoch_record)
        print(epoch_record, flush=True)

        metric = epoch_record[args.selection_metric]
        if metric > best_metric:
            best_metric = metric
            torch.save(
                {
                    "epoch": epoch + 1,
                    "encoder_state_dict": encoder.state_dict(),
                    "config": vars(args),
                    "history": history,
                },
                save_dir / "teacher_projector_best.pt",
            )

        with open(save_dir / "teacher_projector_history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)


def compute_source_top1_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    with torch.no_grad():
        return float((logits.argmax(dim=1) == targets).float().mean().item())


def parse_args():
    parser = argparse.ArgumentParser(description="Train teacher-side source projector on stage-wise sequences")
    parser.add_argument("--prototype_cache_dir", type=str, default=str(resolve_experiment_root() / "stage3_prototype_cache_anchor12"))
    parser.add_argument("--teacher_cache_dir", type=str, default=str(resolve_experiment_root() / "stage3_teacher_cache_full"))
    parser.add_argument("--meta_path", type=str, default=str(resolve_meta_path()))
    parser.add_argument("--save_dir", type=str, default=str(resolve_experiment_root() / "teacher_projector_checkpoints"))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--val_batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--align_weight", type=float, default=1.0)
    parser.add_argument("--classification_weight", type=float, default=0.25)
    parser.add_argument("--pairwise_weight", type=float, default=1.0)
    parser.add_argument("--pairwise_margin", type=float, default=0.0)
    parser.add_argument("--margin_weight", type=float, default=0.25)
    parser.add_argument("--margin", type=float, default=0.10)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_raws", type=int, default=None)
    parser.add_argument("--include_versions", type=int, nargs="*", default=[1, 2, 3, 4])
    parser.add_argument("--anchor_versions", type=int, nargs="*", default=[1, 2])
    parser.add_argument("--shift_versions", type=int, nargs="*", default=[3, 4])
    parser.add_argument("--anchor_weight", type=float, default=1.0)
    parser.add_argument("--shift_weight", type=float, default=2.0)
    parser.add_argument("--classification_scope", type=str, default="batch", choices=["batch", "full"])
    parser.add_argument(
        "--selection_metric",
        type=str,
        default="val_pairwise_auc",
        choices=["val_hard_auc", "val_pairwise_auc", "val_pairwise_win_rate", "val_batch_acc", "val_full_acc"],
    )
    return parser.parse_args()


if __name__ == "__main__":
    train_teacher_sequence_projector(parse_args())
