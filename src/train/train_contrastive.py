import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Sampler, Subset
from torchvision import transforms as T
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.losses import CrossModalGroupInfoNCELoss, CrossModalHybridInfoNCELoss, CrossModalInfoNCELoss
from src.datasets.contrastive_dataset import CrossModalContrastiveDataset
from src.models.markov_encoder import MarkovRandomWalkEncoder
from src.models.visual_encoder import VisualEncoder
from src.tools.data_roots import resolve_experiment_root, resolve_meta_path, resolve_dataset_root


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def compute_cross_modal_top1_accuracy(z_a: torch.Tensor, z_b: torch.Tensor) -> float:
    with torch.no_grad():
        logits = torch.matmul(z_a, z_b.T)
        targets = torch.arange(logits.shape[0], device=logits.device)
        acc_ab = (logits.argmax(dim=1) == targets).float().mean()
        acc_ba = (logits.T.argmax(dim=1) == targets).float().mean()
        return float((0.5 * (acc_ab + acc_ba)).item())


def encode_group_ids(raw_anchors, device: torch.device) -> torch.Tensor:
    raw_to_id = {}
    encoded = []
    for raw_anchor in raw_anchors:
        if raw_anchor not in raw_to_id:
            raw_to_id[raw_anchor] = len(raw_to_id)
        encoded.append(raw_to_id[raw_anchor])
    return torch.tensor(encoded, dtype=torch.long, device=device)


def compute_cross_modal_group_top1_accuracy(z_a: torch.Tensor, z_b: torch.Tensor, raw_anchors) -> float:
    with torch.no_grad():
        logits = torch.matmul(z_a, z_b.T)
        preds_ab = logits.argmax(dim=1).tolist()
        preds_ba = logits.T.argmax(dim=1).tolist()

        acc_ab = np.mean([raw_anchors[i] == raw_anchors[j] for i, j in enumerate(preds_ab)])
        acc_ba = np.mean([raw_anchors[i] == raw_anchors[j] for i, j in enumerate(preds_ba)])
        return float(0.5 * (acc_ab + acc_ba))


def build_group_split(dataset: CrossModalContrastiveDataset, val_ratio: float, seed: int, max_raws: int = None):
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
        self.group_size = len(self.grouped_indices[0]) if self.grouped_indices else 1

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


def build_transforms(preset: str = "strong"):
    if preset == "strong":
        train_transform = T.Compose(
            [
                T.RandomResizedCrop(224, scale=(0.8, 1.0)),
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    elif preset == "mild":
        train_transform = T.Compose(
            [
                T.Resize(256),
                T.RandomCrop(224),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    elif preset == "none":
        train_transform = T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    else:
        raise ValueError(f"Unsupported augmentation preset: {preset}")
    eval_transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, eval_transform


def train_stage3(args):
    set_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    pin_memory = device.type == "cuda"

    train_transform, eval_transform = build_transforms(args.augmentation_preset)

    full_dataset = CrossModalContrastiveDataset(
        teacher_cache_dir=args.teacher_cache_dir,
        meta_path=args.meta_path,
        rgb_dir=args.rgb_dir,
        teacher_key=args.teacher_key,
        transform=eval_transform,
    )

    train_indices, val_indices, train_raws, val_raws = build_group_split(
        dataset=full_dataset,
        val_ratio=args.val_ratio,
        seed=args.seed,
        max_raws=args.max_raws,
    )
    if not train_indices:
        raise RuntimeError("No training samples available after the raw-anchor split.")

    train_dataset = CrossModalContrastiveDataset(
        teacher_cache_dir=args.teacher_cache_dir,
        meta_path=args.meta_path,
        rgb_dir=args.rgb_dir,
        teacher_key=args.teacher_key,
        transform=train_transform,
    )
    val_dataset = CrossModalContrastiveDataset(
        teacher_cache_dir=args.teacher_cache_dir,
        meta_path=args.meta_path,
        rgb_dir=args.rgb_dir,
        teacher_key=args.teacher_key,
        transform=eval_transform,
    )

    train_group_size = 4
    val_group_size = 4
    train_groups_per_batch = max(1, args.batch_size // train_group_size)
    val_groups_per_batch = max(1, args.val_batch_size // val_group_size)

    train_batch_sampler = RawGroupBatchSampler(
        group_ids=train_dataset.group_ids,
        indices=train_indices,
        groups_per_batch=train_groups_per_batch,
        seed=args.seed,
        shuffle=True,
        drop_last=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = None
    if val_indices:
        val_batch_sampler = RawGroupBatchSampler(
            group_ids=val_dataset.group_ids,
            indices=val_indices,
            groups_per_batch=val_groups_per_batch,
            seed=args.seed,
            shuffle=False,
            drop_last=False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_sampler=val_batch_sampler,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            persistent_workers=args.num_workers > 0,
        )

    teacher_dim = full_dataset.teacher_dim
    num_stages = full_dataset.num_stages

    teacher_left = MarkovRandomWalkEncoder(
        d_p=teacher_dim,
        d_A=args.embedding_dim,
        num_stages=num_stages,
        walk_steps=args.walk_steps,
        tau=args.markov_tau,
    ).to(device)
    student_right = VisualEncoder(
        d_out=args.embedding_dim,
        backbone_type=args.backbone_type,
        pretrained=not args.no_pretrained,
    ).to(device)

    if args.init_checkpoint:
        init_path = Path(args.init_checkpoint)
        if not init_path.exists():
            raise FileNotFoundError(f"Initialization checkpoint not found: {init_path}")
        ckpt = torch.load(init_path, map_location=device)
        teacher_left.load_state_dict(ckpt["teacher_state_dict"], strict=True)
        student_right.load_state_dict(ckpt["student_state_dict"], strict=True)
        print(f"Initialized from checkpoint: {init_path}", flush=True)

    optimizer = optim.AdamW(
        list(teacher_left.parameters()) + list(student_right.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    if args.loss_mode == "group":
        criterion = CrossModalGroupInfoNCELoss(temperature=args.temperature).to(device)
    elif args.loss_mode == "hybrid":
        criterion = CrossModalHybridInfoNCELoss(
            temperature=args.temperature,
            pair_weight=args.pair_loss_weight,
            group_weight=args.group_loss_weight,
        ).to(device)
    else:
        criterion = CrossModalInfoNCELoss(temperature=args.temperature).to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "teacher_cache_dir": args.teacher_cache_dir,
        "meta_path": args.meta_path,
        "rgb_dir": args.rgb_dir,
        "teacher_key": args.teacher_key,
        "teacher_dim": teacher_dim,
        "num_stages": num_stages,
        "embedding_dim": args.embedding_dim,
        "batch_size": args.batch_size,
        "val_batch_size": args.val_batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "temperature": args.temperature,
        "walk_steps": args.walk_steps,
        "markov_tau": args.markov_tau,
        "loss_mode": args.loss_mode,
        "pair_loss_weight": args.pair_loss_weight,
        "group_loss_weight": args.group_loss_weight,
        "augmentation_preset": args.augmentation_preset,
        "seed": args.seed,
        "train_raw_count": len(train_raws),
        "val_raw_count": len(val_raws),
        "train_sample_count": len(train_indices),
        "val_sample_count": len(val_indices),
    }

    print("=" * 72, flush=True)
    print("Stage-3 Teacher-Student Training", flush=True)
    print(f"Device           : {device}", flush=True)
    print(f"Teacher cache    : {args.teacher_cache_dir}", flush=True)
    print(f"Teacher sequence : ({num_stages}, {teacher_dim})", flush=True)
    print(f"Train raws       : {len(train_raws)}", flush=True)
    print(f"Val raws         : {len(val_raws)}", flush=True)
    print(f"Train samples    : {len(train_indices)}", flush=True)
    print(f"Val samples      : {len(val_indices)}", flush=True)
    print(f"Train groups/b   : {train_groups_per_batch}", flush=True)
    print(f"Val groups/b     : {val_groups_per_batch}", flush=True)
    print(f"Loss mode        : {args.loss_mode}", flush=True)
    print(f"Augmentation     : {args.augmentation_preset}", flush=True)
    print("=" * 72, flush=True)

    best_metric = float("-inf")
    history = []

    for epoch in range(args.epochs):
        if hasattr(train_loader.batch_sampler, "set_epoch"):
            train_loader.batch_sampler.set_epoch(epoch)
        teacher_left.train()
        student_right.train()
        train_loss = 0.0
        train_exact_acc = 0.0
        train_group_acc = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Train]")
        for batch in pbar:
            teacher_seq = batch["teacher_seq"].to(device, non_blocking=True)
            rgb_image = batch["rgb_image"].to(device, non_blocking=True)
            group_ids = encode_group_ids(batch["raw_anchor"], device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                teacher_repr = teacher_left(teacher_seq)
                student_repr = student_right(rgb_image)
                if args.loss_mode == "group":
                    loss = criterion(teacher_repr, student_repr, group_ids)
                elif args.loss_mode == "hybrid":
                    loss = criterion(teacher_repr, student_repr, group_ids)
                else:
                    loss = criterion(teacher_repr, student_repr)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            exact_acc = compute_cross_modal_top1_accuracy(teacher_repr.detach(), student_repr.detach())
            group_acc = compute_cross_modal_group_top1_accuracy(
                teacher_repr.detach(), student_repr.detach(), batch["raw_anchor"]
            )
            train_loss += float(loss.item())
            train_exact_acc += exact_acc
            train_group_acc += group_acc
            pbar.set_postfix(
                {
                    "Loss": f"{loss.item():.4f}",
                    "GAcc": f"{group_acc * 100:.1f}%",
                    "EAcc": f"{exact_acc * 100:.1f}%",
                }
            )

        scheduler.step()

        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": train_loss / max(len(train_loader), 1),
            "train_exact_acc": train_exact_acc / max(len(train_loader), 1),
            "train_group_acc": train_group_acc / max(len(train_loader), 1),
        }

        if val_loader is not None:
            teacher_left.eval()
            student_right.eval()
            val_loss = 0.0
            val_exact_acc = 0.0
            val_group_acc = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    teacher_seq = batch["teacher_seq"].to(device, non_blocking=True)
                    rgb_image = batch["rgb_image"].to(device, non_blocking=True)
                    group_ids = encode_group_ids(batch["raw_anchor"], device)
                    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                        teacher_repr = teacher_left(teacher_seq)
                        student_repr = student_right(rgb_image)
                        if args.loss_mode == "group":
                            loss = criterion(teacher_repr, student_repr, group_ids)
                        elif args.loss_mode == "hybrid":
                            loss = criterion(teacher_repr, student_repr, group_ids)
                        else:
                            loss = criterion(teacher_repr, student_repr)
                    exact_acc = compute_cross_modal_top1_accuracy(teacher_repr, student_repr)
                    group_acc = compute_cross_modal_group_top1_accuracy(
                        teacher_repr, student_repr, batch["raw_anchor"]
                    )
                    val_loss += float(loss.item())
                    val_exact_acc += exact_acc
                    val_group_acc += group_acc

            epoch_record["val_loss"] = val_loss / max(len(val_loader), 1)
            epoch_record["val_exact_acc"] = val_exact_acc / max(len(val_loader), 1)
            epoch_record["val_group_acc"] = val_group_acc / max(len(val_loader), 1)
            monitor_metric = epoch_record["val_group_acc"] if args.loss_mode == "group" else epoch_record["val_exact_acc"]
        else:
            monitor_metric = epoch_record["train_group_acc"] if args.loss_mode == "group" else epoch_record["train_exact_acc"]

        history.append(epoch_record)
        print(epoch_record, flush=True)

        if monitor_metric > best_metric:
            best_metric = monitor_metric
            torch.save(
                {
                    "epoch": epoch + 1,
                    "teacher_state_dict": teacher_left.state_dict(),
                    "student_state_dict": student_right.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "config": run_config,
                    "history": history,
                },
                save_dir / "stage3_best.pt",
            )

        with open(save_dir / "stage3_history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

    torch.save(
        {
            "teacher_state_dict": teacher_left.state_dict(),
            "student_state_dict": student_right.state_dict(),
            "config": run_config,
            "history": history,
        },
        save_dir / "stage3_last.pt",
    )
    with open(save_dir / "stage3_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Train the stage-3 teacher-student contrastive model")
    parser.add_argument(
        "--teacher_cache_dir",
        type=str,
        default=str(resolve_experiment_root() / "stage3_teacher_cache"),
    )
    parser.add_argument("--meta_path", type=str, default=str(resolve_meta_path()))
    parser.add_argument("--rgb_dir", type=str, default=str(resolve_dataset_root() / "rgb_web_jpg"))
    parser.add_argument("--teacher_key", type=str, default="teacher_seq")
    parser.add_argument("--save_dir", type=str, default=str(resolve_experiment_root() / "stage3_checkpoints"))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--val_batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--walk_steps", type=int, default=3)
    parser.add_argument("--markov_tau", type=float, default=0.1)
    parser.add_argument("--backbone_type", type=str, default="resnet50", choices=["resnet18", "resnet50"])
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--loss_mode", type=str, default="group", choices=["group", "pair", "hybrid"])
    parser.add_argument("--pair_loss_weight", type=float, default=0.5)
    parser.add_argument("--group_loss_weight", type=float, default=1.0)
    parser.add_argument("--augmentation_preset", type=str, default="strong", choices=["strong", "mild", "none"])
    parser.add_argument("--init_checkpoint", type=str, default=None)
    parser.add_argument("--max_raws", type=int, default=None, help="Optional cap for raw-anchor groups, useful for smoke tests")
    return parser.parse_args()


if __name__ == "__main__":
    train_stage3(parse_args())
