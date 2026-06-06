#!/usr/bin/env python3
import os
import sys
import copy
import random
import argparse
import logging
import warnings
from pathlib import Path
from contextlib import nullcontext

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.transformer")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ================= 自定义模块 =================
from src.datasets.path_dataset import PathDataset, path_collate_fn
from src.models.backbone import FeatureBackbone
from src.models.encoders import PathEncoder
from src.core.losses import Stage1JointLoss

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("Stage1_Trainer_Joint")


# =========================================================
# 排序头
# =========================================================
class RankingHead(nn.Module):
    """
    从路径 embedding z 预测一个标量分数，用于 pairwise ranking。
    """
    def __init__(self, d_z: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_z, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)  # (B,)


# =========================================================
# Trainer
# =========================================================
class Stage1Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_type = "cuda" if self.device.type == "cuda" else "cpu"

        logger.info(f"初始化 Stage 1 联合训练器，设备: {self.device}")

        # -------- Backbone / Encoder --------
        self.backbone = FeatureBackbone(
            d_f=args.d_f,
            C_f=args.C_f,
            freeze_bn=True,
        ).to(self.device)

        self.path_encoder = PathEncoder(
            d_f=args.d_f,
            d_z=args.d_z,
            num_stages=5,
        ).to(self.device)

        # -------- Ranking heads --------
        self.rank_names = [
            "gamma",
            "saturation",
            "contrast",
            "denoise_strength",
            "jpeg_quality",
        ]
        self.rank_heads = nn.ModuleDict({
            name: RankingHead(d_z=args.d_z, hidden=args.rank_hidden)
            for name in self.rank_names
        }).to(self.device)

        # -------- Optimizer / Scheduler --------
        params = (
            list(self.backbone.parameters())
            + list(self.path_encoder.parameters())
            + list(self.rank_heads.parameters())
        )
        self.optimizer = optim.AdamW(
            params,
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=args.epochs,
            eta_min=1e-6
        )

        # -------- Loss --------
        self.criterion = Stage1JointLoss(
            temperature=args.temp,
            class_balance=True,
            rank_tau=args.rank_tau,
            rank_weight=args.rank_weight,
        )

        self.scaler = torch.amp.GradScaler(self.device_type, enabled=(self.device.type == "cuda"))
        self.stage_order = ["stage_raw", "stage_demosaic", "stage_denoise", "stage_color", "rgb"]

        self.best_val_loss = float("inf")
        self.save_dir = PROJECT_ROOT / "checkpoints" / "stage1_joint"
        self.save_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------
    # 内部工具
    # -----------------------------------------------------
    def _autocast_ctx(self):
        if self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    def _encode_view(self, view_dict):
        """
        view_dict["stage_name"] -> (B, C, H, W)
        """
        f_seq = []
        for stage_name in self.stage_order:
            img_tensor = view_dict[stage_name].to(self.device, non_blocking=True)
            f_k = self.backbone.extract_global(img_tensor)   # (B, d_f)
            f_seq.append(f_k)

        f_seq_tensor = torch.stack(f_seq, dim=1)            # (B, K, d_f)
        z = self.path_encoder(f_seq_tensor)                 # (B, d_z)
        return z

    def _build_rank_pairs(self, z_a, z_b, rank_targets):
        rank_pairs = {}
        for name in self.rank_names:
            if name not in rank_targets:
                continue
            score_a = self.rank_heads[name](z_a)
            score_b = self.rank_heads[name](z_b)
            target = rank_targets[name].to(self.device, non_blocking=True)

            rank_pairs[name] = {
                "score_a": score_a,
                "score_b": score_b,
                "target": target,
            }
        return rank_pairs

    @torch.no_grad()
    def _compute_type_gap(self, features: torch.Tensor, labels: torch.Tensor):
        """
        计算同路径族 / 异路径族相似度 gap
        features: (N, d)
        labels:   (N,)
        """
        features = torch.nn.functional.normalize(features, dim=1)
        sim = torch.matmul(features, features.T)  # (N, N)

        labels = labels.view(-1, 1)
        same = (labels == labels.T)
        eye = torch.eye(sim.size(0), device=sim.device, dtype=torch.bool)

        pos_mask = same & (~eye)
        neg_mask = (~same) & (~eye)

        pos_sim = sim.masked_select(pos_mask).mean().item() if pos_mask.any() else 0.0
        neg_sim = sim.masked_select(neg_mask).mean().item() if neg_mask.any() else 0.0
        gap = pos_sim - neg_sim
        return pos_sim, neg_sim, gap

    # -----------------------------------------------------
    # 训练 / 验证
    # -----------------------------------------------------
    def train_epoch(self, dataloader, epoch):
        self.backbone.train()
        self.path_encoder.train()
        self.rank_heads.train()

        total_loss = 0.0
        total_supcon = 0.0
        total_rank = 0.0
        total_gap = 0.0

        pbar = tqdm(
            dataloader,
            desc=f"Epoch [{epoch + 1}/{self.args.epochs}] Train",
            leave=False
        )

        for batch in pbar:
            self.optimizer.zero_grad(set_to_none=True)

            with self._autocast_ctx():
                z_a = self._encode_view(batch["view_a"]["stages"])
                z_b = self._encode_view(batch["view_b"]["stages"])

                labels_a = batch["labels_a"].to(self.device, non_blocking=True)
                labels_b = batch["labels_b"].to(self.device, non_blocking=True)

                features = torch.cat([z_a, z_b], dim=0)      # (2B, d)
                labels = torch.cat([labels_a, labels_b], dim=0)

                rank_pairs = self._build_rank_pairs(z_a, z_b, batch["rank_targets"])
                loss, log_dict = self.criterion(features, labels, rank_pairs=rank_pairs)

            self.scaler.scale(loss).backward()

            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.backbone.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.path_encoder.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.rank_heads.parameters(), max_norm=1.0)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            with torch.no_grad():
                pos_sim, neg_sim, gap = self._compute_type_gap(features.detach(), labels.detach())

            total_loss += float(log_dict["total"])
            total_supcon += float(log_dict["supcon"])
            total_rank += float(log_dict.get("rank_mean", 0.0))
            total_gap += gap

            pbar.set_postfix({
                "Loss": f"{log_dict['total']:.4f}",
                "SupCon": f"{log_dict['supcon']:.4f}",
                "Rank": f"{log_dict.get('rank_mean', 0.0):.4f}",
                "Gap": f"{gap:.4f}",
            })

        n = len(dataloader)
        return {
            "loss": total_loss / n,
            "supcon": total_supcon / n,
            "rank": total_rank / n,
            "gap": total_gap / n,
        }

    @torch.no_grad()
    def validate(self, dataloader, epoch):
        self.backbone.eval()
        self.path_encoder.eval()
        self.rank_heads.eval()

        total_loss = 0.0
        total_supcon = 0.0
        total_rank = 0.0
        total_gap = 0.0
        total_pos = 0.0
        total_neg = 0.0

        pbar = tqdm(
            dataloader,
            desc=f"Epoch [{epoch + 1}/{self.args.epochs}] Val  ",
            leave=False
        )

        for batch in pbar:
            with self._autocast_ctx():
                z_a = self._encode_view(batch["view_a"]["stages"])
                z_b = self._encode_view(batch["view_b"]["stages"])

                labels_a = batch["labels_a"].to(self.device, non_blocking=True)
                labels_b = batch["labels_b"].to(self.device, non_blocking=True)

                features = torch.cat([z_a, z_b], dim=0)
                labels = torch.cat([labels_a, labels_b], dim=0)

                rank_pairs = self._build_rank_pairs(z_a, z_b, batch["rank_targets"])
                loss, log_dict = self.criterion(features, labels, rank_pairs=rank_pairs)

            pos_sim, neg_sim, gap = self._compute_type_gap(features, labels)

            total_loss += float(log_dict["total"])
            total_supcon += float(log_dict["supcon"])
            total_rank += float(log_dict.get("rank_mean", 0.0))
            total_gap += gap
            total_pos += pos_sim
            total_neg += neg_sim

            pbar.set_postfix({
                "Loss": f"{log_dict['total']:.4f}",
                "SupCon": f"{log_dict['supcon']:.4f}",
                "Rank": f"{log_dict.get('rank_mean', 0.0):.4f}",
                "Gap": f"{gap:.4f}",
            })

        n = len(dataloader)
        return {
            "loss": total_loss / n,
            "supcon": total_supcon / n,
            "rank": total_rank / n,
            "gap": total_gap / n,
            "pos_sim": total_pos / n,
            "neg_sim": total_neg / n,
        }

    # -----------------------------------------------------
    # Checkpoint
    # -----------------------------------------------------
    def save_checkpoint(self, filename, epoch, is_best=False):
        save_path = self.save_dir / filename
        state = {
            "epoch": epoch,
            "args": vars(self.args),
            "backbone": self.backbone.state_dict(),
            "path_encoder": self.path_encoder.state_dict(),
            "rank_heads": self.rank_heads.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
        }
        torch.save(state, save_path)

        if is_best:
            best_path = self.save_dir / "stage1_joint_best.pt"
            torch.save(state, best_path)
            logger.info(f"新的最佳模型已保存至 {best_path}")

    def load_checkpoint(self, filename="stage1_joint_latest.pt"):
        load_path = self.save_dir / filename
        if not load_path.exists():
            logger.info("未找到历史存档，将从第 1 轮开始全新训练。")
            return 0

        logger.info(f"发现存档文件 {load_path}，正在恢复训练...")
        checkpoint = torch.load(load_path, map_location=self.device)

        self.backbone.load_state_dict(checkpoint["backbone"])
        self.path_encoder.load_state_dict(checkpoint["path_encoder"])
        self.rank_heads.load_state_dict(checkpoint["rank_heads"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))

        start_epoch = checkpoint.get("epoch", -1) + 1
        logger.info(f"恢复完成，将从第 {start_epoch + 1} 轮继续。")
        return start_epoch


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="Stage 1: Path-SupCon + Ranking Joint Training")

    parser.add_argument("--data_dir", type=str, default="data/my_forensics_dataset",
                        help="数据集根目录，应包含 dataset_meta.json / stage_cache.h5")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--d_f", type=int, default=512)
    parser.add_argument("--C_f", type=int, default=256)
    parser.add_argument("--d_z", type=int, default=256)
    parser.add_argument("--temp", type=float, default=0.07)
    parser.add_argument("--rank_tau", type=float, default=1.0)
    parser.add_argument("--rank_weight", type=float, default=0.2)
    parser.add_argument("--rank_hidden", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    args.data_dir = str(PROJECT_ROOT / args.data_dir)

    # -------- Seed --------
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # -------- 加载索引 --------
    from src.datasets.custom_index import CustomForensicsIndex

    logger.info("正在挂载专属物理路径数据集...")
    full_index = CustomForensicsIndex(args.data_dir)

    # 只按 RAW anchor 划分，避免同场景泄漏
    unique_raws = list(set([sample["meta"]["raw_anchor"] for sample in full_index.samples]))
    random.shuffle(unique_raws)

    split_point = int(len(unique_raws) * 0.9)
    train_raws = set(unique_raws[:split_point])
    val_raws = set(unique_raws[split_point:])

    train_index = copy.copy(full_index)
    train_index.samples = [s for s in full_index.samples if s["meta"]["raw_anchor"] in train_raws]
    for s in train_index.samples:
        s["basename"] = s["version_id"]

    val_index = copy.copy(full_index)
    val_index.samples = [s for s in full_index.samples if s["meta"]["raw_anchor"] in val_raws]
    for s in val_index.samples:
        s["basename"] = s["version_id"]

    logger.info(
        f"划分完成 | 训练场景: {len(train_raws)} | 验证场景: {len(val_raws)} | "
        f"训练版本数: {len(train_index.samples)} | 验证版本数: {len(val_index.samples)}"
    )

    # -------- Dataset / Loader --------
    cache_dir = str(Path(args.data_dir) / "stage_cache")

    train_dataset = PathDataset(
        fivek_index=train_index,
        mode="dual_view",
        cache_dir=cache_dir,
        is_val=False,
        deterministic_val=False,
    )

    val_dataset = PathDataset(
        fivek_index=val_index,
        mode="dual_view",
        cache_dir=cache_dir,
        is_val=True,
        deterministic_val=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=path_collate_fn,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(args.num_workers > 0),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=path_collate_fn,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(args.num_workers > 0),
    )

    # -------- Trainer --------
    trainer = Stage1Trainer(args)

    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load_checkpoint("stage1_joint_latest.pt")

    # -------- Training Loop --------
    for epoch in range(start_epoch, args.epochs):
        train_stats = trainer.train_epoch(train_loader, epoch)
        val_stats = trainer.validate(val_loader, epoch)

        trainer.scheduler.step()
        current_lr = trainer.optimizer.param_groups[0]["lr"]

        logger.info(
            f"Epoch [{epoch + 1}/{args.epochs}] | LR: {current_lr:.2e} | "
            f"Train Loss: {train_stats['loss']:.4f} "
            f"(SupCon {train_stats['supcon']:.4f}, Rank {train_stats['rank']:.4f}, Gap {train_stats['gap']:.4f}) | "
            f"Val Loss: {val_stats['loss']:.4f} "
            f"(SupCon {val_stats['supcon']:.4f}, Rank {val_stats['rank']:.4f}, "
            f"Same-Type {val_stats['pos_sim']:.4f}, Diff-Type {val_stats['neg_sim']:.4f}, Gap {val_stats['gap']:.4f})"
        )

        is_best = val_stats["loss"] < trainer.best_val_loss
        if is_best:
            trainer.best_val_loss = val_stats["loss"]

        trainer.save_checkpoint("stage1_joint_latest.pt", epoch, is_best=is_best)

        if (epoch + 1) % 10 == 0:
            trainer.save_checkpoint(f"stage1_joint_epoch_{epoch + 1}.pt", epoch, is_best=False)


if __name__ == "__main__":
    main()
