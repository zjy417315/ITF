import torch
import torch.nn as nn
import torch.nn.functional as F


class PathSupConLoss(nn.Module):
    """
    监督对比损失（Supervised Contrastive Loss）
    输入:
        features: (N, d)
        labels:   (N,)
    作用:
        拉近同路径族标签样本，推远异路径族样本
    """
    def __init__(self, temperature=0.07, eps=1e-8, class_balance=True):
        super().__init__()
        self.temperature = temperature
        self.eps = eps
        self.class_balance = class_balance

    def forward(self, features, labels):
        device = features.device
        N = features.size(0)

        features = F.normalize(features, dim=1)
        sim_matrix = torch.matmul(features, features.T) / self.temperature

        # 数值稳定
        sim_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        logits = sim_matrix - sim_max.detach()

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        logits_mask = torch.ones_like(mask)
        logits_mask.fill_diagonal_(0)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + self.eps)

        mask_sum = mask.sum(1)
        valid = mask_sum > 0

        mean_log_prob_pos = torch.zeros(N, device=device, dtype=features.dtype)
        mean_log_prob_pos[valid] = (
            (mask[valid] * log_prob[valid]).sum(1) / mask_sum[valid]
        )

        if valid.any():
            if self.class_balance:
                flat_labels = labels.view(-1)
                unique_labels, counts = torch.unique(flat_labels, return_counts=True)
                label_to_count = {
                    int(k.item()): int(v.item())
                    for k, v in zip(unique_labels, counts)
                }

                weights = torch.ones(N, device=device, dtype=features.dtype)
                for i in range(N):
                    weights[i] = 1.0 / max(1, label_to_count[int(flat_labels[i].item())])

                weights = weights[valid]
                weights = weights / (weights.sum() + self.eps)

                loss = -(weights * mean_log_prob_pos[valid]).sum()
            else:
                loss = -mean_log_prob_pos[valid].mean()
        else:
            loss = torch.zeros((), device=device, dtype=features.dtype)

        return loss


class PairwiseRankingLoss(nn.Module):
    """
    成对排序损失：
    输入:
        score_a, score_b: (B,)
        target: (B,)
            1.0 -> a > b
            0.0 -> a < b
            0.5 -> a == b（可选）
    """
    def __init__(self, tau=1.0):
        super().__init__()
        self.tau = tau

    def forward(self, score_a, score_b, target):
        if score_a.dim() > 1:
            score_a = score_a.view(-1)
        if score_b.dim() > 1:
            score_b = score_b.view(-1)
        if target.dim() > 1:
            target = target.view(-1)

        logits = (score_a - score_b) / self.tau
        return F.binary_cross_entropy_with_logits(logits, target.float())


class Stage1JointLoss(nn.Module):
    """
    Stage 1 联合损失：
        total = supcon + rank_weight * mean(rank_losses)
    """
    def __init__(
        self,
        temperature=0.07,
        class_balance=True,
        rank_tau=1.0,
        rank_weight=0.2,
    ):
        super().__init__()
        self.supcon = PathSupConLoss(
            temperature=temperature,
            class_balance=class_balance,
        )
        self.rank_loss = PairwiseRankingLoss(tau=rank_tau)
        self.rank_weight = rank_weight

    def forward(self, features, labels, rank_pairs=None):
        supcon_loss = self.supcon(features, labels)
        total_loss = supcon_loss

        log_dict = {
            "supcon": float(supcon_loss.detach().item())
        }

        if rank_pairs is not None and len(rank_pairs) > 0:
            rank_sum = 0.0
            rank_cnt = 0

            for name, pack in rank_pairs.items():
                rank_l = self.rank_loss(
                    pack["score_a"],
                    pack["score_b"],
                    pack["target"],
                )
                total_loss = total_loss + self.rank_weight * rank_l
                rank_sum += rank_l.detach().item()
                rank_cnt += 1
                log_dict[f"rank_{name}"] = float(rank_l.detach().item())

            if rank_cnt > 0:
                log_dict["rank_mean"] = float(rank_sum / rank_cnt)

        log_dict["total"] = float(total_loss.detach().item())
        return total_loss, log_dict


class CrossModalInfoNCELoss(nn.Module):
    """
    预留给后续 Stage 2/3 的跨模态对齐损失
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_a, z_b):
        z_a = F.normalize(z_a, dim=1)
        z_b = F.normalize(z_b, dim=1)

        logits = torch.matmul(z_a, z_b.T) / self.temperature
        targets = torch.arange(z_a.size(0), device=z_a.device)

        loss_ab = F.cross_entropy(logits, targets)
        loss_ba = F.cross_entropy(logits.T, targets)
        return 0.5 * (loss_ab + loss_ba)


class CrossModalGroupInfoNCELoss(nn.Module):
    """
    Cross-modal InfoNCE with group-aware positives.

    Samples that share the same group id are treated as additional positives
    instead of being forced apart as negatives.
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    @staticmethod
    def _directional_loss(logits: torch.Tensor, positive_mask: torch.Tensor) -> torch.Tensor:
        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        pos_count = positive_mask.sum(dim=1).clamp_min(1.0)
        loss = -(positive_mask * log_prob).sum(dim=1) / pos_count
        return loss.mean()

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor, group_ids: torch.Tensor) -> torch.Tensor:
        z_a = F.normalize(z_a, dim=1)
        z_b = F.normalize(z_b, dim=1)

        logits = torch.matmul(z_a, z_b.T) / self.temperature
        group_ids = group_ids.view(-1, 1)
        positive_mask = torch.eq(group_ids, group_ids.T).to(logits.dtype)

        loss_ab = self._directional_loss(logits, positive_mask)
        loss_ba = self._directional_loss(logits.T, positive_mask.T)
        return 0.5 * (loss_ab + loss_ba)


class CrossModalHybridInfoNCELoss(nn.Module):
    """
    Weighted combination of exact-pair InfoNCE and group-aware InfoNCE.
    """

    def __init__(self, temperature=0.07, pair_weight=0.5, group_weight=1.0):
        super().__init__()
        self.pair_loss = CrossModalInfoNCELoss(temperature=temperature)
        self.group_loss = CrossModalGroupInfoNCELoss(temperature=temperature)
        self.pair_weight = pair_weight
        self.group_weight = group_weight

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor, group_ids: torch.Tensor) -> torch.Tensor:
        pair_term = self.pair_loss(z_a, z_b)
        group_term = self.group_loss(z_a, z_b, group_ids)
        return self.pair_weight * pair_term + self.group_weight * group_term
