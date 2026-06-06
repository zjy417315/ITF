import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeVerifier(nn.Module):
    """
    Learnable verification head for claimed-source scoring.
    """

    def __init__(self, d_model: int, hidden_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        self.d_model = int(d_model)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        pair_dim = self.d_model * 4 + 8
        mid_dim = max(self.hidden_dim // 2, 32)
        self.input_norm = nn.LayerNorm(self.d_model)
        self.pair_mlp = nn.Sequential(
            nn.Linear(pair_dim, self.hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, mid_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(self.dropout),
        )
        self.residual_head = nn.Linear(mid_dim, 1)
        self.gate_head = nn.Linear(mid_dim, 1)
        self.scalar_context = nn.Sequential(
            nn.Linear(8, mid_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(mid_dim, mid_dim),
            nn.SiLU(inplace=True),
        )
        self.expert_router = nn.Sequential(
            nn.Linear(8, mid_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(mid_dim, 3),
        )
        self.expert_heads = nn.ModuleList([nn.Linear(mid_dim, 1) for _ in range(3)])
        self.scalar_gate = nn.Linear(mid_dim, 1)
        self.expert_scale = nn.Parameter(torch.tensor(0.5))
        self.cosine_scale = nn.Parameter(torch.tensor(2.0))
        self.dot_scale = nn.Parameter(torch.tensor(0.5))
        self.l2_scale = nn.Parameter(torch.tensor(-0.5))
        self.stats_mlp = nn.Sequential(
            nn.Linear(6, max(self.hidden_dim // 2, 32)),
            nn.SiLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(max(self.hidden_dim // 2, 32), 1),
        )

    def _pair_features(self, student_repr: torch.Tensor, prototype_repr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        student_repr = self.input_norm(student_repr)
        prototype_repr = self.input_norm(prototype_repr)
        student_unit = F.normalize(student_repr, dim=-1)
        prototype_unit = F.normalize(prototype_repr, dim=-1)
        abs_diff = torch.abs(student_repr - prototype_repr)
        sq_diff = (student_repr - prototype_repr).pow(2)
        cosine = (student_unit * prototype_unit).sum(dim=-1, keepdim=True)
        mean_prod = (student_repr * prototype_repr).mean(dim=-1, keepdim=True)
        mean_abs = abs_diff.mean(dim=-1, keepdim=True)
        max_abs = abs_diff.max(dim=-1, keepdim=True).values
        mean_sq = sq_diff.mean(dim=-1, keepdim=True)
        max_sq = sq_diff.max(dim=-1, keepdim=True).values
        repr_norm_gap = (
            student_repr.norm(dim=-1, keepdim=True) - prototype_repr.norm(dim=-1, keepdim=True)
        ).abs()
        signed_mean = (student_repr - prototype_repr).mean(dim=-1, keepdim=True)
        scalar_features = torch.cat(
            [
                cosine,
                mean_prod,
                mean_abs,
                max_abs,
                mean_sq,
                max_sq,
                repr_norm_gap,
                signed_mean,
            ],
            dim=-1,
        )
        return torch.cat(
            [
                student_repr,
                prototype_repr,
                student_repr * prototype_repr,
                abs_diff,
                scalar_features,
            ],
            dim=-1,
        ), scalar_features

    def _score_from_features(self, pair_features: torch.Tensor, scalar_features: torch.Tensor) -> torch.Tensor:
        hidden = self.pair_mlp(pair_features)
        scalar_hidden = self.scalar_context(scalar_features)
        combined_hidden = hidden + 0.5 * scalar_hidden
        expert_weights = F.softmax(self.expert_router(scalar_features), dim=-1)
        expert_residuals = torch.cat(
            [head(combined_hidden) for head in self.expert_heads],
            dim=-1,
        )
        expert_residual = (expert_weights * expert_residuals).sum(dim=-1)
        residual = torch.tanh(
            self.residual_head(combined_hidden).squeeze(-1) + self.expert_scale * expert_residual
        )
        gate = torch.sigmoid(
            self.gate_head(combined_hidden).squeeze(-1) + self.scalar_gate(scalar_hidden).squeeze(-1)
        )
        cosine = scalar_features[..., 0]
        mean_prod = scalar_features[..., 1]
        mean_sq = scalar_features[..., 4]
        base = self.cosine_scale * cosine + self.dot_scale * mean_prod + self.l2_scale * mean_sq
        return base + gate * residual

    def score_pairs(self, student_repr: torch.Tensor, prototype_repr: torch.Tensor) -> torch.Tensor:
        pair_features, scalar_features = self._pair_features(student_repr, prototype_repr)
        return self._score_from_features(pair_features, scalar_features)

    def score_bank(self, student_repr: torch.Tensor, prototype_bank: torch.Tensor, chunk_size: int = 256) -> torch.Tensor:
        scores = []
        for chunk in prototype_bank.split(chunk_size, dim=0):
            batch_size = student_repr.shape[0]
            chunk_size_actual = chunk.shape[0]
            student_expand = student_repr.unsqueeze(1).expand(batch_size, chunk_size_actual, self.d_model)
            proto_expand = chunk.unsqueeze(0).expand(batch_size, chunk_size_actual, self.d_model)
            pair_features, scalar_features = self._pair_features(student_expand, proto_expand)
            chunk_scores = self._score_from_features(pair_features, scalar_features)
            scores.append(chunk_scores)
        return torch.cat(scores, dim=1)

    def score_stats_bank(
        self,
        student_repr: torch.Tensor,
        patch_repr: torch.Tensor,
        prototype_bank: torch.Tensor,
        topk: int = 4,
        temperature: float = 8.0,
    ) -> torch.Tensor:
        if patch_repr is None:
            raise ValueError("patch_repr is required for stats-local verifier mode.")
        student_repr = torch.nn.functional.normalize(student_repr, dim=-1)
        patch_repr = torch.nn.functional.normalize(patch_repr, dim=-1)
        prototype_bank = torch.nn.functional.normalize(prototype_bank, dim=-1)

        global_scores = student_repr @ prototype_bank.T
        patch_scores = torch.einsum("bpd,sd->bps", patch_repr, prototype_bank)
        patch_mean = patch_scores.mean(dim=1)
        patch_max = patch_scores.max(dim=1).values
        patch_std = patch_scores.std(dim=1, unbiased=False)
        topk = max(1, min(int(topk), patch_scores.shape[1]))
        patch_topk_mean = torch.topk(patch_scores, k=topk, dim=1).values.mean(dim=1)
        temperature = max(float(temperature), 1e-6)
        patch_lse = torch.logsumexp(patch_scores * temperature, dim=1) / temperature

        stats = torch.stack(
            [
                global_scores,
                patch_mean,
                patch_max,
                patch_std,
                patch_topk_mean,
                patch_lse,
            ],
            dim=-1,
        )
        return self.stats_mlp(stats).squeeze(-1)
