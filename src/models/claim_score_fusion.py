import torch
import torch.nn as nn


class ClaimScoreFusionHead(nn.Module):
    """
    Learned decision layer for authentication scores.

    It consumes pairwise score features derived from the main claimed-source
    score matrix and the protocol score matrix, then predicts a fused score
    for each query-reference pair.
    """

    def __init__(
        self,
        input_dim: int = 12,
        hidden_dim: int = 32,
        dropout: float = 0.0,
        mode: str = "direct",
        residual_scale: float = 1.0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.mode = str(mode or "direct")
        if self.mode not in {"direct", "residual"}:
            raise ValueError(f"Unsupported claim score fusion mode: {self.mode}")
        self.residual_scale = float(residual_scale)
        mid_dim = max(self.hidden_dim // 2, 8)
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, mid_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(mid_dim, 1),
        )
        if self.mode == "residual":
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, pair_features: torch.Tensor, base_scores: torch.Tensor | None = None) -> torch.Tensor:
        scores = self.mlp(pair_features).squeeze(-1)
        if self.mode == "direct":
            return scores
        if base_scores is None:
            raise ValueError("base_scores must be provided when claim_score_fusion_mode='residual'")
        return base_scores + self.residual_scale * scores
