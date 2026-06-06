import torch
import torch.nn as nn
import torch.nn.functional as F


class AuthCodeHead(nn.Module):
    """
    Teacher-side authentication code head.

    It maps privileged prototype/teacher features into a compact code space that
    is trained jointly with the RGB-side student, instead of using a fixed
    post-hoc projection.
    """

    def __init__(
        self,
        d_in: int,
        code_dim: int = 32,
        hidden_dim: int = 256,
        use_sequence: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_in = int(d_in)
        self.code_dim = int(code_dim)
        self.hidden_dim = int(hidden_dim)
        self.use_sequence = bool(use_sequence)
        self.dropout = float(dropout)

        if self.hidden_dim > 0:
            self.global_mlp = nn.Sequential(
                nn.Linear(self.d_in, self.hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim, self.code_dim),
            )
        else:
            self.global_mlp = nn.Linear(self.d_in, self.code_dim)
        if self.use_sequence:
            if self.hidden_dim > 0:
                self.sequence_mlp = nn.Sequential(
                    nn.Linear(self.d_in, self.hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.hidden_dim, self.code_dim),
                )
            else:
                self.sequence_mlp = nn.Linear(self.d_in, self.code_dim)
        else:
            self.sequence_mlp = None

    def initialize_from_projection(self, mean_vec: torch.Tensor, projection: torch.Tensor):
        if self.hidden_dim > 0:
            raise ValueError("Projection warm-start only supports linear AuthCodeHead (hidden_dim <= 0).")
        mean_vec = mean_vec.detach().float()
        projection = projection.detach().float()
        if projection.shape[0] != self.d_in or projection.shape[1] != self.code_dim:
            raise ValueError(
                f"Projection shape {tuple(projection.shape)} does not match "
                f"({self.d_in}, {self.code_dim})"
            )
        with torch.no_grad():
            self.global_mlp.weight.copy_(projection.T)
            self.global_mlp.bias.copy_(-(mean_vec @ projection))
            if self.sequence_mlp is not None:
                self.sequence_mlp.weight.copy_(projection.T)
                self.sequence_mlp.bias.copy_(-(mean_vec @ projection))

    def forward(
        self,
        feature_vec: torch.Tensor,
        feature_seq: torch.Tensor = None,
        return_sequence: bool = False,
        return_logits: bool = False,
    ):
        global_logits = self.global_mlp(feature_vec)
        stage_logits = None
        if self.sequence_mlp is not None and feature_seq is not None:
            stage_logits = self.sequence_mlp(feature_seq)
            fused_logits = global_logits + stage_logits.mean(dim=1)
        else:
            fused_logits = global_logits

        global_repr = F.normalize(fused_logits, dim=-1)
        if return_sequence or return_logits:
            outputs = {"global_repr": global_repr}
            if return_logits:
                outputs["global_logits"] = fused_logits
            if stage_logits is not None and return_sequence:
                outputs["stage_logits"] = stage_logits
                outputs["stage_repr"] = F.normalize(stage_logits, dim=-1)
            return outputs
        return global_repr
