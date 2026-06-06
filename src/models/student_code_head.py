import torch
import torch.nn as nn
import torch.nn.functional as F


class StudentCodeHead(nn.Module):
    """
    Auxiliary student head for recovering the transmitted authentication code.

    The main VisualEncoder branch remains optimized for verification geometry,
    while this head specializes in decoding the teacher-side protocol code.
    """

    def __init__(
        self,
        d_in: int,
        code_dim: int = 32,
        hidden_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_in = int(d_in)
        self.code_dim = int(code_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        input_dim = self.d_in + self.code_dim

        if self.hidden_dim > 0:
            self.head = nn.Sequential(
                nn.Linear(input_dim, self.hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim, self.code_dim),
            )
            final_layer = self.head[-1]
        else:
            self.head = nn.Linear(input_dim, self.code_dim)
            final_layer = self.head
        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)

    def forward(
        self,
        feature_vec: torch.Tensor,
        base_logits: torch.Tensor = None,
        return_logits: bool = False,
    ):
        if base_logits is None:
            base_logits = torch.zeros(
                feature_vec.shape[0],
                self.code_dim,
                device=feature_vec.device,
                dtype=feature_vec.dtype,
            )
        head_input = torch.cat([feature_vec, base_logits], dim=-1)
        logits = base_logits + self.head(head_input)
        code_repr = F.normalize(logits, dim=-1)
        if return_logits:
            return {"global_repr": code_repr, "global_logits": logits}
        return code_repr
