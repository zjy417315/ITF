import torch
import torch.nn as nn


class StudentTokenHead(nn.Module):
    """
    Student-side token recovery head.

    This head predicts the teacher-side source token from RGB features while the
    residual/code branch keeps modeling the continuous protocol detail.
    """

    def __init__(
        self,
        d_in: int,
        num_tokens: int,
        code_dim: int = 0,
        hidden_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_in = int(d_in)
        self.num_tokens = int(num_tokens)
        self.code_dim = int(code_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        input_dim = self.d_in + max(self.code_dim, 0)

        if self.hidden_dim > 0:
            self.head = nn.Sequential(
                nn.Linear(input_dim, self.hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim, self.num_tokens),
            )
            final_layer = self.head[-1]
        else:
            self.head = nn.Linear(input_dim, self.num_tokens)
            final_layer = self.head
        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)

    def forward(
        self,
        feature_vec: torch.Tensor,
        base_logits: torch.Tensor = None,
    ) -> dict[str, torch.Tensor]:
        if self.code_dim > 0:
            if base_logits is None:
                base_logits = torch.zeros(
                    feature_vec.shape[0],
                    self.code_dim,
                    device=feature_vec.device,
                    dtype=feature_vec.dtype,
                )
            head_input = torch.cat([feature_vec, base_logits], dim=-1)
        else:
            head_input = feature_vec
        token_logits = self.head(head_input)
        return {
            "token_logits": token_logits,
            "token_probs": torch.softmax(token_logits, dim=-1),
        }
