import torch
import torch.nn as nn
import torch.nn.functional as F


class TeacherTokenTokenizer(nn.Module):
    """
    Offline teacher tokenizer for the authentication protocol.

    The tokenizer clusters continuous teacher-side codes into a compact set of
    source tokens, then exposes both hard assignments and soft match
    probabilities for deterministic-gated verification.
    """

    def __init__(
        self,
        code_dim: int,
        num_tokens: int,
        temperature: float = 12.0,
    ):
        super().__init__()
        self.code_dim = int(code_dim)
        self.num_tokens = int(num_tokens)
        self.temperature = float(temperature)
        if self.code_dim <= 0:
            raise ValueError("code_dim must be positive.")
        if self.num_tokens <= 1:
            raise ValueError("num_tokens must be greater than 1.")

        init_codes = torch.sign(torch.randn(self.num_tokens, self.code_dim))
        init_codes[init_codes == 0] = 1.0
        init_codes = F.normalize(init_codes, dim=-1)
        self.prototypes = nn.Parameter(init_codes, requires_grad=False)

    def normalized_prototypes(self) -> torch.Tensor:
        return F.normalize(self.prototypes, dim=-1)

    def similarity(self, logits: torch.Tensor) -> torch.Tensor:
        logits_shape = logits.shape[:-1]
        flat_logits = F.normalize(logits.reshape(-1, self.code_dim), dim=-1)
        sims = flat_logits @ self.normalized_prototypes().T
        return sims.reshape(*logits_shape, self.num_tokens)

    def logits(self, logits: torch.Tensor) -> torch.Tensor:
        return self.similarity(logits) / max(self.temperature, 1e-6)

    def probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.logits(logits), dim=-1)

    def assign(self, logits: torch.Tensor):
        token_logits = self.logits(logits)
        token_indices = token_logits.argmax(dim=-1)
        return token_indices, token_logits

    def lookup(self, indices: torch.Tensor) -> torch.Tensor:
        return F.embedding(indices, self.normalized_prototypes())

    @torch.no_grad()
    def initialize_from_samples(
        self,
        sample_logits: torch.Tensor,
        num_iters: int = 20,
    ):
        if sample_logits.ndim != 2 or sample_logits.shape[1] != self.code_dim:
            raise ValueError(
                f"sample_logits should have shape [N, {self.code_dim}], got {tuple(sample_logits.shape)}."
            )
        sample_logits = F.normalize(sample_logits.float(), dim=-1)
        num_samples = sample_logits.shape[0]
        if num_samples == 0:
            raise ValueError("sample_logits must contain at least one sample.")

        if num_samples >= self.num_tokens:
            perm = torch.randperm(num_samples, device=sample_logits.device)[: self.num_tokens]
            centroids = sample_logits[perm].clone()
        else:
            pad_count = self.num_tokens - num_samples
            pad = torch.sign(torch.randn(pad_count, self.code_dim, device=sample_logits.device))
            pad[pad == 0] = 1.0
            pad = F.normalize(pad, dim=-1)
            centroids = torch.cat([sample_logits, pad], dim=0)

        for _ in range(max(int(num_iters), 1)):
            sims = sample_logits @ F.normalize(centroids, dim=-1).T
            assignments = sims.argmax(dim=1)
            next_centroids = []
            for token_idx in range(self.num_tokens):
                member_mask = assignments == token_idx
                if member_mask.any():
                    centroid = sample_logits[member_mask].mean(dim=0)
                else:
                    centroid = centroids[token_idx]
                next_centroids.append(F.normalize(centroid, dim=-1))
            centroids = torch.stack(next_centroids, dim=0)
        self.prototypes.copy_(centroids)
