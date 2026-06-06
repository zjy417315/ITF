import torch
import torch.nn as nn
import torch.nn.functional as F


class AuthenticationCodebook(nn.Module):
    """
    Learnable authentication codebook on the hypersphere.

    Teacher/student logits are mapped to the nearest prototype so the protocol
    works with stable discrete codewords instead of arbitrary continuous codes.
    """

    def __init__(
        self,
        code_dim: int,
        num_codes: int,
        temperature: float = 12.0,
        learnable: bool = True,
    ):
        super().__init__()
        self.code_dim = int(code_dim)
        self.num_codes = int(num_codes)
        self.temperature = float(temperature)
        if self.code_dim <= 0:
            raise ValueError("code_dim must be positive.")
        if self.num_codes <= 1:
            raise ValueError("num_codes must be greater than 1.")

        init_codes = torch.sign(torch.randn(self.num_codes, self.code_dim))
        init_codes[init_codes == 0] = 1.0
        init_codes = F.normalize(init_codes, dim=-1)
        self.codes = nn.Parameter(init_codes, requires_grad=bool(learnable))

    def normalized_codes(self) -> torch.Tensor:
        return F.normalize(self.codes, dim=-1)

    def similarity(self, logits: torch.Tensor) -> torch.Tensor:
        logits_shape = logits.shape[:-1]
        flat_logits = F.normalize(logits.reshape(-1, self.code_dim), dim=-1)
        sims = flat_logits @ self.normalized_codes().T
        return sims.reshape(*logits_shape, self.num_codes)

    def assign(self, logits: torch.Tensor):
        sims = self.similarity(logits)
        indices = sims.argmax(dim=-1)
        return indices, sims

    def lookup(self, indices: torch.Tensor) -> torch.Tensor:
        return F.embedding(indices, self.normalized_codes())

    def quantize(self, logits: torch.Tensor, straight_through: bool = True):
        indices, sims = self.assign(logits)
        prototypes = self.lookup(indices)
        if straight_through:
            quantized = logits + (prototypes - logits).detach()
        else:
            quantized = prototypes
        return quantized, indices, sims, prototypes

    def classification_loss(
        self,
        logits: torch.Tensor,
        target_indices: torch.Tensor,
        sample_weights: torch.Tensor = None,
    ) -> torch.Tensor:
        sims = self.similarity(logits) / max(self.temperature, 1e-6)
        flat_loss = F.cross_entropy(
            sims.reshape(-1, self.num_codes),
            target_indices.reshape(-1),
            reduction="none",
        )
        if sample_weights is None:
            return flat_loss.mean()
        expanded_weights = sample_weights.reshape(-1)
        if target_indices.ndim > 1:
            repeats = int(target_indices.numel() / sample_weights.numel())
            expanded_weights = expanded_weights.repeat_interleave(repeats)
        weights = expanded_weights.to(flat_loss.device, dtype=flat_loss.dtype)
        return (flat_loss * weights).sum() / weights.sum().clamp_min(1e-6)

    def prototype_alignment_loss(
        self,
        logits: torch.Tensor,
        target_prototypes: torch.Tensor,
        sample_weights: torch.Tensor = None,
    ) -> torch.Tensor:
        logits_norm = F.normalize(logits, dim=-1)
        target_norm = F.normalize(target_prototypes.detach(), dim=-1)
        sample_loss = 1.0 - (logits_norm * target_norm).sum(dim=-1)
        if sample_weights is None:
            return sample_loss.mean()
        weights = sample_weights.to(sample_loss.device, dtype=sample_loss.dtype)
        return (sample_loss * weights).sum() / weights.sum().clamp_min(1e-6)

    def commitment_loss(
        self,
        logits: torch.Tensor,
        prototypes: torch.Tensor,
        sample_weights: torch.Tensor = None,
    ) -> torch.Tensor:
        logits_norm = F.normalize(logits, dim=-1)
        proto_norm = F.normalize(prototypes, dim=-1)
        sample_loss = 1.0 - (logits_norm * proto_norm).sum(dim=-1)
        if sample_weights is None:
            return sample_loss.mean()
        weights = sample_weights.to(sample_loss.device, dtype=sample_loss.dtype)
        return (sample_loss * weights).sum() / weights.sum().clamp_min(1e-6)

    def usage_loss(self, indices: torch.Tensor) -> torch.Tensor:
        hist = torch.bincount(indices.reshape(-1), minlength=self.num_codes).float()
        probs = hist / hist.sum().clamp_min(1.0)
        target = torch.full_like(probs, 1.0 / self.num_codes)
        return (probs - target).pow(2).mean()

    def separation_loss(self) -> torch.Tensor:
        codes = self.normalized_codes()
        sim = codes @ codes.T
        off_diag = sim - torch.diag(torch.diag(sim))
        return off_diag.pow(2).mean()

    @torch.no_grad()
    def initialize_from_samples(self, sample_logits: torch.Tensor):
        if sample_logits.ndim != 2 or sample_logits.shape[1] != self.code_dim:
            raise ValueError(
                f"sample_logits should have shape [N, {self.code_dim}], got {tuple(sample_logits.shape)}."
            )
        sample_logits = F.normalize(sample_logits.float(), dim=-1)
        if sample_logits.shape[0] >= self.num_codes:
            perm = torch.randperm(sample_logits.shape[0], device=sample_logits.device)[: self.num_codes]
            selected = sample_logits[perm]
        else:
            pad_count = self.num_codes - sample_logits.shape[0]
            pad = torch.sign(torch.randn(pad_count, self.code_dim, device=sample_logits.device))
            pad[pad == 0] = 1.0
            pad = F.normalize(pad, dim=-1)
            selected = torch.cat([sample_logits, pad], dim=0)
        self.codes.copy_(selected)
