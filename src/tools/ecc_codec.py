from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ErrorCorrectingCodec:
    scheme: str
    payload_dim: int
    code_dim: int
    repetition: int = 1

    def __post_init__(self):
        self.scheme = str(self.scheme)
        self.payload_dim = int(self.payload_dim)
        self.code_dim = int(self.code_dim)
        self.repetition = int(self.repetition)
        if self.payload_dim <= 0 or self.code_dim <= 0:
            raise ValueError("payload_dim and code_dim must be positive.")
        if self.scheme == "identity":
            if self.payload_dim != self.code_dim:
                raise ValueError("identity codec requires payload_dim == code_dim.")
            self.repetition = 1
        elif self.scheme == "repetition":
            if self.repetition <= 1:
                raise ValueError("repetition codec requires repetition > 1.")
            if self.payload_dim * self.repetition != self.code_dim:
                raise ValueError(
                    f"repetition codec requires payload_dim * repetition == code_dim, got "
                    f"{self.payload_dim} * {self.repetition} != {self.code_dim}"
                )
        else:
            raise ValueError(f"Unsupported ECC scheme: {self.scheme}")

    def encode_logits(self, payload_logits: torch.Tensor | None) -> torch.Tensor | None:
        if payload_logits is None:
            return None
        if self.scheme == "identity":
            return payload_logits
        return payload_logits.repeat_interleave(self.repetition, dim=-1)

    def encode_bits(self, payload_bits: torch.Tensor | None) -> torch.Tensor | None:
        if payload_bits is None:
            return None
        if self.scheme == "identity":
            return payload_bits
        return payload_bits.repeat_interleave(self.repetition, dim=-1)

    def decode_logits(self, codeword_logits: torch.Tensor | None) -> torch.Tensor | None:
        if codeword_logits is None:
            return None
        if self.scheme == "identity":
            return codeword_logits
        shape = codeword_logits.shape[:-1]
        return codeword_logits.reshape(*shape, self.payload_dim, self.repetition).mean(dim=-1)

    def hard_codeword_bits(self, codeword_logits: torch.Tensor | None) -> torch.Tensor | None:
        if codeword_logits is None:
            return None
        return (codeword_logits >= 0).to(torch.int64)

    def hard_payload_bits_from_codeword(self, codeword_logits: torch.Tensor | None) -> torch.Tensor | None:
        decoded_logits = self.decode_logits(codeword_logits)
        if decoded_logits is None:
            return None
        return (decoded_logits >= 0).to(torch.int64)


def resolve_payload_dim(code_dim: int, ecc_scheme: str = "identity", ecc_repetition: int = 2, payload_dim: int | None = None) -> int:
    code_dim = int(code_dim)
    ecc_scheme = str(ecc_scheme)
    ecc_repetition = int(ecc_repetition)
    if payload_dim is not None:
        return int(payload_dim)
    if ecc_scheme == "identity":
        return code_dim
    if ecc_scheme == "repetition":
        if ecc_repetition <= 1 or code_dim % ecc_repetition != 0:
            raise ValueError(
                f"repetition codec requires code_dim divisible by repetition, got "
                f"code_dim={code_dim}, repetition={ecc_repetition}"
            )
        return code_dim // ecc_repetition
    raise ValueError(f"Unsupported ECC scheme: {ecc_scheme}")


def build_auth_codec(code_dim: int, ecc_scheme: str = "identity", ecc_repetition: int = 2, payload_dim: int | None = None) -> ErrorCorrectingCodec:
    payload_dim = resolve_payload_dim(
        code_dim=code_dim,
        ecc_scheme=ecc_scheme,
        ecc_repetition=ecc_repetition,
        payload_dim=payload_dim,
    )
    return ErrorCorrectingCodec(
        scheme=ecc_scheme,
        payload_dim=payload_dim,
        code_dim=int(code_dim),
        repetition=int(ecc_repetition),
    )
