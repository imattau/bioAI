import math

import torch
import torchhd


def pack_bipolar(vec: torch.Tensor) -> torch.Tensor:
    """Pack a bipolar ±1 vector into uint8 bits (1 for +1, 0 for -1)."""
    bits = (vec > 0).to(torch.uint8)
    dim = bits.numel()
    padded = ((dim + 7) // 8) * 8
    if padded > dim:
        bits = torch.nn.functional.pad(bits, (0, padded - dim))
    packed = bits.view(-1, 8).mul(torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], device=vec.device, dtype=torch.uint8)).sum(dim=1)
    return packed


def _popcnt(x: torch.Tensor) -> torch.Tensor:
    """Total popcount of uint8 tensor. For (N, K) returns (N,), for (K,) returns ()."""
    bits = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], device=x.device, dtype=torch.uint8)
    byte_pops = (x.unsqueeze(-1).bitwise_and(bits)).ne(0).sum(dim=-1, dtype=torch.float32)
    return byte_pops.sum(dim=-1) if byte_pops.dim() > 1 else byte_pops.sum()


def packed_similarity(query_packed: torch.Tensor, pattern_packed: torch.Tensor, dim: int) -> torch.Tensor:
    """Cosine similarity of bipolar vectors from packed uint8 tensors.

    sim = 1 - 2 * popcount(query XOR pattern) / dim
    Handles both 1D (K,) and 2D (N, K) inputs.
    """
    xor_bits = query_packed ^ pattern_packed
    popcnt = _popcnt(xor_bits)
    return 1.0 - 2.0 * popcnt / dim


def packed_similarity_batch(query_packed: torch.Tensor, patterns_packed: torch.Tensor, dim: int) -> torch.Tensor:
    """Batch query against N stored patterns. query_packed can be (K,) or (1, K)."""
    if query_packed.dim() == 1:
        query_packed = query_packed.unsqueeze(0)
    xor_bits = patterns_packed ^ query_packed
    return 1.0 - 2.0 * _popcnt(xor_bits) / dim


def unpack_to_float(packed: torch.Tensor, dim: int) -> torch.Tensor:
    """Unpack uint8 bits back to float32 ±1 vector."""
    byte_bits = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], device=packed.device, dtype=torch.uint8)
    bits = (packed.unsqueeze(-1).bitwise_and(byte_bits)).ne(0).flatten()[:dim].float()
    return bits * 2.0 - 1.0


class VSA:
    def __init__(self, dim: int = 10000, device: str | None = None):
        self.dim = dim
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def make_vector(self) -> torch.Tensor:
        return torchhd.random(1, self.dim, device=self.device).squeeze(0)

    def make_vectors(self, n: int) -> torch.Tensor:
        return torchhd.random(n, self.dim, device=self.device)

    def bind(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torchhd.bind(a, b)

    def unbind(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torchhd.bind(a, torchhd.inverse(b))

    def bundle(self, items: list[torch.Tensor]) -> torch.Tensor:
        stacked = torch.stack(items)
        return torchhd.multiset(stacked)

    def permute(self, x: torch.Tensor, shifts: int = 1) -> torch.Tensor:
        return torchhd.permute(x, shifts=shifts)

    def similarity(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a_2d = a.unsqueeze(0) if a.dim() == 1 else a
        b_2d = b.unsqueeze(0) if b.dim() == 1 else b
        return torchhd.cosine_similarity(a_2d, b_2d).squeeze(0)

    def pack(self, vec: torch.Tensor) -> torch.Tensor:
        return pack_bipolar(vec)

    def packed_similarity(self, a, b) -> torch.Tensor:
        if b.dim() == 1:
            return packed_similarity(a, b, self.dim)
        return packed_similarity_batch(a, b, self.dim)

    def unpack(self, packed: torch.Tensor) -> torch.Tensor:
        return unpack_to_float(packed, self.dim)

    def encode_roles(self, roles: list[str]) -> dict[str, torch.Tensor]:
        return {r: self.make_vector() for r in roles}

    def encode_fillers(self, fillers: list[str]) -> dict[str, torch.Tensor]:
        return {f: self.make_vector() for f in fillers}

    def encode_triple(self, role_vectors: dict, filler_vectors: dict,
                      role: str, filler: str) -> torch.Tensor:
        return self.bind(role_vectors[role], filler_vectors[filler])

    def fractional_power_encoding(self, value: float, scale: float = 10.0) -> torch.Tensor:
        base = self.make_vector()
        base = base / base.norm()
        return torchhd.permute(base, shifts=int(value * scale))
