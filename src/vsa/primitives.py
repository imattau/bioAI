import torch
import torchhd


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
