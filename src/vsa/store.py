import torch
import torchhd


class AssociativeStore:
    def __init__(self, dim: int, capacity: int = 1000):
        self.dim = dim
        self.capacity = capacity
        self.keys: list[torch.Tensor] = []
        self.values: list[torch.Tensor] = []

    def insert(self, key: torch.Tensor, value: torch.Tensor | None = None):
        if value is None:
            value = key
        if len(self.keys) >= self.capacity:
            self.keys.pop(0)
            self.values.pop(0)
        self.keys.append(key.detach().clone())
        self.values.append(value.detach().clone())

    def lookup(self, query: torch.Tensor, k: int = 1) -> list[tuple[torch.Tensor, float]]:
        if not self.keys:
            return []
        stack = torch.stack(self.keys, dim=0)
        sims = torchhd.cosine_similarity(query.unsqueeze(0), stack).squeeze(0)
        vals, idxs = sims.topk(min(k, len(sims)))
        return [(self.values[i.item()], vals[j].item()) for j, i in enumerate(idxs)]

    def similarity_to_all(self, query: torch.Tensor) -> torch.Tensor:
        if not self.keys:
            return torch.tensor([])
        stack = torch.stack(self.keys, dim=0)
        return torchhd.cosine_similarity(query.unsqueeze(0), stack).squeeze(0)

    def __len__(self) -> int:
        return len(self.keys)
