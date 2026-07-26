import torch


class AssociativeStore:
    def __init__(self, dim: int, capacity: int = 1000):
        self.dim = dim
        self.capacity = capacity
        self.keys: list[torch.Tensor] = []
        self.values: list[torch.Tensor] = []
        self._key_stack: torch.Tensor | None = None
        self._dirty = True

    def insert(self, key: torch.Tensor, value: torch.Tensor | None = None):
        if value is None:
            value = key
        if len(self.keys) >= self.capacity:
            self.keys.pop(0)
            self.values.pop(0)
        self.keys.append(key.detach().clone())
        self.values.append(value.detach().clone())
        self._dirty = True

    def key_stack(self) -> torch.Tensor:
        if self._dirty or self._key_stack is None:
            self._key_stack = torch.stack(self.keys, dim=0) if self.keys else torch.empty(0, self.dim)
            self._dirty = False
        return self._key_stack

    def lookup(self, query: torch.Tensor, k: int = 1) -> list[tuple[torch.Tensor, float]]:
        if not self.keys:
            return []
        stack = self.key_stack()
        sims = (query.unsqueeze(0) @ stack.T).squeeze(0) / self.dim
        vals, idxs = sims.topk(min(k, len(sims)))
        return [(self.values[i.item()], vals[j].item()) for j, i in enumerate(idxs)]

    def similarity_to_all(self, query: torch.Tensor) -> torch.Tensor:
        if not self.keys:
            return torch.tensor([])
        stack = self.key_stack()
        return (query.unsqueeze(0) @ stack.T).squeeze(0) / self.dim

    def get_state(self) -> dict:
        return {"dim": self.dim, "capacity": self.capacity,
                "keys": self.keys, "values": self.values}

    def set_state(self, keys: list[torch.Tensor], values: list[torch.Tensor]):
        self.keys = keys
        self.values = values
        self._dirty = True

    def __len__(self) -> int:
        return len(self.keys)
