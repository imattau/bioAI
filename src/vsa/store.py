import torch


class AssociativeStore:
    def __init__(self, dim: int, capacity: int = 1000,
                 lsh_bits: int = 16, lsh_threshold: int = 100):
        self.dim = dim
        self.capacity = capacity
        self.keys: list[torch.Tensor] = []
        self.values: list[torch.Tensor] = []
        self._key_stack: torch.Tensor | None = None
        self._dirty = True

        self.lsh_bits = lsh_bits
        self.lsh_threshold = lsh_threshold
        _gen = torch.Generator().manual_seed(0)
        self._lsh_planes = torch.randn(lsh_bits, dim, generator=_gen)
        self._lsh_index: dict[int, list[int]] = {}

    def _device_lsh_planes(self, device: torch.device) -> torch.Tensor:
        if self._lsh_planes.device != device:
            self._lsh_planes = self._lsh_planes.to(device)
        return self._lsh_planes

    def _lsh_hash(self, vec: torch.Tensor) -> int:
        planes = self._device_lsh_planes(vec.device)
        bits = (vec @ planes.T > 0).to(torch.int8)
        h = 0
        if vec.dim() > 1:
            for i in range(self.lsh_bits):
                h = (h << 1) | int(bits[0, i].item())
        else:
            for i in range(self.lsh_bits):
                h = (h << 1) | int(bits[i].item())
        return h

    def _rebuild_lsh(self):
        self._lsh_index.clear()
        for idx, key in enumerate(self.keys):
            h = self._lsh_hash(key)
            if h not in self._lsh_index:
                self._lsh_index[h] = []
            self._lsh_index[h].append(idx)

    def insert(self, key: torch.Tensor, value: torch.Tensor | None = None):
        if value is None:
            value = key
        evicting = len(self.keys) >= self.capacity
        if evicting:
            self.keys.pop(0)
            self.values.pop(0)
        self.keys.append(key.detach().clone())
        self.values.append(value.detach().clone())
        self._dirty = True
        if self.lsh_bits:
            if evicting:
                self._rebuild_lsh()
            else:
                h = self._lsh_hash(key)
                if h not in self._lsh_index:
                    self._lsh_index[h] = []
                self._lsh_index[h].append(len(self.keys) - 1)

    def key_stack(self) -> torch.Tensor:
        if self._dirty or self._key_stack is None:
            self._key_stack = torch.stack(self.keys, dim=0) if self.keys else torch.empty(0, self.dim)
            self._dirty = False
        return self._key_stack

    def _exact_topk(self, query: torch.Tensor,
                     keys: torch.Tensor, values: list[torch.Tensor],
                     k: int) -> list[tuple[torch.Tensor, float]]:
        if keys.numel() == 0:
            return []
        sims = (query.unsqueeze(0) @ keys.T).squeeze(0) / self.dim
        k = min(k, len(sims))
        if k <= 0:
            return []
        vals, idxs = sims.topk(k)
        return [(values[i.item()], vals[j].item()) for j, i in enumerate(idxs)]

    def lookup(self, query: torch.Tensor, k: int = 1) -> list[tuple[torch.Tensor, float]]:
        if not self.keys:
            return []
        n = len(self.keys)
        if n <= self.lsh_threshold or not self.lsh_bits:
            return self._exact_topk(query, self.key_stack(), self.values, k)

        qh = self._lsh_hash(query)
        candidate_idxs = self._lsh_index.get(qh)
        if not candidate_idxs:
            return self._exact_topk(query, self.key_stack(), self.values, k)

        candidate_keys = torch.stack([self.keys[i] for i in candidate_idxs])
        candidate_values = [self.values[i] for i in candidate_idxs]
        return self._exact_topk(query, candidate_keys, candidate_values, k)

    def similarity_to_all(self, query: torch.Tensor) -> torch.Tensor:
        if not self.keys:
            return torch.tensor([])
        stack = self.key_stack()
        return (query.unsqueeze(0) @ stack.T).squeeze(0) / self.dim

    def get_state(self) -> dict:
        return {"capacity": self.capacity, "keys": self.keys,
                "lsh_bits": self.lsh_bits, "lsh_threshold": self.lsh_threshold}

    def set_state(self, keys: list[torch.Tensor]):
        self.keys = keys
        self.values = [k.clone() for k in keys]
        self._dirty = True
        if self.lsh_bits:
            self._rebuild_lsh()

    def __len__(self) -> int:
        return len(self.keys)
