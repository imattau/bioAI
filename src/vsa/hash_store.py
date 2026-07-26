import torch

import torchhd


class VSAHashStore:
    def __init__(self, vsa, num_buckets: int = 10000):
        self.vsa = vsa
        self.num_buckets = num_buckets
        self._data: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {}

    def _bucket(self, key: torch.Tensor) -> int:
        return hash(key.cpu().numpy().tobytes()) % self.num_buckets

    def insert(self, key: torch.Tensor, value: torch.Tensor):
        h = self._bucket(key)
        bound = torchhd.bind(key, value)
        if h in self._data:
            self._data[h].append((key, bound))
        else:
            self._data[h] = [(key, bound)]

    def lookup(self, query: torch.Tensor) -> torch.Tensor | None:
        h = self._bucket(query)
        items = self._data.get(h)
        if items is None:
            return None
        for key, bound in items:
            sim = self.vsa.similarity(query, key).item()
            if sim > 0.99:
                return torchhd.bind(bound, key)
        return None

    def get_state(self) -> dict:
        keys = [k for bucket in self._data.values() for k, _ in bucket]
        return {"num_buckets": self.num_buckets, "all_keys": keys}

    def set_state(self, state: dict):
        self._data = {}
        for k in state["all_keys"]:
            self.insert(k, k)

    def __len__(self) -> int:
        return sum(len(v) for v in self._data.values())
