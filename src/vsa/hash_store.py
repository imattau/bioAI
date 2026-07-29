import torch

import torchhd


class VSAHashStore:
    def __init__(self, vsa, num_buckets: int = 10000, max_items: int = 1000):
        self.vsa = vsa
        self.num_buckets = num_buckets
        self.max_items = max_items
        self._data: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {}
        self._order: list[bytes] = []

    def _bucket(self, key: torch.Tensor) -> int:
        return hash(key.cpu().numpy().tobytes()) % self.num_buckets

    def _key_bytes(self, key: torch.Tensor) -> bytes:
        return key.cpu().numpy().tobytes()

    def _evict_one(self):
        while len(self._order) > self.max_items:
            oldest = self._order.pop(0)
            h = hash(oldest) % self.num_buckets
            bucket = self._data.get(h)
            if bucket is None:
                continue
            for i, (k, _) in enumerate(bucket):
                if self._key_bytes(k) == oldest:
                    bucket.pop(i)
                    break
            if not bucket:
                del self._data[h]

    def insert(self, key: torch.Tensor, value: torch.Tensor):
        h = self._bucket(key)
        bound = torchhd.bind(key, value)
        if h in self._data:
            self._data[h].append((key, bound))
        else:
            self._data[h] = [(key, bound)]
        self._order.append(self._key_bytes(key))
        if len(self._order) > self.max_items:
            self._evict_one()

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
        return {"num_buckets": self.num_buckets, "all_keys": keys,
                "max_items": self.max_items}

    def set_state(self, state: dict):
        self.max_items = state.get("max_items", self.max_items)
        self._data = {}
        self._order = []
        for k in state["all_keys"]:
            self.insert(k, k)

    def __len__(self) -> int:
        return sum(len(v) for v in self._data.values())
