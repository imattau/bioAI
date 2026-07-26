import os
import torch

from src.text.encoder import VSAEncoder
from src.vsa import VSA, HopfieldNet, AssociativeStore


class VSADecoder:
    def __init__(self, vsa: VSA | None = None, store_capacity: int = 1000,
                 hopfield_steps: int = 10,
                 encoder: VSAEncoder | None = None):
        self.vsa = vsa or VSA()
        self.hopfield_steps = hopfield_steps
        self.encoder = encoder or VSAEncoder(vsa=self.vsa)
        self.hopfield = HopfieldNet(dim=self.vsa.dim)
        self.store = AssociativeStore(dim=self.vsa.dim, capacity=store_capacity)
        self._texts: list[str] = []
        torch.set_num_threads(os.cpu_count() or 1)

    def ingest(self, text: str):
        hv = self.encode(text)
        self.hopfield.store(hv)
        self.store.insert(hv, hv)
        self._texts.append(text)

    def ingest_batch(self, texts: list[str]):
        for t in texts:
            self.ingest(t)

    def encode(self, text: str) -> torch.Tensor:
        return self.encoder.encode(text)

    def _similarity(self, query: torch.Tensor) -> torch.Tensor:
        stack = self.store.key_stack()
        if stack.numel() == 0:
            return torch.empty(0, device=query.device)
        return (query.unsqueeze(0) @ stack.T).squeeze(0) / self.vsa.dim

    def decode(self, vector: torch.Tensor,
               k: int = 1) -> list[tuple[str, float]]:
        if not self._texts:
            return []

        sims = self._similarity(vector)

        if k == 1:
            exact_idx = (sims > 0.9999).nonzero()
            if exact_idx.numel() > 0:
                idx = exact_idx[0].item()
                return [(self._texts[idx], sims[idx].item())]

        best_sim = sims.max().item()
        if best_sim < 0.5 and len(self._texts) > 1:
            cleaned = self.hopfield.recall(vector, steps=self.hopfield_steps)
            sims_clean = self._similarity(cleaned)
            if sims_clean.max().item() > best_sim:
                sims = sims_clean

        vals, idxs = sims.topk(min(k, len(sims)))
        return [(self._texts[i.item()], vals[j].item()) for j, i in enumerate(idxs)]

    def decode_from_text(self, text: str,
                         k: int = 1) -> list[tuple[str, float]]:
        hv = self.encode(text)
        return self.decode(hv, k=k)

    def get_state(self) -> dict:
        return {
            "hopfield_steps": self.hopfield_steps,
            "texts": self._texts,
            "store": self.store.get_state(),
            "hopfield": self.hopfield.get_state(),
        }

    def set_state(self, state: dict):
        self.hopfield_steps = state["hopfield_steps"]
        self._texts = state["texts"]
        self.store.set_state(state["store"]["keys"])
        self.hopfield.set_state(state["hopfield"]["patterns"])

    def __len__(self) -> int:
        return len(self._texts)
