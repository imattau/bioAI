import torch
import torchhd

from src.vsa import VSA, HopfieldNet, AssociativeStore


class LookupDecoder:
    def __init__(self, vsa_dim: int = 10000, max_seq_len: int = 64,
                 store_capacity: int = 1000, hopfield_steps: int = 10):
        self.vsa_dim = vsa_dim
        self.max_seq_len = max_seq_len
        self.hopfield_steps = hopfield_steps

        self.vsa = VSA(dim=vsa_dim)
        self.hopfield = HopfieldNet(dim=vsa_dim)
        self.store = AssociativeStore(dim=vsa_dim, capacity=store_capacity)

        device = self.vsa.device
        self._pos_vectors = torchhd.random(max_seq_len, vsa_dim, device=device)
        self._word_cache: dict[str, torch.Tensor] = {}
        self._texts: list[str] = []

    def _ensure_word(self, word: str) -> torch.Tensor:
        if word not in self._word_cache:
            self._word_cache[word] = self.vsa.make_vector()
        return self._word_cache[word]

    def encode(self, sentence: str) -> torch.Tensor:
        words = sentence.lower().strip(".,!?;:'\"").split()
        words = words[:self.max_seq_len]
        if not words:
            return torch.zeros(self.vsa_dim)
        vecs = torch.stack([self._ensure_word(w) for w in words])
        pos = self._pos_vectors[:len(vecs)]
        bound = torchhd.bind(vecs, pos)
        bundle = torchhd.multiset(bound)
        return bundle.sign()

    def ingest(self, sentence: str) -> torch.Tensor:
        hv = self.encode(sentence)
        self.hopfield.store(hv)
        self.store.insert(hv, hv)
        self._texts.append(sentence)
        return hv

    def ingest_batch(self, sentences: list[str]) -> list[torch.Tensor]:
        return [self.ingest(s) for s in sentences]

    def lookup(self, query: torch.Tensor, k: int = 1) -> list[tuple[str, float]]:
        if not self._texts:
            return []
        cleaned = self.hopfield.recall(query, steps=self.hopfield_steps)
        stack = torch.stack(self.store.keys, dim=0)
        sims = torchhd.cosine_similarity(cleaned.unsqueeze(0), stack).squeeze(0)
        vals, idxs = sims.topk(min(k, len(sims)))
        return [(self._texts[i.item()], vals[j].item()) for j, i in enumerate(idxs)]

    def lookup_from_text(self, sentence: str, k: int = 1) -> list[tuple[str, float]]:
        hv = self.encode(sentence)
        return self.lookup(hv, k=k)

    def __len__(self) -> int:
        return len(self._texts)
