"""Real-valued VSA encoder with learned token vectors.

No sign() binarization until the final output layer.
EMA-style bundling: equal-weight sum → L2 normalize to unit sphere.

Architecture:
  token_vector(token) → bind(role_vector) → bundle(sum) → L2 normalize
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict

import torch

from src.text.token_library import TokenLibrary


class VSAEncoder:
    def __init__(
        self,
        dimension: int = 512,
        max_tokens: int = 96,
        seed: int = 137,
        cache_size: int = 50_000,
    ):
        self.dimension = dimension
        self.max_tokens = max_tokens
        self.seed = seed
        self.cache_size = cache_size
        self._token_cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.learned_vectors: dict[str, torch.Tensor] = {}
        generator = torch.Generator().manual_seed(seed)
        self.roles = {
            name: torch.where(
                torch.rand(dimension, generator=generator) >= 0.5,
                torch.ones((), dtype=torch.float32),
                -torch.ones((), dtype=torch.float32),
            )
            for name in ("query", "candidate")
        }

    def _random_vector(self, token: str) -> torch.Tensor:
        vector = self._token_cache.pop(token, None)
        if vector is None:
            digest = hashlib.blake2b(
                f"{self.seed}:{token}".encode(), digest_size=8
            ).digest()
            generator = torch.Generator().manual_seed(
                int.from_bytes(digest, "little") & ((1 << 63) - 1)
            )
            vector = torch.where(
                torch.rand(self.dimension, generator=generator) >= 0.5,
                torch.ones((), dtype=torch.float32),
                -torch.ones((), dtype=torch.float32),
            )
        self._token_cache[token] = vector
        while len(self._token_cache) > self.cache_size:
            self._token_cache.popitem(last=False)
        return vector

    def token_vector(self, token: str) -> torch.Tensor:
        learned = self.learned_vectors.get(token)
        return learned if learned is not None else self._random_vector(token)

    def encode(self, text: str, role: str) -> torch.Tensor:
        tokens = TokenLibrary.tokenize(text)[:self.max_tokens]
        if not tokens:
            return torch.zeros(self.dimension)
        state = torch.zeros(self.dimension)
        rv = self.roles[role]
        for token in tokens:
            item = self.token_vector(token) * rv
            state = state + item
        norm = state.norm(p=2).clamp(min=1e-8)
        return state / norm

    def encode_query(self, prompt: str, evidence: list[str]) -> torch.Tensor:
        pv = self.encode(prompt, "query")
        ev = self.encode(" ".join(evidence), "query")
        joint = pv + ev
        norm = joint.norm(p=2).clamp(min=1e-8)
        return joint / norm

    def learn_associations(
        self,
        examples: list[tuple[str, list[str]]],
        max_context_terms: int = 64,
    ) -> None:
        accumulated: dict[str, torch.Tensor] = {}
        counts: dict[str, int] = {}
        for text, semantic_context in examples:
            text_tokens = list(dict.fromkeys(TokenLibrary.tokenize(text)))
            context_tokens = list(dict.fromkeys(
                TokenLibrary.tokenize(" ".join(semantic_context))
            ))[:max_context_terms]
            if not text_tokens or not context_tokens:
                continue
            context = torch.stack([
                self._random_vector(token) for token in context_tokens
            ]).sum(dim=0)
            for token in text_tokens:
                accumulated[token] = (
                    accumulated.get(token, torch.zeros(self.dimension))
                    + context
                )
                counts[token] = counts.get(token, 0) + 1
        self.learned_vectors = {}
        for token, vec in accumulated.items():
            count = counts[token]
            mean = vec / count
            norm = mean.norm(p=2).clamp(min=1e-8)
            self.learned_vectors[token] = mean / norm
        self._token_cache.clear()

    def get_state(self) -> dict:
        return {
            "dimension": self.dimension,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
            "cache_size": self.cache_size,
            "learned_vectors": self.learned_vectors,
            "roles": self.roles,
        }

    @classmethod
    def from_state(cls, state: dict) -> "VSAEncoder":
        encoder = cls(
            state["dimension"],
            state["max_tokens"],
            state["seed"],
            state.get("cache_size", 50_000),
        )
        encoder.learned_vectors = state.get("learned_vectors", {})
        for k, v in encoder.learned_vectors.items():
            if isinstance(v, torch.Tensor) and v.dtype == torch.int8:
                encoder.learned_vectors[k] = v.float()
        encoder.roles = state["roles"]
        return encoder
