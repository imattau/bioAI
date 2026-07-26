"""Recurrent VSA sequence encoding with Go/No-Go candidate ranking."""

from __future__ import annotations

import hashlib
import re
from collections import OrderedDict

import torch
import torch.nn.functional as F

from src.basal import GoNoGoActorCritic
from src.text.token_library import TokenLibrary


class RecurrentVSASequenceEncoder:
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
        generator = torch.Generator().manual_seed(seed)
        self.roles = {
            name: torch.where(
                torch.rand(dimension, generator=generator) >= 0.5,
                torch.ones((), dtype=torch.float32),
                -torch.ones((), dtype=torch.float32),
            )
            for name in ("query", "evidence", "candidate")
        }

    def _token_vector(self, token: str) -> torch.Tensor:
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

    def encode(self, text: str, role: str) -> torch.Tensor:
        tokens = TokenLibrary.tokenize(text)[:self.max_tokens]
        if not tokens:
            return torch.zeros(self.dimension)
        state = torch.zeros(self.dimension)
        role_vector = self.roles[role]
        for position, token in enumerate(tokens):
            item = torch.roll(
                self._token_vector(token) * role_vector,
                shifts=position % self.dimension,
            )
            state = torch.sign(torch.roll(state, shifts=1) + item)
        return state

    def encode_candidate(
        self, prompt: str, evidence: list[str], candidate: str
    ) -> torch.Tensor:
        query = self.encode(prompt, "query")
        support = self.encode(" ".join(evidence), "evidence")
        response = self.encode(candidate, "candidate")
        joint = (
            query + support + response
            + query * response
            + support * response
        )
        return torch.sign(joint)

    def get_state(self) -> dict:
        return {
            "kind": "random_token",
            "dimension": self.dimension,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
            "cache_size": self.cache_size,
            "roles": self.roles,
        }

    @classmethod
    def from_state(cls, state: dict) -> "RecurrentVSASequenceEncoder":
        encoder = cls(
            state["dimension"], state["max_tokens"],
            state["seed"], state.get("cache_size", 50_000),
        )
        encoder.roles = state["roles"]
        return encoder


class SemanticChunkVSASequenceEncoder(RecurrentVSASequenceEncoder):
    """Recurrent role-bound states over learned semantic chunk VSAs."""

    def __init__(
        self,
        semantic_encoder,
        max_tokens: int = 16,
        seed: int = 137,
        cache_size: int = 50_000,
    ):
        super().__init__(
            dimension=semantic_encoder.vsa_dim,
            max_tokens=max_tokens,
            seed=seed,
            cache_size=cache_size,
        )
        self.semantic_encoder = semantic_encoder
        self._chunk_cache: OrderedDict[str, torch.Tensor] = OrderedDict()

    @staticmethod
    def chunks(text: str) -> list[str]:
        return [
            chunk.strip()
            for chunk in re.split(r"(?<=[.!?;:])\s+|\n+", text)
            if chunk.strip()
        ]

    def _chunk_vector(self, chunk: str) -> torch.Tensor:
        vector = self._chunk_cache.pop(chunk, None)
        if vector is None:
            vector = self.semantic_encoder.encode(chunk).float()
        self._chunk_cache[chunk] = vector
        while len(self._chunk_cache) > self.cache_size:
            self._chunk_cache.popitem(last=False)
        return vector

    def encode(self, text: str, role: str) -> torch.Tensor:
        chunks = self.chunks(text)[:self.max_tokens]
        if not chunks:
            return torch.zeros(self.dimension)
        state = torch.zeros(self.dimension)
        role_vector = self.roles[role]
        for position, chunk in enumerate(chunks):
            item = torch.roll(
                self._chunk_vector(chunk) * role_vector,
                shifts=position % self.dimension,
            )
            state = torch.sign(torch.roll(state, shifts=1) + item)
        return state

    def get_state(self) -> dict:
        state = super().get_state()
        state["kind"] = "semantic_chunk"
        state["semantic_encoder"] = self.semantic_encoder.get_state()
        return state

    @classmethod
    def from_state(
        cls, state: dict, embedder
    ) -> "SemanticChunkVSASequenceEncoder":
        from src.text.semantic_vsa import SemanticVSAEncoder

        semantic_state = state["semantic_encoder"]
        semantic_encoder = SemanticVSAEncoder(
            embedder,
            vsa_dim=semantic_state["vsa_dim"],
            seed=semantic_state["seed"],
            projection=semantic_state["projection"],
        )
        encoder = cls(
            semantic_encoder,
            max_tokens=state["max_tokens"],
            seed=state["seed"],
            cache_size=state.get("cache_size", 50_000),
        )
        encoder.roles = state["roles"]
        return encoder


class VSASequenceRanker:
    def __init__(
        self,
        dimension: int = 512,
        hidden_dimension: int = 64,
        learning_rate: float = 0.001,
        encoder: RecurrentVSASequenceEncoder | None = None,
    ):
        self.encoder = encoder or RecurrentVSASequenceEncoder(dimension)
        self.hidden_dimension = hidden_dimension
        self.learning_rate = learning_rate
        self.gonogo = GoNoGoActorCritic(
            self.encoder.dimension, n_actions=1, hidden_dim=hidden_dimension
        )
        self.optimizer = torch.optim.Adam(
            self.gonogo.parameters(), lr=learning_rate
        )
        self.updates = 0

    def _logit(
        self, prompt: str, candidate: dict, evidence: list[str]
    ) -> torch.Tensor:
        state = self.encoder.encode_candidate(
            prompt, evidence, candidate["text"]
        )
        _, _, go, nogo = self.gonogo(state)
        return (go - nogo).squeeze()

    def encode_state(
        self, prompt: str, candidate: dict, evidence: list[str]
    ) -> torch.Tensor:
        return self.encoder.encode_candidate(
            prompt, evidence, candidate["text"]
        )

    def _state_logits(self, states: torch.Tensor) -> torch.Tensor:
        _, _, go, nogo = self.gonogo(states)
        return (go - nogo).squeeze(-1)

    def learn_encoded_batch(
        self,
        preferred_states: torch.Tensor,
        rejected_states: torch.Tensor,
    ) -> float:
        preferred_logits = self._state_logits(preferred_states.float())
        rejected_logits = self._state_logits(rejected_states.float())
        loss = F.softplus(-(preferred_logits - rejected_logits)).mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.gonogo.parameters(), 1.0)
        self.optimizer.step()
        self.updates += len(preferred_states)
        return loss.item()

    def train_encoded_preferences(
        self,
        preferences: list[tuple[torch.Tensor, torch.Tensor]],
        examples: int,
        batch_size: int = 64,
        seed: int = 191,
    ) -> float:
        if not preferences or examples <= 0:
            return 0.0
        generator = torch.Generator().manual_seed(seed + self.updates)
        losses = []
        remaining = examples
        while remaining > 0:
            size = min(batch_size, remaining)
            indices = torch.randint(
                len(preferences), (size,), generator=generator
            ).tolist()
            preferred = torch.stack([
                preferences[index][0] for index in indices
            ])
            rejected = torch.stack([
                preferences[index][1] for index in indices
            ])
            losses.append(self.learn_encoded_batch(preferred, rejected))
            remaining -= size
        return sum(losses) / len(losses)

    @torch.no_grad()
    def score(self, prompt: str, candidate: dict, evidence: list[str]) -> float:
        return self._logit(prompt, candidate, evidence).item()

    def rank(self, prompt: str, candidates: list[dict], evidence: list[str]):
        ranked = [
            {**candidate, "score": self.score(prompt, candidate, evidence)}
            for candidate in candidates
        ]
        return sorted(ranked, key=lambda item: item["score"], reverse=True)

    def learn_preference(
        self,
        prompt: str,
        preferred: dict,
        rejected: dict,
        evidence: list[str],
        steps: int = 1,
    ) -> float:
        loss_value = 0.0
        for _ in range(steps):
            preferred_logit = self._logit(prompt, preferred, evidence)
            rejected_logit = self._logit(prompt, rejected, evidence)
            loss = F.softplus(-(preferred_logit - rejected_logit))
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.gonogo.parameters(), 1.0)
            self.optimizer.step()
            loss_value = loss.item()
            self.updates += 1
        return loss_value

    def get_state(self) -> dict:
        return {
            "encoder": self.encoder.get_state(),
            "hidden_dimension": self.hidden_dimension,
            "learning_rate": self.learning_rate,
            "gonogo": self.gonogo.state_dict(),
            "updates": self.updates,
        }

    @classmethod
    def from_state(cls, state: dict, embedder=None) -> "VSASequenceRanker":
        encoder_state = state["encoder"]
        if encoder_state.get("kind") == "semantic_chunk":
            if embedder is None:
                raise ValueError("Semantic ranker state requires an embedder")
            encoder = SemanticChunkVSASequenceEncoder.from_state(
                encoder_state, embedder
            )
        else:
            encoder = RecurrentVSASequenceEncoder.from_state(encoder_state)
        ranker = cls(
            encoder=encoder,
            hidden_dimension=state.get("hidden_dimension", 64),
            learning_rate=state.get("learning_rate", 0.001),
        )
        ranker.gonogo.load_state_dict(state["gonogo"])
        ranker.updates = state.get("updates", 0)
        return ranker
