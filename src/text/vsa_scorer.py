"""Unified VSA encoder + Go/No-Go scorer for candidate ranking.

No hand-crafted features, no bag-of-words arithmetic — just learned
VSA encodings scored through a small feedforward network.

Architecture:
  prompt ──┐
           ├── VSA encode ──→ joint state ──→ Go/No-Go network ──→ score
  evidence ─┘                                          ↑
  candidate text ──────────────────────────────────────┘
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from src.basal import GoNoGoActorCritic
from src.text.vsa_sequence_ranker import (
    LearnedTokenVSASequenceEncoder,
    RecurrentVSASequenceEncoder,
)


class VSAEncoderScorer:
    """VSA-encode (prompt, evidence, candidate) and score through Go/No-Go."""

    def __init__(
        self,
        dimension: int = 512,
        hidden_dimension: int = 64,
        learning_rate: float = 0.001,
    ):
        self.encoder = LearnedTokenVSASequenceEncoder(dimension=dimension)
        self.gonogo = GoNoGoActorCritic(
            dimension, n_actions=1, hidden_dim=hidden_dimension
        )
        self.optimizer = torch.optim.Adam(
            self.gonogo.parameters(), lr=learning_rate
        )
        self.updates = 0

    # ── encoding ──────────────────────────────────────────────────────

    def encode_state(
        self, prompt: str, evidence: list[str], candidate_text: str
    ) -> torch.Tensor:
        """Joint VSA encoding of (prompt, evidence, candidate) as a bipolar vector."""
        return self.encoder.encode_candidate(prompt, evidence, candidate_text)

    def encode_query(
        self, prompt: str, evidence: list[str]
    ) -> torch.Tensor:
        """Encode just the query side (prompt + evidence) for batched candidate scoring."""
        query = self.encoder.encode(prompt, "query")
        support = self.encoder.encode(" ".join(evidence), "evidence")
        return torch.sign(query + support)

    def encode_candidate(self, candidate_text: str) -> torch.Tensor:
        """Encode just the candidate text for batched scoring."""
        return self.encoder.encode(candidate_text, "candidate")

    def joint_from_components(
        self, query_state: torch.Tensor, candidate_state: torch.Tensor
    ) -> torch.Tensor:
        """Reconstruct the joint state from pre-computed query and candidate states."""
        joint = (
            query_state
            + candidate_state
            + query_state * candidate_state
        )
        return torch.sign(joint)

    # ── scoring ───────────────────────────────────────────────────────

    def score_state(self, state: torch.Tensor) -> torch.Tensor:
        """Go - NoGo logit for a joint VSA state vector."""
        _, _, go, nogo = self.gonogo(state.unsqueeze(0) if state.dim() == 1 else state)
        return (go - nogo).squeeze(-1)

    def score(
        self, prompt: str, candidate: dict, evidence: list[str]
    ) -> float:
        state = self.encode_state(prompt, evidence, candidate["text"])
        return self.score_state(state).item()

    def score_batch(
        self,
        query_state: torch.Tensor,
        candidate_states: torch.Tensor,
    ) -> torch.Tensor:
        """Score a batch of candidates against a shared query state."""
        batch = candidate_states.shape[0]
        query = query_state.unsqueeze(0).expand(batch, -1)
        joint = torch.sign(query + candidate_states + query * candidate_states)
        return self.score_state(joint)

    def rank(
        self, prompt: str, candidates: list[dict], evidence: list[str]
    ) -> list[dict]:
        qs = self.encode_query(prompt, evidence)
        scored = []
        for c in candidates:
            cs = self.encode_candidate(c["text"])
            score = self.score_batch(qs, cs.unsqueeze(0)).item()
            scored.append({**c, "score": score})
        return sorted(scored, key=lambda x: x["score"], reverse=True)

    # ── learning ──────────────────────────────────────────────────────

    def learn_preference(
        self,
        prompt: str,
        preferred: dict,
        rejected: dict,
        evidence: list[str],
    ) -> float:
        pref_state = self.encode_state(prompt, evidence, preferred["text"])
        rej_state = self.encode_state(prompt, evidence, rejected["text"])
        pref_logit = self.score_state(pref_state)
        rej_logit = self.score_state(rej_state)
        loss = F.softplus(-(pref_logit - rej_logit))
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.gonogo.parameters(), 1.0)
        self.optimizer.step()
        self.updates += 1
        return loss.item()

    def learn_associations(
        self,
        examples: list[tuple[str, list[str]]],
    ) -> None:
        """Learn VSA token vectors from (text, semantic_context) pairs."""
        self.encoder.learn_associations(examples)

    # ── persistence ───────────────────────────────────────────────────

    def get_state(self) -> dict:
        return {
            "encoder": self.encoder.get_state(),
            "gonogo": self.gonogo.state_dict(),
            "updates": self.updates,
        }

    @classmethod
    def from_state(cls, state: dict) -> "VSAEncoderScorer":
        encoder_state = state["encoder"]
        if encoder_state.get("kind") == "learned_token":
            encoder = LearnedTokenVSASequenceEncoder.from_state(encoder_state)
        else:
            encoder = RecurrentVSASequenceEncoder.from_state(encoder_state)
        scorer = cls(dimension=encoder.dimension)
        scorer.encoder = encoder
        scorer.gonogo.load_state_dict(state["gonogo"])
        scorer.updates = state.get("updates", 0)
        return scorer

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.get_state(), path)

    @classmethod
    def load(cls, path: Path) -> "VSAEncoderScorer":
        return cls.from_state(torch.load(path, map_location="cpu", weights_only=False))
