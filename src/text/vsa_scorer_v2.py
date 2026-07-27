"""Go/No-Go scorer over VSA encoded states.

No evidence in the scorer path — evidence is used for candidate
generation only. The scorer sees only (query_state, candidate_state).

Architecture:
  query = encode_query(prompt, evidence)
  candidate = encode(candidate_text, "candidate")
  joint = concat(query, candidate)
  score = go(joint) - nogo(joint)
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.text.vsa_encoder import VSAEncoder


class GoNoGoScorer(nn.Module):
    def __init__(self, dimension: int = 512, hidden: int = 64):
        super().__init__()
        self.go = nn.Sequential(
            nn.Linear(dimension * 2, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )
        self.nogo = nn.Sequential(
            nn.Linear(dimension * 2, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )

    def forward(
        self, query: torch.Tensor, candidate: torch.Tensor
    ) -> torch.Tensor:
        joint = torch.cat([query, candidate], dim=-1)
        return self.go(joint) - self.nogo(joint)


class EncoderScorer:
    """Unified VSA encoder + Go/No-Go scorer.

    Evidence is used for query encoding (encode_query) but is NOT
    passed to the scorer as a separate input. This prevents the
    evidence_coverage shortcut identified in v1.
    """

    def __init__(
        self,
        dimension: int = 512,
        hidden: int = 64,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
    ):
        self.encoder = VSAEncoder(dimension=dimension)
        self.gonogo = GoNoGoScorer(dimension=dimension, hidden=hidden)
        self.optimizer = torch.optim.Adam(
            self.gonogo.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.updates = 0

    def encode_query(
        self, prompt: str, evidence: list[str]
    ) -> torch.Tensor:
        return self.encoder.encode_query(prompt, evidence)

    def score(
        self, prompt: str, candidate: dict, evidence: list[str]
    ) -> float:
        qs = self.encode_query(prompt, evidence).unsqueeze(0)
        cs = self.encoder.encode(
            candidate["text"], "candidate"
        ).unsqueeze(0)
        return self.gonogo(qs, cs).item()

    def rank(
        self, prompt: str, candidates: list[dict], evidence: list[str]
    ) -> list[dict]:
        qs = self.encode_query(prompt, evidence).unsqueeze(0)
        scored = []
        for c in candidates:
            cs = self.encoder.encode(
                c["text"], "candidate"
            ).unsqueeze(0)
            score = self.gonogo(qs, cs).item()
            scored.append({**c, "score": score})
        return sorted(scored, key=lambda x: x["score"], reverse=True)

    def learn_preference(
        self,
        prompt: str,
        preferred: dict,
        rejected: dict,
        evidence: list[str],
    ) -> float:
        qs = self.encode_query(prompt, evidence).unsqueeze(0)
        ps = self.encoder.encode(
            preferred["text"], "candidate"
        ).unsqueeze(0)
        rs = self.encoder.encode(
            rejected["text"], "candidate"
        ).unsqueeze(0)
        pref_logit = self.gonogo(qs, ps)
        rej_logit = self.gonogo(qs, rs)
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
        self.encoder.learn_associations(examples)

    def get_state(self) -> dict:
        return {
            "encoder": self.encoder.get_state(),
            "gonogo": self.gonogo.state_dict(),
            "updates": self.updates,
        }

    @classmethod
    def from_state(cls, state: dict) -> "EncoderScorer":
        encoder_state = state["encoder"]
        scorer = cls(
            dimension=encoder_state.get("dimension", 512),
            hidden=64,
        )
        scorer.encoder = VSAEncoder.from_state(encoder_state)
        scorer.gonogo.load_state_dict(state["gonogo"])
        scorer.updates = state.get("updates", 0)
        return scorer

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.get_state(), path)

    @classmethod
    def load(cls, path: Path) -> "EncoderScorer":
        return cls.from_state(
            torch.load(path, map_location="cpu", weights_only=False)
        )
