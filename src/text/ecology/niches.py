"""Functional niches for the response ecosystem.

Each niche is a different answer *shape* preference, expressed purely as a
weighting over `SequenceCandidateScorer`'s existing named features
(`response_candidates.py`) plus a length-based bonus -- no new feature
extraction, per the response-ecosystem plan's Phase 1 scope.

`cautious` is the one niche whose real job (preserving/expressing
uncertainty) needs an uncertainty-qualifier mutation operator that doesn't
exist until Phase 2 -- Phase 1 can only *select* among candidates the
existing generator already produces, none of which hedge. Its weights here
are an honest approximation (favor evidence coverage without committing to
one dominant source) rather than a claim that it expresses real caution.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.text.response_candidates import SequenceCandidateScorer

LEARNED_RANK_FEATURE = "learned_rank"
FEATURE_NAMES = SequenceCandidateScorer.FEATURE_NAMES + (LEARNED_RANK_FEATURE,)


@dataclass(frozen=True)
class Niche:
    name: str
    feature_weights: dict[str, float]
    preferred_tokens: tuple[int, int]
    length_weight: float = 0.5

    def length_bonus(self, token_count: int) -> float:
        low, high = self.preferred_tokens
        if low <= token_count <= high:
            return 1.0
        if token_count < low:
            distance, scale = low - token_count, max(1, low)
        else:
            distance, scale = token_count - high, max(1, high)
        return max(0.0, 1.0 - distance / scale)

    def score(self, feature_values: dict[str, float], token_count: int) -> float:
        weighted = sum(
            weight * feature_values.get(feature, 0.0)
            for feature, weight in self.feature_weights.items()
        )
        return weighted + self.length_weight * self.length_bonus(token_count)


NICHES: dict[str, Niche] = {
    "direct": Niche(
        name="direct",
        feature_weights={
            "query_overlap": 1.0, "evidence_coverage": 0.3, "single_source": 1.0,
            "complete_response": 0.4, "retrieval_priority": 0.8,
            "transition_coherence": 0.2, "length_fit": 0.3, "repetition": -0.6,
            LEARNED_RANK_FEATURE: 0.6,
        },
        preferred_tokens=(1, 12),
        length_weight=1.0,
    ),
    "explanatory": Niche(
        name="explanatory",
        feature_weights={
            "query_overlap": 0.8, "evidence_coverage": 0.8, "single_source": 0.3,
            "complete_response": 0.5, "retrieval_priority": 0.4,
            "transition_coherence": 0.8, "length_fit": 0.5, "repetition": -0.6,
            LEARNED_RANK_FEATURE: 0.6,
        },
        preferred_tokens=(10, 35),
    ),
    "integrative": Niche(
        name="integrative",
        feature_weights={
            "query_overlap": 0.6, "evidence_coverage": 1.2, "single_source": -0.5,
            "complete_response": 0.2, "retrieval_priority": 0.2,
            "transition_coherence": 0.6, "length_fit": 0.4, "repetition": -0.6,
            LEARNED_RANK_FEATURE: 0.5,
        },
        preferred_tokens=(15, 50),
    ),
    "cautious": Niche(
        name="cautious",
        feature_weights={
            "query_overlap": 0.5, "evidence_coverage": 0.7, "single_source": 0.0,
            "complete_response": 0.0, "retrieval_priority": 0.0,
            "transition_coherence": 0.5, "length_fit": 0.6, "repetition": -0.8,
            LEARNED_RANK_FEATURE: 0.4,
        },
        preferred_tokens=(5, 25),
    ),
}
