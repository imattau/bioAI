"""Functional niches for the response ecosystem.

Each niche is a different answer *shape* preference, expressed as a
weighting over `SequenceCandidateScorer`'s existing named features
(`response_candidates.py`), Phase 2's proposition-level fitness terms
(`coverage`, `grounding`, `contradiction`, `uncertainty_expressed` --
see `fitness.py`), plus a length-based bonus.

`cautious`'s weight on `uncertainty_expressed` is the honest resolution of
a gap disclosed in Phase 1: that niche could only *select* among
candidates the generator already produced, none of which hedge. Phase 2's
`add_uncertainty_qualifier` mutation actually produces a hedging variant
when `RelationalMemory.ground_truth_ambiguity` shows a genuine collision,
and this weight is what makes `cautious` prefer it once it exists.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.text.response_candidates import SequenceCandidateScorer

LEARNED_RANK_FEATURE = "learned_rank"
PROPOSITION_FEATURES = (
    "coverage", "grounding", "contradiction", "uncertainty_expressed",
)
FEATURE_NAMES = (
    SequenceCandidateScorer.FEATURE_NAMES
    + (LEARNED_RANK_FEATURE,)
    + PROPOSITION_FEATURES
)


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
            "query_overlap": 1.0, "evidence_coverage": 0.3, "single_source": 1.5,
            "complete_response": 0.4, "retrieval_priority": 0.2,
            "transition_coherence": 0.2, "length_fit": 0.3, "repetition": -0.6,
            LEARNED_RANK_FEATURE: 0.6,
            "coverage": 0.3, "grounding": 1.0, "contradiction": -2.0,
        },
        preferred_tokens=(1, 8),
        length_weight=1.0,
    ),
    "explanatory": Niche(
        name="explanatory",
        feature_weights={
            "query_overlap": 0.8, "evidence_coverage": 0.8, "single_source": 0.3,
            "complete_response": 0.5, "retrieval_priority": 0.4,
            "transition_coherence": 0.8, "length_fit": 0.5, "repetition": -0.6,
            LEARNED_RANK_FEATURE: 0.6,
            "coverage": 0.8, "grounding": 1.0, "contradiction": -2.0,
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
            "coverage": 1.2, "grounding": 1.0, "contradiction": -2.0,
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
            "coverage": 0.6, "grounding": 1.0, "contradiction": -2.0,
            "uncertainty_expressed": 1.5,
        },
        preferred_tokens=(5, 25),
    ),
}
