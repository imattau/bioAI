"""Seed and prune a bounded response-organism population.

Phase 1 seeds directly from `FixedSpliceCandidateGenerator`'s existing
output and only selects among those candidates across a few rounds -- no
new organism is synthesized (see `organism.py`/`niches.py` and the
response-ecosystem plan's Phase 1 scope).
"""

from __future__ import annotations

from src.text.response_candidates import SequenceCandidateScorer

from .niches import LEARNED_RANK_FEATURE, NICHES
from .organism import ResponseOrganism


def _token_count(text: str) -> int:
    return len(SequenceCandidateScorer._tokens(text))


def _best_fit_niche(feature_values: dict[str, float], token_count: int) -> tuple[str, float]:
    best_name, best_score = None, float("-inf")
    for name, niche in NICHES.items():
        score = niche.score(feature_values, token_count)
        if score > best_score:
            best_name, best_score = name, score
    return best_name, best_score


def seed_population(
    prompt: str,
    candidates: list[dict],
    evidence: list[str],
    scorer: SequenceCandidateScorer,
    sequence_ranker=None,
) -> list[ResponseOrganism]:
    """Wrap each already-generated candidate dict as an organism, scoring
    it against every niche and assigning it to whichever niche it fits
    best (not simply the highest-scoring niche overall for every
    candidate -- that would collapse every organism into one niche)."""
    population = []
    for candidate in candidates:
        features = scorer.features(prompt, candidate, evidence)
        fitness = dict(zip(SequenceCandidateScorer.FEATURE_NAMES, features))
        if sequence_ranker is not None and sequence_ranker.updates > 0:
            fitness[LEARNED_RANK_FEATURE] = sequence_ranker.score(
                prompt, candidate, evidence
            )
        else:
            fitness[LEARNED_RANK_FEATURE] = 0.0
        token_count = _token_count(candidate["text"])
        niche_name, niche_score = _best_fit_niche(fitness, token_count)
        fitness["niche_score"] = niche_score
        population.append(ResponseOrganism(
            text=candidate["text"],
            kind=candidate.get("kind", "unknown"),
            source_ids=tuple(candidate.get("source_ids", ())),
            niche=niche_name,
            fitness=fitness,
        ))
    return population


def select_survivors(
    population: list[ResponseOrganism], survivors_per_niche: int = 2
) -> list[ResponseOrganism]:
    """Keep the top `survivors_per_niche` organisms per niche, so a short
    extractive candidate can't eliminate every longer candidate just
    because a single undifferentiated pool would have ranked it higher."""
    survivors = []
    for niche_name in NICHES:
        members = sorted(
            (organism for organism in population if organism.niche == niche_name),
            key=lambda organism: organism.niche_score,
            reverse=True,
        )
        survivors.extend(members[:survivors_per_niche])
    return survivors
