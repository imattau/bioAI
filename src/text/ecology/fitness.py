"""Fitness scoring shared by Phase 1 raw-candidate organisms and Phase 2
proposition-genotype organisms.

Combines `SequenceCandidateScorer`'s existing 8 features (computed on the
*realised* text) with proposition-level signals when propositions exist:
coverage (fraction of all extracted propositions this organism
represents), grounding (fraction of its propositions with valid
provenance), contradiction (an internal (subject, relation) conflict),
and uncertainty_expressed (whether it includes a hedging proposition --
see `operators.add_uncertainty_qualifier`). Every component stays visible
in the returned dict; per the response-ecosystem plan, collapsing these
into one scalar immediately would hide why an organism survived.
"""

from __future__ import annotations

from src.text.response_candidates import SequenceCandidateScorer

from .niches import LEARNED_RANK_FEATURE
from .operators import _has_internal_conflict
from .proposition_extractor import Proposition, UNCERTAINTY_RELATION


def score_organism(
    prompt: str,
    text: str,
    kind: str,
    source_ids: tuple[int, ...],
    propositions: tuple[Proposition, ...],
    evidence: list[str],
    all_propositions: list[Proposition],
    scorer: SequenceCandidateScorer,
    sequence_ranker=None,
) -> dict[str, float]:
    candidate = {"text": text, "kind": kind, "source_ids": list(source_ids)}
    features = scorer.features(prompt, candidate, evidence)
    fitness = dict(zip(SequenceCandidateScorer.FEATURE_NAMES, features))
    if sequence_ranker is not None and sequence_ranker.updates > 0:
        fitness[LEARNED_RANK_FEATURE] = sequence_ranker.score(
            prompt, candidate, evidence
        )
    else:
        fitness[LEARNED_RANK_FEATURE] = 0.0

    if propositions:
        represented = {(p.subject, p.relation, p.object) for p in propositions}
        total = (
            {(p.subject, p.relation, p.object) for p in all_propositions}
            or represented
        )
        fitness["coverage"] = len(represented & total) / max(1, len(total))
        fitness["grounding"] = sum(
            1 for p in propositions if p.source_id >= 0
        ) / len(propositions)
        fitness["contradiction"] = (
            1.0 if _has_internal_conflict(propositions) else 0.0
        )
        fitness["uncertainty_expressed"] = 1.0 if any(
            p.relation == UNCERTAINTY_RELATION for p in propositions
        ) else 0.0
    else:
        # Phase-1-style raw candidates: the text itself IS the source, so
        # it's trivially fully "grounded"; coverage/contradiction don't
        # apply without a proposition genotype to measure them against.
        fitness["coverage"] = 0.0
        fitness["grounding"] = 1.0
        fitness["contradiction"] = 0.0
        fitness["uncertainty_expressed"] = 0.0
    return fitness
