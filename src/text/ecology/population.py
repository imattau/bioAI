"""Seed and prune a bounded response-organism population.

Phase 1 seeds from `FixedSpliceCandidateGenerator`'s existing output and
only selects among those candidates (no propositions). Phase 2 adds
`seed_proposition_population` (one organism per source's extracted
propositions) and `build_proposition_organism`, the shared "realise, score,
assign niche" step used both for initial proposition seeds and for every
new organism `ecosystem.py`'s succession loop produces via
`operators.py`'s mutation/recombination.
"""

from __future__ import annotations

from src.text.response_candidates import SequenceCandidateScorer

from .fitness import score_organism
from .niches import NICHES
from .organism import ResponseOrganism
from .proposition_extractor import Proposition
from .realiser import PropositionRealiser


def _token_count(text: str) -> int:
    return len(SequenceCandidateScorer._tokens(text))


def _best_fit_niche(
    feature_values: dict[str, float], token_count: int
) -> tuple[str, float]:
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
        text = candidate["text"]
        source_ids = tuple(candidate.get("source_ids", ()))
        fitness = score_organism(
            prompt, text, candidate.get("kind", "unknown"), source_ids,
            (), evidence, [], scorer, sequence_ranker,
        )
        niche_name, niche_score = _best_fit_niche(fitness, _token_count(text))
        fitness["niche_score"] = niche_score
        population.append(ResponseOrganism(
            text=text, kind=candidate.get("kind", "unknown"),
            source_ids=source_ids, niche=niche_name, fitness=fitness,
        ))
    return population


def build_proposition_organism(
    prompt: str,
    propositions: tuple[Proposition, ...],
    evidence: list[str],
    scorer: SequenceCandidateScorer,
    all_propositions: list[Proposition],
    sequence_ranker=None,
    kind: str = "proposition_composition",
    generation: int = 0,
    lineage: tuple[str, ...] = (),
) -> ResponseOrganism:
    text = PropositionRealiser.realise(propositions)
    source_ids = tuple(sorted({
        prop.source_id for prop in propositions if prop.source_id >= 0
    }))
    fitness = score_organism(
        prompt, text, kind, source_ids, propositions, evidence,
        all_propositions, scorer, sequence_ranker,
    )
    niche_name, niche_score = _best_fit_niche(fitness, _token_count(text))
    fitness["niche_score"] = niche_score
    return ResponseOrganism(
        text=text, kind=kind, source_ids=source_ids, niche=niche_name,
        generation=generation, lineage=lineage, propositions=propositions,
        fitness=fitness,
    )


def seed_proposition_population(
    prompt: str,
    propositions: list[Proposition],
    evidence: list[str],
    scorer: SequenceCandidateScorer,
    sequence_ranker=None,
) -> list[ResponseOrganism]:
    """One organism per source: everything a single retrieved memory
    asserted. Cross-source integration isn't pre-seeded -- it emerges
    through `operators.recombine` during succession instead, which is a
    more honest demonstration of real synthesis than pre-building an
    "integrative" seed by hand."""
    if not propositions:
        return []
    by_source: dict[int, list[Proposition]] = {}
    for prop in propositions:
        by_source.setdefault(prop.source_id, []).append(prop)
    return [
        build_proposition_organism(
            prompt, tuple(props), evidence, scorer, propositions, sequence_ranker,
        )
        for props in by_source.values()
    ]


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


def deduplicate_population(
    population: list[ResponseOrganism],
) -> list[ResponseOrganism]:
    seen: set[str] = set()
    unique = []
    for organism in population:
        key = " ".join(organism.text.split()).lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(organism)
    return unique
