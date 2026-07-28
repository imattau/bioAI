"""Extract propositions (subject, relation, object triples with
provenance) from retrieved memories -- the genotype for Phase 2 of the
response-ecosystem plan.

Reuses, rather than reimplements, two existing pieces:
`FixedSpliceCandidateGenerator._sentences` for sentence splitting, and
`ConsolidationMemory.extract_relations` for the actual (subject, relation,
object) extraction. That extractor is regex-based and returns at most one
triple per sentence, first-pattern-wins -- sentences it can't parse (e.g.
"Marsupials carry their young in a pouch", which has no copula any of its
5 patterns match) yield zero propositions. That content isn't silently
dropped: `FixedSpliceCandidateGenerator`'s own raw-candidate output already
covers it in parallel (see `ecosystem.py`), so the ecosystem degrades to
Phase 1 behavior for anything this extractor can't decompose.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.text.consolidation import ConsolidationMemory
from src.text.response_candidates import FixedSpliceCandidateGenerator

# Sentinel relation for a synthesized hedging proposition (see
# operators.add_uncertainty_qualifier) -- not something extract_propositions
# ever produces itself.
UNCERTAINTY_RELATION = "_uncertainty"


@dataclass(frozen=True)
class Proposition:
    subject: str
    relation: str
    object: str
    source_id: int
    source_text: str


def extract_propositions(memories: list[str]) -> list[Proposition]:
    propositions: list[Proposition] = []
    for source_id, memory in enumerate(memories):
        for sentence in FixedSpliceCandidateGenerator._sentences(memory):
            for subject, relation, obj in ConsolidationMemory.extract_relations(sentence):
                propositions.append(Proposition(
                    subject=subject, relation=relation, object=obj,
                    source_id=source_id, source_text=sentence,
                ))
    return propositions
