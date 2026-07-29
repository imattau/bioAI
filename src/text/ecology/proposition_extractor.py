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

Phase 7 addition: compound-predicate splitting. `PropositionRealiser`'s
compound-clause merging (`realiser.py`) can now produce sentences like
"Cows were farm animals and sit in farmyard and have long horns." --
three different relations sharing one subject, joined with "and" rather
than ". ". `ConsolidationMemory.extract_relations`'s catch-all pattern
still *matches* a sentence like this (it just needs some "is/are/was/were"
followed by anything), but garbles it into one bogus object string
instead of failing cleanly -- found via real testing (the frozen-spec
regression check the response-ecosystem plan calls for), not assumed.
Fixed by detecting the specific shape compound-clause merging produces
(every "and"-segment *after* the first itself starts with a recognizable
verb -- "sit in farmyard", "have long horns") and reconstructing/
extracting each conjunct separately, while leaving alone the pre-existing,
legitimate case of one relation with multiple "and"-joined *objects*
("Wombats are marsupials and mammals.", where the second segment
"mammals" does NOT start with a verb) exactly as before.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.text.consolidation import ConsolidationMemory
from src.text.response_candidates import FixedSpliceCandidateGenerator

# Sentinel relation for a synthesized hedging proposition (see
# operators.add_uncertainty_qualifier) -- not something extract_propositions
# ever produces itself.
UNCERTAINTY_RELATION = "_uncertainty"

# Matches the start of a bare verb phrase (no subject) -- "sit(s) in X",
# "lie(s) within X", "has/have X", "is/are/was/were X". A second-or-later
# "and"-segment matching this is the signal that the whole sentence is a
# compound predicate (multiple relations, one shared subject), not a
# single relation with an "and"-joined multi-value object.
_VERB_PHRASE_START = re.compile(
    r"^(?:is|are|was|were|has|have|sits?|lies?|located)\b", re.IGNORECASE
)


@dataclass(frozen=True)
class Proposition:
    subject: str
    relation: str
    object: str
    source_id: int
    source_text: str


def _extract_compound(sentence: str) -> list[tuple[str, str, str]]:
    """Try compound-predicate splitting; return [] if the sentence isn't
    that shape (caller falls back to plain `extract_relations`)."""
    stripped = sentence.strip().rstrip(".!?")
    segments = re.split(r"\s+and\s+", stripped)
    if len(segments) < 2:
        return []
    if not all(_VERB_PHRASE_START.match(segment.strip()) for segment in segments[1:]):
        return []
    first = ConsolidationMemory.extract_relations(segments[0])
    if not first:
        return []
    subject = first[0][0]
    triples = list(first)
    for segment in segments[1:]:
        triples.extend(ConsolidationMemory.extract_relations(f"{subject} {segment}"))
    return triples if len(triples) > 1 else []


def extract_propositions(memories: list[str]) -> list[Proposition]:
    propositions: list[Proposition] = []
    for source_id, memory in enumerate(memories):
        for sentence in FixedSpliceCandidateGenerator._sentences(memory):
            triples = _extract_compound(sentence) or ConsolidationMemory.extract_relations(sentence)
            for subject, relation, obj in triples:
                propositions.append(Proposition(
                    subject=subject, relation=relation, object=obj,
                    source_id=source_id, source_text=sentence,
                ))
    return propositions
