"""Conservative mutation and recombination operators for proposition-
genotype organisms (Phase 2 of the response-ecosystem plan). Each operator
returns a new `ResponseOrganism` (with `text=""`, to be realised and
re-scored by the caller -- see `population.build_proposition_organism`) or
`None` when it doesn't apply. `None` means "not applicable," not
"failure" -- callers should skip it, not treat it as an error.
"""

from __future__ import annotations

import re
from dataclasses import replace

from .organism import ResponseOrganism
from .proposition_extractor import Proposition, UNCERTAINTY_RELATION

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def _relevance(prompt_tokens: set[str], prop: Proposition) -> float:
    prop_tokens = _tokens(prop.subject) | _tokens(prop.object)
    if not prop_tokens:
        return 0.0
    return len(prompt_tokens & prop_tokens) / len(prop_tokens)


def _has_internal_conflict(propositions: tuple[Proposition, ...]) -> bool:
    seen: dict[tuple[str, str], str] = {}
    for prop in propositions:
        key = (prop.subject, prop.relation)
        if key in seen and seen[key] != prop.object:
            return True
        seen[key] = prop.object
    return False


def _conflicts_between(
    props_a: tuple[Proposition, ...], props_b: tuple[Proposition, ...]
) -> bool:
    lookup: dict[tuple[str, str], set[str]] = {}
    for prop in props_a:
        lookup.setdefault((prop.subject, prop.relation), set()).add(prop.object)
    for prop in props_b:
        existing = lookup.get((prop.subject, prop.relation))
        if existing and prop.object not in existing:
            return True
    return False


def is_compatible(
    organism_a: ResponseOrganism,
    organism_b: ResponseOrganism,
    relational_memory=None,
) -> bool:
    """Reject a merge if the two organisms assert different objects for
    the same (subject, relation) pair -- checked directly between the two
    organisms' own propositions first (always available), then against
    RelationalMemory's broader stored knowledge when one is supplied, per
    the response-ecosystem plan's Phase 2 recombination gating."""
    if _conflicts_between(organism_a.propositions, organism_b.propositions):
        return False
    if relational_memory is None:
        return True
    for prop in organism_a.propositions + organism_b.propositions:
        ties = relational_memory.ground_truth_ambiguity(
            {"subject": prop.subject, "relation": prop.relation}
        )
        conflicting = {obj for _, _, obj in ties if obj != prop.object}
        if conflicting:
            return False
    return True


def survives_predation(organism: ResponseOrganism) -> bool:
    """Reject an organism with lost provenance, an internal contradiction,
    or a realised phenotype with an adjacent duplicate token (the real
    "Earth is is..." class of bug found via LLM fuzzing earlier in this
    branch's history -- see the response-ecosystem plan)."""
    if any(prop.source_id < 0 for prop in organism.propositions):
        return False
    if _has_internal_conflict(organism.propositions):
        return False
    if organism.text:
        words = organism.text.lower().split()
        if any(a == b for a, b in zip(words, words[1:])):
            return False
    return True


def add_supported_proposition(
    organism: ResponseOrganism, available: list[Proposition], prompt: str
) -> ResponseOrganism | None:
    if not organism.propositions:
        return None
    prompt_tokens = _tokens(prompt)
    existing_subjects = {prop.subject for prop in organism.propositions}
    existing_objects = {prop.object for prop in organism.propositions}
    existing = set(organism.propositions)
    best, best_score = None, 0.0
    for prop in available:
        if prop in existing:
            continue
        if prop.subject not in existing_subjects and prop.object not in existing_objects:
            continue
        if _conflicts_between(organism.propositions, (prop,)):
            continue
        score = _relevance(prompt_tokens, prop)
        if score > best_score:
            best, best_score = prop, score
    if best is None:
        return None
    return replace(
        organism,
        propositions=organism.propositions + (best,),
        text="", kind="mutated_add", generation=organism.generation + 1,
        lineage=organism.lineage + (organism.kind,),
    )


def remove_low_relevance_proposition(
    organism: ResponseOrganism, prompt: str
) -> ResponseOrganism | None:
    if len(organism.propositions) <= 1:
        return None
    prompt_tokens = _tokens(prompt)
    weakest = min(
        organism.propositions, key=lambda prop: _relevance(prompt_tokens, prop)
    )
    return replace(
        organism,
        propositions=tuple(p for p in organism.propositions if p != weakest),
        text="", kind="mutated_remove", generation=organism.generation + 1,
        lineage=organism.lineage + (organism.kind,),
    )


def reorder_for_coherence(
    organism: ResponseOrganism, prompt: str
) -> ResponseOrganism | None:
    if len(organism.propositions) <= 1:
        return None
    prompt_tokens = _tokens(prompt)
    reordered = tuple(sorted(
        organism.propositions,
        key=lambda prop: _relevance(prompt_tokens, prop),
        reverse=True,
    ))
    if reordered == organism.propositions:
        return None
    return replace(
        organism, propositions=reordered, text="",
        kind="mutated_reorder", generation=organism.generation + 1,
        lineage=organism.lineage + (organism.kind,),
    )


def compress_shared_subject(organism: ResponseOrganism) -> ResponseOrganism | None:
    """Reorders propositions so same-subject entries are adjacent. With
    today's realiser (which already groups by (subject, relation)
    regardless of order) this mostly changes clause *order*, not content
    -- an honest, disclosed limitation pending Phase 4's frame-based
    realiser, which is where adjacency will actually start mattering."""
    if len(organism.propositions) <= 1:
        return None
    order: list[str] = []
    for prop in organism.propositions:
        if prop.subject not in order:
            order.append(prop.subject)
    reordered = tuple(
        sorted(organism.propositions, key=lambda prop: order.index(prop.subject))
    )
    if reordered == organism.propositions:
        return None
    return replace(
        organism, propositions=reordered, text="",
        kind="mutated_compress", generation=organism.generation + 1,
        lineage=organism.lineage + (organism.kind,),
    )


def add_uncertainty_qualifier(
    organism: ResponseOrganism, relational_memory
) -> ResponseOrganism | None:
    if relational_memory is None or not organism.propositions:
        return None
    for prop in organism.propositions:
        if prop.relation == UNCERTAINTY_RELATION:
            continue
        ties = relational_memory.ground_truth_ambiguity(
            {"subject": prop.subject, "relation": prop.relation}
        )
        objects = sorted({obj for _, _, obj in ties} | {prop.object})
        if len(objects) > 1:
            qualifier = Proposition(
                subject=prop.subject, relation=UNCERTAINTY_RELATION,
                object=" or ".join(objects),
                source_id=prop.source_id, source_text=prop.source_text,
            )
            return replace(
                organism, propositions=organism.propositions + (qualifier,),
                text="", kind="mutated_uncertainty",
                generation=organism.generation + 1,
                lineage=organism.lineage + (organism.kind,),
            )
    return None


def recombine(
    organism_a: ResponseOrganism,
    organism_b: ResponseOrganism,
    relational_memory=None,
) -> ResponseOrganism | None:
    if not organism_a.propositions or not organism_b.propositions:
        return None
    if not is_compatible(organism_a, organism_b, relational_memory):
        return None
    merged = tuple(dict.fromkeys(organism_a.propositions + organism_b.propositions))
    if merged == organism_a.propositions or merged == organism_b.propositions:
        return None
    return ResponseOrganism(
        text="", kind="recombined",
        source_ids=tuple(sorted(
            set(organism_a.source_ids) | set(organism_b.source_ids)
        )),
        propositions=merged,
        generation=max(organism_a.generation, organism_b.generation) + 1,
        lineage=organism_a.lineage + organism_b.lineage,
    )
