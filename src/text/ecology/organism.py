"""Response organism: the unit of selection for the response ecosystem.

Phase 1 (see the response-ecosystem plan) only selects among candidates the
existing `FixedSpliceCandidateGenerator` already produces -- no new organism
is synthesized here, so `lineage`/`generation` are mostly inert bookkeeping
for now. They exist so Phase 2's real reproduction/recombination can slot
in without changing this shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ResponseOrganism:
    text: str
    kind: str
    source_ids: tuple[int, ...]
    niche: str = ""
    generation: int = 0
    lineage: tuple[int, ...] = ()
    fitness: dict[str, float] = field(default_factory=dict)

    @property
    def niche_score(self) -> float:
        return self.fitness.get("niche_score", 0.0)
