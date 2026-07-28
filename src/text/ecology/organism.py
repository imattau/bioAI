"""Response organism: the unit of selection for the response ecosystem.

Phase 1 (see the response-ecosystem plan) only selects among candidates the
existing `FixedSpliceCandidateGenerator` already produces -- `propositions`
stays empty for those, and `lineage`/`generation` are mostly inert
bookkeeping. Phase 2 adds real reproduction: `propositions` becomes the
genotype for organisms built from `proposition_extractor.extract_propositions`,
`text` is the phenotype rendered by `realiser.PropositionRealiser`, and
`lineage` records the chain of parent `kind` values through mutation/
recombination (see `operators.py`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .proposition_extractor import Proposition


@dataclass
class ResponseOrganism:
    text: str
    kind: str
    source_ids: tuple[int, ...]
    niche: str = ""
    generation: int = 0
    lineage: tuple[str, ...] = ()
    propositions: tuple[Proposition, ...] = ()
    fitness: dict[str, float] = field(default_factory=dict)

    @property
    def niche_score(self) -> float:
        return self.fitness.get("niche_score", 0.0)
