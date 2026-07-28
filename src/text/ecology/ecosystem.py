"""Response ecosystem: Phase 1 -- ecological *selection* over the existing
`FixedSpliceCandidateGenerator` output.

See the response-ecosystem plan for the full multi-phase design. Phase 1
tests whether keeping multiple answer-shape niches alive across a few
survival rounds beats a single rank-and-pick-top-1 pass, using only
candidates the system already knows how to generate -- no propositions, no
mutation, no recombination yet (Phase 2). Because Phase 1 never creates a
new organism, a second round of selection over an already-pruned
population is idempotent by construction; `generate()` runs until the
population stops shrinking (typically one round) rather than always
running a fixed number of no-op rounds. This makes the round loop an
honest scaffold for Phase 2's real reproduction step, not decoration.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.text.response_candidates import (
    FixedSpliceCandidateGenerator,
    SequenceCandidateScorer,
)

from .organism import ResponseOrganism
from .population import seed_population, select_survivors


@dataclass
class EcosystemResult:
    response: str
    winner: ResponseOrganism | None
    niche_winners: dict[str, ResponseOrganism] = field(default_factory=dict)
    population_sizes: list[int] = field(default_factory=list)


class ResponseEcosystem:
    def __init__(
        self,
        candidate_generator: FixedSpliceCandidateGenerator | None = None,
        scorer: SequenceCandidateScorer | None = None,
        survivors_per_niche: int = 2,
        max_rounds: int = 3,
    ):
        self.candidate_generator = (
            candidate_generator or FixedSpliceCandidateGenerator()
        )
        self.scorer = scorer or SequenceCandidateScorer()
        self.survivors_per_niche = survivors_per_niche
        self.max_rounds = max_rounds

    def generate(
        self,
        prompt: str,
        evidence: list[str],
        composer=None,
        sequence_ranker=None,
        fallback: str = "",
    ) -> EcosystemResult:
        candidates = self.candidate_generator.generate(prompt, evidence, composer)
        if not candidates:
            return EcosystemResult(
                response=fallback, winner=None, population_sizes=[0]
            )

        population = seed_population(
            prompt, candidates, evidence, self.scorer, sequence_ranker
        )
        population_sizes = [len(population)]
        for _ in range(1, self.max_rounds):
            survivors = select_survivors(population, self.survivors_per_niche)
            if len(survivors) == len(population):
                break
            population = survivors
            population_sizes.append(len(population))

        niche_winners: dict[str, ResponseOrganism] = {}
        for organism in population:
            current = niche_winners.get(organism.niche)
            if current is None or organism.niche_score > current.niche_score:
                niche_winners[organism.niche] = organism

        winner = max(population, key=lambda organism: organism.niche_score, default=None)
        response = winner.text if winner is not None else fallback
        return EcosystemResult(
            response=response,
            winner=winner,
            niche_winners=niche_winners,
            population_sizes=population_sizes,
        )
