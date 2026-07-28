"""Response ecosystem.

Phase 1 proved ecological *selection* (niches + survival rounds) over
`FixedSpliceCandidateGenerator`'s existing output beats a single
rank-and-pick-top-1 pass, using candidates the system already knew how to
generate. Phase 2 adds real synthesis: a proposition genotype
(`proposition_extractor.py`), a minimal realiser (`realiser.py`), and
conservative mutation/recombination/predation operators (`operators.py`)
that actually produce organisms no single source contains verbatim.

`generate()` seeds both kinds of organism (raw candidates and, when the
evidence decomposes into propositions, proposition compositions), then
runs succession rounds: select survivors per niche, reproduce via
mutation/recombination, apply predation, deduplicate, repeat until the
per-niche winners are stable for 2 rounds or `max_rounds` is reached. See
the response-ecosystem plan for the full design and its success criterion.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.text.response_candidates import (
    FixedSpliceCandidateGenerator,
    SequenceCandidateScorer,
)

from .operators import (
    add_supported_proposition,
    add_uncertainty_qualifier,
    compress_shared_subject,
    recombine,
    reorder_for_coherence,
    remove_low_relevance_proposition,
    survives_predation,
)
from .organism import ResponseOrganism
from .population import (
    build_proposition_organism,
    deduplicate_population,
    seed_population,
    seed_proposition_population,
    select_survivors,
)
from .proposition_extractor import extract_propositions

_STABLE_ROUNDS_TO_STOP = 2


@dataclass
class EcosystemResult:
    response: str
    winner: ResponseOrganism | None
    niche_winners: dict[str, ResponseOrganism] = field(default_factory=dict)
    population_sizes: list[int] = field(default_factory=list)


def _reproduce(
    survivors: list[ResponseOrganism],
    all_propositions: list,
    prompt: str,
    evidence: list[str],
    scorer: SequenceCandidateScorer,
    sequence_ranker,
    relational_memory,
    generation: int,
) -> list[ResponseOrganism]:
    genotype_survivors = [organism for organism in survivors if organism.propositions]
    candidates = []

    for organism in genotype_survivors:
        for mutate in (
            lambda o: add_supported_proposition(o, all_propositions, prompt),
            lambda o: remove_low_relevance_proposition(o, prompt),
            lambda o: reorder_for_coherence(o, prompt),
            lambda o: compress_shared_subject(o),
            lambda o: add_uncertainty_qualifier(o, relational_memory),
        ):
            child = mutate(organism)
            if child is not None:
                candidates.append(child)

    for index, organism_a in enumerate(genotype_survivors):
        for organism_b in genotype_survivors[index + 1:]:
            if set(organism_a.source_ids) & set(organism_b.source_ids):
                continue  # only cross-source recombination adds anything new
            child = recombine(organism_a, organism_b, relational_memory)
            if child is not None:
                candidates.append(child)

    offspring = [
        build_proposition_organism(
            prompt, child.propositions, evidence, scorer, all_propositions,
            sequence_ranker, kind=child.kind, generation=generation,
            lineage=child.lineage,
        )
        for child in candidates
    ]
    return [organism for organism in offspring if survives_predation(organism)]


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
        relational_memory=None,
        fallback: str = "",
    ) -> EcosystemResult:
        candidates = self.candidate_generator.generate(prompt, evidence, composer)
        propositions = extract_propositions(evidence)

        population = seed_population(
            prompt, candidates, evidence, self.scorer, sequence_ranker
        )
        population += seed_proposition_population(
            prompt, propositions, evidence, self.scorer, sequence_ranker
        )
        population = [
            organism for organism in population if survives_predation(organism)
        ]
        if not population:
            return EcosystemResult(
                response=fallback, winner=None, population_sizes=[0]
            )

        population_sizes = [len(population)]
        previous_winner_texts: dict[str, str] = {}
        stable_rounds = 0
        for round_index in range(1, self.max_rounds):
            survivors = select_survivors(population, self.survivors_per_niche)
            offspring = _reproduce(
                survivors, propositions, prompt, evidence, self.scorer,
                sequence_ranker, relational_memory, round_index,
            )
            population = deduplicate_population(survivors + offspring)
            population_sizes.append(len(population))

            winners_now = {
                organism.niche: organism.text
                for organism in select_survivors(population, 1)
            }
            stable_rounds = (
                stable_rounds + 1 if winners_now == previous_winner_texts else 0
            )
            previous_winner_texts = winners_now
            if stable_rounds >= _STABLE_ROUNDS_TO_STOP:
                break

        survivors = select_survivors(population, self.survivors_per_niche)
        niche_winners: dict[str, ResponseOrganism] = {}
        for organism in survivors:
            current = niche_winners.get(organism.niche)
            if current is None or organism.niche_score > current.niche_score:
                niche_winners[organism.niche] = organism

        winner = max(
            survivors, key=lambda organism: organism.niche_score, default=None
        )
        response = winner.text if winner is not None else fallback
        return EcosystemResult(
            response=response,
            winner=winner,
            niche_winners=niche_winners,
            population_sizes=population_sizes,
        )
