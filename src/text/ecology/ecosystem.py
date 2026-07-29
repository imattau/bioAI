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

Phase 9's `evidence_propositions` parameter lets a caller supply
propositions directly, bypassing `extract_propositions(evidence)` --
isolating generation quality from extraction accuracy, which every
earlier phase's evaluation conflated by always deriving propositions
from evidence text via the same regex extractor under test elsewhere.
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


def _supported_proposition_count(text: str, all_propositions: list) -> int:
    """How many of the evidence's known propositions this organism's
    *realised text* actually asserts -- computed the same way for every
    organism kind (raw candidate or proposition genotype) by re-extracting
    from the text itself, not by reading `organism.propositions` (which is
    always empty for Phase-1-style raw candidates even when their text
    happens to state the same facts). Used to pick the final winner across
    niches; see the response-ecosystem plan's Phase 3 finding: comparing
    raw `niche_score` across niches with different weight vectors is
    apples-to-oranges and let a short "direct" answer win even when an
    "integrative"/"explanatory" niche winner correctly composed more of
    the evidence."""
    if not all_propositions or not text:
        return 0
    asserted = {
        (prop.subject, prop.relation, prop.object)
        for prop in extract_propositions([text])
    }
    known = {
        (prop.subject, prop.relation, prop.object) for prop in all_propositions
    }
    return len(asserted & known)


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
    frame_library=None,
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
            lineage=child.lineage, frame_library=frame_library,
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
        enable_synthesis: bool = True,
    ):
        self.candidate_generator = (
            candidate_generator or FixedSpliceCandidateGenerator()
        )
        self.scorer = scorer or SequenceCandidateScorer()
        self.survivors_per_niche = survivors_per_niche
        self.max_rounds = max_rounds
        # False reproduces Phase 1 exactly: ecological *selection* only,
        # over FixedSpliceCandidateGenerator's existing candidates, no
        # proposition genotype or mutation/recombination. Exists so
        # experiments/ecology_benchmark.py can isolate that data point from
        # full Phase 2 synthesis in the same class, rather than needing a
        # second, drifting implementation to compare against.
        self.enable_synthesis = enable_synthesis

    def generate(
        self,
        prompt: str,
        evidence: list[str],
        composer=None,
        sequence_ranker=None,
        relational_memory=None,
        frame_library=None,
        fallback: str = "",
        evidence_propositions: list | None = None,
    ) -> EcosystemResult:
        candidates = self.candidate_generator.generate(prompt, evidence, composer)
        if evidence_propositions is not None:
            # Phase 9: bypass extract_propositions(evidence) entirely when
            # the caller already has structured propositions (e.g. a
            # teacher's frozen labels) -- isolates "can this compose
            # fluent text from known facts" from "can it parse them out
            # of free text first," which every earlier proof in this plan
            # conflated by always re-deriving propositions from evidence
            # text via the same regex extractor being tested elsewhere.
            propositions = evidence_propositions if self.enable_synthesis else []
        else:
            propositions = (
                extract_propositions(evidence) if self.enable_synthesis else []
            )

        population = seed_population(
            prompt, candidates, evidence, self.scorer, sequence_ranker
        )
        if self.enable_synthesis:
            population += seed_proposition_population(
                prompt, propositions, evidence, self.scorer, sequence_ranker,
                frame_library=frame_library,
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
            if self.enable_synthesis:
                # Genotype organisms are breeding stock, not just competing
                # finalists: a short single-proposition organism can easily
                # lose its niche slot to an already-good raw candidate
                # before ever getting a chance to recombine with another
                # source's genotype organism -- which would silently defeat
                # the whole point of Phase 2 for exactly the short,
                # single-fact-per-source scenarios it's meant to help with.
                # So every surviving genotype organism still in the
                # population gets to participate in reproduction this
                # round, whether or not it won a niche slot.
                survivor_ids = {id(organism) for organism in survivors}
                reproduction_pool = survivors + [
                    organism for organism in population
                    if organism.propositions and id(organism) not in survivor_ids
                ]
                offspring = _reproduce(
                    reproduction_pool, propositions, prompt, evidence,
                    self.scorer, sequence_ranker, relational_memory, round_index,
                    frame_library=frame_library,
                )
            else:
                offspring = []
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
            survivors,
            key=lambda organism: (
                _supported_proposition_count(organism.text, propositions),
                organism.niche_score,
            ),
            default=None,
        )
        response = winner.text if winner is not None else fallback
        return EcosystemResult(
            response=response,
            winner=winner,
            niche_winners=niche_winners,
            population_sizes=population_sizes,
        )
