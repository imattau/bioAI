"""Tests for Phase 1 of the response-ecosystem plan: ecological *selection*
over FixedSpliceCandidateGenerator's existing output, via niches that
preserve distinct answer shapes across a few survival rounds instead of a
single rank-and-pick-top-1 pass. No propositions, mutation, or
recombination yet -- see src/text/ecology/ docstrings and the
response-ecosystem plan for the full multi-phase design.
"""

import torch

from src.text import BioAIDialogueAgent
from src.text.ecology import NICHES, ResponseEcosystem
from src.text.ecology.niches import LEARNED_RANK_FEATURE
from src.text.ecology.organism import ResponseOrganism
from src.text.ecology.population import seed_population, select_survivors
from src.text.response_candidates import SequenceCandidateScorer


def test_niche_assignment_gives_distinct_winners_for_short_vs_long_evidence():
    eco = ResponseEcosystem()
    prompt = "How do wombats care for their young?"
    evidence = [
        "Wombats are marsupials. Marsupials carry their young in a pouch. "
        "The pouch faces backwards in wombats to avoid dirt while digging.",
        "Wombats are burrowing animals native to Australia. They are "
        "herbivores that eat grasses and roots.",
        "Marsupial young are born at an early stage of development and "
        "continue developing in the pouch.",
    ]
    result = eco.generate(prompt, evidence)

    assert result.winner is not None
    # At least two distinct niches must be populated for this evidence --
    # if everything collapsed into one niche, niches wouldn't be doing
    # anything beyond what a single ranked pool already does.
    assert len(result.niche_winners) >= 2
    texts = {organism.text for organism in result.niche_winners.values()}
    assert len(texts) >= 2


def test_survivors_per_niche_bound_is_respected():
    scorer = SequenceCandidateScorer()
    prompt = "What is the capital of France?"
    evidence = [
        "Paris is the capital of France and it is beautiful.",
        "France is a country in Europe with a rich history and culture.",
    ]
    candidates = [
        {"text": f"Paris is the capital of France, fact {i}.",
         "kind": "complete", "source_ids": [0]}
        for i in range(10)
    ]
    population = seed_population(prompt, candidates, evidence, scorer)
    survivors = select_survivors(population, survivors_per_niche=2)

    for niche_name in NICHES:
        count = sum(1 for organism in survivors if organism.niche == niche_name)
        assert count <= 2


def test_ecosystem_returns_fallback_when_no_candidates():
    eco = ResponseEcosystem()
    result = eco.generate("anything", [], fallback="fallback text")
    assert result.response == "fallback text"
    assert result.winner is None
    assert result.population_sizes == [0]


def test_population_stabilizes_and_stops_early():
    eco = ResponseEcosystem(survivors_per_niche=1, max_rounds=5)
    prompt = "What is the capital of France?"
    evidence = ["Paris is the capital of France and it is beautiful."]
    result = eco.generate(prompt, evidence)
    # Phase 1 never creates new organisms, so pruning an already-pruned
    # population is a no-op -- the loop must stop well short of max_rounds.
    assert len(result.population_sizes) < 5


def test_seed_population_uses_learned_rank_only_once_ranker_has_trained():
    class StubRanker:
        def __init__(self, updates):
            self.updates = updates

        def score(self, prompt, candidate, evidence):
            return 42.0

    scorer = SequenceCandidateScorer()
    candidates = [{"text": "Paris is the capital of France.",
                   "kind": "complete", "source_ids": [0]}]
    evidence = ["Paris is the capital of France."]

    untrained = seed_population(
        "q", candidates, evidence, scorer, sequence_ranker=StubRanker(updates=0)
    )
    assert untrained[0].fitness[LEARNED_RANK_FEATURE] == 0.0

    trained = seed_population(
        "q", candidates, evidence, scorer, sequence_ranker=StubRanker(updates=5)
    )
    assert trained[0].fitness[LEARNED_RANK_FEATURE] == 42.0


def test_organism_niche_score_defaults_to_zero_without_fitness():
    organism = ResponseOrganism(text="x", kind="complete", source_ids=(0,))
    assert organism.niche_score == 0.0


def test_agent_ecological_generation_end_to_end():
    torch.manual_seed(0)  # unseeded VSA vectors make retrieval margin flaky
    agent = BioAIDialogueAgent(vsa_dim=1000)
    agent.enable_ecological_generation()
    agent.process_turn(
        "The weather today is unusually cold for this time of year."
    )
    result = agent.process_turn("What is the weather like today?")

    assert result["response_mode"] == "ecological_generation"
    assert result["response_generated"] is True
    assert result["response"]


def test_agent_ecosystem_state_persists_across_save_load():
    import tempfile
    from pathlib import Path

    agent = BioAIDialogueAgent(vsa_dim=64)
    agent.enable_ecological_generation(survivors_per_niche=3, max_rounds=2)
    path = Path(tempfile.mktemp(suffix=".pt"))
    try:
        agent.save(path)
        restored = BioAIDialogueAgent.load(path)
        assert restored.response_ecosystem is not None
        assert restored.response_ecosystem.survivors_per_niche == 3
        assert restored.response_ecosystem.max_rounds == 2
    finally:
        path.unlink(missing_ok=True)


def test_agent_without_ecological_generation_has_none_after_save_load():
    import tempfile
    from pathlib import Path

    agent = BioAIDialogueAgent(vsa_dim=64)
    path = Path(tempfile.mktemp(suffix=".pt"))
    try:
        agent.save(path)
        restored = BioAIDialogueAgent.load(path)
        assert restored.response_ecosystem is None
    finally:
        path.unlink(missing_ok=True)
