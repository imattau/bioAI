from pathlib import Path
import tempfile

import torch

from src.text import BioAIDialogueAgent, ConsolidationMemory, TokenLibrary


def test_concept_requires_repeated_evidence_and_queries_sources():
    memory = ConsolidationMemory(promotion_threshold=3)
    vectors = [
        torch.tensor([1, 1, -1, -1], dtype=torch.int8),
        torch.tensor([1, 1, -1, 1], dtype=torch.int8),
        torch.tensor([1, 1, -1, -1], dtype=torch.int8),
    ]
    assert not memory.observe_concept("river", vectors[0], 0)
    assert "river" not in memory.concepts
    assert memory.observe_concept(
        "river", vectors[-1], 2,
        initial_vectors=vectors,
        initial_source_ids=[0, 1, 2],
    )
    result = memory.query_concepts(vectors[0], limit=1)[0]
    assert result["term"] == "river"
    assert result["count"] == 3
    assert result["source_ids"] == [0, 1, 2]


def test_prototype_index_tracks_updates():
    memory = ConsolidationMemory(promotion_threshold=1)
    left = torch.tensor([1, 1, -1, -1, 1, 1, -1, -1], dtype=torch.int8)
    right = -left
    memory.observe_concept(
        "left", left, 0, initial_vectors=[left], initial_source_ids=[0]
    )
    assert memory.query_concepts(left, limit=1)[0]["term"] == "left"
    memory.observe_concept(
        "right", right, 1, initial_vectors=[right], initial_source_ids=[1]
    )
    assert memory.query_concepts(right, limit=1)[0]["term"] == "right"


def test_relation_conflicts_are_counted_not_overwritten():
    memory = ConsolidationMemory()
    memory.observe_relation("The capital of Australia is Canberra", 0)
    memory.observe_relation("The capital of Australia is Canberra", 1)
    memory.observe_relation("The capital of Australia is Sydney", 2)
    claims = memory.relation_claims("Australia", "capital")
    assert [(claim["object"], claim["count"]) for claim in claims] == [
        ("canberra", 2),
        ("sydney", 1),
    ]
    assert claims[0]["source_ids"] == [0, 1]
    assert memory.relation_candidates(
        "What is the capital of Australia?"
    ) == [(0, 2 / 3), (1, 2 / 3), (2, 1 / 3)]


def test_consolidation_and_semantic_vectors_round_trip():
    library = TokenLibrary()
    vector = torch.tensor([1, -1, 1, -1, -1, 1, -1, 1], dtype=torch.int8)
    sentence_id = library.add("river evidence", vector)
    assert torch.equal(library.semantic_vector(sentence_id), vector)

    memory = ConsolidationMemory(promotion_threshold=1)
    memory.observe_concept(
        "river", vector, sentence_id,
        initial_vectors=[vector],
        initial_source_ids=[sentence_id],
    )
    restored = ConsolidationMemory.from_state(memory.get_state())
    assert restored.query_concepts(vector, limit=1)[0]["term"] == "river"


class _SemanticStub:
    def encode(self, text):
        text = text.lower()
        if "river" in text or "waterway" in text:
            return torch.tensor([1, 1, -1, -1, 1, -1, 1, -1], dtype=torch.int8)
        return torch.tensor([-1, -1, 1, 1, -1, 1, -1, 1], dtype=torch.int8)


def test_agent_promotes_repeated_concepts():
    agent = BioAIDialogueAgent(vsa_dim=64)
    agent.semantic_encoder = _SemanticStub()
    agent.process_turn("A river carries water")
    agent.process_turn("This river crosses the valley")
    agent.process_turn("The river supports wildlife")
    assert agent.consolidation.concepts["river"]["count"] == 3
    match = agent.consolidation.query_concepts(
        agent.semantic_encoder.encode("waterway"), limit=1
    )[0]
    assert match["term"] == "river"
    assert match["source_ids"] == [0, 1, 2]


def test_relations_survive_agent_save_load():
    agent = BioAIDialogueAgent(vsa_dim=64)
    agent.process_turn("The capital of France is Paris")
    path = Path(tempfile.mktemp(suffix=".pt"))
    try:
        agent.save(path)
        restored = BioAIDialogueAgent.load(path)
        claims = restored.consolidation.relation_claims("France", "capital")
        assert claims[0]["object"] == "paris"
        assert claims[0]["source_ids"] == [0]
    finally:
        path.unlink(missing_ok=True)


def test_agent_uses_consolidated_relation_evidence():
    agent = BioAIDialogueAgent(vsa_dim=64)
    agent.process_turn("The capital of France is Paris")
    result = agent.process_turn("What is the capital of France?")
    assert result["retrieval_accepted"]
    assert result["response"] == "The capital of France is Paris"
    assert result["sources"][0]["id"] == 0


def test_agent_composes_two_hop_answer_with_evidence():
    agent = BioAIDialogueAgent(vsa_dim=64)
    agent.process_turn("Canberra is the capital of Australia")
    agent.process_turn("Australia is located in Oceania")
    result = agent.process_turn("Which continent is Canberra in?")
    assert result["response"] == "canberra is in oceania."
    assert result["response_mode"] == "consolidated_reasoning"
    assert result["reasoning"]["path"] == ["canberra", "australia", "oceania"]
    assert [source["id"] for source in result["sources"]] == [0, 1]


def test_reasoning_abstains_on_tied_conflict():
    memory = ConsolidationMemory()
    memory.observe_relation("Springfield is in Illinois", 0)
    memory.observe_relation("Springfield is in Massachusetts", 1)
    assert memory.reason_path("Where is Springfield?") is None
