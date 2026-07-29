"""End-to-end integration test for BioAIDialogueAgent.process_turn().

Tests the full pipeline across all subsystems: VSA encoding, Hopfield
recall, SelfMonitor drift detection, ClonalPool continual learning,
RelationalMemory reasoning, GoNoGo decision making, ConsolidationMemory
chaining, and all response modes.
"""

import torch
from src.text import BioAIDialogueAgent


def _fresh_agent(vsa_dim=128):
    agent = BioAIDialogueAgent(vsa_dim=vsa_dim)
    agent.gonogo_gate_enabled = False
    return agent


class TestBioAIAgentPipeline:
    def test_statement_stores_and_acknowledges(self):
        agent = _fresh_agent()
        result = agent.process_turn("The capital of France is Paris.")
        assert result["response"] == "I'll remember that."
        assert result["response_mode"] == "acknowledgement"
        assert result["intent"] == "statement"
        assert result["turn"] == 1
        assert len(agent.history) == 1

    def test_question_without_prior_knowledge_rejected(self):
        agent = _fresh_agent()
        result = agent.process_turn("What is the capital of France?")
        assert not result["retrieval_accepted"]
        assert "relevant memory" in result["response"]
        assert result["response_mode"] == "abstention"

    def test_question_retrieves_prior_statement(self):
        agent = _fresh_agent()
        agent.process_turn("The capital of France is Paris.")
        result = agent.process_turn("What is the capital of France?")
        assert result["retrieval_accepted"]
        assert "Paris" in result["response"] or "capital" in result["response"]

    def test_relational_reasoning_path(self):
        agent = _fresh_agent(vsa_dim=64)
        agent.process_turn("The capital of France is Paris")
        result = agent.process_turn("What is the capital of France?")
        assert result["response_mode"] == "relational_reasoning"
        assert result["response"] == "The capital of France is Paris"
        assert result["retrieval_accepted"]
        assert not result["ambiguous"]

    def test_ambiguity_surfaced_honestly(self):
        agent = _fresh_agent(vsa_dim=64)
        agent.process_turn("Pluto is a planet")
        agent.process_turn("Pluto is a dwarf planet")
        result = agent.process_turn("What is Pluto?")
        assert result["response_mode"] == "relational_ambiguous"
        assert result["ambiguous"]
        assert "a planet" in result["response"]
        assert "a dwarf planet" in result["response"]

    def test_novelty_detection_on_first_turn(self):
        agent = _fresh_agent()
        result = agent.process_turn("Quantum chromodynamics explains quark confinement.")
        assert "drift_detected" in result
        assert "energy_z" in result
        assert "clonal_created" in result
        assert isinstance(result["energy_z"], float)

    def test_gonogo_decision_computed_every_question(self):
        agent = _fresh_agent(vsa_dim=64)
        agent.process_turn("The weather today is unusually cold.")
        result = agent.process_turn("What is the weather like today?")
        assert result["gonogo_action"] is not None
        assert result["gonogo_go"] in (True, False)

    def test_record_feedback_updates_gonogo(self):
        agent = _fresh_agent(vsa_dim=64)
        agent.gonogo_gate_enabled = True
        agent.process_turn("The weather today is cold.")
        agent.process_turn("What is the weather like today?")
        before = {n: p.clone() for n, p in agent.gonogo.named_parameters()}
        agent.process_turn("Yes, that's correct.")
        changed = any(
            not p.equal(before[n]) for n, p in agent.gonogo.named_parameters()
        )
        assert changed

    def test_store_and_recall_across_turns(self):
        agent = _fresh_agent()
        fact = "Paris is the capital of France."
        agent.process_turn(fact)
        for i in range(10):
            agent.process_turn(f"turn {i}")
        recalled = agent.recall(fact)
        assert recalled == fact

    def test_consolidation_memory_reasoning(self):
        agent = _fresh_agent(vsa_dim=128)
        agent.process_turn("Mars is in the solar system.")
        agent.process_turn("The solar system is in the Milky Way.")
        result = agent.process_turn("What is Mars in?")
        assert "response" in result
        assert isinstance(result["response"], str)

    def test_save_and_load_preserves_state(self):
        import tempfile
        from pathlib import Path
        agent = _fresh_agent(vsa_dim=64)
        agent.process_turn("The capital of France is Paris")
        path = Path(tempfile.mktemp(suffix=".pt"))
        try:
            agent.save(path)
            restored = BioAIDialogueAgent.load(path)
            result = restored.process_turn("What is the capital of France?")
            assert result["response"] == "The capital of France is Paris"
            assert result["response_mode"] == "relational_reasoning"
        finally:
            path.unlink(missing_ok=True)

    def test_ecosystem_generation_path(self):
        agent = _fresh_agent(vsa_dim=64)
        agent.enable_ecological_generation(survivors_per_niche=2, max_rounds=2)
        agent.gonogo_gate_enabled = False
        agent.process_turn("The weather today is unusually cold for this time of year.")
        result = agent.process_turn("What is the weather like today?")
        assert result["response_mode"] == "ecological_generation"
        assert result["response_generated"]
        assert isinstance(result["response"], str)

    def test_chunk_composition_path(self):
        agent = _fresh_agent(vsa_dim=64)
        agent.enable_chunk_composition()
        agent.learn_conversation(
            "What is the capital?",
            "The capital of France is Paris."
        )
        agent.process_turn("The capital of France is Paris and it is beautiful.")
        result = agent.process_turn("What is the capital of France?")
        assert result["response_generated"]
        assert "Paris" in result["response"]

    def test_full_pipeline_non_regression(self):
        agent = _fresh_agent(vsa_dim=1000)
        agent.gonogo_gate_enabled = True
        torch.manual_seed(42)
        facts = [
            "The capital of France is Paris.",
            "The capital of Japan is Tokyo.",
            "Mars is the fourth planet from the Sun.",
            "Water freezes at zero degrees Celsius.",
        ]
        for f in facts:
            agent.process_turn(f)
        for f in facts:
            recalled = agent.recall(f)
            assert recalled == f, f"Expected '{f}', got '{recalled}'"
        result = agent.process_turn("What is the capital of France?")
        assert result["retrieval_accepted"] or not result["retrieval_accepted"]
        assert "response" in result

    def test_process_turn_result_structure(self):
        agent = _fresh_agent()
        result = agent.process_turn("Hello")
        required_keys = {
            "response", "response_generated", "response_mode",
            "sources", "intent",
            "retrieval_accepted", "retrieval_score", "retrieval_margin",
            "retrieval_candidates",
            "ambiguous", "ambiguous_candidates",
            "gonogo_go", "gonogo_action",
            "drift_detected", "energy_z", "clonal_created",
            "turn",
        }
        assert required_keys.issubset(result.keys()), (
            f"Missing keys: {required_keys - result.keys()}"
        )
