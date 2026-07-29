"""Tests for the warm VSA cache tier (Phase 3).

Verifies that evicted hot-store items can still be retrieved via the
warm VSA cache or re-encoded from cold text storage.
"""

import tempfile
from pathlib import Path

import torch

from src.text import BioAIDialogueAgent


class TestWarmCache:
    def test_warm_cache_populated_on_insert(self):
        agent = BioAIDialogueAgent(vsa_dim=64)
        agent.gonogo_gate_enabled = False
        agent.process_turn("The capital of France is Paris.")
        assert len(agent._warm_vsa_cache) == 1

    def test_warm_cache_lru_eviction(self):
        agent = BioAIDialogueAgent(vsa_dim=64)
        agent.gonogo_gate_enabled = False
        agent._warm_cache_capacity = 10
        for i in range(20):
            agent.process_turn(f"Statement number {i}.")
        assert len(agent._warm_vsa_cache) == 10

    def test_recall_from_warm_cache_after_hot_eviction(self):
        agent = BioAIDialogueAgent(vsa_dim=64)
        agent.gonogo_gate_enabled = False
        fact = "The capital of France is Paris."
        agent.process_turn(fact)
        # Evict from hash_store by filling it
        for i in range(3000):
            agent.hash_store.insert(torch.sign(torch.randn(64)),
                                    torch.sign(torch.randn(64)))
        # The fact's VSA vector should still be in warm cache
        recalled = agent.recall(fact)
        assert recalled == fact, (
            f"Warm cache should preserve recall after hash_store eviction: "
            f"got '{recalled}'"
        )

    def test_warm_cache_survives_save_load(self):
        agent = BioAIDialogueAgent(vsa_dim=64)
        agent.gonogo_gate_enabled = False
        for i in range(50):
            agent.process_turn(f"Fact {i}.")
        path = Path(tempfile.mktemp(suffix=".pt"))
        try:
            agent.save(path)
            restored = BioAIDialogueAgent.load(path)
            assert len(restored._warm_vsa_cache) == 50
            assert restored._warm_cache_capacity == 5000
        finally:
            path.unlink(missing_ok=True)

    def test_warm_cache_capacity_configurable(self):
        agent = BioAIDialogueAgent(vsa_dim=64)
        assert agent._warm_cache_capacity == 5000

    def test_retrieval_still_works_after_thousands_of_turns(self):
        agent = BioAIDialogueAgent(vsa_dim=64)
        agent.gonogo_gate_enabled = False
        agent.process_turn("The capital of France is Paris.")
        for i in range(1000):
            agent.process_turn(f"Conversation turn {i} about random topics.")
        # Should still find it via cold text + warm cache
        result = agent.process_turn("What is the capital of France?")
        assert "Paris" in result["response"]
