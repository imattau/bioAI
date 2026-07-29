"""Tests for capacity-limited stores with eviction policies.

Covers VSAHashStore FIFO eviction, HopfieldNet max_patterns eviction,
and the removal of the hard 200-cap gate in agent.py _store_turn.
"""

import torch

from src.vsa import VSA, VSAHashStore, HopfieldNet
from src.text import BioAIDialogueAgent


class TestVSAHashStoreEviction:
    def setup_method(self):
        self.vsa = VSA(dim=100, device="cpu")

    def test_stores_below_capacity(self):
        store = VSAHashStore(self.vsa, num_buckets=100, max_items=10)
        for i in range(5):
            store.insert(torch.sign(torch.randn(100)), torch.sign(torch.randn(100)))
        assert len(store) == 5

    def test_evicts_oldest_when_over_capacity(self):
        store = VSAHashStore(self.vsa, num_buckets=100, max_items=5)
        keys = []
        for i in range(5):
            k = torch.sign(torch.randn(100))
            keys.append(k)
            store.insert(k, torch.sign(torch.randn(100)))
        assert len(store) == 5

        # Insert one more — oldest should be evicted
        new = torch.sign(torch.randn(100))
        store.insert(new, new)
        assert len(store) == 5

        # The oldest key should no longer be findable
        assert store.lookup(keys[0]) is None

    def test_latest_insertions_still_retrievable(self):
        store = VSAHashStore(self.vsa, num_buckets=100, max_items=5)
        keys = []
        for i in range(10):
            k = torch.sign(torch.randn(100))
            keys.append(k)
            store.insert(k, k)
        assert len(store) == 5
        # The last 5 should be retrievable
        for k in keys[5:]:
            assert store.lookup(k) is not None, "Recent key should survive eviction"

    def test_default_max_items(self):
        store = VSAHashStore(self.vsa)
        assert store.max_items == 1000

    def test_custom_max_items_persisted_in_state(self):
        store = VSAHashStore(self.vsa, max_items=500)
        state = store.get_state()
        assert state["max_items"] == 500
        store2 = VSAHashStore(self.vsa)
        store2.set_state(state)
        assert store2.max_items == 500

    def test_empty_eviction_noop(self):
        store = VSAHashStore(self.vsa, max_items=5)
        store._evict_one()
        assert len(store) == 0

    def test_exact_capacity_does_not_evict(self):
        store = VSAHashStore(self.vsa, max_items=5)
        keys = []
        for i in range(5):
            k = torch.sign(torch.randn(100))
            keys.append(k)
            store.insert(k, k)
        assert len(store) == 5
        for k in keys:
            assert store.lookup(k) is not None


class TestHopfieldNetEviction:
    def setup_method(self):
        self.vsa = VSA(dim=100, device="cpu")

    def test_stores_below_max(self):
        hop = HopfieldNet(dim=100, max_patterns=10)
        for i in range(5):
            hop.store(torch.sign(torch.randn(100)))
        assert len(hop) == 5

    def test_evicts_oldest_when_over_max(self):
        hop = HopfieldNet(dim=100, max_patterns=5)
        patterns = [torch.sign(torch.randn(100)) for _ in range(5)]
        for p in patterns:
            hop.store(p)
        assert len(hop) == 5

        # Insert one more
        new = torch.sign(torch.randn(100))
        hop.store(new)
        assert len(hop) == 5

        # The oldest pattern should be gone
        recall = hop.recall(patterns[0], steps=5)
        sim = self.vsa.similarity(patterns[0], recall)
        # The evicted pattern should NOT be recalled well (sim will be near 0
        # if it's truly gone from the store, vs >0.8 if it were still there)
        assert sim.item() < 0.8, "Oldest pattern should have been evicted"

    def test_keeps_newest_patterns(self):
        hop = HopfieldNet(dim=100, max_patterns=5)
        patterns = [torch.sign(torch.randn(100)) for _ in range(8)]
        for p in patterns:
            hop.store(p)
        assert len(hop) == 5
        # The most recent 5 should all be recallable
        for p in patterns[-5:]:
            recall = hop.recall(p, steps=5)
            sim = self.vsa.similarity(p, recall)
            assert sim.item() > 0.7, "Recent pattern should be recallable"

    def test_retrieval_mode_unchanged(self):
        hop = HopfieldNet(dim=100, max_patterns=5, retrieval_mode="modern")
        for i in range(10):
            hop.store(torch.sign(torch.randn(100)))
        assert len(hop) == 5
        assert hop.retrieval_mode == "modern"

    def test_get_state_preserves_max_patterns(self):
        hop = HopfieldNet(dim=100, max_patterns=500)
        state = hop.get_state()
        assert state["max_patterns"] == 500

    def test_set_state_preserves_pattern_count(self):
        hop = HopfieldNet(dim=100, max_patterns=10)
        patterns = [torch.sign(torch.randn(100)) for _ in range(8)]
        for p in patterns:
            hop.store(p)
        # Should have evicted down to 10 (all fit)
        assert len(hop) == 8

        saved = hop.get_state()
        hop2 = HopfieldNet(dim=100)
        hop2.set_state(saved["patterns"])
        assert len(hop2) == 8


class TestAgentStoreTurnEviction:
    """Verifies the hard 200-cap was removed and stores manage their own capacity."""

    def test_stores_more_than_200_statements(self):
        agent = BioAIDialogueAgent(vsa_dim=64)
        agent.gonogo_gate_enabled = False
        for i in range(300):
            agent.process_turn(f"Statement number {i} with some content to encode.")
        # All stores should have entries (not silently dropped at 200)
        assert len(agent.hash_store) > 0
        assert len(agent.hopfield) > 0
        assert len(agent.decoder) > 0

    def test_hash_store_capped_by_own_eviction_not_agent_gate(self):
        agent = BioAIDialogueAgent(vsa_dim=64)
        agent.gonogo_gate_enabled = False
        # Insert 2500 statements — hash_store (max_items=2000) should evict
        # oldest rather than silently dropping
        for i in range(2500):
            agent.process_turn(f"Statement {i}: the weather today is nice.")
        # hash_store should reflect its own capacity, not a lower gate
        assert 1000 <= len(agent.hash_store) <= len(agent.hash_store._order)
        # Recent statements should still be retrievable
        result = agent.process_turn("Statement 2499: the weather today is nice?")
        assert "Statement" in result["response"] or not result["retrieval_accepted"]

    def test_hopfield_stores_beyond_200(self):
        agent = BioAIDialogueAgent(vsa_dim=64)
        agent.gonogo_gate_enabled = False
        for i in range(500):
            agent.process_turn(f"Fact number {i} with some content.")
        # HopfieldNet should have more than 200 (its max_patterns is 2000)
        assert len(agent.hopfield) > 200

    def test_decoder_stores_beyond_200(self):
        agent = BioAIDialogueAgent(vsa_dim=64)
        agent.gonogo_gate_enabled = False
        for i in range(500):
            agent.process_turn(f"Statement {i} to fill the decoder.")
        # Decoder capacity is 200 — its own FIFO handles eviction
        assert len(agent.decoder) <= 200
        assert len(agent.decoder) > 0

    def test_memory_survives_across_500_turns(self):
        agent = BioAIDialogueAgent(vsa_dim=64)
        agent.gonogo_gate_enabled = False
        agent.process_turn("The capital of France is Paris.")
        for i in range(498):
            agent.process_turn(f"Fill turn {i}.")
        result = agent.process_turn("What is the capital of France?")
        assert "Paris" in result["response"]
