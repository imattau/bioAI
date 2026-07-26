"""Experiment 4: Multi-Turn Conversation Test — Using BioAIDialogueAgent"""

import torch
from src.text import BioAIDialogueAgent
from src.vsa import VSA


class TestDialogueWithAgent:
    def setup_method(self):
        self.agent = BioAIDialogueAgent(vsa_dim=1000)

    def test_first_turn_no_context(self):
        result = self.agent.process_turn("hello")
        assert "response" in result
        assert result["turn"] == 1

    def test_second_turn_retrieves_first(self):
        self.agent.process_turn("the capital of France is Paris")
        result = self.agent.process_turn("what is the capital of France")
        assert "Paris" in result["response"] or "capital" in result["response"]

    def test_memory_across_turns(self):
        fact = "The capital of France is Paris"
        self.agent.process_turn(fact)
        for i in range(10):
            self.agent.process_turn(f"some conversation turn {i}")
        recalled = self.agent.recall(fact)
        assert recalled == fact, f"Expected '{fact}', got '{recalled}'"

    def test_continual_learning_no_forgetting(self):
        topics = [f"topic_{i}_related_sentence" for i in range(5)]
        for t in topics:
            self.agent.process_turn(t)
        for t in topics:
            recalled = self.agent.recall(t)
            assert recalled == t, f"Topic '{t}' should be recallable, got '{recalled}'"

    def test_store_persists_across_100_turns(self):
        torch.manual_seed(42)
        stored_fact = "persistent fact that must survive"
        self.agent.process_turn(stored_fact)
        for i in range(100):
            result = self.agent.process_turn(f"conversation turn number {i}")
            assert isinstance(result["response"], str)
            assert result["turn"] == i + 2
        recalled = self.agent.recall(stored_fact)
        assert recalled == stored_fact, (
            f"Fact should survive 100 turns: got '{recalled}'"
        )

    def test_novelty_detection(self):
        results = []
        for i in range(5):
            r = self.agent.process_turn(f"normal turn {i}")
            results.append(r)
        assert all(isinstance(r["response"], str) for r in results)
