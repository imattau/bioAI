"""Experiment 4: Multi-Turn Conversation Test — Using BioAIDialogueAgent"""

import torch
from src.text import BioAIDialogueAgent
from src.vsa import VSA


class TestDialogueWithAgent:
    def setup_method(self):
        self.agent = BioAIDialogueAgent(vsa_dim=1000)
        # These tests check the deterministic threshold/margin retrieval
        # rule itself, not gonogo -- an untrained gonogo network vetoes
        # close to randomly, which would make this class flaky.
        self.agent.gonogo_gate_enabled = False

    def test_first_turn_no_context(self):
        result = self.agent.process_turn("hello")
        assert "response" in result
        assert result["turn"] == 1

    def test_second_turn_retrieves_first(self):
        self.agent.process_turn("the capital of France is Paris")
        result = self.agent.process_turn("what is the capital of France")
        assert "Paris" in result["response"] or "capital" in result["response"]
        assert result["retrieval_accepted"]
        assert result["intent"] == "question"

    def test_unrelated_statement_does_not_repeat_previous_memory(self):
        fact = "the capital of France is Paris"
        self.agent.process_turn(fact)
        result = self.agent.process_turn(
            "Mars has seasons that change its climate"
        )
        assert result["response"] == "I'll remember that."
        assert fact not in result["response"]
        assert result["intent"] == "statement"

    def test_unrelated_question_is_rejected(self):
        self.agent.process_turn("the capital of France is Paris")
        result = self.agent.process_turn("what causes tides on the moon?")
        assert not result["retrieval_accepted"]
        assert "relevant memory" in result["response"]

    def test_ambiguous_candidates_are_rejected_by_margin(self):
        self.agent.process_turn("Mars is a red planet")
        self.agent.process_turn("Mars is a cold planet")
        result = self.agent.process_turn("what kind of planet is Mars?")
        assert len(result["retrieval_candidates"]) >= 2
        assert not result["retrieval_accepted"]

    def test_semantic_paraphrase_retrieves_fact(self):
        self.agent.process_turn("Paris is the capital of France")
        result = self.agent.process_turn(
            "Which city is France's administrative centre?"
        )
        assert result["retrieval_accepted"]
        assert "Paris" in result["response"]

    def test_questions_remain_history_but_not_long_term_facts(self):
        self.agent.process_turn("Paris is the capital of France")
        self.agent.process_turn("What is the capital of France?")
        assert len(self.agent.history) == 2
        assert len(self.agent.library) == 1
        assert self.agent.library.exact_lookup(
            "What is the capital of France?"
        ) is None

    def test_grounded_generator_receives_accepted_memory_and_exposes_source(self):
        calls = []

        def generator(question, memories):
            calls.append((question, memories))
            return f"Paris is the answer [memory:{memories[0]['id']}]"

        self.agent.response_generator = generator
        self.agent.process_turn("Paris is the capital of France")
        result = self.agent.process_turn("What is the capital of France?")
        assert result["response_generated"]
        assert result["response"] == "Paris is the answer [memory:0]"
        assert result["sources"] == [{
            "id": 0,
            "text": "Paris is the capital of France",
            "score": result["retrieval_score"],
        }]
        assert calls[0][0] == "What is the capital of France?"

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
