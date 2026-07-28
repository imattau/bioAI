import torch

from src.text import BioAIDialogueAgent, LearnedChunkComposer


def test_learns_chunk_order_and_stopping():
    composer = LearnedChunkComposer()
    composer.learn(
        "How do I bake bread?",
        "Mix flour with water. Let the dough rise. Bake until golden.",
    )
    response = composer.generate("How can I bake bread?")
    assert response == (
        "Mix flour with water. Let the dough rise. Bake until golden."
    )


def test_combines_learned_chunks_for_shared_subject():
    composer = LearnedChunkComposer()
    composer.learn("Tell me about solar power", "Solar panels produce power.")
    composer.learn(
        "Are solar panels clean?",
        "Solar panels produce power. They operate without direct emissions.",
    )
    response = composer.generate("Does solar power produce clean energy?")
    assert "Solar panels produce power." in response
    assert "They operate without direct emissions." in response


def test_state_round_trip():
    composer = LearnedChunkComposer()
    composer.learn("Explain gravity", "Mass attracts mass.")
    restored = LearnedChunkComposer.from_state(composer.get_state())
    assert restored.generate("Explain gravity") == "Mass attracts mass."


def test_agent_exposes_learned_composer():
    agent = BioAIDialogueAgent(vsa_dim=64)
    agent.learn_conversation("How should bread be baked?", "Heat it in an oven.")
    assert agent.chunk_composer.pairs == 1


def test_chunk_composed_response_reports_response_generated():
    """Regression test: response_generated used to be true only when
    response_generator was set, so a turn answered via chunk_composer
    (response_mode="learned_chunk_composition") wrongly reported
    response_generated=False even though the text was synthesized, not
    quoted verbatim -- see ARCHITECTURE_STATUS.md/RELATIONAL_MEMORY.md's
    response-ecosystem planning notes.
    """
    # Deliberately not a "capital of X" / "X is Y" shape: those are answered
    # by the relational path, which overrides response_mode before this
    # block runs, regardless of chunk_composer -- see
    # tests/test_text/test_gonogo_wiring.py for the same avoidance.
    torch.manual_seed(0)  # unseeded VSA vectors make retrieval margin flaky
    agent = BioAIDialogueAgent(vsa_dim=1000)
    agent.enable_chunk_composition()
    agent.process_turn("The weather today is unusually cold for this time of year.")
    result = agent.process_turn("What is the weather like today?")
    assert result["response_mode"] == "learned_chunk_composition"
    assert result["response_generated"] is True
