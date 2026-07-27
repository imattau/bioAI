from src.text import BioAIDialogueAgent


def answered_agent():
    agent = BioAIDialogueAgent(vsa_dim=64)
    agent.process_turn("The capital of France is Paris")
    result = agent.process_turn("What is the capital of France?")
    assert result["retrieval_accepted"]
    return agent


def test_positive_feedback_teaches_composer_without_storing_feedback_as_fact():
    agent = answered_agent()
    before = len(agent.library)
    result = agent.process_turn("Correct")
    assert result["response_mode"] == "feedback_learning"
    assert result["feedback"]["learned"]
    assert agent.chunk_composer.pairs == 1
    assert len(agent.library) == before
    assert agent.library.exact_lookup("Correct") is None


def test_correction_updates_memory_relations_and_rankers():
    agent = answered_agent()
    agent.enable_vsa_sequence_ranking(dimension=64)
    result = agent.record_feedback(
        preferred_response="The capital of France is Lyon"
    )
    assert result["correction_stored"]
    assert result["candidate_scorer_updated"]
    assert result["sequence_ranker_updated"]
    assert agent.library.exact_lookup("The capital of France is Lyon")
    claims = agent.consolidation.relation_claims("France", "capital")
    assert {claim["object"] for claim in claims} == {"paris", "lyon"}
    assert agent.sequence_ranker.updates == 1


def test_negative_feedback_requests_a_correction():
    agent = answered_agent()
    result = agent.process_turn("That's wrong")
    assert result["response_mode"] == "feedback_learning"
    assert not result["feedback"]["learned"]
    assert "corrected answer" in result["response"]


def test_natural_language_correction_closes_feedback_loop():
    agent = answered_agent()
    result = agent.process_turn(
        "Actually: The capital of France is Marseille"
    )
    assert result["response_mode"] == "feedback_learning"
    assert result["feedback"]["correction_stored"]
    assert agent.library.exact_lookup(
        "The capital of France is Marseille"
    )
    assert agent.chunk_composer.pairs == 1
