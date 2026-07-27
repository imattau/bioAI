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
    composer = LearnedChunkComposer(
        max_candidate_chunks=128,
        seed_term_limit=2,
        rarity_multiplier=5,
    )
    composer.learn("Explain gravity", "Mass attracts mass.")
    restored = LearnedChunkComposer.from_state(composer.get_state())
    assert restored.generate("Explain gravity") == "Mass attracts mass."
    assert restored.max_candidate_chunks == 128
    assert restored.seed_term_limit == 2
    assert restored.rarity_multiplier == 5


def test_rare_terms_bound_common_term_scoring_candidates():
    composer = LearnedChunkComposer(max_candidate_chunks=16)
    for index in range(100):
        prompt = f"common subject example {index}"
        if index == 73:
            prompt += " distinctive"
        composer.learn(prompt, f"Response number {index}.")

    scores = composer._candidate_scores(
        {"common", "subject", "distinctive"}, starts=True
    )

    assert list(scores) == [73]


def test_agent_exposes_learned_composer():
    agent = BioAIDialogueAgent(vsa_dim=64)
    agent.learn_conversation("How should bread be baked?", "Heat it in an oven.")
    assert agent.chunk_composer.pairs == 1
