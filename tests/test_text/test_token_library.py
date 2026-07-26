from src.text import BioAIDialogueAgent, TokenLibrary, VSAEncoder
from src.vsa import VSA


def test_compact_contiguous_token_storage_and_exact_lookup():
    library = TokenLibrary()
    library.add("Mars is the red planet")
    library.add("Venus is a hot planet")
    assert len(library) == 2
    assert library.exact_lookup("Mars is the red planet") == "Mars is the red planet"
    assert library.exact_lookup("unknown sentence") is None
    assert library.token_storage_bytes == 10 * 4
    assert list(library.sequence(0)) != list(library.sequence(1))


def test_inverted_index_prioritises_rare_relevant_sentence():
    library = TokenLibrary()
    for index in range(100):
        library.add(f"common background sentence number {index}")
    target_id = library.add("Paris is the capital of France")
    candidates = library.candidate_ids("what is the capital of France?", limit=5)
    assert candidates[0][0] == target_id
    assert len(candidates) <= 5


def test_semantic_alias_retrieves_paraphrase():
    library = TokenLibrary()
    target_id = library.add("Paris is the capital of France")
    library.add("Berlin has many museums")
    candidates = library.candidate_ids(
        "Which city is France's administrative centre?", limit=5
    )
    assert candidates[0][0] == target_id


def test_all_common_query_has_bounded_candidate_work():
    library = TokenLibrary()
    for index in range(25_000):
        library.add(f"common shared text group {index % 10}")
    candidates = library.candidate_ids(
        "common shared text", limit=20, max_candidates=500
    )
    assert len(candidates) == 20


def test_lazy_vector_cache_is_bounded():
    library = TokenLibrary(vector_cache_size=2)
    encoder = VSAEncoder(vsa=VSA(dim=32, device="cpu"))
    for index in range(3):
        library.add(f"sentence {index}")
        library.vector(index, encoder)
    assert len(library._vector_cache) == 2
    assert 0 not in library._vector_cache


def test_state_round_trip():
    library = TokenLibrary()
    for text in ("alpha beta", "beta gamma", "delta epsilon"):
        library.add(text)
    restored = TokenLibrary.from_state(library.get_state())
    assert restored.texts == library.texts
    assert restored.exact_lookup("beta gamma") == "beta gamma"
    assert restored.candidate_ids("epsilon", limit=1)[0][0] == 2


def test_agent_long_term_library_outgrows_bounded_vector_memory():
    agent = BioAIDialogueAgent(vsa_dim=32)
    for index in range(210):
        text = f"stored sentence number {index}"
        agent.turn_count += 1
        agent._store_turn(text, agent.encoder.encode(text))
    assert len(agent.library) == 210
    assert len(agent.decoder) == 200
    assert len(agent.hash_store) == 200
    assert len(agent.hopfield) == 200
    assert agent.recall("stored sentence number 209") == "stored sentence number 209"
