import torch

from src.text import SemanticVSAEncoder, TokenLibrary


EMBEDDINGS = {
    "river": torch.tensor([1.0, 0.0, 0.0, 0.0]),
    "waterway": torch.tensor([0.95, 0.05, 0.0, 0.0]),
    "piano": torch.tensor([0.0, 0.0, 1.0, 0.0]),
}


def embed(text: str) -> torch.Tensor:
    return EMBEDDINGS[text]


def test_projection_preserves_learned_semantic_neighbourhood():
    encoder = SemanticVSAEncoder(embed, vsa_dim=256, seed=42)
    river = encoder.encode("river")
    waterway = encoder.encode("waterway")
    piano = encoder.encode("piano")
    assert encoder.similarity(river, waterway) > 0.8
    assert encoder.similarity(river, waterway) > encoder.similarity(river, piano)


def test_semantic_lsh_retrieves_nearby_vsa():
    encoder = SemanticVSAEncoder(embed, vsa_dim=256, seed=42)
    library = TokenLibrary()
    target = library.add("stored river fact", encoder.encode("river"))
    library.add("stored piano fact", encoder.encode("piano"))
    candidates = library.semantic_candidate_ids(
        encoder.encode("waterway"), limit=2
    )
    assert candidates[0][0] == target
    assert candidates[0][1] > 0.8


def test_semantic_state_round_trip():
    encoder = SemanticVSAEncoder(embed, vsa_dim=64, seed=7)
    vector = encoder.encode("river")
    library = TokenLibrary()
    library.add("river memory", vector)
    restored = TokenLibrary.from_state(library.get_state())
    assert restored.semantic_storage_bytes == 8
    assert restored.semantic_candidate_ids(vector, limit=1)[0][0] == 0
