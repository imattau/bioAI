import torch

from src.text import (
    RecurrentVSASequenceEncoder,
    SemanticChunkVSASequenceEncoder,
    SemanticVSAEncoder,
    VSASequenceRanker,
)


def candidate(text):
    return {"text": text, "kind": "complete", "source_ids": [0]}


def test_sequence_encoder_is_order_and_role_sensitive():
    encoder = RecurrentVSASequenceEncoder(dimension=128, seed=3)
    forward = encoder.encode("dog bites person", "candidate")
    reverse = encoder.encode("person bites dog", "candidate")
    query = encoder.encode("dog bites person", "query")
    assert not torch.equal(forward, reverse)
    assert not torch.equal(forward, query)


def test_pairwise_training_increases_preference_margin():
    torch.manual_seed(5)
    ranker = VSASequenceRanker(dimension=128, hidden_dimension=32)
    preferred = candidate("Paris is the capital of France.")
    rejected = candidate("Football teams compete for points.")
    evidence = ["France has its capital in Paris."]
    before = (
        ranker.score("What is France's capital?", preferred, evidence)
        - ranker.score("What is France's capital?", rejected, evidence)
    )
    ranker.learn_preference(
        "What is France's capital?", preferred, rejected, evidence, steps=20
    )
    after = (
        ranker.score("What is France's capital?", preferred, evidence)
        - ranker.score("What is France's capital?", rejected, evidence)
    )
    assert after > before
    assert ranker.rank(
        "What is France's capital?", [rejected, preferred], evidence
    )[0]["text"] == preferred["text"]


def test_ranker_state_round_trip():
    ranker = VSASequenceRanker(dimension=64, hidden_dimension=16)
    preferred = candidate("Supported response.")
    rejected = candidate("Unrelated response.")
    ranker.learn_preference("question", preferred, rejected, ["support"])
    restored = VSASequenceRanker.from_state(ranker.get_state())
    assert restored.updates == 1
    assert abs(
        restored.score("question", preferred, ["support"])
        - ranker.score("question", preferred, ["support"])
    ) < 1e-6


def test_cached_minibatch_training_scales_preference_updates():
    ranker = VSASequenceRanker(dimension=64, hidden_dimension=16)
    preferred = candidate("The supported relevant answer.")
    rejected = candidate("An unrelated answer.")
    evidence = ["The relevant evidence."]
    preferences = [(
        ranker.encode_state("relevant question", preferred, evidence),
        ranker.encode_state("relevant question", rejected, evidence),
    )]
    loss = ranker.train_encoded_preferences(
        preferences, examples=128, batch_size=16
    )
    assert loss > 0
    assert ranker.updates == 128
    assert ranker.rank(
        "relevant question", [rejected, preferred], evidence
    )[0]["text"] == preferred["text"]


def test_semantic_chunk_encoder_preserves_paraphrase_similarity():
    embeddings = {
        "Cars are fast.": torch.tensor([1.0, 0.0, 0.0, 0.0]),
        "Automobiles move quickly.": torch.tensor([0.95, 0.05, 0.0, 0.0]),
        "Bread uses flour.": torch.tensor([0.0, 0.0, 1.0, 0.0]),
    }
    semantic = SemanticVSAEncoder(
        lambda text: embeddings[text], vsa_dim=128, seed=11
    )
    encoder = SemanticChunkVSASequenceEncoder(semantic, seed=7)
    cars = encoder.encode("Cars are fast.", "candidate")
    automobiles = encoder.encode("Automobiles move quickly.", "candidate")
    bread = encoder.encode("Bread uses flour.", "candidate")
    similar = torch.nn.functional.cosine_similarity(cars, automobiles, dim=0)
    unrelated = torch.nn.functional.cosine_similarity(cars, bread, dim=0)
    assert similar > unrelated


def test_semantic_ranker_state_round_trip_with_embedder():
    embed = lambda text: torch.tensor([1.0, 0.0, 0.0, 0.0])
    semantic = SemanticVSAEncoder(embed, vsa_dim=64, seed=13)
    ranker = VSASequenceRanker(
        encoder=SemanticChunkVSASequenceEncoder(semantic),
        hidden_dimension=16,
    )
    state = ranker.get_state()
    restored = VSASequenceRanker.from_state(state, embedder=embed)
    assert isinstance(restored.encoder, SemanticChunkVSASequenceEncoder)
