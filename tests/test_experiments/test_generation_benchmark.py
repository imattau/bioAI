import torch
import pytest

from experiments.generation_benchmark import (
    ControlledCorpus,
    build_splits,
    compute_metrics,
    levenshtein,
)


def test_compositional_split_has_no_overlap_and_full_vocabulary():
    seen, unseen = build_splits()
    assert len(seen) == 48
    assert len(unseen) == 16
    assert not {tuple(row.tolist()) for row in seen} & {
        tuple(row.tolist()) for row in unseen
    }
    assert set(seen.flatten().tolist()) == set(range(12))


def test_controlled_encoder_preserves_word_order():
    corpus = ControlledCorpus(vsa_dim=128, seed=42)
    forward = torch.tensor([0, 4, 8])
    reverse = torch.tensor([8, 4, 0])
    assert not torch.equal(
        corpus.encode_tokens(forward),
        corpus.encode_tokens(reverse),
    )


def test_levenshtein():
    assert levenshtein([1, 2, 3], [1, 2, 3]) == 0
    assert levenshtein([1, 2, 3], [3, 2, 1]) == 2


def test_metrics_separate_token_and_order_accuracy():
    targets = torch.tensor([[0, 4, 8], [1, 5, 9]])
    predictions = torch.tensor([[0, 4, 8], [9, 5, 1]])
    metrics = compute_metrics(predictions, targets, "test", 0.0, 0.002)
    assert metrics.exact_match == 0.5
    assert metrics.token_accuracy == pytest.approx(4 / 6)
    assert metrics.mean_edit_distance == 1.0
    assert metrics.order_accuracy == 0.5
