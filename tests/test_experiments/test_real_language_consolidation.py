from experiments.real_language_consolidation import (
    LABELS,
    project_bipolar,
    stratified_examples,
)

import torch


class _Rows:
    def __init__(self, rows):
        self.rows = rows

    def shuffle(self, seed):
        return self.rows


def test_stratified_examples_balances_real_labels():
    rows = _Rows([
        {"text": f"article {label}-{index}", "label": label}
        for index in range(3)
        for label in range(len(LABELS))
    ])
    selected = stratified_examples(rows, per_label=2, seed=1)
    assert len(selected) == 8
    assert [row["label"] for row in selected].count(0) == 2


def test_bipolar_projection_is_deterministic():
    embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    first = project_bipolar(embeddings, dimension=64, seed=9)
    second = project_bipolar(embeddings, dimension=64, seed=9)
    assert torch.equal(first, second)
    assert set(first.flatten().tolist()) == {-1, 1}
