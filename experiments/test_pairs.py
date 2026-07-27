"""Test 2-slot VSA pair encoding with tokenised and atomic vectors.

A pair (input, output) is encoded as:
  bind(INPUT_ROLE, input) + bind(OUTPUT_ROLE, output)

Query with input only → retrieve the stored pair → decode output.
Only 1 slot is ever missing, so there is never a collision.
"""

from __future__ import annotations

import random
from pathlib import Path

import torch

from src.vsa.primitives import VSA
from src.vsa.hopfield import HopfieldNet


def tokenise(name: str, mode: str) -> list[str]:
    if mode == "atomic":
        return [name]
    if mode == "char-1gram":
        return list(name)
    if mode == "char-bigram":
        return [name[i:i+2] for i in range(len(name) - 1)] or [name]
    raise ValueError(f"Unknown mode: {mode}")


def build_entities(
    vsa: VSA, names: list[str], mode: str
) -> dict[str, torch.Tensor]:
    token_cache: dict[str, torch.Tensor] = {}
    entities: dict[str, torch.Tensor] = {}
    for name in names:
        tokens = tokenise(name, mode)
        vecs = []
        for t in tokens:
            if t not in token_cache:
                token_cache[t] = vsa.make_vector()
            vecs.append(token_cache[t])
        if len(vecs) == 1:
            entities[name] = vecs[0]
        else:
            entities[name] = vsa.bundle(vecs)
    return entities


def test_pairs(
    n_pairs: int = 200,
    dim: int = 10000,
    modern_beta: float = 50.0,
    mode: str = "atomic",
    entity_names: list[str] | None = None,
) -> dict:
    vsa = VSA(dim=dim)
    input_role = vsa.make_vector()
    output_role = vsa.make_vector()

    if entity_names is None:
        entity_names = [f"S{i}" for i in range(n_pairs * 2)]
    entities = build_entities(vsa, entity_names, mode)

    random.seed(42)
    available = list(entities.keys())
    used_inputs = set()
    pairs = []
    while len(pairs) < n_pairs:
        i = random.choice(available)
        o = random.choice(available)
        if i != o and i not in used_inputs:
            used_inputs.add(i)
            pairs.append((i, o))

    def encode_pair(in_name: str, out_name: str) -> torch.Tensor:
        return vsa.bundle([
            vsa.bind(input_role, entities[in_name]),
            vsa.bind(output_role, entities[out_name]),
        ])

    def encode_input_only(in_name: str) -> torch.Tensor:
        return vsa.bind(input_role, entities[in_name])

    hopfield = HopfieldNet(dim, retrieval_mode="modern", modern_beta=modern_beta)
    for in_name, out_name in pairs:
        hopfield.store(encode_pair(in_name, out_name))

    all_vecs = torch.stack(list(entities.values()))
    all_names = list(entities.keys())

    correct_out = 0
    correct_in = 0
    for in_name, out_name in pairs:
        q = encode_input_only(in_name)
        recalled = hopfield.recall(q)

        unbound_input = vsa.unbind(recalled, input_role)
        unbound_output = vsa.unbind(recalled, output_role)

        sim_in = vsa.similarity(unbound_input, all_vecs)
        sim_out = vsa.similarity(unbound_output, all_vecs)

        decoded_in = all_names[sim_in.argmax().item()]
        decoded_out = all_names[sim_out.argmax().item()]

        if decoded_in == in_name:
            correct_in += 1
        if decoded_out == out_name:
            correct_out += 1

    return {
        "n_pairs": n_pairs,
        "input_accuracy": round(100 * correct_in / n_pairs, 1),
        "output_accuracy": round(100 * correct_out / n_pairs, 1),
    }


def run():
    # Use meaningful entity names that share tokens
    ENTITY_NAMES = [
        "cat", "dog", "mouse", "bird", "fish",
        "lion", "tiger", "wolf", "fox", "eagle",
        "shark", "whale", "bear", "deer", "rabbit",
        "horse", "sheep", "goat", "owl", "hawk",
        "elephant", "monkey", "turtle", "frog", "lizard",
        "beetle", "spider", "ant", "bee", "fly",
        "salmon", "trout", "bass", "carp", "perch",
    ]

    print(f"{'N':>5s} {'mode':15s} {'input':>7s} {'output':>8s}  {'tokens':>6s}")
    print("-" * 55)

    for n in [10, 50, 100, 200]:
        for mode in ("atomic", "char-1gram", "char-bigram"):
            names = ENTITY_NAMES[:max(n * 2 + 5, len(ENTITY_NAMES))]
            r = test_pairs(n_pairs=n, mode=mode, entity_names=names)
            # Count unique tokens
            tokens = set()
            for name in names:
                for t in tokenise(name, mode):
                    tokens.add(t)
            print(
                f"{n:5d} {mode:15s} {r['input_accuracy']:>6.1f}% "
                f"{r['output_accuracy']:>7.1f}%  {len(tokens):>4d}"
            )

    # Also test with larger N on atomic only
    print("\n--- Atomic scaled ---")
    for n in [500, 1000]:
        names = [f"S{i}" for i in range(n * 2)]
        r = test_pairs(n_pairs=n, mode="atomic", entity_names=names)
        print(f"{n:5d} {'atomic':15s} {r['input_accuracy']:>6.1f}% {r['output_accuracy']:>7.1f}%")


if __name__ == "__main__":
    run()
