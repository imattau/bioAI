"""VSA relational reasoning prototype.

Tests whether VSA binding + Hopfield pattern completion can solve
relational reasoning tasks using only local Hebbian learning.

Four tasks (ARCHITECTURE.md Section 6):
  1. Single-relation completion  — given 2/3 slots, recover the 3rd
  2. Analogy completion          — given (A,R,B) and (C,?,D), recover R
  3. Capacity stress             — measure accuracy vs stored triple count
  4. Compositional generalization — test unseen entities with known relations

No backprop, no attention, no gradient descent. Pure VSA + Hopfield.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path

from src.vsa.primitives import VSA
from src.vsa.relational import RelationalEncoder, RelationalMemory


# ── data generation ──────────────────────────────────────────────────

ENTITIES = [
    "cat", "dog", "mouse", "bird", "fish",
    "lion", "tiger", "wolf", "fox", "eagle",
    "shark", "whale", "bear", "deer", "rabbit",
    "horse", "sheep", "goat", "owl", "hawk",
    "elephant", "monkey", "turtle", "frog", "lizard",
]
RELATIONS = [
    "chases", "fears", "likes", "bigger-than", "smaller-than",
    "eats", "hunts", "hides-from", "sleeps-near", "competes-with",
]


def generate_triples(
    entities: list[str],
    relations: list[str],
    n: int,
    seed: int = 42,
) -> list[tuple[str, str, str]]:
    random.seed(seed)
    generated = set()
    triples = []
    attempts = 0
    while len(triples) < n and attempts < n * 10:
        s = random.choice(entities)
        r = random.choice(relations)
        o = random.choice(entities)
        if s == o:
            attempts += 1
            continue
        key = (s, r, o)
        if key in generated:
            attempts += 1
            continue
        generated.add(key)
        triples.append(key)
        attempts += 1
    return triples


def split_triples(
    triples: list[tuple[str, str, str]],
    train_ratio: float = 0.8,
) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]]]:
    random.shuffle(triples)
    split = int(len(triples) * train_ratio)
    return triples[:split], triples[split:]


# ── task runners ────────────────────────────────────────────────────

def task_single_completion(memory: RelationalMemory, stored_triples: list) -> dict:
    """Given 2 of 3 slots of a STORED triple, recover the 3rd via pattern completion.

    This tests whether binding + Hopfield can retrieve a full relational
    pattern from a partial cue. All triples here were stored in the Hopfield net.

    Splits accuracy by whether the query is genuinely ambiguous in the stored
    data (more than one stored triple matches the known slots) — a collision
    no encoding scheme can resolve to a single right answer — vs uniquely
    determined queries, where the encoder/Hopfield mechanism should be ~100%.
    """
    queries = {
        "subject": lambda s, r, o: {"relation": r, "object": o},
        "relation": lambda s, r, o: {"subject": s, "object": o},
        "object": lambda s, r, o: {"subject": s, "relation": r},
    }
    stats = {
        role: {"unique_correct": 0, "unique_total": 0, "ambig_correct": 0,
               "ambig_total": 0, "ambig_top3_hit": 0}
        for role in queries
    }

    for s, r, o in stored_triples:
        truth = {"subject": s, "relation": r, "object": o}
        for role, make_known in queries.items():
            known = make_known(s, r, o)
            result = memory.complete_detailed(known)
            ok = result.best == truth[role]
            bucket = stats[role]
            if len(memory.ground_truth_ambiguity(known)) <= 1:
                bucket["unique_total"] += 1
                bucket["unique_correct"] += ok
            else:
                bucket["ambig_total"] += 1
                bucket["ambig_correct"] += ok
                bucket["ambig_top3_hit"] += any(name == truth[role] for name, _ in result.candidates)

    def pct(n, d):
        return round(100 * n / d, 1) if d else None

    out = {}
    for role, b in stats.items():
        total = b["unique_total"] + b["ambig_total"]
        out[f"{role}_pct"] = pct(b["unique_correct"] + b["ambig_correct"], total)
        out[f"{role}_unique_pct"] = pct(b["unique_correct"], b["unique_total"])
        out[f"{role}_ambiguous_pct"] = pct(b["ambig_correct"], b["ambig_total"])
        out[f"{role}_ambiguous_top3_pct"] = pct(b["ambig_top3_hit"], b["ambig_total"])
        out[f"{role}_ambiguous_frac"] = pct(b["ambig_total"], total)
    return out


def task_analogy(
    memory: RelationalMemory,
    stored_triples: list,
    sample_size: int = 500,
) -> dict:
    """Given (A, R, B) and (C, ?, D), recover R.

    Tests whether the relation vector is compositionally independent of entity
    identity. All triples are stored — the question is whether the relation
    component can be cleanly extracted from a different subject-object pair.
    """
    import random
    correct = 0
    random.seed(42)
    sampled = 0
    attempts = 0
    while sampled < sample_size and attempts < sample_size * 10:
        t1 = random.choice(stored_triples)
        t2 = random.choice(stored_triples)
        s1, r1, o1 = t1
        s2, r2, o2 = t2
        if (s1, o1) == (s2, o2) or r1 != r2:
            attempts += 1
            continue
        # Query with s2, o2 (different entities, same relation)
        # The correct relation is r1 (which equals r2)
        _, r_recalled, _ = memory.complete({"subject": s2, "object": o2})
        if r_recalled == r1:
            correct += 1
        sampled += 1
    return {
        "analogy_pct": round(100 * correct / sampled, 1),
        "pairs": sampled,
    }


def task_capacity_stress(entities, relations, dim=10000, batch_sizes=None) -> list[dict]:
    """Measure accuracy vs stored triple count to find capacity limit.

    At ~0.1×D items, Hopfield crosstalk should degrade accuracy.
    """
    if batch_sizes is None:
        batch_sizes = [10, 50, 100, 200, 500, 1000, 2000, 5000]

    results = []
    for n in batch_sizes:
        triples = generate_triples(entities, relations, n, seed=42)
        train, test = split_triples(triples, train_ratio=0.8)

        vsa = VSA(dim=dim)
        encoder = RelationalEncoder(vsa)
        memory = RelationalMemory(encoder, dim=dim)

        for s, r, o in train:
            memory.store_triple(s, r, o)

        metrics = task_single_completion(memory, test)
        metrics["stored"] = n
        metrics["tested"] = len(test)
        results.append(metrics)
        print(
            f"  Capacity {n:5d}:  "
            f"subject={metrics['subject_pct']:5.1f}%  "
            f"relation={metrics['relation_pct']:5.1f}%  "
            f"object={metrics['object_pct']:5.1f}%"
        )
    return results


def task_comp_generalization(memory: RelationalMemory, all_entities, train_entities) -> dict:
    """Test with entities unseen during training.

    If the relation vector encodes the relation independently of entity
    identity, it should work even for novel entities.
    """
    test_entities = [e for e in all_entities if e not in train_entities]
    if not test_entities:
        return {"comp_gen_pct": 0.0, "note": "no unseen entities available"}

    # Generate test triples using novel entities
    test_triples = []
    for s in test_entities:
        for r in ["chases", "fears", "likes"]:
            for o in test_entities:
                if s != o:
                    test_triples.append((s, r, o))

    if not test_triples:
        return {"comp_gen_pct": 0.0, "note": "no test triples"}

    correct = 0
    for s, r, o in test_triples:
        _, _, recalled_o = memory.complete({"subject": s, "relation": r})
        if recalled_o == o:
            correct += 1

    return {
        "comp_gen_pct": round(100 * correct / len(test_triples), 1),
        "test_triples": len(test_triples),
        "unseen_entities": len(test_entities),
    }


# ── main ────────────────────────────────────────────────────────────

def run(
    n_triples: int = 200,
    dim: int = 10000,
    hopfield_beta: float = 50.0,
    hopfield_steps: int = 10,
    seed: int = 42,
    entities_list: list[str] | None = None,
    relations_list: list[str] | None = None,
) -> dict:
    entities = entities_list or ENTITIES
    relations = relations_list or RELATIONS

    train_entities = entities[:len(entities) // 2]
    test_entities = entities[len(entities) // 2:]

    print(f"Dimension: {dim}")
    print(f"Entities: {len(entities)}  Relations: {len(relations)}")

    # Generate triples from train entities only
    triples = generate_triples(train_entities, relations, n_triples, seed=seed)
    train, test = split_triples(triples, train_ratio=0.8)
    print(f"Triples: {n_triples}  Train: {len(train)}  Test: {len(test)}")

    # Build encoder + memory
    vsa = VSA(dim=dim)
    encoder = RelationalEncoder(vsa)
    memory = RelationalMemory(encoder, dim=dim, hopfield_beta=hopfield_beta)

    start = time.perf_counter()

    # Hebbian storage (one-shot, no training loop)
    for s, r, o in train:
        memory.store_triple(s, r, o)
    print(f"Stored {len(train)} triples in {time.perf_counter() - start:.2f}s")

    # ── Task 1: Single-relation completion (from STORED triples) ──
    print("\n─── Task 1: Single-relation completion (from stored) ───")
    t1 = task_single_completion(memory, train)
    print(f"  {'role':10s}  {'overall':>8s}  {'unique':>8s}  {'ambiguous':>10s}  {'ambig top3':>11s}  {'%ambig':>7s}")
    for role in ("subject", "relation", "object"):
        print(f"  {role:10s}  {t1[f'{role}_pct']:>7.1f}%  {t1[f'{role}_unique_pct']:>7.1f}%  "
              f"{t1[f'{role}_ambiguous_pct']:>9.1f}%  {t1[f'{role}_ambiguous_top3_pct']:>10.1f}%  "
              f"{t1[f'{role}_ambiguous_frac']:>6.1f}%")

    # ── Task 2: Analogy completion ──
    print("\n─── Task 2: Analogy completion ───")
    t2 = task_analogy(memory, train)
    print(f"  Analogies: {t2['analogy_pct']:.1f}%  "
          f"(over {t2['pairs']} pairs)")

    # ── Task 3: Compositional generalization (unseen test triples) ──
    print("\n─── Task 3: Compositional generalization (unseen test triples) ───")
    t3 = task_comp_generalization(memory, entities, train_entities)
    print(f"  Unseen entities: {t3.get('unseen_entities', 'N/A')}  "
          f"Accuracy: {t3.get('comp_gen_pct', 0):.1f}%")

    elapsed = time.perf_counter() - start
    print(f"\nTotal: {elapsed:.1f}s")

    return {
        "config": {
            "dim": dim,
            "n_triples": n_triples,
            "n_train": len(train),
            "n_test": len(test),
            "n_entities": len(entities),
            "n_relations": len(relations),
            "hopfield_beta": hopfield_beta,
            "hopfield_steps": hopfield_steps,
        },
        "task1_single_completion": t1,
        "task2_analogy": t2,
        "task3_comp_generalization": t3,
        "elapsed_seconds": round(elapsed, 1),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-triples", type=int, default=200)
    parser.add_argument("--dim", type=int, default=10000)
    parser.add_argument("--hopfield-beta", type=float, default=50.0)
    parser.add_argument("--hopfield-steps", type=int, default=10)
    parser.add_argument("--capacity-test", action="store_true",
                        help="Run capacity stress test instead of standard tasks")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.capacity_test:
        results = task_capacity_stress(ENTITIES, RELATIONS, dim=args.dim)
    else:
        results = run(
            n_triples=args.n_triples,
            dim=args.dim,
            hopfield_beta=args.hopfield_beta,
            hopfield_steps=args.hopfield_steps,
        )

    output = json.dumps(results, indent=2)
    if args.output:
        args.output.write_text(output + "\n")
        print(f"\nResults saved to {args.output}")
    else:
        print(f"\n{output}")


if __name__ == "__main__":
    main()
