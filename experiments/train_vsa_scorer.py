"""Train the VSA encoder/scorer from teacher records and evaluate on frozen validation."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from experiments.library_response_scaling import (
    load_frozen_validation,
    token_f0_5,
    token_f1,
)
from src.text import TokenLibrary, FixedSpliceCandidateGenerator, read_teacher_records
from src.text.vsa_scorer import VSAEncoderScorer


def build_associations(
    records: list,
) -> list[tuple[str, list[str]]]:
    """Build (text, semantic_context) pairs for VSA token-vector learning."""
    associations = []
    for rec in records:
        context = [
            *rec.get("required_facts", []),
            *rec.get("semantic_concepts", []),
            *rec.get("chunks", []),
            rec["candidates"][rec["preferred_index"]]["text"],
        ]
        associations.append((rec["prompt"], context))
        for c in rec["candidates"]:
            associations.append((c["text"], context))
    return associations


def train(
    records_path: Path,
    output_path: Path,
    epochs: int = 5,
    dimension: int = 512,
    hidden_dimension: int = 64,
    learning_rate: float = 0.001,
    validation_pairs: int = 200,
) -> dict:
    records = read_teacher_records(records_path)
    print(f"Loaded {len(records)} teacher records")

    associations = build_associations(records)
    print(f"Built {len(associations)} token associations")

    scorer = VSAEncoderScorer(
        dimension=dimension,
        hidden_dimension=hidden_dimension,
        learning_rate=learning_rate,
    )
    scorer.learn_associations(associations)
    print(f"Learned {len(scorer.encoder.learned_vectors)} token vectors")

    total_pairs = 0
    start = time.perf_counter()
    for epoch in range(epochs):
        epoch_losses = []
        epoch_pairs = 0
        for rec in records:
            pref_idx = rec["preferred_index"]
            preferred = rec["candidates"][pref_idx]
            for idx, rejected in enumerate(rec["candidates"]):
                if idx == pref_idx:
                    continue
                loss = scorer.learn_preference(
                    rec["prompt"], preferred, rejected, rec["evidence"]
                )
                epoch_losses.append(loss)
                epoch_pairs += 1
        total_pairs += epoch_pairs
        mean_loss = statistics.mean(epoch_losses) if epoch_losses else 0.0
        elapsed = time.perf_counter() - start
        print(
            f"  Epoch {epoch + 1}/{epochs}: {epoch_pairs} pairs, "
            f"loss={mean_loss:.4f}, updates={scorer.updates}, {elapsed:.1f}s"
        )

    # ── evaluate on frozen validation ──
    print("\nEvaluating on frozen validation set...")
    pairs = load_frozen_validation()[:validation_pairs]

    # load the response library for candidate generation
    learning = torch.load(
        "checkpoints/live_feedback_scale_state.pt",
        map_location="cpu",
        weights_only=False,
    )
    library = TokenLibrary.from_state(learning.get("response_library", learning.get("prompts")))
    responses = learning["responses"]
    prompts = TokenLibrary.from_state(learning["prompts"])
    generator = FixedSpliceCandidateGenerator()

    preds, refs = [], []
    for prompt, reference in pairs:
        ids = [sid for sid, _ in prompts.candidate_ids(prompt, limit=5)]
        if not ids:
            continue
        evidence = [responses[sid] for sid in ids]
        candidates = generator.generate(prompt, evidence)
        if not candidates:
            continue
        ranked = scorer.rank(prompt, candidates, evidence)
        preds.append(ranked[0]["text"] if ranked else "")
        refs.append(reference)

    f05 = 100 * statistics.mean(
        token_f0_5(p, r) for p, r in zip(preds, refs)
    )
    f1 = 100 * statistics.mean(
        token_f1(p, r) for p, r in zip(preds, refs)
    )
    print(f"Frozen validation ({len(preds)} pairs):")
    print(f"  F0.5 = {f05:.2f}%")
    print(f"  F1   = {f1:.2f}%")

    scorer.save(output_path)
    print(f"Saved to {output_path}")

    return {
        "records": len(records),
        "epochs": epochs,
        "dimension": dimension,
        "hidden_dimension": hidden_dimension,
        "learning_rate": learning_rate,
        "total_pairs": total_pairs,
        "updates": scorer.updates,
        "learned_token_vectors": len(scorer.encoder.learned_vectors),
        "validation_f0_5_pct": round(f05, 2),
        "validation_f1_pct": round(f1, 2),
        "elapsed_seconds": round(time.perf_counter() - start, 1),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records", type=Path, default=Path("checkpoints/teacher_records_even500.jsonl")
    )
    parser.add_argument("--output", type=Path, default=Path("checkpoints/vsa_scorer.pt"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--dimension", type=int, default=512)
    parser.add_argument("--hidden-dimension", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--validation-pairs", type=int, default=200)
    args = parser.parse_args()

    result = train(
        records_path=args.records,
        output_path=args.output,
        epochs=args.epochs,
        dimension=args.dimension,
        hidden_dimension=args.hidden_dimension,
        learning_rate=args.learning_rate,
        validation_pairs=args.validation_pairs,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
