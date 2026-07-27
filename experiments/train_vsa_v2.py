"""Train the v2 VSA encoder/scorer on splice-generated candidate pairs.

Key changes from v1 training (train_vsa_scorer.py):
  1. Training pairs are generated from FixedSpliceCandidateGenerator
     (matching inference distribution), not from teacher-generated alternatives.
  2. Evidence is used ONLY for candidate generation, NOT in the scorer.
  3. VSA encoding is real-valued (no sign()), with L2 normalization.
  4. Go/No-Go scorer has weight decay + dropout.
  5. Frozen validation set is evaluated every epoch with early stopping.
  6. Per-dimension attribution diagnostic monitors for shortcut exploitation.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from experiments.library_response_scaling import (
    load_frozen_validation,
    token_f0_5,
)
from src.text import (
    FixedSpliceCandidateGenerator,
    LearnedChunkComposer,
    TokenLibrary,
)
from src.text.vsa_scorer_v2 import EncoderScorer


def load_records(path: Path) -> list[dict]:
    """Load teacher records from JSONL (simple format, no teacher_bootstrap dependency)."""
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def build_associations(
    records: list[dict],
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


def generate_training_pairs(
    records: list[dict],
    generator: FixedSpliceCandidateGenerator,
    composer: LearnedChunkComposer | None,
    responses: list[str],
    f05_threshold: float = 0.05,
    max_per_record: int = 5,
) -> list[tuple[str, dict, dict, list[str]]]:
    """Generate preference pairs from splice candidates.

    For each record: generate candidates from evidence, score them
    against the teacher's preferred text using token_f0_5,
    and create pairs from the best and worst candidates.

    This matches the inference distribution exactly — both preferred
    and rejected are splice/composition candidates from the same evidence.
    """
    pairs = []
    skipped_no_candidates = 0
    skipped_ties = 0
    for rec in records:
        prompt = rec["prompt"]
        evidence = rec["evidence"]
        pref_text = rec["candidates"][rec["preferred_index"]]["text"]

        candidates = generator.generate(prompt, evidence, composer)
        if len(candidates) < 2:
            skipped_no_candidates += 1
            continue

        scored = [
            (token_f0_5(c["text"], pref_text), c)
            for c in candidates
        ]
        scored.sort(key=lambda x: x[0], reverse=True)

        best_score, best = scored[0]
        worst_score, worst = scored[-1]
        delta = best_score - worst_score

        if delta < f05_threshold:
            skipped_ties += 1
            continue

        pairs.append((prompt, best, worst, evidence))

        # Optionally add more pairs from the same candidate pool
        if max_per_record > 1:
            added = 1
            for i in range(len(scored) - 1):
                if added >= max_per_record:
                    break
                if scored[i][0] - worst_score >= f05_threshold:
                    pairs.append((prompt, scored[i][1], worst, evidence))
                    added += 1

    if skipped_no_candidates or skipped_ties:
        print(
            f"  Skipped {skipped_no_candidates} records (no/insufficient candidates), "
            f"{skipped_ties} records (near-tie)"
        )
    return pairs


def evaluate(
    scorer: EncoderScorer,
    pairs: list[tuple[str, str]],
    generator: FixedSpliceCandidateGenerator,
    prompts_lib: TokenLibrary,
    responses: list[str],
    composer: LearnedChunkComposer | None = None,
) -> dict:
    """Evaluate on frozen validation pairs. Returns metrics dict."""
    preds, refs = [], []
    for prompt, reference in pairs:
        ids = [sid for sid, _ in prompts_lib.candidate_ids(prompt, limit=5)]
        if not ids:
            continue
        evidence = [responses[sid] for sid in ids]
        candidates = generator.generate(prompt, evidence, composer)
        if not candidates:
            continue
        ranked = scorer.rank(prompt, candidates, evidence)
        preds.append(ranked[0]["text"] if ranked else "")
        refs.append(reference)

    if not preds:
        return {"f0_5_pct": 0.0, "f1_pct": 0.0, "count": 0}

    from experiments.library_response_scaling import token_f1
    f05 = 100 * statistics.mean(
        token_f0_5(p, r) for p, r in zip(preds, refs)
    )
    f1 = 100 * statistics.mean(
        token_f1(p, r) for p, r in zip(preds, refs)
    )
    return {"f0_5_pct": round(f05, 2), "f1_pct": round(f1, 2), "count": len(preds)}


def diagnose(
    scorer: EncoderScorer,
    pairs: list[tuple[str, dict, dict, list[str]]],
) -> dict:
    """Per-dimension attribution diagnostic (InfoRM-style)."""
    query_states = []
    cand_states = []
    for prompt, preferred, rejected, evidence in pairs[:100]:
        qs = scorer.encode_query(prompt, evidence)
        ps = scorer.encoder.encode(preferred["text"], "candidate")
        rs = scorer.encoder.encode(rejected["text"], "candidate")
        query_states.append(qs)
        cand_states.append(ps)
        cand_states.append(rs)

    if not query_states:
        return {"top2_attribution_pct": 0.0}

    # Estimate per-dimension contribution via gradient sensitivity
    qs = torch.stack(query_states[:50])
    cs = torch.stack(cand_states[:50])
    joint = torch.cat([qs, cs], dim=-1)

    # Forward pass to get gradients
    joint.requires_grad_(True)
    scores = scorer.gonogo.go(joint) - scorer.gonogo.nogo(joint)
    scores.sum().backward()

    attrib = joint.grad.abs().mean(dim=0)
    total = attrib.sum()
    top2 = attrib.topk(2).values.sum()
    top2_pct = (top2 / total * 100).item() if total > 0 else 0.0

    return {"top2_attribution_pct": round(top2_pct, 1)}


def train(
    records_path: Path,
    learning_state_path: Path,
    output_path: Path,
    epochs: int = 10,
    dimension: int = 512,
    hidden: int = 64,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-5,
    batch_size: int = 32,
    f05_threshold: float = 0.05,
    max_train_pairs: int | None = None,
    early_stop_patience: int = 3,
) -> dict:
    # ── load data ────────────────────────────────────────────────────
    records = load_records(records_path)
    print(f"Loaded {len(records)} teacher records")

    # Load learning state for response library
    learning = torch.load(
        learning_state_path, map_location="cpu", weights_only=False
    )
    prompts_lib = TokenLibrary.from_state(learning["prompts"])
    responses = learning["responses"]

    composer = LearnedChunkComposer.from_state(learning["composer"])

    generator = FixedSpliceCandidateGenerator(max_candidates=24)

    # ── generate training pairs from splice candidates ───────────────
    print("Generating training pairs from splice candidates...")
    start = time.perf_counter()
    pairs = generate_training_pairs(
        records, generator, composer, responses,
        f05_threshold=f05_threshold,
    )
    if max_train_pairs is not None:
        pairs = pairs[:max_train_pairs]
    print(
        f"  Generated {len(pairs)} preference pairs in "
        f"{time.perf_counter() - start:.1f}s"
    )

    if not pairs:
        raise ValueError("No training pairs could be generated")

    # ── build VSA token associations ─────────────────────────────────
    print("Building VSA token associations...")
    associations = build_associations(records)
    print(f"  {len(associations)} (text, context) pairs")

    # ── create scorer ────────────────────────────────────────────────
    scorer = EncoderScorer(
        dimension=dimension,
        hidden=hidden,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    scorer.learn_associations(associations)
    print(f"  Learned {len(scorer.encoder.learned_vectors)} token vectors")

    # ── training loop with validation ────────────────────────────────
    frozen_val = load_frozen_validation()[:200]
    best_val_f05 = 0.0
    stall_count = 0
    train_start = time.perf_counter()
    epoch_metrics = []

    for epoch in range(epochs):
        epoch_losses = []
        indices = list(range(len(pairs)))
        torch.manual_seed(epoch + 42)
        perm = torch.randperm(len(pairs)).tolist()

        for i in range(0, len(pairs), batch_size):
            batch_indices = perm[i:i + batch_size]
            batch_losses = []
            for idx in batch_indices:
                prompt, pref, rej, ev = pairs[idx]
                loss = scorer.learn_preference(prompt, pref, rej, ev)
                batch_losses.append(loss)
            if batch_losses:
                epoch_losses.extend(batch_losses)

        mean_loss = statistics.mean(epoch_losses) if epoch_losses else 0.0
        elapsed = time.perf_counter() - train_start

        # Validation
        val = evaluate(scorer, frozen_val, generator, prompts_lib, responses, composer)
        diag = diagnose(scorer, pairs[:200])

        print(
            f"Epoch {epoch + 1:2d}/{epochs}:  loss={mean_loss:.4f}  "
            f"val F0.5={val['f0_5_pct']:.2f}%  "
            f"F1={val['f1_pct']:.2f}%  "
            f"top2_attrib={diag['top2_attribution_pct']:.1f}%  "
            f"updates={scorer.updates}  {elapsed:.1f}s"
        )

        epoch_metrics.append({
            "epoch": epoch + 1,
            "loss": round(mean_loss, 4),
            "val_f0_5_pct": val["f0_5_pct"],
            "val_f1_pct": val["f1_pct"],
            "top2_attribution_pct": diag["top2_attribution_pct"],
            "updates": scorer.updates,
        })

        if val["f0_5_pct"] > best_val_f05:
            best_val_f05 = val["f0_5_pct"]
            stall_count = 0
            scorer.save(output_path)
            print(f"  → Saved checkpoint (best val F0.5 = {best_val_f05:.2f}%)")
        else:
            stall_count += 1
            if stall_count >= early_stop_patience:
                print(
                    f"  → Early stopping (no improvement for {early_stop_patience} epochs)"
                )
                break

    # ── final evaluation ────────────────────────────────────────────
    scorer = EncoderScorer.load(output_path)
    print(f"\nFinal evaluation (best checkpoint):")
    val = evaluate(scorer, frozen_val, generator, prompts_lib, responses, composer)
    print(f"  F0.5 = {val['f0_5_pct']:.2f}%  F1 = {val['f1_pct']:.2f}%  ({val['count']} pairs)")

    return {
        "records": len(records),
        "training_pairs": len(pairs),
        "epochs": epoch + 1,
        "dimension": dimension,
        "hidden": hidden,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "f05_threshold": f05_threshold,
        "learned_token_vectors": len(scorer.encoder.learned_vectors),
        "best_val_f0_5_pct": best_val_f05,
        "final_val_f0_5_pct": val["f0_5_pct"],
        "final_val_f1_pct": val["f1_pct"],
        "best_val_top2_attribution_pct": epoch_metrics[-1]["top2_attribution_pct"],
        "total_updates": scorer.updates,
        "epoch_metrics": epoch_metrics,
        "elapsed_seconds": round(time.perf_counter() - train_start, 1),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records", type=Path, default=Path("checkpoints/teacher_records_even500.jsonl")
    )
    parser.add_argument(
        "--learning-state", type=Path,
        default=Path("checkpoints/live_feedback_scale_state.pt"),
    )
    parser.add_argument("--output", type=Path, default=Path("checkpoints/vsa_v2.pt"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--dimension", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--f05-threshold", type=float, default=0.05)
    parser.add_argument("--max-train-pairs", type=int, default=None)
    parser.add_argument("--early-stop-patience", type=int, default=3)
    args = parser.parse_args()

    result = train(
        records_path=args.records,
        learning_state_path=args.learning_state,
        output_path=args.output,
        epochs=args.epochs,
        dimension=args.dimension,
        hidden=args.hidden,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        f05_threshold=args.f05_threshold,
        max_train_pairs=args.max_train_pairs,
        early_stop_patience=args.early_stop_patience,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
