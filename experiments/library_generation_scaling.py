"""Measure generation quality as a function of acquired library size.

Loads a frozen spec (lessons + holdout) and sweeps across lesson-count
prefixes, recording native_generation_proof_rate (the primary metric),
its component rates, relational_retrieval_accuracy on taught subjects,
frames_learned, and latency at each size.

This is the experiment that bridges the gap between:
  - library_scaling_sweep.py (sizes 5/10/19, all metrics saturated at 1.0)
  - library_response_scaling.py (2.5K-30K pairs, token-f1 ~15%, no
    proposition-level evaluation)

Usage:
    1. Generate a frozen spec (requires Ollama with an LLM):
       python experiments/llm_teacher_curriculum.py \\
           --lessons 300 --holdout 100 --model qwen2.5-coder:1.5b \\
           --save-spec checkpoints/large_spec.json

    2. Run the sweep (no LLM calls):
       python experiments/library_generation_scaling.py \\
           --spec checkpoints/large_spec.json \\
           --sizes 10 30 60 100 150 200 250 300
"""

from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path

from llm_teacher_curriculum import (
    _norm,
    _object_for,
    evaluate_holdout,
    run_acquisition,
)
from src.text import BioAIDialogueAgent


def relational_retrieval_accuracy(
    agent: BioAIDialogueAgent, lesson_contents: list[dict],
    sample_size: int = 40, seed: int = 0,
) -> float:
    rng = random.Random(seed)
    sample = rng.sample(lesson_contents, min(sample_size, len(lesson_contents)))
    correct = 0
    for content in sample:
        subject = _norm(content["subject"])
        expected = _norm(_object_for("is", content))
        result = agent.relational.complete_detailed(
            {"subject": subject, "relation": "is"}
        )
        if result.best == expected and not result.ambiguous:
            correct += 1
    return correct / len(sample) if sample else 0.0


def sweep(spec: dict, sizes: list[int]) -> list[dict]:
    results = []
    for size in sizes:
        lesson_contents = spec["lessons"][:size]
        if len(lesson_contents) < size:
            print(
                f"  [size {size}] spec only has {len(lesson_contents)} "
                f"lessons, skipping"
            )
            continue

        print(f"\n{'='*60}")
        print(f"Size: {size} lessons")
        print(f"{'='*60}")

        agent = BioAIDialogueAgent(vsa_dim=64)
        agent.gonogo_gate_enabled = False

        t0 = time.perf_counter()
        acquisition_records = run_acquisition(agent, lesson_contents)
        acquisition_latency = time.perf_counter() - t0
        print(
            f"  Acquisition: {len(acquisition_records)} lessons "
            f"({3 * len(acquisition_records)} sentences) "
            f"in {acquisition_latency:.2f}s"
        )

        taught_sentences = [
            s for r in acquisition_records for s in r.sentences
        ]
        taught_subjects = {r.subject for r in acquisition_records}

        t0 = time.perf_counter()
        holdout_results = evaluate_holdout(
            spec["holdout"], agent.frame_library,
            taught_sentences, taught_subjects,
        )
        holdout_latency = time.perf_counter() - t0

        def rate(flag: str) -> float:
            if not holdout_results:
                return 0.0
            return (
                sum(getattr(r, flag) for r in holdout_results)
                / len(holdout_results)
            )

        all_three_rate = (
            sum(
                r.generation_success and r.no_library_overlap
                and r.propositions_match
                for r in holdout_results
            ) / len(holdout_results)
            if holdout_results else 0.0
        )

        frames_learned = len(agent.frame_library.frames)
        retrieval_acc = relational_retrieval_accuracy(agent, lesson_contents)

        row = {
            "size": size,
            "lessons_taught": len(acquisition_records),
            "acquisition_latency_seconds": round(acquisition_latency, 3),
            "holdout_latency_seconds": round(holdout_latency, 3),
            "holdout_evaluated": len(holdout_results),
            "frames_learned": frames_learned,
            "generation_success_rate": round(rate("generation_success"), 4),
            "no_library_overlap_rate": round(rate("no_library_overlap"), 4),
            "propositions_match_rate": round(rate("propositions_match"), 4),
            "native_generation_proof_rate": round(all_three_rate, 4),
            "relational_retrieval_accuracy": round(retrieval_acc, 4),
        }
        results.append(row)

        print(
            f"  Holdout: {len(holdout_results)} targets in "
            f"{holdout_latency:.2f}s"
        )
        print(
            f"  native_generation_proof_rate={all_three_rate:.4f}, "
            f"propositions_match={rate('propositions_match'):.4f}, "
            f"relational_retrieval_accuracy={retrieval_acc:.4f}, "
            f"frames_learned={frames_learned}"
        )

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec", type=Path, required=True,
        help="Path to frozen spec JSON (lessons + holdout lists)",
    )
    parser.add_argument(
        "--sizes", type=int, nargs="+",
        default=[10, 30, 60, 100, 150, 200, 250, 300],
        help="Lesson-count prefixes to evaluate (default: full sweep)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("checkpoints"),
    )
    args = parser.parse_args()

    spec = json.loads(args.spec.read_text())
    print(
        f"Loaded spec: {len(spec['lessons'])} lessons, "
        f"{len(spec['holdout'])} holdout"
    )
    print(f"Sweep sizes: {args.sizes}")

    results = sweep(spec, args.sizes)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / (
        f"library_generation_scaling_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    report = {
        "spec_path": str(args.spec),
        "spec_lessons": len(spec["lessons"]),
        "spec_holdout": len(spec["holdout"]),
        "results": results,
    }
    path.write_text(json.dumps(report, indent=2) + "\n")

    print(f"\n{'='*60}")
    print(f"Report: {path}")
    print(f"{'='*60}")
    for row in results:
        print(
            f"  size={row['size']:>4d}: "
            f"proof={row['native_generation_proof_rate']:.4f} "
            f"propositions_match={row['propositions_match_rate']:.4f} "
            f"success={row['generation_success_rate']:.4f} "
            f"overlap={row['no_library_overlap_rate']:.4f} "
            f"relational_acc={row['relational_retrieval_accuracy']:.4f} "
            f"frames={row['frames_learned']} "
            f"acq={row['acquisition_latency_seconds']:.2f}s "
            f"eval={row['holdout_latency_seconds']:.2f}s"
        )


if __name__ == "__main__":
    main()
