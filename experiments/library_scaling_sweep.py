"""Phase 6 item 6: library scaling sweep.

Measures `native_generation_proof_rate`, latency, and `RelationalMemory`
retrieval accuracy (the established crosstalk proxy from
`RELATIONAL_MEMORY.md` SS2.1-2.2 -- reused here, not reinvented) as the
acquisition library grows. Uses `llm_teacher_curriculum.py`'s frozen spec
so every size level tests a *prefix* of the same underlying content, not
fresh LLM samples per size -- otherwise "does more data help" would be
confounded with "was this particular batch easier or harder."

"Collision rate" in `FrameLibrary` terms isn't a meaningful metric here:
templates are the literal observed phrasing, not a hash, so two
observations producing the same template is correct evidence
accumulation, not a collision. The metric that actually matters for
scaling is whether `RelationalMemory`'s per-relation Hopfield nets start
losing retrieval accuracy as more triples share a net -- so this script
measures that directly, on TAUGHT (not held-out) subjects, at each size.

Confirmed via an early validation run: `frames_learned` saturates almost
immediately (11/11 possible templates observed by ~5 lessons) because the
fixed template rotation deliberately cycles through every phrasing
variant quickly to establish diversity early -- it is NOT a useful
scaling signal on its own, reported for completeness only.
`relational_retrieval_accuracy` is the metric that actually moves with
scale: it dropped from 0.80 to 0.70 to 0.37 across 5/10/19 taught
subjects on the same content in that validation run, consistent with
expected Hopfield capacity/crosstalk behavior at this benchmark's default
`vsa_dim=64` -- a genuine, disclosed finding, not a target to silently
paper over by increasing `vsa_dim` in this script.

Usage:
    python experiments/library_scaling_sweep.py --spec checkpoints/llm_teacher_spec_v1.json --sizes 10 50 200
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

from llm_teacher_curriculum import _norm, _object_for, run

from src.text import BioAIDialogueAgent


def _relational_retrieval_accuracy(
    agent: BioAIDialogueAgent, lesson_contents: list[dict], sample_size: int = 20,
    seed: int = 0,
) -> float:
    rng = random.Random(seed)
    sample = rng.sample(lesson_contents, min(sample_size, len(lesson_contents)))
    correct = 0
    for content in sample:
        subject = _norm(content["subject"])
        expected = _norm(_object_for("is", content))
        result = agent.relational.complete_detailed({"subject": subject, "relation": "is"})
        if result.best == expected and not result.ambiguous:
            correct += 1
    return correct / len(sample) if sample else 0.0


def sweep(spec: dict, sizes: list[int]) -> list[dict]:
    results = []
    holdout_contents = spec["holdout"]
    for size in sizes:
        lesson_contents = spec["lessons"][:size]
        if len(lesson_contents) < size:
            print(f"  [size {size}] spec only has {len(lesson_contents)} lessons, skipping")
            continue
        sub_spec = {
            "lessons": lesson_contents, "holdout": holdout_contents,
            "adversarial_a": [], "adversarial_b": [],
        }
        result = run(
            model="unused (frozen spec, no LLM calls)", n_lessons=size,
            n_holdout=len(holdout_contents), spec=sub_spec,
        )
        agent = BioAIDialogueAgent(vsa_dim=64)
        from llm_teacher_curriculum import run_acquisition
        run_acquisition(agent, lesson_contents)
        retrieval_accuracy = _relational_retrieval_accuracy(agent, lesson_contents)

        row = {
            "size": size,
            "native_generation_proof_rate": result["native_generation_proof_rate"],
            "generation_success_rate": result["generation_success_rate"],
            "no_library_overlap_rate": result["no_library_overlap_rate"],
            "propositions_match_rate": result["propositions_match_rate"],
            "frames_learned": result["frames_learned"],
            "acquisition_latency_seconds": result["acquisition_latency_seconds"],
            "holdout_latency_seconds": result["holdout_latency_seconds"],
            "relational_retrieval_accuracy": retrieval_accuracy,
        }
        results.append(row)
        print(
            f"size={size}: native_generation_proof_rate="
            f"{row['native_generation_proof_rate']:.3f}, "
            f"relational_retrieval_accuracy={retrieval_accuracy:.3f}, "
            f"frames_learned={row['frames_learned']}, "
            f"acquisition_latency={row['acquisition_latency_seconds']:.3f}s"
        )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[10, 50, 200])
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()

    spec = json.loads(args.spec.read_text())
    results = sweep(spec, args.sizes)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / (
        f"library_scaling_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    path.write_text(json.dumps({"spec_path": str(args.spec), "results": results}, indent=2) + "\n")
    print(f"Report: {path}")


if __name__ == "__main__":
    main()
