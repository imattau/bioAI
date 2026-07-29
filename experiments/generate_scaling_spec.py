"""Generate a large frozen spec for library_generation_scaling.py.

Separate lesson and holdout generation into two non-overlapping subject
pools so holdout evaluation has enough unique targets. Uses two rounds
with different start indices and deduplication by subject.

Usage:
    python experiments/generate_scaling_spec.py \\
        --model qwen2.5-coder:1.5b \\
        --lessons 350 --holdout 120 \\
        --output checkpoints/scaling_spec.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

# Must run from project root or set PYTHONPATH
from llm_teacher_curriculum import generate_contents


def generate_disjoint_spec(
    model: str,
    n_lessons: int,
    n_holdout: int,
) -> dict:
    seen_subjects: set[str] = set()
    _norm = str.lower

    # Phase 1: generate lessons
    print(f"Generating {n_lessons} lessons (start_index=0)...")
    raw_lessons = generate_contents(model, n_lessons, start_index=0)
    # Deduplicate by subject, keeping first occurrence
    lessons, lesson_subjects = [], set()
    for item in raw_lessons:
        subj = _norm(item["subject"])
        if subj not in lesson_subjects:
            lesson_subjects.add(subj)
            lessons.append(item)
    print(f"  -> {len(lessons)} unique-subject lessons (dropped "
          f"{len(raw_lessons) - len(lessons)} duplicates)")

    # Phase 2: generate holdout items with subjects NOT in lessons
    seen_subjects.update(lesson_subjects)
    print(f"Generating {int(n_holdout * 2.5)} raw items for holdout "
          f"(start_index=50000)...")
    raw_holdout = generate_contents(model, int(n_holdout * 2.5), start_index=50000)
    holdout = []
    for item in raw_holdout:
        subj = _norm(item["subject"])
        if subj not in seen_subjects:
            seen_subjects.add(subj)
            holdout.append(item)
            if len(holdout) >= n_holdout:
                break
    print(f"  -> {len(holdout)} unique holdout subjects (out of "
          f"{len(raw_holdout)} generated)")

    return {"lessons": lessons, "holdout": holdout}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen2.5-coder:1.5b")
    parser.add_argument("--lessons", type=int, default=350)
    parser.add_argument("--holdout", type=int, default=120)
    parser.add_argument("--output", type=Path,
                        default=Path("checkpoints/scaling_spec.json"))
    args = parser.parse_args()

    t0 = time.perf_counter()
    spec = generate_disjoint_spec(args.model, args.lessons, args.holdout)
    elapsed = time.perf_counter() - t0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(spec, indent=2) + "\n")
    print(f"\nSaved {len(spec['lessons'])} lessons + {len(spec['holdout'])} "
          f"holdout to {args.output} in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
