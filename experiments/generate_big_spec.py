"""Generate a large spec with guaranteed disjoint lessons and holdout.

Strategy: generate a big pool of items, collect unique subjects,
then split the first N for lessons and the remaining for holdout.

Usage:
    python experiments/generate_big_spec.py \\
        --model qwen2.5-coder:1.5b \\
        --total 500 \\
        --output checkpoints/scaling_spec.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from llm_teacher_curriculum import generate_contents


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen2.5-coder:1.5b")
    parser.add_argument("--total", type=int, default=500)
    parser.add_argument("--output", type=Path,
                        default=Path("checkpoints/scaling_spec.json"))
    args = parser.parse_args()

    # Generate extra because ~10-15% fail validation
    t0 = time.perf_counter()
    raw = generate_contents(args.model, int(args.total * 1.3), start_index=0)
    elapsed = time.perf_counter() - t0
    print(f"Generated {len(raw)} items in {elapsed:.1f}s")

    # Deduplicate by subject, keep first occurrence
    seen: set[str] = set()
    unique = []
    for item in raw:
        subj = item["subject"].lower()
        if subj not in seen:
            seen.add(subj)
            unique.append(item)
    print(f"{len(unique)} unique subjects after dedup")

    if len(unique) < args.total:
        print(f"WARNING: only {len(unique)} unique subjects, "
              f"requested {args.total}")

    # Split: 80% lessons, 20% holdout
    n_holdout = max(50, int(len(unique) * 0.2))
    n_lessons = len(unique) - n_holdout
    lessons = unique[:n_lessons]
    holdout = unique[n_lessons:]

    spec = {"lessons": lessons, "holdout": holdout}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(spec, indent=2) + "\n")
    print(f"Saved: {len(lessons)} lessons + {len(holdout)} holdout "
          f"to {args.output}")


if __name__ == "__main__":
    main()
