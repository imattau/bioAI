"""Regenerate holdout for the large spec ensuring disjoint subjects.

The existing large_spec_300.json has 328 lessons but only 9 usable holdout
items because most holdout subjects were also taught. This script keeps the
328 lessons and generates fresh holdout items with guaranteed novel subjects.

Usage:
    python experiments/fix_scaling_spec.py \\
        --model qwen2.5-coder:1.5b \\
        --spec checkpoints/large_spec_300.json \\
        --output checkpoints/scaling_spec.json \\
        --holdout 100
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
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=Path("checkpoints/scaling_spec.json"))
    parser.add_argument("--holdout", type=int, default=100)
    args = parser.parse_args()

    spec = json.loads(args.spec.read_text())
    lessons = spec["lessons"]

    seen_subjects = {c["subject"].lower() for c in lessons}
    print(f"Using {len(lessons)} lessons with {len(seen_subjects)} unique subjects")

    # Generate extra to account for duplicates and overlap
    t0 = time.perf_counter()
    raw = generate_contents(args.model, int(args.holdout * 2.5), start_index=50000)
    elapsed = time.perf_counter() - t0
    print(f"Generated {len(raw)} raw holdout items in {elapsed:.1f}s")

    holdout = []
    for item in raw:
        subj = item["subject"].lower()
        if subj not in seen_subjects:
            seen_subjects.add(subj)
            holdout.append(item)
            if len(holdout) >= args.holdout:
                break

    print(f"Got {len(holdout)} unique-subject holdout items")

    out = {"lessons": lessons, "holdout": holdout}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(f"Saved spec to {args.output}")


if __name__ == "__main__":
    main()
