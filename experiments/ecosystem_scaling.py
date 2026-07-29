"""Measure ecosystem generation quality as evidence count scales.

For each held-out subject, builds evidence at increasing sizes:
the subject's own 3 facts always included + N-3 distractor facts
about unrelated subjects. The ecosystem must compose a response
from this evidence — the subject's facts are the ground truth.

Measures recall, unsupported_rate, and supported_count across
evidence sizes to determine whether more evidence helps or harms
the ecosystem's composition quality.

Usage:
    python experiments/ecosystem_scaling.py \\
        --spec checkpoints/scaling_spec.json \\
        --sizes 3 5 10 20 50
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from src.text.ecology import ResponseEcosystem, extract_propositions
from src.text.response_candidates import (
    FixedSpliceCandidateGenerator,
    SequenceCandidateScorer,
)


@dataclass
class SizeResult:
    evidence_count: int
    recall: float
    unsupported_rate: float
    contradiction: bool
    provenance_correct: bool
    supported_count: int
    total_ground_truth: int
    ecosystem_rounds: int
    latency_seconds: float


def build_sentences_for(subject: str, lessons_by_subject: dict) -> list[str]:
    """Build all evidence sentences about a subject from its lesson data."""
    content = lessons_by_subject[subject]
    name = content["subject"].capitalize()
    sentences = [f"{name} is {content['category']}."]
    if "place" in content:
        sentences.append(f"{name} is in {content['place']}.")
    if content.get("property"):
        sentences.append(f"{name} has {content['property']}.")
    else:
        sentences.append(f"The capital of {name} is {content['capital_city']}.")
    return sentences


def build_evidence(
    target_sentences: list[str],
    distractors: list[str],
    n: int,
    rng: random.Random,
) -> list[str]:
    """Build evidence of size n: all target sentences + n - len(target) distractors."""
    if n <= len(target_sentences):
        return target_sentences[:n]
    needed = n - len(target_sentences)
    chosen = rng.sample(distractors, min(needed, len(distractors)))
    return target_sentences + chosen


def evaluate_response(response_text: str, ground_truth: list) -> dict:
    asserted = extract_propositions([response_text])
    asserted_triples = {(p.subject, p.relation, p.object) for p in asserted}
    gt_triples = {(p.subject, p.relation, p.object) for p in ground_truth}
    gt_lookup = {(p.subject, p.relation): p.object for p in ground_truth}

    recall = (
        len(asserted_triples & gt_triples) / len(gt_triples)
        if gt_triples else 0.0
    )
    unsupported = asserted_triples - gt_triples
    unsupported_rate = (
        len(unsupported) / len(asserted_triples) if asserted_triples else 0.0
    )
    contradiction = any(
        (s, r) in gt_lookup and gt_lookup[(s, r)] != o
        for s, r, o in asserted_triples
    )
    return {
        "recall": recall,
        "unsupported_rate": unsupported_rate,
        "contradiction": contradiction,
        "provenance_correct": unsupported_rate == 0.0,
        "supported_count": len(asserted_triples & gt_triples),
    }


def run(
    spec: dict,
    sizes: list[int],
    seed: int = 0,
    max_rounds: int = 4,
    survivors_per_niche: int = 2,
) -> list[SizeResult]:
    rng = random.Random(seed)

    # Index lessons by subject
    lessons_by_subject: dict[str, dict] = {}
    for lesson in spec["lessons"]:
        lessons_by_subject[lesson["subject"]] = lesson
    holdout = spec["holdout"]

    # Build distractor pool: sentences about subjects NOT in holdout
    holdout_subjects = {h["subject"] for h in holdout}
    all_distractors: list[str] = []
    for subject, content in lessons_by_subject.items():
        if subject not in holdout_subjects:
            all_distractors.extend(build_sentences_for(subject, lessons_by_subject))

    scorer = SequenceCandidateScorer()
    generator = FixedSpliceCandidateGenerator()
    ecosystem = ResponseEcosystem(
        candidate_generator=generator,
        scorer=scorer,
        survivors_per_niche=survivors_per_niche,
        max_rounds=max_rounds,
        enable_synthesis=True,
    )

    results: list[SizeResult] = []
    for eval_subject in holdout:
        subject_name = eval_subject["subject"].capitalize()

        # Ground truth: all extractable propositions about this subject
        target_sentences = build_sentences_for(eval_subject["subject"],
                                                {eval_subject["subject"]: eval_subject})
        ground_truth = extract_propositions(target_sentences)
        if len(ground_truth) == 0:
            continue

        # Filter distractors that reference the target subject
        distractor_pool = [
            s for s in all_distractors
            if subject_name.lower() not in s.lower()
        ]

        for n in sizes:
            if n < len(target_sentences):
                continue
            evidence = build_evidence(target_sentences, distractor_pool, n, rng)

            t0 = time.perf_counter()
            result = ecosystem.generate(
                f"What do you know about {subject_name}?",
                evidence,
            )
            latency = time.perf_counter() - t0

            metrics = evaluate_response(result.response, ground_truth)
            size_result = SizeResult(
                evidence_count=n,
                ecosystem_rounds=len(result.population_sizes),
                latency_seconds=latency,
                total_ground_truth=len(ground_truth),
                **metrics,
            )
            results.append(size_result)

        print(f"  {eval_subject['subject']}: {len(sizes)} sizes done")

    return results


def summarize(results: list[SizeResult]) -> dict:
    by_size: dict[int, list[SizeResult]] = {}
    for r in results:
        by_size.setdefault(r.evidence_count, []).append(r)

    summary = {}
    for size in sorted(by_size):
        group = by_size[size]
        summary[size] = {
            "n": len(group),
            "mean_recall": statistics.mean(r.recall for r in group),
            "mean_supported_count": statistics.mean(r.supported_count for r in group),
            "mean_unsupported_rate": statistics.mean(r.unsupported_rate for r in group),
            "contradiction_rate": (
                sum(r.contradiction for r in group) / len(group)
            ),
            "provenance_correct_rate": (
                sum(r.provenance_correct for r in group) / len(group)
            ),
            "mean_ecosystem_rounds": (
                statistics.mean(r.ecosystem_rounds for r in group)
            ),
            "mean_latency_seconds": (
                statistics.mean(r.latency_seconds for r in group)
            ),
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=[3, 5, 10, 20, 50]
    )
    parser.add_argument("--max-rounds", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()

    spec = json.loads(args.spec.read_text())
    print(
        f"Spec: {len(spec['lessons'])} lessons, {len(spec['holdout'])} holdout"
    )
    print(f"Sizes: {args.sizes}")

    results = run(spec, args.sizes, max_rounds=args.max_rounds)
    summary = summarize(results)

    print(f"\nResults ({len(results)} evaluations):")
    print(json.dumps(summary, indent=2))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / (
        f"ecosystem_scaling_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    path.write_text(json.dumps({
        "config": {
            "spec_path": str(args.spec),
            "spec_lessons": len(spec["lessons"]),
            "spec_holdout": len(spec["holdout"]),
            "sizes": args.sizes,
            "max_rounds": args.max_rounds,
        },
        "summary": summary,
        "results": [asdict(r) for r in results],
    }, indent=2) + "\n")
    print(f"Report: {path}")


if __name__ == "__main__":
    main()
