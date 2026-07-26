"""Incremental retrieval experiment from 100 to one million memories."""

from __future__ import annotations

import argparse
import json
import resource
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from src.text import TokenLibrary


CHECKPOINTS = (100, 1_000, 10_000, 100_000, 500_000, 1_000_000)
QUESTIONS_PER_BAND = 20


def target_fact(question_id: int) -> str:
    return (
        f"The capital of regioncode{question_id} "
        f"is citycode{question_id}."
    )


def question_text(question_id: int) -> str:
    return (
        f"Which city serves as the administrative centre "
        f"of regioncode{question_id}?"
    )


def insertion_schedule() -> dict[int, int]:
    schedule = {}
    previous = 0
    question_id = 0
    for checkpoint in CHECKPOINTS:
        span = checkpoint - previous
        for offset in range(QUESTIONS_PER_BAND):
            rank = previous + 1 + (offset * max(1, span - 1)) // QUESTIONS_PER_BAND
            schedule[rank] = question_id
            question_id += 1
        previous = checkpoint
    return schedule


def distractor_text(index: int) -> str:
    region = index % (len(CHECKPOINTS) * QUESTIONS_PER_BAND)
    metro = index % 10_000
    relation = ("largest", "oldest", "nearest", "coastal")[index % 4]
    return (
        f"The {relation} city associated with regioncode{region} "
        f"is metrocode{metro}."
    )


def content_words(library: TokenLibrary, text: str) -> set[str]:
    stop = {
        "a", "an", "the", "is", "are", "was", "were", "of", "to",
        "in", "on", "at", "for", "and", "or", "what", "who", "where",
        "when", "why", "how", "which", "do", "does", "did",
    }
    return {word for word in library.tokenize(text) if word not in stop}


def retrieve(library: TokenLibrary, question: str) -> tuple[int | None, bool, float, float]:
    indexed = library.candidate_ids(question, limit=100)
    query_words = content_words(library, question)
    query_concepts = library.semantic_terms(question)
    reranked = []
    for sentence_id, _ in indexed:
        text = library.texts[sentence_id]
        memory_words = content_words(library, text)
        lexical = len(query_words & memory_words) / len(query_words)
        memory_concepts = library.semantic_terms(text)
        semantic = (
            len(query_concepts & memory_concepts)
            / min(len(query_concepts), len(memory_concepts))
        )
        score = 0.375 * lexical + 0.625 * semantic
        reranked.append((sentence_id, score))
    reranked.sort(key=lambda item: item[1], reverse=True)
    if not reranked:
        return None, False, 0.0, 0.0
    best_id, best_score = reranked[0]
    second_score = reranked[1][1] if len(reranked) > 1 else -1.0
    margin = best_score - second_score
    return best_id, best_score >= 0.38 and margin >= 0.1, best_score, margin


def evaluate(library: TokenLibrary, target_ids: dict[int, int]) -> dict:
    latencies = []
    correct = 0
    present = 0
    rejected_present = 0
    false_accepted_absent = 0
    accepted = 0
    total_questions = len(CHECKPOINTS) * QUESTIONS_PER_BAND
    for question_id in range(total_questions):
        started = time.perf_counter_ns()
        predicted_id, was_accepted, _, _ = retrieve(
            library, question_text(question_id)
        )
        latencies.append((time.perf_counter_ns() - started) / 1_000_000)
        target_id = target_ids.get(question_id)
        if target_id is not None:
            present += 1
            correct += int(was_accepted and predicted_id == target_id)
            rejected_present += int(not was_accepted)
        else:
            false_accepted_absent += int(was_accepted)
        accepted += int(was_accepted)
    ordered = sorted(latencies)
    return {
        "library_size": len(library),
        "questions": total_questions,
        "answerable": present,
        "coverage_pct": 100 * present / total_questions,
        "correct": correct,
        "answerable_recall_pct": 100 * correct / present,
        "end_to_end_accuracy_pct": 100 * correct / total_questions,
        "rejected_answerable": rejected_present,
        "false_accepted_unanswerable": false_accepted_absent,
        "accepted": accepted,
        "latency_p50_ms": statistics.median(ordered),
        "latency_p95_ms": ordered[int(0.95 * (len(ordered) - 1))],
        "token_storage_mib": library.token_storage_bytes / (1024 * 1024),
    }


def run(max_size: int = 1_000_000) -> dict:
    library = TokenLibrary(vector_cache_size=0)
    schedule = insertion_schedule()
    target_ids = {}
    results = []
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    started = time.perf_counter()
    for rank in range(1, max_size + 1):
        question_id = schedule.get(rank)
        if question_id is None:
            library.add(distractor_text(rank))
        else:
            target_ids[question_id] = library.add(target_fact(question_id))
        if rank in CHECKPOINTS:
            result = evaluate(library, target_ids)
            result["elapsed_ingestion_seconds"] = time.perf_counter() - started
            result["process_peak_rss_delta_mib"] = (
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - rss_before
            ) / 1024
            results.append(result)
    return {
        "checkpoints": list(CHECKPOINTS),
        "questions_per_band": QUESTIONS_PER_BAND,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-size", type=int, default=1_000_000)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()
    report = run(args.max_size)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = args.output_dir / f"library_size_sweep_{stamp}.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Report: {path}")


if __name__ == "__main__":
    main()
