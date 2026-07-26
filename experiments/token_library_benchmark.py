"""Scale benchmark for compact tokenised long-term storage."""

import argparse
import json
import statistics
import time
import resource
from datetime import datetime, timezone
from pathlib import Path

from src.text import TokenLibrary


def run(size: int, queries: int) -> dict:
    rss_before_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    library = TokenLibrary(vector_cache_size=128)
    semantic_id = size // 2

    def text_for(index):
        if index == semantic_id:
            return "Paris is the capital of France"
        return (
            f"record {index} describes topic {index % 1000} "
            f"with unique value item{index}"
        )

    started = time.perf_counter()
    for index in range(size):
        library.add(text_for(index))
    ingestion_seconds = time.perf_counter() - started
    rss_after_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    exact_latencies = []
    candidate_latencies = []
    common_latencies = []
    correct = 0
    for query_index in range(queries):
        index = (query_index * 7919) % size
        text = text_for(index)
        started = time.perf_counter_ns()
        exact = library.exact_lookup(text)
        exact_latencies.append((time.perf_counter_ns() - started) / 1_000_000)
        started = time.perf_counter_ns()
        candidates = library.candidate_ids(
            f"what has unique value item{index}", limit=10
        )
        candidate_latencies.append(
            (time.perf_counter_ns() - started) / 1_000_000
        )
        started = time.perf_counter_ns()
        library.candidate_ids("record describes topic", limit=10)
        common_latencies.append(
            (time.perf_counter_ns() - started) / 1_000_000
        )
        correct += int(exact == text and candidates and candidates[0][0] == index)

    started = time.perf_counter_ns()
    semantic_candidates = library.candidate_ids(
        "Which city is France's administrative centre?", limit=10
    )
    semantic_latency = (time.perf_counter_ns() - started) / 1_000_000

    return {
        "sentences": size,
        "tokens": len(library.token_ids),
        "vocabulary": len(library.id_to_token),
        "token_storage_bytes": library.token_storage_bytes,
        "offset_storage_bytes": (
            library.offsets.buffer_info()[1] * library.offsets.itemsize
        ),
        "process_peak_rss_delta_mib": (rss_after_kib - rss_before_kib) / 1024,
        "ingestion_seconds": ingestion_seconds,
        "ingestion_sentences_per_second": size / ingestion_seconds,
        "queries": queries,
        "correct": correct,
        "accuracy": correct / queries,
        "exact_latency_p50_ms": statistics.median(exact_latencies),
        "exact_latency_p95_ms": sorted(exact_latencies)[int(0.95 * (queries - 1))],
        "candidate_latency_p50_ms": statistics.median(candidate_latencies),
        "candidate_latency_p95_ms": sorted(candidate_latencies)[
            int(0.95 * (queries - 1))
        ],
        "common_latency_p50_ms": statistics.median(common_latencies),
        "common_latency_p95_ms": sorted(common_latencies)[
            int(0.95 * (queries - 1))
        ],
        "semantic_correct": bool(
            semantic_candidates and semantic_candidates[0][0] == semantic_id
        ),
        "semantic_latency_ms": semantic_latency,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sentences", type=int, default=500_000)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()
    result = run(args.sentences, args.queries)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = args.output_dir / f"token_library_{stamp}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"Report: {path}")


if __name__ == "__main__":
    main()
