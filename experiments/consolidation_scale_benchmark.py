"""Large-scale benchmark for evidence-backed semantic consolidation.

Measures noisy concept learning, relation conflict retention, lookup latency,
memory growth, and persisted size across an observation sweep.
"""

from __future__ import annotations

import argparse
import json
import resource
import statistics
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch

from src.text import ConsolidationMemory


DEFAULT_CHECKPOINTS = (10_000, 100_000, 1_000_000)


@dataclass(frozen=True)
class Config:
    observations: int = 1_000_000
    concepts: int = 10_000
    dimension: int = 256
    queries: int = 20
    noise: float = 0.1
    relation_every: int = 10
    relation_subjects: int = 10_000
    seed: int = 47


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[int(fraction * (len(ordered) - 1))]


def noisy(vector: torch.Tensor, generator: torch.Generator, rate: float):
    if rate <= 0:
        return vector
    flips = torch.rand(len(vector), generator=generator) < rate
    return torch.where(flips, -vector, vector)


def evaluate(
    memory: ConsolidationMemory,
    bases: torch.Tensor,
    query_count: int,
) -> dict:
    latencies = []
    correct = 0
    tested = min(query_count, len(memory.concepts))
    if tested == 0:
        return {
            "prototype_queries": 0,
            "prototype_top1_accuracy_pct": 0.0,
            "prototype_query_cold_ms": 0.0,
            "prototype_query_p50_ms": 0.0,
            "prototype_query_p95_ms": 0.0,
        }
    concept_ids = torch.linspace(
        0, len(memory.concepts) - 1, steps=tested
    ).to(torch.int64)
    for concept_id in concept_ids.tolist():
        started = time.perf_counter_ns()
        matches = memory.query_concepts(bases[concept_id], limit=1)
        latencies.append((time.perf_counter_ns() - started) / 1_000_000)
        correct += int(
            bool(matches) and matches[0]["term"] == f"concept{concept_id}"
        )
    return {
        "prototype_queries": tested,
        "prototype_top1_accuracy_pct": 100 * correct / tested,
        "prototype_query_cold_ms": latencies[0],
        "prototype_query_warm_p50_ms": statistics.median(
            latencies[1:] or latencies
        ),
        "prototype_query_p50_ms": statistics.median(latencies),
        "prototype_query_p95_ms": percentile(latencies, 0.95),
    }


def run(config: Config, checkpoints: tuple[int, ...] | None = None) -> dict:
    checkpoints = checkpoints or tuple(
        point for point in DEFAULT_CHECKPOINTS
        if point <= config.observations
    )
    if not checkpoints or checkpoints[-1] != config.observations:
        checkpoints = (*checkpoints, config.observations)
    generator = torch.Generator().manual_seed(config.seed)
    bases = torch.where(
        torch.rand(
            (config.concepts, config.dimension), generator=generator
        ) >= 0.5,
        torch.ones((), dtype=torch.int8),
        -torch.ones((), dtype=torch.int8),
    )
    memory = ConsolidationMemory(
        promotion_threshold=3,
        max_prototypes=config.concepts,
    )
    pending: dict[int, list[tuple[int, torch.Tensor]]] = {}
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    started = time.perf_counter()
    results = []

    for observation in range(config.observations):
        concept_id = observation % config.concepts
        term = f"concept{concept_id}"
        vector = noisy(bases[concept_id], generator, config.noise)
        if term in memory.concepts:
            memory.observe_concept(term, vector, observation)
        else:
            evidence = pending.setdefault(concept_id, [])
            evidence.append((observation, vector))
            if len(evidence) == memory.promotion_threshold:
                memory.observe_concept(
                    term,
                    vector,
                    observation,
                    initial_vectors=[item[1] for item in evidence],
                    initial_source_ids=[item[0] for item in evidence],
                )
                del pending[concept_id]

        if observation % config.relation_every == 0:
            claim_id = observation // config.relation_every
            subject_id = claim_id % config.relation_subjects
            # A systematic minority conflict is retained as a second object.
            object_id = (
                subject_id + 1
                if claim_id % 11 == 0 else subject_id
            )
            memory.observe_relation(
                f"The capital of region{subject_id} is city{object_id}",
                observation,
            )

        count = observation + 1
        if count in checkpoints:
            elapsed = time.perf_counter() - started
            result = {
                "observations": count,
                "promoted_concepts": len(memory.concepts),
                "relation_keys": len(memory.relations),
                "relation_alternatives": sum(
                    len(alternatives)
                    for alternatives in memory.relations.values()
                ),
                "ingestion_seconds": elapsed,
                "ingestion_observations_per_second": count / elapsed,
                "process_peak_rss_delta_mib": (
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    - rss_before
                ) / 1024,
                **evaluate(memory, bases, config.queries),
            }
            results.append(result)

    with tempfile.NamedTemporaryFile(suffix=".pt") as destination:
        save_started = time.perf_counter()
        torch.save(memory.get_state(), destination.name)
        save_seconds = time.perf_counter() - save_started
        persisted_bytes = Path(destination.name).stat().st_size
        load_started = time.perf_counter()
        restored = ConsolidationMemory.from_state(
            torch.load(destination.name, weights_only=False)
        )
        load_seconds = time.perf_counter() - load_started

    conflicting = sum(
        len(alternatives) > 1
        for alternatives in memory.relations.values()
    )
    final = results[-1]
    final["persisted_mib"] = persisted_bytes / (1024 * 1024)
    final["bytes_per_observation"] = persisted_bytes / config.observations
    raw_semantic_bytes = config.observations * config.dimension / 8
    final["raw_bitpacked_episode_semantic_mib"] = (
        raw_semantic_bytes / (1024 * 1024)
    )
    final["semantic_compression_ratio"] = raw_semantic_bytes / persisted_bytes
    final["save_seconds"] = save_seconds
    final["load_seconds"] = load_seconds
    final["reload_counts_match"] = (
        len(restored.concepts) == len(memory.concepts)
        and len(restored.relations) == len(memory.relations)
    )
    final["conflicting_relation_keys"] = conflicting
    final["conflicts_retained_pct"] = (
        100 * conflicting / len(memory.relations)
        if memory.relations else 0.0
    )
    return {
        "benchmark": "semantic_consolidation_scale",
        "config": config.__dict__,
        "checkpoints": list(checkpoints),
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=int, default=1_000_000)
    parser.add_argument("--concepts", type=int, default=10_000)
    parser.add_argument("--dimension", type=int, default=256)
    parser.add_argument("--queries", type=int, default=20)
    parser.add_argument("--noise", type=float, default=0.1)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()
    config = Config(
        observations=args.observations,
        concepts=args.concepts,
        dimension=args.dimension,
        queries=args.queries,
        noise=args.noise,
    )
    report = run(config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = args.output_dir / f"consolidation_scale_{stamp}.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Report: {path}")


if __name__ == "__main__":
    main()
