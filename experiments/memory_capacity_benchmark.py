"""Reproducible Hopfield memory capacity benchmark.

Sweeps hypervector dimension, memory load, and cue noise. Each configuration
is repeated with independent seeds and reports recall accuracy with a Wilson
95% confidence interval, p50/p95 recall latency, and storage memory.

Example:
    python -m experiments.memory_capacity_benchmark --output-dir checkpoints
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch

from src.vsa import HopfieldNet, VSA


@dataclass(frozen=True)
class BenchmarkConfig:
    dimensions: tuple[int, ...] = (64, 128, 256)
    load_ratios: tuple[float, ...] = (0.05, 0.10, 0.20)
    noise_levels: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0)
    repeats: int = 5
    queries_per_repeat: int = 8
    recall_steps: int = 10
    base_seed: int = 42
    pattern_source: str = "random"
    retrieval_mode: str = "classical"


@dataclass(frozen=True)
class BenchmarkResult:
    pattern_source: str
    retrieval_mode: str
    dimension: int
    capacity: int
    load_ratio: float
    noise: float
    repeats: int
    trials: int
    correct: int
    accuracy: float
    accuracy_ci95_low: float
    accuracy_ci95_high: float
    latency_p50_ms: float
    latency_p95_ms: float
    storage_mib: float


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    if trials <= 0:
        return 0.0, 0.0
    proportion = successes / trials
    denominator = 1 + z * z / trials
    centre = (proportion + z * z / (2 * trials)) / denominator
    margin = z * math.sqrt(
        proportion * (1 - proportion) / trials + z * z / (4 * trials * trials)
    ) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def _validate_config(config: BenchmarkConfig) -> None:
    if not config.dimensions or any(d <= 0 for d in config.dimensions):
        raise ValueError("dimensions must contain positive integers")
    if not config.load_ratios or any(r <= 0 for r in config.load_ratios):
        raise ValueError("load_ratios must contain positive values")
    if not config.noise_levels or any(n < 0 for n in config.noise_levels):
        raise ValueError("noise_levels must contain non-negative values")
    if config.repeats <= 0 or config.queries_per_repeat <= 0:
        raise ValueError("repeats and queries_per_repeat must be positive")
    if config.pattern_source not in {"random", "text"}:
        raise ValueError("pattern_source must be 'random' or 'text'")
    if config.retrieval_mode not in {"classical", "modern"}:
        raise ValueError("retrieval_mode must be 'classical' or 'modern'")


def _make_patterns(vsa: VSA, capacity: int, source: str) -> torch.Tensor:
    if source == "random":
        return vsa.make_vectors(capacity)

    # Deliberately overlapping sentences exercise structured, correlated inputs.
    from src.text.encoder import VSAEncoder

    encoder = VSAEncoder(vsa=vsa)
    texts = [
        f"entity {index} has color {index % 7} in region {index % 11}"
        for index in range(capacity)
    ]
    return encoder.encode_batch(texts)


def run_benchmark(config: BenchmarkConfig) -> list[BenchmarkResult]:
    _validate_config(config)
    results = []

    for dimension in config.dimensions:
        for load_index, load_ratio in enumerate(config.load_ratios):
            capacity = max(1, round(dimension * load_ratio))
            cells = {
                noise: {"correct": 0, "latencies": []}
                for noise in config.noise_levels
            }

            for repeat in range(config.repeats):
                seed = config.base_seed + dimension * 10_000 + load_index * 1_000 + repeat
                torch.manual_seed(seed)
                vsa = VSA(dim=dimension, device="cpu")
                patterns = _make_patterns(vsa, capacity, config.pattern_source)
                memory = HopfieldNet(
                    dim=dimension, retrieval_mode=config.retrieval_mode
                )
                memory.store_batch(list(patterns))
                query_count = min(capacity, config.queries_per_repeat)
                query_indices = torch.randperm(capacity)[:query_count]

                for noise in config.noise_levels:
                    for target_index in query_indices.tolist():
                        cue = patterns[target_index] + noise * torch.randn(dimension)
                        started = time.perf_counter_ns()
                        recalled = memory.recall(cue, steps=config.recall_steps)
                        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
                        predicted = torch.mv(patterns, recalled).argmax().item()
                        cells[noise]["correct"] += int(predicted == target_index)
                        cells[noise]["latencies"].append(elapsed_ms)

            storage_bytes = (dimension * dimension + capacity * dimension) * 4
            for noise in config.noise_levels:
                latencies = cells[noise]["latencies"]
                trials = len(latencies)
                correct = cells[noise]["correct"]
                ci_low, ci_high = wilson_interval(correct, trials)
                results.append(BenchmarkResult(
                    pattern_source=config.pattern_source,
                    retrieval_mode=config.retrieval_mode,
                    dimension=dimension,
                    capacity=capacity,
                    load_ratio=load_ratio,
                    noise=noise,
                    repeats=config.repeats,
                    trials=trials,
                    correct=correct,
                    accuracy=correct / trials,
                    accuracy_ci95_low=ci_low,
                    accuracy_ci95_high=ci_high,
                    latency_p50_ms=statistics.median(latencies),
                    latency_p95_ms=percentile(latencies, 0.95),
                    storage_mib=storage_bytes / (1024 * 1024),
                ))
    return results


def save_results(
    config: BenchmarkConfig,
    results: list[BenchmarkResult],
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = output_dir / f"memory_capacity_{stamp}.json"
    csv_path = output_dir / f"memory_capacity_{stamp}.csv"
    payload = {
        "benchmark": "hopfield_capacity_noise_dimension",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "results": [asdict(result) for result in results],
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)
    return json_path, csv_path


def _csv_numbers(value: str, cast):
    return tuple(cast(item.strip()) for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimensions", default="64,128,256")
    parser.add_argument("--load-ratios", default="0.05,0.10,0.20")
    parser.add_argument("--noise-levels", default="0,0.25,0.5,1.0")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--queries-per-repeat", type=int, default=8)
    parser.add_argument("--recall-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pattern-source",
        choices=("random", "text"),
        default="random",
        help="Use independent random vectors or correlated position-bound text vectors.",
    )
    parser.add_argument(
        "--retrieval-mode",
        choices=("classical", "modern"),
        default="classical",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()
    config = BenchmarkConfig(
        dimensions=_csv_numbers(args.dimensions, int),
        load_ratios=_csv_numbers(args.load_ratios, float),
        noise_levels=_csv_numbers(args.noise_levels, float),
        repeats=args.repeats,
        queries_per_repeat=args.queries_per_repeat,
        recall_steps=args.recall_steps,
        base_seed=args.seed,
        pattern_source=args.pattern_source,
        retrieval_mode=args.retrieval_mode,
    )
    results = run_benchmark(config)
    json_path, csv_path = save_results(config, results, args.output_dir)
    print(f"Completed {len(results)} configurations.")
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")


if __name__ == "__main__":
    main()
