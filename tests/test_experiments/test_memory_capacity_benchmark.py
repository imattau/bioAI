import json

import pytest

from experiments.memory_capacity_benchmark import (
    BenchmarkConfig,
    run_benchmark,
    save_results,
    wilson_interval,
)


def test_wilson_interval_contains_observed_accuracy():
    low, high = wilson_interval(8, 10)
    assert low < 0.8 < high


def test_small_benchmark_is_reproducible():
    config = BenchmarkConfig(
        dimensions=(16,),
        load_ratios=(0.125,),
        noise_levels=(0.0, 0.5),
        repeats=2,
        queries_per_repeat=2,
        recall_steps=2,
        base_seed=7,
    )
    first = run_benchmark(config)
    second = run_benchmark(config)
    assert len(first) == 2
    for left, right in zip(first, second):
        assert left.correct == right.correct
        assert left.trials == right.trials == 4
        assert left.accuracy == right.accuracy
        assert 0 <= left.accuracy_ci95_low <= left.accuracy_ci95_high <= 1
        assert left.latency_p50_ms >= 0
        assert left.storage_mib > 0


def test_save_results_writes_json_and_csv(tmp_path):
    config = BenchmarkConfig(
        dimensions=(8,),
        load_ratios=(0.125,),
        noise_levels=(0.0,),
        repeats=1,
        queries_per_repeat=1,
        recall_steps=1,
    )
    results = run_benchmark(config)
    json_path, csv_path = save_results(config, results, tmp_path)
    payload = json.loads(json_path.read_text())
    assert payload["config"]["base_seed"] == 42
    assert payload["results"][0]["trials"] == 1
    assert csv_path.read_text().startswith(
        "pattern_source,retrieval_mode,dimension,capacity,"
    )


def test_invalid_config_is_rejected():
    with pytest.raises(ValueError, match="repeats"):
        run_benchmark(BenchmarkConfig(repeats=0))


def test_text_pattern_source_runs():
    results = run_benchmark(BenchmarkConfig(
        dimensions=(16,),
        load_ratios=(0.125,),
        noise_levels=(0.0,),
        repeats=1,
        queries_per_repeat=1,
        recall_steps=1,
        pattern_source="text",
    ))
    assert results[0].pattern_source == "text"
