from experiments.consolidation_scale_benchmark import Config, run


def test_small_consolidation_scale_benchmark():
    report = run(
        Config(
            observations=120,
            concepts=10,
            dimension=64,
            queries=5,
            noise=0.05,
            relation_every=2,
            relation_subjects=10,
        ),
        checkpoints=(30, 120),
    )
    assert report["benchmark"] == "semantic_consolidation_scale"
    assert [row["observations"] for row in report["results"]] == [30, 120]
    final = report["results"][-1]
    assert final["promoted_concepts"] == 10
    assert final["prototype_top1_accuracy_pct"] == 100
    assert final["conflicting_relation_keys"] > 0
    assert final["persisted_mib"] > 0
    assert final["semantic_compression_ratio"] > 0
    assert final["reload_counts_match"]
