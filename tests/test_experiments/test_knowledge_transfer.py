from experiments.knowledge_transfer_benchmark import CHAINS, run


def test_real_fact_multi_hop_transfer():
    report = run(vsa_dim=64)
    assert report["questions"] == len(CHAINS)
    assert report["accuracy_pct"] == 100
    assert report["evidence_completeness_pct"] == 100


def test_transfer_survives_real_language_library_noise():
    corpus = [
        {"text": (
            "Markets rallied after the company reported stronger quarterly "
            f"earnings and revised its annual forecast number {index}."
        )}
        for index in range(100)
    ]
    report = run(vsa_dim=64, corpus_tokens=300, corpus=corpus)
    assert report["corpus_tokens"] >= 300
    assert report["corpus_articles"] > 0
    assert report["accuracy_pct"] == 100
    assert report["evidence_completeness_pct"] == 100
