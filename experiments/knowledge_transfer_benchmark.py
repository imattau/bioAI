"""Functional transfer benchmark over real geographic knowledge."""

from __future__ import annotations

import argparse
import json
import resource
import time
from datetime import datetime, timezone
from pathlib import Path

from src.text import BioAIDialogueAgent


CHAINS = (
    ("Canberra", "Australia", "Oceania"),
    ("Paris", "France", "Europe"),
    ("Tokyo", "Japan", "Asia"),
    ("Ottawa", "Canada", "North America"),
    ("Brasilia", "Brazil", "South America"),
    ("Nairobi", "Kenya", "Africa"),
    ("Cairo", "Egypt", "Africa"),
    ("Wellington", "New Zealand", "Oceania"),
    ("Madrid", "Spain", "Europe"),
    ("Seoul", "South Korea", "Asia"),
    ("Buenos Aires", "Argentina", "South America"),
    ("Mexico City", "Mexico", "North America"),
)


def load_real_corpus(
    agent: BioAIDialogueAgent,
    target_tokens: int,
    corpus=None,
) -> dict:
    if target_tokens <= 0:
        return {
            "corpus": None,
            "corpus_articles": 0,
            "corpus_tokens": 0,
            "corpus_load_seconds": 0.0,
        }
    if corpus is None:
        from datasets import load_dataset
        corpus = load_dataset("fancyzhx/ag_news")["train"]
        corpus_name = "fancyzhx/ag_news:train"
    else:
        corpus_name = "provided_test_corpus"
    started = time.perf_counter()
    articles = 0
    for row in corpus:
        agent.library.add(row["text"])
        articles += 1
        if len(agent.library.token_ids) >= target_tokens:
            break
    elapsed = time.perf_counter() - started
    if len(agent.library.token_ids) < target_tokens:
        raise ValueError("Corpus ended before the requested token target")
    return {
        "corpus": corpus_name,
        "corpus_articles": articles,
        "corpus_tokens": len(agent.library.token_ids),
        "corpus_sentences": len(agent.library),
        "corpus_load_seconds": elapsed,
        "corpus_tokens_per_second": len(agent.library.token_ids) / elapsed,
        "token_array_mib": agent.library.token_storage_bytes / (1024 * 1024),
    }


def run(
    vsa_dim: int = 256,
    corpus_tokens: int = 0,
    corpus=None,
) -> dict:
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    agent = BioAIDialogueAgent(vsa_dim=vsa_dim)
    corpus_metrics = load_real_corpus(agent, corpus_tokens, corpus)
    fact_styles = (
        lambda city, country: f"{city} is the capital of {country}.",
        lambda city, country: f"The capital of {country} is {city}.",
    )
    location_styles = (
        lambda country, continent: f"{country} is located in {continent}.",
        lambda country, continent: f"{country} lies within {continent}.",
        lambda country, continent: f"{country} sits in {continent}.",
    )
    for index, (city, country, continent) in enumerate(CHAINS):
        agent.process_turn(fact_styles[index % len(fact_styles)](city, country))
        agent.process_turn(
            location_styles[index % len(location_styles)](country, continent)
        )

    records = []
    correct = 0
    evidence_complete = 0
    latencies = []
    for index, (city, country, continent) in enumerate(CHAINS):
        question = (
            f"What continent is {city} in?"
            if index % 2 == 0
            else f"Which continent contains {city}?"
        )
        started = time.perf_counter_ns()
        result = agent.process_turn(question)
        latencies.append((time.perf_counter_ns() - started) / 1_000_000)
        answer_correct = (
            result["reasoning"] is not None
            and result["reasoning"]["answer"] == continent.lower()
        )
        sources_complete = len(result["sources"]) == 2
        correct += int(answer_correct)
        evidence_complete += int(sources_complete)
        records.append({
            "question": question,
            "expected": continent,
            "response": result["response"],
            "path": (
                result["reasoning"]["path"] if result["reasoning"] else None
            ),
            "source_texts": [source["text"] for source in result["sources"]],
            "correct": answer_correct,
        })
    return {
        "benchmark": "real_fact_multi_hop_transfer",
        **corpus_metrics,
        "library_sentences_after_facts": len(agent.library),
        "library_tokens_after_facts": len(agent.library.token_ids),
        "process_peak_rss_delta_mib": (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - rss_before
        ) / 1024,
        "facts": len(CHAINS) * 2,
        "questions": len(CHAINS),
        "correct": correct,
        "accuracy_pct": 100 * correct / len(CHAINS),
        "complete_evidence_paths": evidence_complete,
        "evidence_completeness_pct": 100 * evidence_complete / len(CHAINS),
        "latency_p50_ms": sorted(latencies)[len(latencies) // 2],
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--corpus-tokens", type=int, default=0)
    args = parser.parse_args()
    report = run(corpus_tokens=args.corpus_tokens)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = args.output_dir / f"knowledge_transfer_{stamp}.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        key: value for key, value in report.items() if key != "records"
    }, indent=2))
    print(f"Report: {path}")


if __name__ == "__main__":
    main()
