"""Evaluate semantic consolidation on held-out AG News articles."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

from src.text import ConsolidationMemory


LABELS = ("World", "Sports", "Business", "Science and Technology")


def stratified_examples(dataset, per_label: int, seed: int) -> list[dict]:
    shuffled = dataset.shuffle(seed=seed)
    counts = [0] * len(LABELS)
    selected = []
    for row in shuffled:
        label = int(row["label"])
        if counts[label] >= per_label:
            continue
        selected.append({"text": row["text"], "label": label})
        counts[label] += 1
        if min(counts) == per_label:
            break
    if min(counts) != per_label:
        raise ValueError(f"Dataset lacks {per_label} examples for every label")
    return selected


def embed_batches(
    texts: list[str], model: str, batch_size: int
) -> tuple[torch.Tensor, float]:
    import ollama

    batches = []
    started = time.perf_counter()
    for start in range(0, len(texts), batch_size):
        response = ollama.embed(
            model=model, input=texts[start:start + batch_size]
        )
        embeddings = (
            response["embeddings"]
            if isinstance(response, dict) else response.embeddings
        )
        batches.append(torch.tensor(embeddings, dtype=torch.float32))
    elapsed = time.perf_counter() - started
    return torch.cat(batches), elapsed


def project_bipolar(
    embeddings: torch.Tensor, dimension: int, seed: int
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    projection = torch.randn(
        dimension, embeddings.shape[1], generator=generator
    ) / embeddings.shape[1] ** 0.5
    projected = F.normalize(embeddings, dim=1) @ projection.T
    return torch.where(projected >= 0, 1, -1).to(torch.int8)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[int(fraction * (len(ordered) - 1))]


def run(
    train_per_label: int = 2_000,
    test_per_label: int = 400,
    model: str = "qwen3-embedding:0.6b",
    dimension: int = 1024,
    batch_size: int = 64,
    seed: int = 73,
) -> dict:
    from datasets import load_dataset

    dataset = load_dataset("fancyzhx/ag_news")
    train = stratified_examples(dataset["train"], train_per_label, seed)
    test = stratified_examples(dataset["test"], test_per_label, seed + 1)
    all_rows = train + test
    embeddings, embedding_seconds = embed_batches(
        [row["text"] for row in all_rows], model, batch_size
    )
    train_embeddings = embeddings[:len(train)]
    test_embeddings = embeddings[len(train):]
    bipolar = project_bipolar(embeddings, dimension, seed)
    train_bipolar = bipolar[:len(train)]
    test_bipolar = bipolar[len(train):]

    float_centroids = torch.stack([
        F.normalize(train_embeddings[[
            row["label"] == label for row in train
        ]].mean(dim=0), dim=0)
        for label in range(len(LABELS))
    ])
    memory = ConsolidationMemory(
        promotion_threshold=3,
        max_prototypes=len(LABELS),
        max_sources_per_item=16,
    )
    pending = {label: [] for label in range(len(LABELS))}
    consolidation_started = time.perf_counter()
    for source_id, (row, vector) in enumerate(zip(train, train_bipolar)):
        term = LABELS[row["label"]]
        if term in memory.concepts:
            memory.observe_concept(term, vector, source_id)
        else:
            pending[row["label"]].append((source_id, vector))
            evidence = pending[row["label"]]
            if len(evidence) == memory.promotion_threshold:
                memory.observe_concept(
                    term, vector, source_id,
                    initial_vectors=[item[1] for item in evidence],
                    initial_source_ids=[item[0] for item in evidence],
                )
    consolidation_seconds = time.perf_counter() - consolidation_started

    vsa_correct = 0
    float_correct = 0
    latencies = []
    errors = []
    for row, vector, embedding in zip(test, test_bipolar, test_embeddings):
        started = time.perf_counter_ns()
        match = memory.query_concepts(vector, limit=1)[0]
        latencies.append((time.perf_counter_ns() - started) / 1_000_000)
        predicted = LABELS.index(match["term"])
        float_predicted = int(torch.argmax(
            float_centroids @ F.normalize(embedding, dim=0)
        ))
        vsa_correct += int(predicted == row["label"])
        float_correct += int(float_predicted == row["label"])
        if predicted != row["label"] and len(errors) < 20:
            errors.append({
                "expected": LABELS[row["label"]],
                "predicted": LABELS[predicted],
                "text": row["text"],
            })

    warm = latencies[1:] or latencies
    total = len(test)
    return {
        "benchmark": "ag_news_real_language_consolidation",
        "dataset": "fancyzhx/ag_news",
        "model": model,
        "dimension": dimension,
        "train_articles": len(train),
        "test_articles": total,
        "classes": list(LABELS),
        "embedding_seconds": embedding_seconds,
        "embedding_articles_per_second": len(all_rows) / embedding_seconds,
        "consolidation_seconds": consolidation_seconds,
        "vsa_correct": vsa_correct,
        "vsa_accuracy_pct": 100 * vsa_correct / total,
        "float_centroid_correct": float_correct,
        "float_centroid_accuracy_pct": 100 * float_correct / total,
        "vsa_vs_float_delta_points": (
            100 * (vsa_correct - float_correct) / total
        ),
        "query_cold_ms": latencies[0],
        "query_warm_p50_ms": statistics.median(warm),
        "query_warm_p95_ms": percentile(warm, 0.95),
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-per-label", type=int, default=2_000)
    parser.add_argument("--test-per-label", type=int, default=400)
    parser.add_argument("--model", default="qwen3-embedding:0.6b")
    parser.add_argument("--dimension", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()
    report = run(
        train_per_label=args.train_per_label,
        test_per_label=args.test_per_label,
        model=args.model,
        dimension=args.dimension,
        batch_size=args.batch_size,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = args.output_dir / f"real_language_consolidation_{stamp}.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    summary = {key: value for key, value in report.items() if key != "errors"}
    print(json.dumps(summary, indent=2))
    print(f"Report: {path}")


if __name__ == "__main__":
    main()
