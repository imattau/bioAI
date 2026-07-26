"""A/B test random-token versus semantic-chunk VSA sequence ranking."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from experiments.library_response_scaling import (
    conversation_pairs,
    semantic_tfidf,
    token_f1,
)
from src.text import (
    FixedSpliceCandidateGenerator,
    LearnedChunkComposer,
    OllamaEmbedder,
    SemanticChunkVSASequenceEncoder,
    SemanticVSAEncoder,
    TokenLibrary,
    VSASequenceRanker,
)


def candidates_for(prompt, library, responses, composer):
    ids = [source_id for source_id, _ in library.candidate_ids(prompt, limit=5)]
    evidence = [responses[source_id] for source_id in ids]
    candidates = FixedSpliceCandidateGenerator().generate(
        prompt, evidence, composer
    )
    return evidence, candidates


def evaluate(ranker, pairs, library, responses, composer):
    predictions, references = [], []
    for prompt, reference in pairs:
        evidence, candidates = candidates_for(
            prompt, library, responses, composer
        )
        ranked = ranker.rank(prompt, candidates, evidence)
        predictions.append(ranked[0]["text"] if ranked else "")
        references.append(reference)
    return {
        "token_f1_pct": 100 * sum(
            token_f1(left, right)
            for left, right in zip(predictions, references)
        ) / len(references),
        "tfidf_similarity_pct": 100 * semantic_tfidf(
            predictions, references
        ),
    }


def run(
    library_pairs: int = 2_000,
    preference_pairs: int = 300,
    test_pairs: int = 100,
    presentations: int = 10_000,
    model: str = "qwen3-embedding:0.6b",
    dimension: int = 256,
) -> dict:
    from datasets import load_dataset

    dataset = load_dataset("OpenAssistant/oasst1")
    training = conversation_pairs(dataset["train"])
    testing = conversation_pairs(dataset["validation"])[:test_pairs]
    library = TokenLibrary(vector_cache_size=0)
    responses = []
    composer = LearnedChunkComposer()
    for prompt, response in training[:library_pairs]:
        library.add(prompt)
        responses.append(response)
        composer.learn(prompt, response)

    semantic = SemanticVSAEncoder(
        OllamaEmbedder(model), vsa_dim=dimension, seed=42
    )
    random_ranker = VSASequenceRanker(
        dimension=dimension, hidden_dimension=32
    )
    semantic_ranker = VSASequenceRanker(
        encoder=SemanticChunkVSASequenceEncoder(semantic),
        hidden_dimension=32,
    )
    random_preferences = []
    semantic_preferences = []
    usable = 0
    for prompt, reference in training[
        library_pairs:library_pairs + preference_pairs
    ]:
        evidence, candidates = candidates_for(
            prompt, library, responses, composer
        )
        if len(candidates) < 2:
            continue
        ordered = sorted(
            candidates,
            key=lambda candidate: token_f1(candidate["text"], reference),
            reverse=True,
        )
        if token_f1(ordered[0]["text"], reference) == token_f1(
            ordered[-1]["text"], reference
        ):
            continue
        preferred, rejected = ordered[0], ordered[-1]
        random_preferences.append((
            random_ranker.encode_state(prompt, preferred, evidence).to(torch.int8),
            random_ranker.encode_state(prompt, rejected, evidence).to(torch.int8),
        ))
        semantic_preferences.append((
            semantic_ranker.encode_state(
                prompt, preferred, evidence
            ).to(torch.int8),
            semantic_ranker.encode_state(
                prompt, rejected, evidence
            ).to(torch.int8),
        ))
        usable += 1

    random_ranker.train_encoded_preferences(
        random_preferences, presentations, batch_size=64
    )
    semantic_ranker.train_encoded_preferences(
        semantic_preferences, presentations, batch_size=64
    )
    return {
        "benchmark": "semantic_chunk_vsa_sequence_ab",
        "dataset": "OpenAssistant/oasst1",
        "model": model,
        "dimension": dimension,
        "library_pairs": library_pairs,
        "preference_pairs_requested": preference_pairs,
        "preference_pairs_usable": usable,
        "presentations": presentations,
        "test_pairs": len(testing),
        "random_token_vsa": evaluate(
            random_ranker, testing, library, responses, composer
        ),
        "semantic_chunk_vsa": evaluate(
            semantic_ranker, testing, library, responses, composer
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library-pairs", type=int, default=2_000)
    parser.add_argument("--preferences", type=int, default=300)
    parser.add_argument("--test-pairs", type=int, default=100)
    parser.add_argument("--presentations", type=int, default=10_000)
    parser.add_argument("--model", default="qwen3-embedding:0.6b")
    parser.add_argument("--dimension", type=int, default=256)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()
    report = run(
        args.library_pairs, args.preferences, args.test_pairs,
        args.presentations, args.model, args.dimension,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = args.output_dir / f"semantic_sequence_{stamp}.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Report: {path}")


if __name__ == "__main__":
    main()
