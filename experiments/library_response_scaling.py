"""Held-out response reconstruction as a real conversation library grows."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from sklearn.feature_extraction.text import TfidfVectorizer

from src.text import (
    FixedSpliceCandidateGenerator,
    LearnedChunkComposer,
    SequenceCandidateScorer,
    TokenLibrary,
    VSASequenceRanker,
)


DEFAULT_CHECKPOINTS = (100_000, 500_000, 1_000_000)
WILDCHAT_CHECKPOINTS = (1_000_000, 3_000_000, 10_000_000)


def conversation_pairs(split) -> list[tuple[str, str]]:
    rows = {
        row["message_id"]: row for row in split
        if not row["deleted"] and row["lang"] == "en"
    }
    pairs = []
    for row in rows.values():
        if row["role"] != "assistant" or row["parent_id"] not in rows:
            continue
        parent = rows[row["parent_id"]]
        if parent["role"] != "prompter":
            continue
        if row["rank"] not in (None, 0):
            continue
        pairs.append((parent["text"], row["text"]))
    return pairs


def wildchat_pairs(row):
    if (
        row.get("language") != "English"
        or row.get("toxic")
        or row.get("redacted")
    ):
        return []
    conversation = row.get("conversation") or []
    return [
        (left["content"], right["content"])
        for left, right in zip(conversation, conversation[1:])
        if (
            left.get("role") == "user"
            and right.get("role") == "assistant"
            and left.get("content")
            and right.get("content")
        )
    ]


def token_f1(predicted: str, reference: str) -> float:
    predicted_tokens = TokenLibrary.tokenize(predicted)
    reference_tokens = TokenLibrary.tokenize(reference)
    if not predicted_tokens or not reference_tokens:
        return 0.0
    from collections import Counter
    overlap = sum(
        (Counter(predicted_tokens) & Counter(reference_tokens)).values()
    )
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall) if overlap else 0.0


def compose_response(
    query: str,
    candidates: list[int],
    responses: list[str],
    max_sentences: int = 3,
) -> str:
    query_terms = set(TokenLibrary.tokenize(query))
    sentences = []
    for source_id in candidates:
        for sentence in re.split(r"(?<=[.!?])\s+", responses[source_id]):
            terms = set(TokenLibrary.tokenize(sentence))
            if not terms:
                continue
            score = len(query_terms & terms) / len(query_terms | terms)
            sentences.append((score, source_id, sentence.strip()))
    sentences.sort(reverse=True)
    selected = []
    seen = set()
    for _, _, sentence in sentences:
        normalised = sentence.lower()
        if normalised in seen:
            continue
        seen.add(normalised)
        selected.append(sentence)
        if len(selected) == max_sentences:
            break
    return " ".join(selected)


def semantic_tfidf(predictions: list[str], references: list[str]) -> float:
    matrix = TfidfVectorizer(
        ngram_range=(1, 2), min_df=1, max_features=50_000
    ).fit_transform(predictions + references)
    count = len(predictions)
    similarities = matrix[:count].multiply(matrix[count:]).sum(axis=1)
    return float(torch.tensor(similarities.A1).mean())


def evaluate_vsa_ranker(
    library: TokenLibrary,
    responses: list[str],
    pairs: list[tuple[str, str]],
    composer: LearnedChunkComposer,
    ranker: VSASequenceRanker,
) -> dict:
    generator = FixedSpliceCandidateGenerator()
    predictions, references = [], []
    for prompt, reference in pairs:
        ids = [
            source_id for source_id, _ in library.candidate_ids(prompt, limit=5)
        ]
        evidence = [responses[source_id] for source_id in ids]
        candidates = generator.generate(prompt, evidence, composer)
        ranked = ranker.rank(prompt, candidates, evidence)
        predictions.append(ranked[0]["text"] if ranked else "")
        references.append(reference)
    return {
        "token_f1_pct": 100 * statistics.mean(
            token_f1(prediction, reference)
            for prediction, reference in zip(predictions, references)
        ),
        "tfidf_similarity_pct": 100 * semantic_tfidf(
            predictions, references
        ),
    }


def evaluate(
    library: TokenLibrary,
    response_library: TokenLibrary,
    responses: list[str],
    held_out: list[tuple[str, str]],
    composer: LearnedChunkComposer | None = None,
    candidate_scorer: SequenceCandidateScorer | None = None,
    sequence_ranker: VSASequenceRanker | None = None,
) -> dict:
    nearest_predictions = []
    composed_predictions = []
    references = []
    learned_predictions = []
    learned_latencies = []
    ranked_predictions = []
    ranked_kinds = []
    sequence_predictions = []
    sequence_kinds = []
    candidate_generator = FixedSpliceCandidateGenerator()
    candidate_scorer = candidate_scorer or SequenceCandidateScorer()
    latencies = []
    accepted = 0
    for prompt, reference in held_out:
        started = time.perf_counter_ns()
        candidates = library.candidate_ids(prompt, limit=5)
        latencies.append((time.perf_counter_ns() - started) / 1_000_000)
        ids = [source_id for source_id, _ in candidates]
        if ids:
            accepted += 1
            nearest_predictions.append(responses[ids[0]])
            composed_predictions.append(compose_response(prompt, ids, responses))
            if composer is not None:
                learned_started = time.perf_counter_ns()
                learned_predictions.append(composer.generate(
                    prompt, evidence=[responses[source_id] for source_id in ids]
                ))
                learned_latencies.append(
                    (time.perf_counter_ns() - learned_started) / 1_000_000
                )
                response_candidates = candidate_generator.generate(
                    prompt,
                    [responses[source_id] for source_id in ids],
                    composer,
                )
                ranked = candidate_scorer.rank(
                    prompt,
                    response_candidates,
                    [responses[source_id] for source_id in ids],
                )
                ranked_predictions.append(ranked[0]["text"])
                ranked_kinds.append(ranked[0]["kind"])
                if sequence_ranker is not None:
                    sequence_ranked = sequence_ranker.rank(
                        prompt, response_candidates,
                        [responses[source_id] for source_id in ids],
                    )
                    sequence_predictions.append(sequence_ranked[0]["text"])
                    sequence_kinds.append(sequence_ranked[0]["kind"])
        else:
            nearest_predictions.append("")
            composed_predictions.append("")
            if composer is not None:
                learned_predictions.append("")
                ranked_predictions.append("")
                ranked_kinds.append("none")
                if sequence_ranker is not None:
                    sequence_predictions.append("")
                    sequence_kinds.append("none")
        references.append(reference)
    result = {
        "library_tokens": (
            len(library.token_ids) + len(response_library.token_ids)
        ),
        "prompt_tokens": len(library.token_ids),
        "response_tokens": len(response_library.token_ids),
        "library_pairs": len(library),
        "held_out_pairs": len(held_out),
        "acceptance_pct": 100 * accepted / len(held_out),
        "nearest_token_f1_pct": 100 * statistics.mean(
            token_f1(prediction, reference)
            for prediction, reference in zip(nearest_predictions, references)
        ),
        "composed_token_f1_pct": 100 * statistics.mean(
            token_f1(prediction, reference)
            for prediction, reference in zip(composed_predictions, references)
        ),
        "nearest_tfidf_similarity_pct": 100 * semantic_tfidf(
            nearest_predictions, references
        ),
        "composed_tfidf_similarity_pct": 100 * semantic_tfidf(
            composed_predictions, references
        ),
        "nearest_exact_match_pct": 100 * statistics.mean(
            prediction.strip() == reference.strip()
            for prediction, reference in zip(nearest_predictions, references)
        ),
        "query_p50_ms": statistics.median(latencies),
        "query_p95_ms": sorted(latencies)[int(0.95 * (len(latencies) - 1))],
        "composition_is_source_extractive": True,
    }
    if composer is not None:
        response_set = set(responses)
        result.update({
            "learned_composer_token_f1_pct": 100 * statistics.mean(
                token_f1(prediction, reference)
                for prediction, reference in zip(
                    learned_predictions, references
                )
            ),
            "learned_composer_tfidf_similarity_pct": 100 * semantic_tfidf(
                learned_predictions, references
            ),
            "learned_composer_novel_combination_pct": 100 * statistics.mean(
                bool(prediction) and prediction not in response_set
                for prediction in learned_predictions
            ),
            "learned_composer_latency_p50_ms": statistics.median(
                learned_latencies
            ) if learned_latencies else 0.0,
            "learned_composer_pairs": composer.pairs,
            "global_ranker_token_f1_pct": 100 * statistics.mean(
                token_f1(prediction, reference)
                for prediction, reference in zip(
                    ranked_predictions, references
                )
            ),
            "global_ranker_tfidf_similarity_pct": 100 * semantic_tfidf(
                ranked_predictions, references
            ),
            "global_ranker_selection_counts": {
                kind: ranked_kinds.count(kind) for kind in sorted(set(ranked_kinds))
            },
            "global_ranker_updates": candidate_scorer.updates,
            "global_ranker_weights": dict(zip(
                candidate_scorer.FEATURE_NAMES, candidate_scorer.weights
            )),
        })
        if sequence_ranker is not None:
            result.update({
                "vsa_sequence_token_f1_pct": 100 * statistics.mean(
                    token_f1(prediction, reference)
                    for prediction, reference in zip(
                        sequence_predictions, references
                    )
                ),
                "vsa_sequence_tfidf_similarity_pct": 100 * semantic_tfidf(
                    sequence_predictions, references
                ),
                "vsa_sequence_selection_counts": {
                    kind: sequence_kinds.count(kind)
                    for kind in sorted(set(sequence_kinds))
                },
                "vsa_sequence_updates": sequence_ranker.updates,
            })
    return result


def run(
    held_out_count: int = 200,
    checkpoints: tuple[int, ...] = DEFAULT_CHECKPOINTS,
    seed: int = 83,
) -> dict:
    from datasets import load_dataset

    dataset = load_dataset("OpenAssistant/oasst1")
    training = conversation_pairs(dataset["train"])
    held_out = conversation_pairs(dataset["validation"])
    generator = torch.Generator().manual_seed(seed)
    training_order = torch.randperm(
        len(training), generator=generator
    ).tolist()
    held_out_order = torch.randperm(
        len(held_out), generator=generator
    )[:held_out_count].tolist()
    held_out = [held_out[index] for index in held_out_order]

    library = TokenLibrary(vector_cache_size=0)
    response_library = TokenLibrary(vector_cache_size=0)
    responses = []
    composer = LearnedChunkComposer()
    candidate_scorer = SequenceCandidateScorer()
    results = []
    pending = list(checkpoints)
    started = time.perf_counter()
    for index in training_order:
        prompt, response = training[index]
        library.add(prompt)
        response_library.add(response)
        responses.append(response)
        composer.learn(prompt, response)
        total_tokens = (
            len(library.token_ids) + len(response_library.token_ids)
        )
        while pending and total_tokens >= pending[0]:
            row = evaluate(
                library, response_library, responses, held_out, composer,
                candidate_scorer,
            )
            row["target_checkpoint_tokens"] = pending.pop(0)
            row["elapsed_seconds"] = time.perf_counter() - started
            results.append(row)
    return {
        "benchmark": "real_conversation_library_response_scaling",
        "dataset": "OpenAssistant/oasst1",
        "generator": "none",
        "retrieval": "TokenLibrary lexical/semantic-term index",
        "composition": "top-retrieved source sentence splicing",
        "available_training_pairs": len(training),
        "available_held_out_pairs": len(conversation_pairs(
            dataset["validation"]
        )),
        "unreached_checkpoints": pending,
        "results": results,
    }


def run_wildchat(
    held_out_count: int = 200,
    checkpoints: tuple[int, ...] = WILDCHAT_CHECKPOINTS,
    seed: int = 83,
) -> dict:
    from datasets import load_dataset

    held_out_dataset = load_dataset("OpenAssistant/oasst1")
    held_out_all = conversation_pairs(held_out_dataset["validation"])
    generator = torch.Generator().manual_seed(seed)
    held_out_order = torch.randperm(
        len(held_out_all), generator=generator
    )[:held_out_count + 100].tolist()
    validation = [held_out_all[index] for index in held_out_order[:100]]
    held_out = [
        held_out_all[index] for index in held_out_order[100:]
    ]
    stream = load_dataset(
        "allenai/WildChat-1M", split="train", streaming=True
    )

    library = TokenLibrary(vector_cache_size=0)
    response_library = TokenLibrary(vector_cache_size=0)
    responses = []
    composer = LearnedChunkComposer()
    candidate_generator = FixedSpliceCandidateGenerator()
    candidate_scorer = SequenceCandidateScorer()
    sequence_ranker = VSASequenceRanker(
        dimension=256, hidden_dimension=32
    )
    maximum_preference_updates = 5_000
    encoded_preferences = []
    results = []
    pending = list(checkpoints)
    conversations = 0
    started = time.perf_counter()
    for row in stream:
        conversations += 1
        for prompt, response in wildchat_pairs(row):
            pair_number = len(responses)
            if (
                pair_number >= 100
                and pair_number % 5 == 0
                and candidate_scorer.updates < maximum_preference_updates
            ):
                retrieved = library.candidate_ids(prompt, limit=5)
                source_ids = [source_id for source_id, _ in retrieved]
                evidence = [responses[source_id] for source_id in source_ids]
                candidates = candidate_generator.generate(
                    prompt, evidence, composer
                )
                ranked = candidate_scorer.rank(
                    prompt, candidates, evidence
                )
                preferred = max(
                    candidates,
                    key=lambda candidate: token_f1(
                        candidate["text"], response
                    ),
                    default=None,
                )
                if (
                    ranked and preferred is not None
                    and ranked[0]["text"] != preferred["text"]
                ):
                    candidate_scorer.learn_preference(
                        prompt,
                        preferred,
                        ranked[0],
                        evidence,
                    )
                    if sequence_ranker.updates < 2_000:
                        sequence_ranker.learn_preference(
                            prompt, preferred, ranked[0], evidence
                        )
                    if len(encoded_preferences) < 10_000:
                        encoded_preferences.append((
                            sequence_ranker.encode_state(
                                prompt, preferred, evidence
                            ).to(torch.int8),
                            sequence_ranker.encode_state(
                                prompt, ranked[0], evidence
                            ).to(torch.int8),
                        ))
            library.add(prompt)
            response_library.add(response)
            responses.append(response)
            composer.learn(prompt, response)
            total_tokens = (
                len(library.token_ids) + len(response_library.token_ids)
            )
            while pending and total_tokens >= pending[0]:
                result = evaluate(
                    library, response_library, responses, held_out, composer,
                    candidate_scorer,
                    sequence_ranker,
                )
                result["target_checkpoint_tokens"] = pending.pop(0)
                result["source_conversations_seen"] = conversations
                result["elapsed_seconds"] = time.perf_counter() - started
                results.append(result)
            if not pending:
                break
        if not pending:
            break
    training_curve = []
    torch.manual_seed(seed)
    trained_ranker = VSASequenceRanker(
        dimension=256, hidden_dimension=32
    )
    previous_budget = 0
    best_validation = -1.0
    best_state = None
    for budget in (2_000, 10_000, 50_000, 200_000):
        loss = trained_ranker.train_encoded_preferences(
            encoded_preferences,
            examples=budget - previous_budget,
            batch_size=64,
            seed=seed,
        )
        metrics = evaluate_vsa_ranker(
            library, responses, validation, composer, trained_ranker
        )
        training_curve.append({
            "examples_presented": budget,
            "mean_pairwise_loss": loss,
            **metrics,
        })
        if metrics["token_f1_pct"] > best_validation:
            best_validation = metrics["token_f1_pct"]
            best_state = trained_ranker.get_state()
        previous_budget = budget
    selected_ranker = VSASequenceRanker.from_state(best_state)
    selected_test = evaluate_vsa_ranker(
        library, responses, held_out, composer, selected_ranker
    )
    return {
        "benchmark": "wildchat_real_conversation_library_scaling",
        "dataset": "allenai/WildChat-1M",
        "held_out_dataset": "OpenAssistant/oasst1:validation",
        "filters": "English, non-toxic, non-redacted",
        "generator": "none",
        "retrieval": "TokenLibrary lexical/semantic-term index",
        "composition": "top-retrieved source sentence splicing",
        "unreached_checkpoints": pending,
        "encoded_training_preferences": len(encoded_preferences),
        "vsa_training_curve_validation": training_curve,
        "vsa_selected_validation_f1_pct": best_validation,
        "vsa_selected_test": selected_test,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--held-out", type=int, default=200)
    parser.add_argument(
        "--dataset", choices=("openassistant", "wildchat"),
        default="openassistant",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()
    report = (
        run_wildchat(held_out_count=args.held_out)
        if args.dataset == "wildchat"
        else run(held_out_count=args.held_out)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = args.output_dir / f"library_response_scaling_{stamp}.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Report: {path}")


if __name__ == "__main__":
    main()
