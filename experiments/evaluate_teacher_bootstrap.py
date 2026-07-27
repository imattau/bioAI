"""Compare a teacher-distilled scorer with BioAI's default held-out policy."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

import torch

from experiments.library_response_scaling import (
    conversation_pairs,
    semantic_tfidf,
    token_f1,
)
from experiments.live_feedback_scale_benchmark import proposed_responses
from src.text import (
    LearnedChunkComposer,
    SequenceCandidateScorer,
    TokenLibrary,
    load_bootstrap,
)


def paired_bootstrap_interval(
    differences: list[float],
    samples: int = 5_000,
    seed: int = 211,
) -> dict:
    if not differences:
        raise ValueError("At least one paired difference is required")
    if samples < 100:
        raise ValueError("At least 100 bootstrap samples are required")
    generator = random.Random(seed)
    count = len(differences)
    estimates = sorted(
        statistics.mean(
            differences[generator.randrange(count)] for _ in range(count)
        )
        for _ in range(samples)
    )
    return {
        "mean_points": 100 * statistics.mean(differences),
        "ci95_low_points": 100 * estimates[int(samples * 0.025)],
        "ci95_high_points": 100 * estimates[int(samples * 0.975)],
        "probability_improvement": (
            sum(value > 0 for value in estimates) / samples
        ),
    }


def evaluate(
    learning_state_path: Path,
    bootstrap_path: Path,
    held_out_count: int = 200,
    seed: int = 211,
    bootstrap_samples: int = 5_000,
) -> dict:
    from datasets import load_dataset

    learning = torch.load(
        learning_state_path, map_location="cpu", weights_only=False
    )
    prompts = TokenLibrary.from_state(learning["prompts"])
    responses = learning["responses"]
    composer = LearnedChunkComposer.from_state(learning["composer"])
    default = SequenceCandidateScorer()
    trained = SequenceCandidateScorer.from_state(
        load_bootstrap(bootstrap_path)["candidate_scorer"]
    )
    all_pairs = conversation_pairs(
        load_dataset("OpenAssistant/oasst1")["validation"]
    )
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(
        len(all_pairs), generator=generator
    )[:held_out_count].tolist()
    pairs = [all_pairs[index] for index in indices]

    predictions = {"default": [], "teacher_bootstrap": []}
    references = []
    for prompt, reference in pairs:
        for name, scorer in (
            ("default", default),
            ("teacher_bootstrap", trained),
        ):
            _, _, prediction, _, _, _, _ = proposed_responses(
                prompt, prompts, responses, composer, scorer
            )
            predictions[name].append(prediction)
        references.append(reference)

    result = {"held_out_count": len(pairs), "seed": seed}
    for name, values in predictions.items():
        result[name] = {
            "token_f1_pct": 100 * statistics.mean(
                token_f1(value, reference)
                for value, reference in zip(values, references)
            ),
            "tfidf_pct": 100 * semantic_tfidf(values, references),
        }
    differences = [
        token_f1(trained_value, reference)
        - token_f1(default_value, reference)
        for trained_value, default_value, reference in zip(
            predictions["teacher_bootstrap"],
            predictions["default"],
            references,
        )
    ]
    result["paired_token_f1"] = paired_bootstrap_interval(
        differences, bootstrap_samples, seed
    )
    result["confirmed"] = (
        result["paired_token_f1"]["ci95_low_points"] > 0
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--learning-state",
        type=Path,
        default=Path("checkpoints/live_feedback_scale_state.pt"),
    )
    parser.add_argument("--bootstrap", type=Path, required=True)
    parser.add_argument("--held-out", type=int, default=200)
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--bootstrap-samples", type=int, default=5_000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate(
        args.learning_state,
        args.bootstrap,
        args.held_out,
        args.seed,
        args.bootstrap_samples,
    )
    payload = json.dumps(result, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
