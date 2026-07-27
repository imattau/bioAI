"""Chronological live-feedback learning benchmark over WildChat."""

from __future__ import annotations

import argparse
import json
import os
import resource
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import torch

from experiments.library_response_scaling import (
    compose_response,
    conversation_pairs,
    semantic_tfidf,
    token_f1,
    wildchat_pairs,
)
from src.text import (
    EpisodicPreferenceScorer,
    FixedSpliceCandidateGenerator,
    LearnedChunkComposer,
    SequenceCandidateScorer,
    TokenLibrary,
)


CHECKPOINTS = (1_000, 10_000, 30_000, 50_000, 75_000, 100_000)


def save_learning_state(path: Path, state: dict) -> None:
    """Atomically persist resumable benchmark learning state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_learning_state(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def proposed_responses(
    prompt,
    library,
    responses,
    composer,
    scorer,
    episodic_scorer=None,
    frozen_scorer=None,
):
    ids = [source_id for source_id, _ in library.candidate_ids(prompt, limit=5)]
    evidence = [responses[source_id] for source_id in ids]
    static = compose_response(prompt, ids, responses) if ids else ""
    candidates = FixedSpliceCandidateGenerator().generate(
        prompt, evidence, composer
    )
    ranked = scorer.rank(prompt, candidates, evidence)
    global_live = ranked[0]["text"] if ranked else ""
    episodic_ranked = (
        episodic_scorer.rank(prompt, candidates, evidence)
        if episodic_scorer is not None else []
    )
    frozen_ranked = (
        frozen_scorer.rank(prompt, candidates, evidence)
        if frozen_scorer is not None else []
    )
    episodic_live = (
        episodic_ranked[0]["text"] if episodic_ranked else global_live
    )
    frozen = frozen_ranked[0]["text"] if frozen_ranked else global_live
    return (
        static, frozen, global_live, episodic_live,
        evidence, candidates, ranked,
    )


def evaluate_transfer(
    pairs, library, responses, composer, scorer,
    episodic_scorer, frozen_scorer, workers: int = 1,
):
    def predict(pair):
        prompt, reference = pair
        static, frozen, global_live, episodic, _, _, _ = proposed_responses(
            prompt, library, responses, composer, scorer,
            episodic_scorer, frozen_scorer,
        )
        return static, frozen, global_live, episodic, reference

    if workers < 1:
        raise ValueError("workers must be at least 1")
    if workers == 1:
        predictions = map(predict, pairs)
    else:
        # Evaluation is read-only, so all workers can safely share the large
        # token libraries without copying their state into separate processes.
        executor = ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="bioai-eval",
        )
        predictions = executor.map(predict, pairs)

    static_predictions, frozen_predictions = [], []
    global_predictions, episodic_predictions, references = [], [], []
    for static, frozen, global_live, episodic, reference in predictions:
        static_predictions.append(static)
        frozen_predictions.append(frozen)
        global_predictions.append(global_live)
        episodic_predictions.append(episodic)
        references.append(reference)
    if workers > 1:
        executor.shutdown()
    result = {}
    for name, predictions in (
        ("fixed_splice", static_predictions),
        ("frozen_candidate", frozen_predictions),
        ("global_update", global_predictions),
        ("episodic_local", episodic_predictions),
    ):
        result[f"{name}_token_f1_pct"] = 100 * statistics.mean(
            token_f1(left, right)
            for left, right in zip(predictions, references)
        )
        result[f"{name}_tfidf_pct"] = 100 * semantic_tfidf(
            predictions, references
        )
    return result


def run(
    max_interactions: int = 30_000,
    held_out_count: int = 200,
    seed: int = 211,
    workers: int = 1,
    state_path: Path | None = None,
    resume: bool = False,
) -> dict:
    from datasets import load_dataset

    held_out_rows = load_dataset("OpenAssistant/oasst1")
    held_out_all = conversation_pairs(held_out_rows["validation"])
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(
        len(held_out_all), generator=generator
    )[:held_out_count].tolist()
    held_out = [held_out_all[index] for index in indices]
    stream = load_dataset(
        "allenai/WildChat-1M", split="train", streaming=True
    )

    if resume:
        if state_path is None or not state_path.exists():
            raise FileNotFoundError(
                "A saved --state-path is required with --resume"
            )
        saved = load_learning_state(state_path)
        if saved.get("seed") != seed:
            raise ValueError("Resume seed does not match saved benchmark")
        if saved.get("held_out_count") != held_out_count:
            raise ValueError(
                "Resume held-out count does not match saved benchmark"
            )
        prompts = TokenLibrary.from_state(saved["prompts"])
        response_library = TokenLibrary.from_state(
            saved["response_library"]
        )
        responses = saved["responses"]
        composer = LearnedChunkComposer.from_state(saved["composer"])
        frozen_scorer = SequenceCandidateScorer.from_state(
            saved["frozen_scorer"]
        )
        scorer = SequenceCandidateScorer.from_state(saved["scorer"])
        episodic_scorer = EpisodicPreferenceScorer.from_state(
            saved["episodic_scorer"]
        )
        static_online_f1 = saved["static_online_f1"]
        frozen_online_f1 = saved["frozen_online_f1"]
        global_online_f1 = saved["global_online_f1"]
        episodic_online_f1 = saved["episodic_online_f1"]
        query_latencies = saved["query_latencies"]
        reports = saved["reports"]
        elapsed_before = saved.get("elapsed_seconds", 0.0)
    else:
        prompts = TokenLibrary(vector_cache_size=0)
        response_library = TokenLibrary(vector_cache_size=0)
        responses = []
        composer = LearnedChunkComposer()
        frozen_scorer = SequenceCandidateScorer()
        scorer = SequenceCandidateScorer()
        episodic_scorer = EpisodicPreferenceScorer()
        static_online_f1 = []
        frozen_online_f1 = []
        global_online_f1 = []
        episodic_online_f1 = []
        query_latencies = []
        reports = []
        elapsed_before = 0.0
    completed_interactions = len(responses)
    if completed_interactions > max_interactions:
        raise ValueError(
            "Saved benchmark already exceeds requested interactions"
        )
    if completed_interactions == max_interactions:
        return {
            "benchmark": "wildchat_live_feedback_scale",
            "training_dataset": "allenai/WildChat-1M",
            "held_out_dataset": "OpenAssistant/oasst1:validation",
            "filters": "English, non-toxic, non-redacted",
            "evaluation_workers": workers,
            "resumed_from_interactions": completed_interactions,
            "state_path": str(state_path),
            "checkpoints": list(CHECKPOINTS),
            "results": reports,
        }
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    started = time.perf_counter()
    stream_interaction = 0

    for row in stream:
        for prompt, recorded_response in wildchat_pairs(row):
            stream_interaction += 1
            if stream_interaction <= completed_interactions:
                continue
            interaction = len(responses) + 1
            query_started = time.perf_counter_ns()
            (
                static, frozen, global_live, episodic_live,
                evidence, candidates, ranked,
            ) = proposed_responses(
                prompt, prompts, responses, composer, scorer,
                episodic_scorer, frozen_scorer,
            )
            query_latencies.append(
                (time.perf_counter_ns() - query_started) / 1_000_000
            )
            static_online_f1.append(token_f1(static, recorded_response))
            frozen_online_f1.append(token_f1(frozen, recorded_response))
            global_online_f1.append(token_f1(global_live, recorded_response))
            episodic_online_f1.append(
                token_f1(episodic_live, recorded_response)
            )

            # Reveal the recorded answer: this is the simulated live correction.
            if candidates:
                preferred = max(
                    candidates,
                    key=lambda candidate: token_f1(
                        candidate["text"], recorded_response
                    ),
                )
                if ranked and preferred["text"] != ranked[0]["text"]:
                    scorer.learn_preference(
                        prompt, preferred, ranked[0], evidence
                    )
                episodic_ranked = episodic_scorer.rank(
                    prompt, candidates, evidence
                )
                if (
                    episodic_ranked
                    and preferred["text"] != episodic_ranked[0]["text"]
                ):
                    episodic_scorer.learn_preference(
                        prompt, preferred, episodic_ranked[0], evidence
                    )
            composer.learn(prompt, recorded_response)
            prompts.add(prompt)
            response_library.add(recorded_response)
            responses.append(recorded_response)

            if interaction in CHECKPOINTS or interaction == max_interactions:
                window = min(1_000, interaction)
                retention_indices = torch.linspace(
                    0, interaction - 1,
                    steps=min(100, interaction),
                ).to(torch.int64).tolist()
                retained = sum(
                    index in prompts.exact_sentence_ids(
                        prompts.texts[index]
                    )
                    and index in response_library.exact_sentence_ids(
                        responses[index]
                    )
                    for index in retention_indices
                )
                reports.append({
                    "interactions": interaction,
                    "library_tokens": (
                        len(prompts.token_ids)
                        + len(response_library.token_ids)
                    ),
                    "online_window": window,
                    "static_online_token_f1_pct": 100 * statistics.mean(
                        static_online_f1[-window:]
                    ),
                    "frozen_online_token_f1_pct": 100 * statistics.mean(
                        frozen_online_f1[-window:]
                    ),
                    "global_online_token_f1_pct": 100 * statistics.mean(
                        global_online_f1[-window:]
                    ),
                    "episodic_online_token_f1_pct": 100 * statistics.mean(
                        episodic_online_f1[-window:]
                    ),
                    "episodic_minus_static_online_points": 100 * (
                        statistics.mean(episodic_online_f1[-window:])
                        - statistics.mean(static_online_f1[-window:])
                    ),
                    "composer_pairs": composer.pairs,
                    "global_scorer_updates": scorer.updates,
                    "episodic_preferences": episodic_scorer.updates,
                    "retention_audit_pct": 100 * retained / len(
                        retention_indices
                    ),
                    "query_p50_ms": statistics.median(
                        query_latencies[-window:]
                    ),
                    "process_peak_rss_delta_mib": (
                        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                        - rss_before
                    ) / 1024,
                    "elapsed_seconds": (
                        elapsed_before + time.perf_counter() - started
                    ),
                    "held_out_transfer": evaluate_transfer(
                        held_out, prompts, responses, composer, scorer,
                        episodic_scorer, frozen_scorer, workers,
                    ),
                })
                if state_path is not None:
                    elapsed = elapsed_before + time.perf_counter() - started
                    save_learning_state(state_path, {
                        "version": 1,
                        "seed": seed,
                        "held_out_count": held_out_count,
                        "interactions": interaction,
                        "prompts": prompts.get_state(),
                        "response_library": response_library.get_state(),
                        "responses": responses,
                        "composer": composer.get_state(),
                        "frozen_scorer": frozen_scorer.get_state(),
                        "scorer": scorer.get_state(),
                        "episodic_scorer": episodic_scorer.get_state(),
                        "static_online_f1": static_online_f1,
                        "frozen_online_f1": frozen_online_f1,
                        "global_online_f1": global_online_f1,
                        "episodic_online_f1": episodic_online_f1,
                        "query_latencies": query_latencies,
                        "reports": reports,
                        "elapsed_seconds": elapsed,
                    })
            if interaction >= max_interactions:
                return {
                    "benchmark": "wildchat_live_feedback_scale",
                    "training_dataset": "allenai/WildChat-1M",
                    "held_out_dataset": "OpenAssistant/oasst1:validation",
                    "filters": "English, non-toxic, non-redacted",
                    "evaluation_workers": workers,
                    "resumed_from_interactions": completed_interactions,
                    "state_path": (
                        str(state_path) if state_path is not None else None
                    ),
                    "checkpoints": list(CHECKPOINTS),
                    "results": reports,
                }
    raise RuntimeError("WildChat stream ended before target interactions")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interactions", type=int, default=30_000)
    parser.add_argument("--held-out", type=int, default=200)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Threads for read-only held-out evaluation (online learning stays ordered)",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument(
        "--state-path",
        type=Path,
        default=None,
        help="Write resumable learned state at every benchmark checkpoint",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from --state-path instead of starting from empty state",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    state_path = (
        args.state_path
        if args.state_path is not None
        else args.output_dir / "live_feedback_scale_state.pt"
    )
    report = run(
        args.interactions,
        args.held_out,
        workers=args.workers,
        state_path=state_path,
        resume=args.resume,
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = args.output_dir / f"live_feedback_scale_{stamp}.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Report: {path}")


if __name__ == "__main__":
    main()
