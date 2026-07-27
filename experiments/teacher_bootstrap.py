"""Distil an offline LLM teacher into a teacher-free BioAI policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.text import (
    OllamaBootstrapTeacher,
    build_teacher_records,
    read_teacher_records,
    save_bootstrap,
    train_bootstrap,
)


def conversation_rows(path: Path, limit: int | None = None):
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            try:
                prompt = str(value["prompt"])
                response = str(value["response"])
            except KeyError as exc:
                raise ValueError(
                    f"Input line {line_number} requires prompt and response"
                ) from exc
            evidence = [str(item) for item in value.get("evidence", [])]
            yield prompt, response, evidence
            if limit is not None and line_number >= limit:
                return


def learning_state_rows(
    path: Path,
    limit: int | None = None,
    max_response_chars: int | None = None,
    sampling: str = "sequential",
    extra: int = 0,
):
    state = torch.load(path, map_location="cpu", weights_only=False)
    prompts = state["prompts"]["texts"]
    responses = state["responses"]
    count = min(len(prompts), len(responses))
    eligible = []
    for index in range(count):
        prompt = str(prompts[index])
        response = str(responses[index])
        if not prompt.strip() or not response.strip():
            continue
        if max_response_chars is not None and len(response) > max_response_chars:
            continue
        eligible.append((prompt, response, [response]))
    if sampling not in {"sequential", "even"}:
        raise ValueError(f"Unsupported sampling strategy: {sampling}")
    if sampling == "even" and limit is not None and len(eligible) > limit:
        if limit == 1:
            selected_indices = [len(eligible) // 2]
        else:
            selected_indices = [
                round(index * (len(eligible) - 1) / (limit - 1))
                for index in range(limit)
            ]
        selected = set(selected_indices)
        replacement_indices = [
            index for index in range(len(eligible))
            if index not in selected
        ][:extra]
        eligible = [
            eligible[index]
            for index in [*selected_indices, *replacement_indices]
        ]
    emitted = 0
    for prompt, response, evidence in eligible:
        yield prompt, response, [response]
        emitted += 1
        if limit is not None and emitted >= limit + extra:
            return


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        help="JSONL with prompt, response, and optional evidence",
    )
    parser.add_argument(
        "--learning-state",
        type=Path,
        help="Use conversations directly from a saved scale-benchmark state",
    )
    parser.add_argument(
        "--records",
        type=Path,
        default=Path("checkpoints/teacher_records.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("checkpoints/bioai_bootstrap.pt"),
    )
    parser.add_argument("--model", default="gemma4:12b")
    parser.add_argument("--teacher-retries", type=int, default=3)
    parser.add_argument("--teacher-timeout", type=float, default=60.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--max-response-chars",
        type=int,
        default=2_000,
        help="Skip oversized saved responses during teacher labeling",
    )
    parser.add_argument(
        "--sampling",
        choices=("sequential", "even"),
        default="sequential",
        help="Select saved conversations sequentially or across the full state",
    )
    parser.add_argument(
        "--sampling-buffer",
        type=int,
        default=50,
        help="Deterministic replacement samples available after refusals",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument(
        "--reuse-records",
        action="store_true",
        help="Train from existing --records without calling an LLM",
    )
    parser.add_argument(
        "--resume-records",
        action="store_true",
        help="Verify and continue an interrupted teacher-record JSONL",
    )
    parser.add_argument(
        "--skip-teacher-failures",
        action="store_true",
        help="Skip exhausted invalid outputs and use replacement samples",
    )
    args = parser.parse_args()

    if args.reuse_records:
        records = read_teacher_records(args.records)
    else:
        if args.input is not None and args.learning_state is not None:
            parser.error("use only one of --input and --learning-state")
        if args.input is None and args.learning_state is None:
            parser.error(
                "--input or --learning-state is required unless "
                "--reuse-records is used"
            )
        teacher = OllamaBootstrapTeacher(
            args.model,
            retries=args.teacher_retries,
            timeout_seconds=args.teacher_timeout,
        )
        conversations = (
            learning_state_rows(
                args.learning_state,
                args.limit,
                args.max_response_chars,
                args.sampling,
                args.sampling_buffer,
            )
            if args.learning_state is not None
            else conversation_rows(args.input, args.limit)
        )
        records = build_teacher_records(
            teacher,
            conversations,
            args.records,
            target_count=args.limit,
            resume=args.resume_records,
            progress=lambda count, target: print(
                f"Teacher records: {count}/{target or '?'}",
                flush=True,
            ),
            skipped=lambda prompt, reason: print(
                f"Teacher skipped sample: {reason}",
                flush=True,
            ),
            skip_failures=args.skip_teacher_failures,
        )

    state = train_bootstrap(records, epochs=args.epochs)
    save_bootstrap(args.output, state)
    print(json.dumps({
        "records": state["records"],
        "epochs": state["epochs"],
        "scorer_updates": state["candidate_scorer"]["updates"],
        "teacher_free": state["teacher_free"],
        "records_path": str(args.records),
        "checkpoint_path": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
