"""LLM-generated multi-turn question/answer conversation benchmark."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import ollama

from src.text import BioAIDialogueAgent


TOPICS = (
    "astronomy", "geography", "biology", "history", "music",
    "mathematics", "physics", "computing", "animals", "architecture",
)


@dataclass
class ConversationRecord:
    turn: int
    phase: str
    user: str
    expected: str
    response: str
    correct: bool
    retrieval_accepted: bool
    retrieval_score: float
    retrieval_margin: float
    latency_ms: float


def parse_pair(text: str) -> tuple[str, str]:
    question = ""
    fact = ""
    for line in text.splitlines():
        label, separator, value = line.partition(":")
        if not separator:
            continue
        if label.strip().upper() == "FACT":
            fact = value.strip()
        elif label.strip().upper() == "QUESTION":
            question = value.strip()
    if not fact or not question:
        raise ValueError(f"Could not parse FACT/QUESTION pair: {text!r}")
    return fact, question


def generate_pair(model: str, topic: str, index: int) -> tuple[str, str]:
    response = ollama.chat(model, messages=[
        {
            "role": "system",
            "content": (
                "Generate one short invented memory fact and a natural question "
                "whose answer is exactly that complete fact. The question must "
                "paraphrase the fact rather than repeat its word order. Output "
                "exactly two lines: FACT: ... and QUESTION: ..."
            ),
        },
        {
            "role": "user",
            "content": f"Topic: {topic}. Pair number: {index}.",
        },
    ], options={"temperature": 0.7, "num_predict": 100})
    return parse_pair(response["message"]["content"])


def run(model: str, pairs: int, vsa_dim: int,
        semantic_model: str | None = None) -> tuple[dict, list[ConversationRecord]]:
    agent = BioAIDialogueAgent(vsa_dim=vsa_dim)
    if semantic_model:
        agent.enable_semantic_retrieval(semantic_model)
    records = []
    pair_failures = 0

    for index in range(pairs):
        try:
            fact, question = generate_pair(
                model, TOPICS[index % len(TOPICS)], index
            )
        except (ValueError, KeyError):
            pair_failures += 1
            continue

        started = time.perf_counter()
        learned = agent.process_turn(fact)
        latency = (time.perf_counter() - started) * 1000
        records.append(ConversationRecord(
            turn=agent.turn_count,
            phase="learn",
            user=fact,
            expected="I'll remember that.",
            response=learned["response"],
            correct=learned["response"] == "I'll remember that.",
            retrieval_accepted=learned["retrieval_accepted"],
            retrieval_score=learned["retrieval_score"],
            retrieval_margin=learned["retrieval_margin"],
            latency_ms=latency,
        ))

        started = time.perf_counter()
        answered = agent.process_turn(question)
        latency = (time.perf_counter() - started) * 1000
        records.append(ConversationRecord(
            turn=agent.turn_count,
            phase="question",
            user=question,
            expected=fact,
            response=answered["response"],
            correct=answered["response"] == fact,
            retrieval_accepted=answered["retrieval_accepted"],
            retrieval_score=answered["retrieval_score"],
            retrieval_margin=answered["retrieval_margin"],
            latency_ms=latency,
        ))

    questions = [record for record in records if record.phase == "question"]
    correct = sum(record.correct for record in questions)
    latencies = sorted(record.latency_ms for record in records)
    summary = {
        "requested_pairs": pairs,
        "completed_pairs": len(questions),
        "generation_parse_failures": pair_failures,
        "total_conversation_turns": len(records),
        "question_correct": correct,
        "question_accuracy_pct": (
            100 * correct / len(questions) if questions else 0.0
        ),
        "retrieval_acceptance_pct": (
            100 * sum(record.retrieval_accepted for record in questions)
            / len(questions) if questions else 0.0
        ),
        "latency_p50_ms": statistics.median(latencies) if latencies else 0.0,
        "latency_p95_ms": (
            latencies[int(0.95 * (len(latencies) - 1))] if latencies else 0.0
        ),
        "library_size": len(agent.library),
        "hot_memory_size": len(agent.decoder),
        "semantic_storage_bytes": agent.library.semantic_storage_bytes,
    }
    return summary, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=50)
    parser.add_argument("--model", default="qwen2.5-coder:1.5b")
    parser.add_argument("--vsa-dim", type=int, default=1000)
    parser.add_argument("--semantic-model", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()
    summary, records = run(
        args.model, args.pairs, args.vsa_dim, args.semantic_model
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / (
        f"llm_conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    path.write_text(json.dumps({
        "config": {
            "model": args.model,
            "pairs": args.pairs,
            "vsa_dim": args.vsa_dim,
            "semantic_model": args.semantic_model,
        },
        "summary": summary,
        "records": [asdict(record) for record in records],
    }, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Report: {path}")


if __name__ == "__main__":
    main()
