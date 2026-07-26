"""Replay an identical conversation transcript through two retrieval modes."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from src.text import BioAIDialogueAgent


@dataclass
class ReplayResult:
    pair: int
    mode: str
    fact: str
    question: str
    expected: str
    response: str
    correct: bool
    accepted: bool
    score: float
    margin: float
    latency_ms: float


def extract_pairs(report: dict) -> list[tuple[str, str, str]]:
    records = report["records"]
    pending_fact = None
    pairs = []
    for record in records:
        if record["phase"] == "learn":
            pending_fact = record["user"]
        elif record["phase"] == "question" and pending_fact is not None:
            pairs.append((pending_fact, record["user"], record["expected"]))
            pending_fact = None
    return pairs


def replay(
    pairs: list[tuple[str, str, str]],
    mode: str,
    semantic_model: str | None,
    vsa_dim: int,
) -> list[ReplayResult]:
    agent = BioAIDialogueAgent(vsa_dim=vsa_dim)
    if semantic_model:
        agent.enable_semantic_retrieval(semantic_model)
    results = []
    for index, (fact, question, expected) in enumerate(pairs):
        agent.process_turn(fact)
        started = time.perf_counter()
        answer = agent.process_turn(question)
        latency = (time.perf_counter() - started) * 1000
        results.append(ReplayResult(
            pair=index,
            mode=mode,
            fact=fact,
            question=question,
            expected=expected,
            response=answer["response"],
            correct=answer["response"] == expected,
            accepted=answer["retrieval_accepted"],
            score=answer["retrieval_score"],
            margin=answer["retrieval_margin"],
            latency_ms=latency,
        ))
    return results


def summarise(results: list[ReplayResult]) -> dict:
    total = len(results)
    correct = sum(result.correct for result in results)
    accepted = sum(result.accepted for result in results)
    wrong_accepted = sum(
        result.accepted and not result.correct for result in results
    )
    latencies = sorted(result.latency_ms for result in results)
    return {
        "pairs": total,
        "correct": correct,
        "accuracy_pct": 100 * correct / total,
        "accepted": accepted,
        "acceptance_pct": 100 * accepted / total,
        "wrong_accepted": wrong_accepted,
        "latency_p50_ms": latencies[len(latencies) // 2],
        "latency_p95_ms": latencies[int(0.95 * (total - 1))],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("transcript", type=Path)
    parser.add_argument("--semantic-model", default="qwen3-embedding:0.6b")
    parser.add_argument("--vsa-dim", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()
    source = json.loads(args.transcript.read_text())
    pairs = extract_pairs(source)
    lexical = replay(pairs, "lexical_vsa", None, args.vsa_dim)
    semantic = replay(
        pairs, "semantic_vsa", args.semantic_model, args.vsa_dim
    )
    lexical_by_pair = {result.pair: result for result in lexical}
    semantic_by_pair = {result.pair: result for result in semantic}
    transitions = {
        "fixed": sum(
            not lexical_by_pair[index].correct
            and semantic_by_pair[index].correct
            for index in lexical_by_pair
        ),
        "regressed": sum(
            lexical_by_pair[index].correct
            and not semantic_by_pair[index].correct
            for index in lexical_by_pair
        ),
        "both_correct": sum(
            lexical_by_pair[index].correct
            and semantic_by_pair[index].correct
            for index in lexical_by_pair
        ),
        "both_incorrect": sum(
            not lexical_by_pair[index].correct
            and not semantic_by_pair[index].correct
            for index in lexical_by_pair
        ),
    }
    payload = {
        "source_transcript": str(args.transcript),
        "pairs": len(pairs),
        "lexical_vsa": summarise(lexical),
        "semantic_vsa": summarise(semantic),
        "paired_transitions": transitions,
        "results": [asdict(result) for result in lexical + semantic],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / (
        f"conversation_replay_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({key: value for key, value in payload.items()
                      if key != "results"}, indent=2))
    print(f"Report: {path}")


if __name__ == "__main__":
    main()
