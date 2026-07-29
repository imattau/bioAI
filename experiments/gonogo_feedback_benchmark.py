"""Benchmark for self.gonogo's record_feedback learning loop.

self.gonogo (src/text/agent.py) makes a real Go/NoGo decision on every
question turn that reaches general retrieval, but its weights only change
when a caller with real outcome knowledge calls agent.record_feedback --
there's no ambient ground-truth signal in ordinary conversation to train it
from safely (see the docstrings in __init__/_retrieve_context/record_feedback,
and RELATIONAL_MEMORY.md/ARCHITECTURE_STATUS.md for why this is deliberate,
not a limitation). This benchmark is that external caller: it supplies real
feedback and measures whether gonogo's own judgment actually improves,
rather than assuming the mechanism works.

Design: an LLM generates, per scenario, one fact plus two questions --
a MATCHING question (paraphrase of the fact, answerable, correct behavior
is accept+correct) and a TRAP question (superficially similar topic,
shares vocabulary, but genuinely unanswerable from that fact -- correct
behavior is abstain). This creates a stream of borderline retrieval
decisions on purpose: trap questions are designed to score close to the
acceptance threshold precisely because they're topically similar, which is
exactly the situation a fixed two-number cutoff handles worst and where a
learned gate has room to actually add value.

One agent persists across all scenarios (unlike llm_relational_benchmark.py's
fresh-agent-per-scenario isolation) so gonogo's weights accumulate real
training across the whole run -- the point here is observing a learning
curve, not scenario isolation. After every question turn, agent.record_feedback
is called with the ground truth this script computed. The heuristic's own
accept/reject decision is never touched (gonogo_gate_enabled stays False
throughout) so this measures gonogo's *shadow* judgment (gonogo_go) against
ground truth, comparing an early window of the run against a late one --
an honest read on whether the learning signal helps, not an assumption
that it does.

Usage:
    python experiments/gonogo_feedback_benchmark.py --scenarios 30 --model qwen2.5-coder:1.5b
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import ollama

from src.text import BioAIDialogueAgent
from src.text.consolidation import ConsolidationMemory

_norm = ConsolidationMemory.normalise


def _extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in: {text!r}")
    return json.loads(match.group(0))


def _as_str(value, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"expected a plain string for {field!r}, got {value!r}")
    return value.strip()


def generate_scenario(model: str, index: int, topic: str) -> dict:
    response = ollama.chat(model, messages=[
        {
            "role": "system",
            "content": (
                "Output exactly one JSON object with keys: fact, "
                "matching_question, trap_question. \"fact\" is one short "
                f"invented factual sentence about {topic}. "
                "\"matching_question\" is a natural question, phrased "
                "differently from the fact (paraphrased, not word-for-word), "
                "whose answer is exactly that fact. \"trap_question\" is a "
                "DIFFERENT question about the same general topic, sharing "
                "some of the same words, but asking about something the fact "
                "does NOT answer (a different specific detail the fact says "
                "nothing about) -- it must NOT be answerable from the fact. "
                "Every value must be a plain string. Output only the JSON "
                "object, nothing else."
            ),
        },
        {"role": "user", "content": f"Scenario number: {index}."},
    ], options={"temperature": 0.9, "num_predict": 150})
    data = _extract_json(response["message"]["content"])
    for key in ("fact", "matching_question", "trap_question"):
        data[key] = _as_str(data[key], key)
    return data


@dataclass
class TurnRecord:
    index: int
    kind: str  # "matching" or "trap"
    question: str
    accepted: bool
    correct: bool
    gonogo_go: bool | None
    gonogo_agrees: bool | None


def run(model: str, scenarios: int) -> tuple[dict, list[TurnRecord]]:
    topics = ("astronomy", "geography", "invented history", "biology",
              "technology", "invented folklore", "chemistry", "music")
    agent = BioAIDialogueAgent(vsa_dim=64)
    agent.gonogo_gate_enabled = False  # measure gonogo's shadow judgment, untouched by its own vetoes
    records: list[TurnRecord] = []
    generation_failures = 0
    turn_index = 0

    for i in range(scenarios):
        try:
            data = generate_scenario(model, i, topics[i % len(topics)])
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            generation_failures += 1
            print(f"  [scenario {i}] generation failed: {exc}")
            continue

        agent.process_turn(data["fact"])

        for kind, question in (
            ("matching", data["matching_question"]),
            ("trap", data["trap_question"]),
        ):
            result = agent.process_turn(question)
            accepted = bool(result["retrieval_accepted"])
            if kind == "matching":
                correct = accepted and _norm(data["fact"]) in _norm(result["response"])
                should_accept = True
            else:
                correct = not accepted
                should_accept = False

            gonogo_go = result.get("gonogo_go")
            gonogo_agrees = (
                (gonogo_go == should_accept) if gonogo_go is not None else None
            )

            agent.record_feedback(correct=correct)

            records.append(TurnRecord(
                index=turn_index, kind=kind, question=question,
                accepted=accepted, correct=correct,
                gonogo_go=gonogo_go, gonogo_agrees=gonogo_agrees,
            ))
            turn_index += 1

    scored = [r for r in records if r.gonogo_agrees is not None]
    third = max(1, len(scored) // 3)
    early = scored[:third]
    late = scored[-third:]

    def agree_pct(window: list[TurnRecord]) -> float:
        return 100 * sum(r.gonogo_agrees for r in window) / len(window) if window else 0.0

    summary = {
        "requested_scenarios": scenarios,
        "generation_failures": generation_failures,
        "total_question_turns": len(records),
        "heuristic_correct_pct": (
            100 * sum(r.correct for r in records) / len(records) if records else 0.0
        ),
        "gonogo_agreement_early_pct": agree_pct(early),
        "gonogo_agreement_late_pct": agree_pct(late),
        "gonogo_agreement_overall_pct": agree_pct(scored),
        "window_size": third,
    }
    return summary, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", type=int, default=30)
    parser.add_argument("--model", default="qwen2.5-coder:1.5b")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()

    summary, records = run(args.model, args.scenarios)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / (
        f"gonogo_feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    path.write_text(json.dumps({
        "config": {"model": args.model, "scenarios": args.scenarios},
        "summary": summary,
        "records": [asdict(r) for r in records],
    }, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Report: {path}")


if __name__ == "__main__":
    main()
