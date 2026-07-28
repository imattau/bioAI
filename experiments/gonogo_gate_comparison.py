"""Before/after comparison: does enabling gonogo_gate_enabled actually help?

gonogo_feedback_benchmark.py showed gonogo's own shadow judgment
(gonogo_go) sometimes converges toward ground truth and sometimes doesn't
at ~40-60 scenario training scales -- but that measures agreement with
ground truth in isolation, not what actually matters: does turning the
gate ON change end-to-end retrieval accuracy for better or worse, versus
just leaving the calibrated heuristic alone?

This trains one agent with live feedback (gate OFF throughout training,
exactly as designed -- see record_feedback), then forks it via save/load
into two identical copies that diverge only in gonogo_gate_enabled, and
evaluates both on the SAME held-out scenarios (generated fresh, not seen
during training). Training happens once; evaluation is frozen (no further
record_feedback calls) so the comparison is a fair, fixed-weights,
side-by-side reading of the trained gate's actual effect -- not an
anecdote from a single conversation.

Usage:
    python experiments/gonogo_gate_comparison.py --train-scenarios 50 --eval-scenarios 25 --model qwen2.5-coder:1.5b
"""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from src.text import BioAIDialogueAgent

from gonogo_feedback_benchmark import generate_scenario, _norm

_TOPICS = ("astronomy", "geography", "invented history", "biology",
           "technology", "invented folklore", "chemistry", "music",
           "sports", "cooking")


@dataclass
class EvalTurn:
    scenario: int
    kind: str  # "matching" or "trap"
    question: str
    accepted: bool
    correct: bool


def train(agent: BioAIDialogueAgent, model: str, n_scenarios: int, start_index: int) -> int:
    """Live-feedback training loop, gate OFF throughout (as record_feedback's
    design requires -- see its docstring). Returns count of scenarios that
    actually generated successfully."""
    trained = 0
    for i in range(start_index, start_index + n_scenarios):
        try:
            data = generate_scenario(model, i, _TOPICS[i % len(_TOPICS)])
        except Exception as exc:  # noqa: BLE001 -- generation can fail many ways
            print(f"  [train {i}] generation failed: {exc}")
            continue

        agent.process_turn(data["fact"])
        for kind, question, should_accept in (
            ("matching", data["matching_question"], True),
            ("trap", data["trap_question"], False),
        ):
            result = agent.process_turn(question)
            accepted = bool(result["retrieval_accepted"])
            correct = (
                accepted and _norm(data["fact"]) in _norm(result["response"])
                if should_accept else not accepted
            )
            agent.record_feedback(correct=correct)
        trained += 1
    return trained


def evaluate(agent: BioAIDialogueAgent, model: str, n_scenarios: int, start_index: int) -> list[EvalTurn]:
    """Frozen evaluation: no record_feedback calls, so this reads the
    gate's effect at fixed weights, not a moving target."""
    records: list[EvalTurn] = []
    for i in range(start_index, start_index + n_scenarios):
        try:
            data = generate_scenario(model, i, _TOPICS[i % len(_TOPICS)])
        except Exception as exc:  # noqa: BLE001
            print(f"  [eval {i}] generation failed: {exc}")
            continue

        agent.process_turn(data["fact"])
        for kind, question, should_accept in (
            ("matching", data["matching_question"], True),
            ("trap", data["trap_question"], False),
        ):
            result = agent.process_turn(question)
            accepted = bool(result["retrieval_accepted"])
            correct = (
                accepted and _norm(data["fact"]) in _norm(result["response"])
                if should_accept else not accepted
            )
            records.append(EvalTurn(
                scenario=i, kind=kind, question=question,
                accepted=accepted, correct=correct,
            ))
    return records


def run(model: str, train_scenarios: int, eval_scenarios: int) -> dict:
    agent = BioAIDialogueAgent(vsa_dim=64)
    agent.gonogo_gate_enabled = False  # gate OFF throughout training, by design -- see record_feedback
    trained = train(agent, model, train_scenarios, start_index=0)
    print(f"Trained on {trained}/{train_scenarios} scenarios.")

    path = Path(tempfile.mktemp(suffix=".pt"))
    agent.save(path)
    try:
        agent_off = BioAIDialogueAgent.load(path)
        agent_on = BioAIDialogueAgent.load(path)
        agent_off.gonogo_gate_enabled = False
        agent_on.gonogo_gate_enabled = True
        assert not agent_off.gonogo_gate_enabled

        # Same held-out scenarios (by index) applied independently to each
        # fork, starting well past the training range so nothing overlaps.
        eval_start = 10_000
        records_off = evaluate(agent_off, model, eval_scenarios, eval_start)
        records_on = evaluate(agent_on, model, eval_scenarios, eval_start)
    finally:
        path.unlink(missing_ok=True)

    def pct(records: list[EvalTurn]) -> float:
        return 100 * sum(r.correct for r in records) / len(records) if records else 0.0

    def kind_pct(records: list[EvalTurn], kind: str) -> float:
        subset = [r for r in records if r.kind == kind]
        return pct(subset)

    return {
        "config": {
            "model": model, "train_scenarios": train_scenarios,
            "eval_scenarios": eval_scenarios,
        },
        "trained_scenarios_actual": trained,
        "eval_turns": len(records_off),
        "gate_off_accuracy_pct": pct(records_off),
        "gate_on_accuracy_pct": pct(records_on),
        "gate_off_matching_pct": kind_pct(records_off, "matching"),
        "gate_on_matching_pct": kind_pct(records_on, "matching"),
        "gate_off_trap_pct": kind_pct(records_off, "trap"),
        "gate_on_trap_pct": kind_pct(records_on, "trap"),
        "records_off": [asdict(r) for r in records_off],
        "records_on": [asdict(r) for r in records_on],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-scenarios", type=int, default=50)
    parser.add_argument("--eval-scenarios", type=int, default=25)
    parser.add_argument("--model", default="qwen2.5-coder:1.5b")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()

    result = run(args.model, args.train_scenarios, args.eval_scenarios)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / (
        f"gonogo_gate_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    path.write_text(json.dumps(result, indent=2) + "\n")
    summary = {k: v for k, v in result.items() if not k.startswith("records_")}
    print(json.dumps(summary, indent=2))
    print(f"Report: {path}")


if __name__ == "__main__":
    main()
