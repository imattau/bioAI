"""LLM-generated relational-memory benchmark.

llm_conversation_benchmark.py tests BioAIDialogueAgent's general free-text
retrieval path with LLM-generated facts, but free-form LLM prose almost
never matches the narrow sentence patterns ConsolidationMemory.extract_relations
requires (confirmed empirically: 0/10 facts in that benchmark's own output
triggered the relational path at all). This benchmark specifically
exercises RelationalMemory / resolve() / resolve_auto instead.

The LLM generates *content* (entities, properties, countries/capitals,
domains) via constrained JSON prompts; sentences are then assembled
programmatically in the exact "X is Y." / "The capital of X is Y." forms
extract_relations needs, rather than trusting the LLM's literal phrasing.

Two scenario types:
  1. Unique facts (capital-of pairs): no collision, tests confident
     relational_reasoning retrieval.
  2. Collision scenarios (two entities, two shared properties, one unique
     property each): tests genuine ambiguity surfacing on the shared
     properties, then resolution via a distinguishing clue -- exercising
     resolve()'s candidate intersection and the resolve_auto escalation
     path in _answer_relational_query.

Each scenario uses a fresh agent to keep measurements isolated (a shared
agent across scenarios risks cross-scenario entity collisions, e.g. two
different LLM-generated scenarios both picking "cat").

Usage:
    python experiments/llm_relational_benchmark.py --scenarios 5 --model qwen2.5-coder:1.5b
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import ollama

from src.text import BioAIDialogueAgent
from src.text.consolidation import ConsolidationMemory

# Use the agent's own normalization (lowercase, strip punctuation) for every
# comparison against agent output below -- LLM-generated names often contain
# punctuation ConsolidationMemory strips at storage time (e.g. "Mercedes-Benz"
# is stored/returned as "mercedes benz"), so a plain .lower() comparison
# produces false-negative "failures" that aren't actually bugs in the agent.
_norm = ConsolidationMemory._normalise


def _extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in: {text!r}")
    return json.loads(match.group(0))


def _as_str(value, field: str) -> str:
    """Reject rather than coerce: the LLM sometimes nests a dict where a
    plain string was asked for (e.g. {"property": "fur"} instead of "fur").
    str()-coercing that would silently store the literal dict repr as an
    "entity" or "property" name, which is a real observed failure mode, not
    a hypothetical one -- so this is a hard requirement, not a formatting
    nicety, and violations are counted as generation failures like any
    other malformed output.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"expected a plain string for {field!r}, got {value!r}")
    return value.strip()


def generate_capital_pair(model: str, index: int) -> tuple[str, str]:
    response = ollama.chat(model, messages=[
        {
            "role": "system",
            "content": (
                "Output exactly one JSON object with keys \"country\" and "
                "\"city\", naming a real or plausible invented country and "
                "its capital city. Each value must be a plain string, 1-2 "
                "words, no punctuation, no nested objects. Output only the "
                "JSON object, nothing else."
            ),
        },
        {"role": "user", "content": f"Pair number: {index}."},
    ], options={"temperature": 0.8, "num_predict": 60})
    data = _extract_json(response["message"]["content"])
    return _as_str(data["country"], "country"), _as_str(data["city"], "city")


def generate_collision_scenario(model: str, index: int, domain: str) -> dict:
    response = ollama.chat(model, messages=[
        {
            "role": "system",
            "content": (
                "Output exactly one JSON object describing two entities in "
                f"the domain of {domain}, with keys: entity_a, entity_b "
                "(each 1-2 words), shared (a list of exactly 2 short "
                "properties, 1-3 words each, that are TRUE of BOTH entities), "
                "unique_a and unique_b (one short property, 1-3 words, true "
                "of ONLY that entity and not the other). Every value must be "
                "a plain string or a list of plain strings -- never a nested "
                "object. All property values must be plain adjectives or "
                "short noun phrases, no punctuation. Output only the JSON "
                "object, nothing else."
            ),
        },
        {"role": "user", "content": f"Scenario number: {index}."},
    ], options={"temperature": 0.9, "num_predict": 150})
    data = _extract_json(response["message"]["content"])
    for key in ("entity_a", "entity_b", "unique_a", "unique_b"):
        data[key] = _as_str(data[key], key)
    shared = data["shared"]
    if not isinstance(shared, list):
        raise ValueError(f"expected 'shared' to be a list, got {shared!r}")
    data["shared"] = [_as_str(p, "shared[]") for p in shared][:2]
    if len(data["shared"]) != 2:
        raise ValueError(f"Expected exactly 2 shared properties: {data}")
    if _norm(data["entity_a"]) == _norm(data["entity_b"]):
        raise ValueError(f"entity_a == entity_b: {data}")
    return data


@dataclass
class ScenarioResult:
    kind: str
    detail: str
    passed: bool
    response_mode: str
    response: str


def run_capital_scenario(model: str, index: int) -> ScenarioResult:
    country, city = generate_capital_pair(model, index)
    agent = BioAIDialogueAgent(vsa_dim=64)
    agent.process_turn(f"The capital of {country} is {city}.")
    result = agent.process_turn(f"What is the capital of {country}?")
    passed = (
        result["response_mode"] == "relational_reasoning"
        and _norm(city) in _norm(result["response"])
    )
    return ScenarioResult(
        kind="capital", detail=f"{country} -> {city}",
        passed=passed, response_mode=result["response_mode"],
        response=result["response"],
    )


def run_collision_scenario(model: str, index: int, domain: str) -> list[ScenarioResult]:
    data = generate_collision_scenario(model, index, domain)
    a, b = data["entity_a"], data["entity_b"]
    shared1, shared2 = data["shared"]
    unique_a, unique_b = data["unique_a"], data["unique_b"]
    detail = f"{a}/{b} share [{shared1}, {shared2}], unique [{unique_a}]/[{unique_b}]"
    results = []

    agent = BioAIDialogueAgent(vsa_dim=64)
    for entity in (a, b):
        agent.process_turn(f"{entity} is {shared1}.")
        agent.process_turn(f"{entity} is {shared2}.")

    # Genuinely tied: no distinguishing fact about either entity is known
    # yet, so nothing to escalate to either -- must stay ambiguous.
    tied = agent.process_turn(f"Who is {shared1} and is {shared2}?")
    tied_candidates = {_norm(c) for c in tied.get("ambiguous_candidates", [])}
    tied_passed = (
        tied["response_mode"] == "relational_ambiguous"
        and {_norm(a), _norm(b)} <= tied_candidates
    )
    results.append(ScenarioResult(
        kind="collision_ambiguous", detail=detail, passed=tied_passed,
        response_mode=tied["response_mode"], response=tied["response"],
    ))

    # A's distinguishing fact becomes known; asking with it as an EXPLICIT
    # clue must resolve deterministically to A (B's fact still isn't known
    # at all yet, so there's no ambiguity in which fact gets used).
    agent.process_turn(f"{a} is {unique_a}.")
    resolved = agent.process_turn(f"Who is {shared1} and is {unique_a}?")
    resolved_passed = (
        resolved["response_mode"] == "relational_reasoning"
        and _norm(a) in _norm(resolved["response"])
    )
    results.append(ScenarioResult(
        kind="collision_resolved_explicit", detail=detail, passed=resolved_passed,
        response_mode=resolved["response_mode"], response=resolved["response"],
    ))

    # B's distinguishing fact becomes known too, without ever being stated
    # in a question. Re-asking the ORIGINAL bare tied question now, the
    # agent must escalate on its own (resolve_auto) to one of the two
    # known distinguishing facts and resolve it -- to EITHER A or B is
    # correct (both are genuinely valid: the original question still
    # matches both, and using either entity's own extra property to settle
    # it is a legitimate, if arbitrary, tie-break, not an error). What
    # would be wrong is staying ambiguous when a real answer is available,
    # or answering something that's neither A nor B.
    agent.process_turn(f"{b} is {unique_b}.")
    escalated = agent.process_turn(f"Who is {shared1} and is {shared2}?")
    escalated_passed = (
        escalated["response_mode"] == "relational_reasoning"
        and (_norm(a) in _norm(escalated["response"])
             or _norm(b) in _norm(escalated["response"]))
    )
    results.append(ScenarioResult(
        kind="collision_escalated", detail=detail, passed=escalated_passed,
        response_mode=escalated["response_mode"], response=escalated["response"],
    ))

    return results


def run(model: str, scenarios: int) -> tuple[dict, list[ScenarioResult]]:
    domains = ("animals", "kitchen gadgets", "musical instruments",
               "vehicles", "sports", "jobs", "planets", "fruits")
    all_results: list[ScenarioResult] = []
    generation_failures = 0

    for i in range(scenarios):
        try:
            all_results.append(run_capital_scenario(model, i))
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            generation_failures += 1
            print(f"  [capital {i}] generation failed: {exc}")

        try:
            all_results.extend(
                run_collision_scenario(model, i, domains[i % len(domains)])
            )
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            generation_failures += 1
            print(f"  [collision {i}] generation failed: {exc}")

    by_kind: dict[str, list[ScenarioResult]] = {}
    for r in all_results:
        by_kind.setdefault(r.kind, []).append(r)

    summary = {
        "requested_scenarios": scenarios,
        "generation_failures": generation_failures,
        "total_checks": len(all_results),
        "overall_pass_pct": (
            100 * sum(r.passed for r in all_results) / len(all_results)
            if all_results else 0.0
        ),
        "by_kind": {
            kind: {
                "n": len(results),
                "pass_pct": 100 * sum(r.passed for r in results) / len(results),
            }
            for kind, results in by_kind.items()
        },
    }
    return summary, all_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", type=int, default=5)
    parser.add_argument("--model", default="qwen2.5-coder:1.5b")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()

    summary, results = run(args.model, args.scenarios)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / (
        f"llm_relational_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    path.write_text(json.dumps({
        "config": {"model": args.model, "scenarios": args.scenarios},
        "summary": summary,
        "results": [asdict(r) for r in results],
    }, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Report: {path}")


if __name__ == "__main__":
    main()
