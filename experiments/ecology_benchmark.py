"""Phase 3 of the response-ecosystem plan: does real synthesis actually
compose more supported propositions than the current single-pass
generator, without increasing unsupported (hallucinated) ones?

Compares four systems on the same LLM-generated scenarios, each requiring
two facts about one subject to answer fully (the wombat/marsupial example
from the plan, generalized: "{subject} is {category}." + "{subject} is in
{place}." -- both sentences are assembled programmatically from
LLM-supplied short phrases, not trusted as LLM-written full sentences,
following llm_relational_benchmark.py's precedent, so ground truth is
guaranteed extractable rather than hoped for):

  1. retrieval_only          -- best single whole-memory candidate, never
                                 combines sources (today's fallback when
                                 neither chunk_composer nor an ecosystem
                                 is enabled).
  2. chunk_composer_single_pass -- today's actual live path
                                 (FixedSpliceCandidateGenerator + a single
                                 SequenceCandidateScorer ranking pass, no
                                 composer training data since each
                                 scenario is independent -- see the
                                 module docstring note below).
  3. ecosystem_selection_only -- Phase 1: niches + survival rounds over
                                 the same raw candidates as #2, no
                                 proposition genotype.
  4. ecosystem_full          -- Phase 2: proposition genotype, mutation,
                                 recombination, predation.

Ground truth for each scenario is exactly the two propositions the
constructed memories encode -- computed via extract_propositions() itself
(the same function every system's proposition-aware machinery uses), so
"supported"/"unsupported" is measured against the same criterion the
system is built around, not an external LLM judge. A response's own
asserted propositions are recovered the same way: extract_propositions()
applied to the response text.

Note: chunk_composer_single_pass passes composer=None. LearnedChunkComposer
only contributes once it has seen prior (prompt, response) pairs via
.learn() -- irrelevant here since every scenario is independent and novel,
exactly LearnedChunkComposer's realistic behavior on unfamiliar content.
This isolates the mechanism actually being tested (splicing), not an
artifact of a cold composer.

Usage:
    python experiments/ecology_benchmark.py --scenarios 20 --model qwen2.5-coder:1.5b
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import ollama

from src.text.ecology import ResponseEcosystem, extract_propositions
from src.text.response_candidates import (
    FixedSpliceCandidateGenerator,
    SequenceCandidateScorer,
)

SYSTEMS = (
    "retrieval_only",
    "chunk_composer_single_pass",
    "ecosystem_selection_only",
    "ecosystem_full",
)

_FORBIDDEN_WORDS = {"and", "or", "is", "are", "was", "were"}
_PLAIN_PHRASE_RE = re.compile(r"^[a-zA-Z][a-zA-Z\s-]*$")


def _extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in: {text!r}")
    return json.loads(match.group(0))


def _as_plain_phrase(value, field: str) -> str:
    """Reject rather than coerce, per this repo's established precedent
    (llm_relational_benchmark.py): a phrase containing a copula or
    conjunction would break the fixed-template sentence assembly's
    guarantee that ground truth extracts cleanly."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"expected a plain string for {field!r}, got {value!r}")
    value = value.strip()
    if not _PLAIN_PHRASE_RE.match(value):
        raise ValueError(f"{field!r} has disallowed punctuation: {value!r}")
    words = value.lower().split()
    if len(words) > 3 or any(word in _FORBIDDEN_WORDS for word in words):
        raise ValueError(f"{field!r} isn't a plain short phrase: {value!r}")
    return value


def generate_scenario(model: str, index: int) -> dict:
    response = ollama.chat(model, messages=[
        {
            "role": "system",
            "content": (
                "Output exactly one JSON object with keys: subject, "
                "category, place, question. \"subject\" is a plain plural "
                "noun (1-2 words, e.g. \"wombats\") naming a real or "
                "invented kind of animal or object. \"category\" is a "
                "plain plural noun phrase (1-2 words, e.g. \"marsupials\") "
                "naming a broader class the subject belongs to. \"place\" "
                "is a plain noun phrase (1-3 words, e.g. \"australia\") "
                "naming a place associated with the subject. \"question\" "
                "is a natural question asking generally about the subject "
                "(e.g. \"What do you know about wombats?\"), whose "
                "complete answer requires knowing both its category and "
                "its place. Every value must be a plain string with no "
                "punctuation and none of the words \"is\", \"are\", "
                "\"was\", \"were\", \"and\", \"or\". Output only the JSON "
                "object, nothing else."
            ),
        },
        {"role": "user", "content": f"Scenario number: {index}."},
    ], options={"temperature": 0.9, "num_predict": 100})
    data = _extract_json(response["message"]["content"])
    return {
        "subject": _as_plain_phrase(data["subject"], "subject"),
        "category": _as_plain_phrase(data["category"], "category"),
        "place": _as_plain_phrase(data["place"], "place"),
        "question": data["question"].strip() if isinstance(data.get("question"), str) else "",
    }


def build_evidence(scenario: dict) -> list[str]:
    subject = scenario["subject"].capitalize()
    return [
        f"{subject} is {scenario['category']}.",
        f"{subject} is in {scenario['place']}.",
    ]


def evaluate_response(response_text: str, ground_truth: list) -> dict:
    asserted = extract_propositions([response_text])
    asserted_triples = {(p.subject, p.relation, p.object) for p in asserted}
    ground_truth_triples = {(p.subject, p.relation, p.object) for p in ground_truth}
    ground_truth_lookup = {
        (p.subject, p.relation): p.object for p in ground_truth
    }

    recall = (
        len(asserted_triples & ground_truth_triples) / len(ground_truth_triples)
        if ground_truth_triples else 0.0
    )
    unsupported = asserted_triples - ground_truth_triples
    unsupported_rate = (
        len(unsupported) / len(asserted_triples) if asserted_triples else 0.0
    )
    contradiction = any(
        (subject, relation) in ground_truth_lookup
        and ground_truth_lookup[(subject, relation)] != obj
        for subject, relation, obj in asserted_triples
    )
    return {
        "recall": recall,
        "unsupported_rate": unsupported_rate,
        "contradiction": contradiction,
        "provenance_correct": unsupported_rate == 0.0,
        "supported_count": len(asserted_triples & ground_truth_triples),
    }


def run_retrieval_only(
    prompt: str, evidence: list[str], scorer: SequenceCandidateScorer
) -> tuple[str, None]:
    candidates = [
        {"text": memory, "kind": "complete", "source_ids": [index]}
        for index, memory in enumerate(evidence)
    ]
    ranked = scorer.rank(prompt, candidates, evidence)
    return (ranked[0]["text"] if ranked else ""), None


def run_chunk_composer_single_pass(
    prompt: str,
    evidence: list[str],
    generator: FixedSpliceCandidateGenerator,
    scorer: SequenceCandidateScorer,
) -> tuple[str, None]:
    candidates = generator.generate(prompt, evidence, composer=None)
    ranked = scorer.rank(prompt, candidates, evidence)
    return (ranked[0]["text"] if ranked else ""), None


def run_ecosystem(
    ecosystem: ResponseEcosystem, prompt: str, evidence: list[str]
) -> tuple[str, int]:
    result = ecosystem.generate(prompt, evidence)
    return result.response, len(result.population_sizes)


@dataclass
class ScenarioResult:
    scenario: int
    system: str
    response: str
    recall: float
    unsupported_rate: float
    contradiction: bool
    provenance_correct: bool
    supported_count: int
    generations: int | None
    latency_seconds: float


def run(model: str, scenarios: int) -> tuple[list[ScenarioResult], int, int]:
    scorer = SequenceCandidateScorer()
    generator = FixedSpliceCandidateGenerator()
    ecosystem_selection_only = ResponseEcosystem(
        enable_synthesis=False, max_rounds=4
    )
    ecosystem_full = ResponseEcosystem(enable_synthesis=True, max_rounds=4)

    records: list[ScenarioResult] = []
    generated = 0
    ground_truth_failures = 0
    for index in range(scenarios):
        try:
            scenario = generate_scenario(model, index)
        except Exception as exc:  # noqa: BLE001 -- generation can fail many ways
            print(f"  [scenario {index}] generation failed: {exc}")
            continue
        generated += 1

        evidence = build_evidence(scenario)
        ground_truth = extract_propositions(evidence)
        if len(ground_truth) != 2:
            ground_truth_failures += 1
            print(
                f"  [scenario {index}] ground truth didn't extract cleanly "
                f"(got {len(ground_truth)} propositions), skipping"
            )
            continue

        prompt = scenario["question"] or f"What do you know about {scenario['subject']}?"

        for system_name, run_fn in (
            ("retrieval_only",
             lambda: run_retrieval_only(prompt, evidence, scorer)),
            ("chunk_composer_single_pass",
             lambda: run_chunk_composer_single_pass(prompt, evidence, generator, scorer)),
            ("ecosystem_selection_only",
             lambda: run_ecosystem(ecosystem_selection_only, prompt, evidence)),
            ("ecosystem_full",
             lambda: run_ecosystem(ecosystem_full, prompt, evidence)),
        ):
            start = time.perf_counter()
            response_text, generations = run_fn()
            latency = time.perf_counter() - start
            metrics = evaluate_response(response_text, ground_truth)
            records.append(ScenarioResult(
                scenario=index, system=system_name, response=response_text,
                generations=generations, latency_seconds=latency, **metrics,
            ))
    return records, generated, ground_truth_failures


def summarize(records: list[ScenarioResult], system_name: str) -> dict:
    subset = [record for record in records if record.system == system_name]
    if not subset:
        return {"n": 0}
    generations = [r.generations for r in subset if r.generations is not None]
    return {
        "n": len(subset),
        "mean_recall": statistics.mean(r.recall for r in subset),
        "mean_supported_count": statistics.mean(r.supported_count for r in subset),
        "mean_unsupported_rate": statistics.mean(r.unsupported_rate for r in subset),
        "contradiction_rate": sum(r.contradiction for r in subset) / len(subset),
        "provenance_correct_rate": (
            sum(r.provenance_correct for r in subset) / len(subset)
        ),
        "mean_generations": statistics.mean(generations) if generations else None,
        "mean_latency_seconds": statistics.mean(r.latency_seconds for r in subset),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", type=int, default=20)
    parser.add_argument("--model", default="qwen2.5-coder:1.5b")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()

    records, generated, ground_truth_failures = run(args.model, args.scenarios)
    print(f"Generated {generated}/{args.scenarios} scenarios "
          f"({ground_truth_failures} failed ground-truth extraction).")

    summary = {name: summarize(records, name) for name in SYSTEMS}
    print(json.dumps(summary, indent=2))

    full = summary["ecosystem_full"]
    baseline = summary["chunk_composer_single_pass"]
    success_criterion_met = bool(
        full.get("n") and baseline.get("n")
        and full["mean_supported_count"] > baseline["mean_supported_count"]
        and full["mean_unsupported_rate"] <= baseline["mean_unsupported_rate"]
    )
    print(
        "\nSuccess criterion (composes more supported propositions than "
        f"chunk_composer_single_pass without more unsupported ones): "
        f"{success_criterion_met}"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / (
        f"ecology_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    path.write_text(json.dumps({
        "config": {"model": args.model, "scenarios": args.scenarios},
        "generated": generated,
        "ground_truth_failures": ground_truth_failures,
        "summary": summary,
        "success_criterion_met": success_criterion_met,
        "records": [asdict(record) for record in records],
    }, indent=2) + "\n")
    print(f"Report: {path}")


if __name__ == "__main__":
    main()
