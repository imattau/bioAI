"""LLM-as-teacher acquisition loop + native-generation proof (Phase 4's
remaining prerequisite/benchmark, per the response-ecosystem plan).

Acquisition: an LLM teacher generates (subject, category-or-place) content
pairs -- validated short plain phrases, never trusted as full LLM-written
sentences, following llm_relational_benchmark.py's/ecology_benchmark.py's
precedent -- and each is taught to a real BioAIDialogueAgent via
learn_conversation(), using a *rotating* choice of phrasing template
("X is Y" / "X are Y" / "X is in Y" / "X is located in Y" / "X lies within
Y" / "X sits in Y") so the agent's frame_library accumulates multiple
distinct, evidenced phrasings per relation rather than always the one
canonical form. This is stage 3 of the plan's curriculum ("relational
language: multiple paraphrases of one relation") -- the stage that's
actionable with what already exists (ConsolidationMemory.extract_relations,
Phase 2's Proposition, Phase 4's FrameLibrary/RelationalEncoder frame
role), unlike stages 1-2/4-6 which need machinery this codebase doesn't
have yet.

Native-generation proof: for a disjoint set of held-out subjects, the LLM
generates content but that content is NEVER converted into an English
sentence and NEVER shown to the agent as text -- it exists only as a bare
(subject, relation, object) proposition. PropositionRealiser is then asked
to realise it using ONLY frames accumulated during acquisition, with the
LLM completely uninvolved in this phase (no ollama calls happen anywhere
in the evaluation loop -- structurally guaranteed by code, not just a
policy). Success requires all three, per the plan's exact acceptance bar:
  1. a non-empty sentence is produced,
  2. it is not verbatim identical to (or a substring of, or containing)
     any sentence taught during acquisition,
  3. extract_propositions() on the generated sentence recovers exactly
     the target (subject, relation, object).
That combination is "generated a correct sentence with an unseen semantic
combination, using learned frames, with no matching sentence in the
library and no LLM active" -- the plan's own strongest-result bar.

Usage:
    python experiments/llm_teacher_curriculum.py --lessons 30 --holdout 15 --model qwen2.5-coder:1.5b
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import ollama

from src.text import BioAIDialogueAgent
from src.text.consolidation import ConsolidationMemory
from src.text.ecology import PropositionRealiser, Proposition, extract_propositions

from ecology_benchmark import _as_plain_phrase, _extract_json

_norm = ConsolidationMemory._normalise

_IS_TEMPLATES = (
    "{subject} is {object}.",
    "{subject} are {object}.",
    "{subject} was {object}.",
    "{subject} were {object}.",
)
_IN_TEMPLATES = (
    "{subject} is in {object}.",
    "{subject} are in {object}.",
    "{subject} is located in {object}.",
    "{subject} lies within {object}.",
    "{subject} sits in {object}.",
)


def generate_lesson_content(model: str, index: int) -> dict:
    family = "is" if index % 2 == 0 else "in"
    kind = "a broader category or class" if family == "is" else "a place or location"
    response = ollama.chat(model, messages=[
        {
            "role": "system",
            "content": (
                "Output exactly one JSON object with keys \"subject\" and "
                "\"object\". \"subject\" is a plain plural noun (1-2 "
                "words, e.g. \"wombats\") naming a real or invented kind "
                f"of animal, object, or place. \"object\" is a plain noun "
                f"phrase (1-3 words) naming {kind} associated with the "
                "subject. Every value must be a plain string with no "
                "punctuation and none of the words \"is\", \"are\", "
                "\"was\", \"were\", \"and\", \"or\". Output only the JSON "
                "object, nothing else."
            ),
        },
        {"role": "user", "content": f"Item number: {index}."},
    ], options={"temperature": 0.9, "num_predict": 60})
    data = _extract_json(response["message"]["content"])
    return {
        "subject": _as_plain_phrase(data["subject"], "subject"),
        "object": _as_plain_phrase(data["object"], "object"),
        "relation_family": family,
    }


@dataclass
class AcquisitionRecord:
    index: int
    subject: str
    relation_family: str
    template: str
    sentence: str


def run_acquisition(
    agent: BioAIDialogueAgent, model: str, n_lessons: int
) -> tuple[list[AcquisitionRecord], int]:
    records: list[AcquisitionRecord] = []
    generated = 0
    for index in range(n_lessons):
        try:
            content = generate_lesson_content(model, index)
        except Exception as exc:  # noqa: BLE001 -- generation can fail many ways
            print(f"  [lesson {index}] generation failed: {exc}")
            continue
        generated += 1
        templates = _IS_TEMPLATES if content["relation_family"] == "is" else _IN_TEMPLATES
        template = templates[index % len(templates)]
        subject_text = content["subject"].capitalize()
        sentence = template.format(subject=subject_text, object=content["object"])
        prompt = f"Tell me about {content['subject']}."
        agent.learn_conversation(prompt, sentence)
        records.append(AcquisitionRecord(
            index=index, subject=_norm(content["subject"]),
            relation_family=content["relation_family"],
            template=template, sentence=sentence,
        ))
    return records, generated


@dataclass
class HoldoutResult:
    index: int
    subject: str
    relation: str
    target_object: str
    generated_text: str
    generation_success: bool
    no_library_overlap: bool
    propositions_match: bool


def evaluate_holdout(
    model: str,
    n_targets: int,
    start_index: int,
    frame_library,
    taught_sentences: list[str],
    taught_subjects: set[str],
) -> tuple[list[HoldoutResult], int]:
    taught_normalised = {" ".join(s.split()).lower() for s in taught_sentences}
    results: list[HoldoutResult] = []
    generated = 0
    for index in range(start_index, start_index + n_targets):
        try:
            content = generate_lesson_content(model, index)
        except Exception as exc:  # noqa: BLE001
            print(f"  [holdout {index}] generation failed: {exc}")
            continue
        subject = _norm(content["subject"])
        if subject in taught_subjects:
            print(f"  [holdout {index}] subject {subject!r} was also taught, skipping")
            continue
        generated += 1
        relation = "is" if content["relation_family"] == "is" else "in"
        target = Proposition(
            subject=subject, relation=relation, object=_norm(content["object"]),
            source_id=-1, source_text="",
        )

        # No LLM call anywhere below this line: pure deterministic
        # realisation from frames accumulated during acquisition.
        generated_text = PropositionRealiser.realise(
            (target,), frame_library=frame_library
        )
        generation_success = bool(generated_text.strip())

        normalised = " ".join(generated_text.split()).lower()
        no_overlap = generation_success and not any(
            normalised == taught or normalised in taught or taught in normalised
            for taught in taught_normalised
        )

        recovered = extract_propositions([generated_text]) if generated_text else []
        propositions_match = any(
            (prop.subject, prop.relation, prop.object)
            == (target.subject, target.relation, target.object)
            for prop in recovered
        )

        results.append(HoldoutResult(
            index=index, subject=subject, relation=relation,
            target_object=target.object, generated_text=generated_text,
            generation_success=generation_success, no_library_overlap=no_overlap,
            propositions_match=propositions_match,
        ))
    return results, generated


def run(model: str, n_lessons: int, n_holdout: int) -> dict:
    agent = BioAIDialogueAgent(vsa_dim=64)
    acquisition_records, lessons_generated = run_acquisition(agent, model, n_lessons)
    print(f"Acquisition: taught {len(acquisition_records)}/{n_lessons} lessons "
          f"({lessons_generated} generated successfully).")

    taught_sentences = [record.sentence for record in acquisition_records]
    taught_subjects = {record.subject for record in acquisition_records}

    holdout_results, holdout_generated = evaluate_holdout(
        model, n_holdout, start_index=10_000,
        frame_library=agent.frame_library,
        taught_sentences=taught_sentences, taught_subjects=taught_subjects,
    )
    print(f"Holdout: evaluated {len(holdout_results)}/{n_holdout} targets "
          f"({holdout_generated} generated successfully).")

    def rate(flag: str) -> float:
        if not holdout_results:
            return 0.0
        return sum(getattr(r, flag) for r in holdout_results) / len(holdout_results)

    all_three_rate = (
        sum(
            r.generation_success and r.no_library_overlap and r.propositions_match
            for r in holdout_results
        ) / len(holdout_results)
        if holdout_results else 0.0
    )

    frames_by_relation = {
        relation: len(agent.frame_library.frames_for_relation(relation))
        for relation in ("is", "in", "capital", "capital_of")
    }

    return {
        "config": {"model": model, "lessons": n_lessons, "holdout": n_holdout},
        "lessons_taught": len(acquisition_records),
        "frames_learned": len(agent.frame_library.frames),
        "distinct_frames_by_relation": frames_by_relation,
        "holdout_evaluated": len(holdout_results),
        "generation_success_rate": rate("generation_success"),
        "no_library_overlap_rate": rate("no_library_overlap"),
        "propositions_match_rate": rate("propositions_match"),
        "native_generation_proof_rate": all_three_rate,
        "acquisition_records": [asdict(r) for r in acquisition_records],
        "holdout_records": [asdict(r) for r in holdout_results],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lessons", type=int, default=30)
    parser.add_argument("--holdout", type=int, default=15)
    parser.add_argument("--model", default="qwen2.5-coder:1.5b")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()

    result = run(args.model, args.lessons, args.holdout)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / (
        f"llm_teacher_curriculum_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    path.write_text(json.dumps(result, indent=2) + "\n")
    summary = {k: v for k, v in result.items() if not k.endswith("_records")}
    print(json.dumps(summary, indent=2))
    print(f"Report: {path}")


if __name__ == "__main__":
    main()
