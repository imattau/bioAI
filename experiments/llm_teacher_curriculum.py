"""LLM-as-teacher acquisition loop + native-generation proof (Phase 4),
extended in Phase 6 to map the generalization boundary rather than just
prove existence: multi-proposition lessons, a frozen test specification
for reproducible scale-up, paraphrase-diversity measurement, and
adversarial near-miss subjects. See the response-ecosystem plan.

Acquisition: an LLM teacher generates, per lesson, ONE subject plus a
category/place/property triple -- validated short plain phrases, never
trusted as full LLM-written sentences, following llm_relational_benchmark.py's/
ecology_benchmark.py's precedent. All three facts about that one subject
("X is CATEGORY", "X is in PLACE", "X has PROPERTY") are taught via a
*rotating* choice of phrasing template per relation, so frame_library
accumulates several distinct, evidenced phrasings per relation rather
than always the one canonical form, AND every held-out target now
requires composing 3 propositions, not 1 -- this is stage 3 of the plan's
curriculum ("relational language: multiple paraphrases of one relation")
plus Phase 6 item 3 (multi-proposition generation), built together since
they share the same content shape.

Phase 8 adds a second lesson kind, "country" (roughly 1 in 3 lessons),
alongside the original "animal" kind: country lessons teach "is"
(category, fixed as "country" -- not worth an LLM call to confirm what's
definitionally true), "in" (a continent/region), and "capital" (capital
city), instead of "is"/"in"/"has". This gives Phase 8's possessive-pronoun
substitution in PropositionRealiser (a "capital" clause after an earlier
compound is/in clause for the same subject renders "Its capital is
Paris." instead of "The capital of France is Paris.") real content to
exercise -- the prior "animal" curriculum never taught "capital" at all
(`distinct_frames_by_relation` always showed 0 for it).

Native-generation proof: for a disjoint set of held-out subjects, the LLM
generates content that is NEVER converted into an English sentence or
shown to the agent as text -- only as bare (subject, relation, object)
propositions. PropositionRealiser then realises all 3 using ONLY frames
accumulated during acquisition, with zero LLM calls anywhere in the
evaluation path. Success requires, per subject:
  1. a non-empty sentence is produced,
  2. it is not verbatim identical to (or a substring of, or containing)
     any sentence taught during acquisition,
  3. extract_propositions() on the generated sentence recovers ALL 3
     target (subject, relation, object) triples, not just one.

Phase 6 additions:
  --save-spec/--load-spec: freeze the LLM-generated content to a JSON
    file so repeated runs test code changes against stable content, not
    a moving target re-sampled from the LLM every time.
  --paraphrase-diversity-samples: for one relation with >=2 observed
    frames, sample select_frame() this many times and report the
    resulting output distribution -- confirms weighted-random frame
    selection produces real variety in practice, not just in one-shot
    manual checks.
  --adversarial-pairs: after acquisition, generate this many pairs of
    subjects sharing their "is"/"in" facts but differing in "has", then
    confirm realising one subject's held-out "has" property never
    surfaces the other subject's differing one -- a cross-contamination
    check on the deterministic realiser using the same near-miss shape
    llm_relational_benchmark.py already uses for RelationalMemory
    directly.

Usage:
    python experiments/llm_teacher_curriculum.py --lessons 30 --holdout 15 --model qwen2.5-coder:1.5b
    python experiments/llm_teacher_curriculum.py --lessons 30 --holdout 15 --save-spec checkpoints/spec.json
    python experiments/llm_teacher_curriculum.py --load-spec checkpoints/spec.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import ollama

from src.text import BioAIDialogueAgent
from src.text.consolidation import ConsolidationMemory
from src.text.ecology import PropositionRealiser, Proposition, extract_propositions

from ecology_benchmark import _as_plain_phrase, _extract_json

_norm = ConsolidationMemory.normalise

# Deliberately local to this benchmark's target-vs-recovered comparison,
# NOT a change to ConsolidationMemory.normalise itself: that function is
# also used for literal storage/display content elsewhere (e.g. "Pluto is
# a planet" is correctly stored and shown AS "a planet", article
# included -- tests/test_text/test_relational_agent.py depends on this).
# The mismatch this strips is narrower and specific to Phase 7: taught
# objects never have an article ("household item", per generate_lesson_content's
# validation), but PropositionRealiser's article insertion (realiser.py)
# adds one when rendering a singular is/has object, so re-extracting the
# realised text captures "a household item" -- same fact, different
# surface form, needs an article-tolerant comparison here specifically.
_LEADING_ARTICLE_RE = re.compile(r"^(?:a|an|the)\s+")


def _strip_article(text: str) -> str:
    return _LEADING_ARTICLE_RE.sub("", text)


# Phase 8's possessive-pronoun rendering for "capital" ("Its capital is
# Paris." instead of "The capital of France is Paris.") is a real,
# disclosed extraction gap, not a bug to hide: `extract_propositions`
# processes one sentence at a time with no cross-sentence discourse
# state (paragraph-level discourse tracking was explicitly named
# out-of-scope for this phase, see the plan), so it recovers
# ("its capital", "is", "paris") -- wrong subject, wrong relation label --
# instead of ("france", "capital", "paris"). This benchmark, unlike
# extract_propositions in general, DOES know the one subject a given
# holdout item concerns (`target_propositions` is built for exactly one
# subject), so resolving the pronoun back to it here is safe and doesn't
# require general coreference resolution -- the same narrowly-scoped,
# benchmark-local fix `_strip_article` above already established the
# precedent for.
_POSSESSIVE_CAPITAL_SUBJECT_RE = re.compile(r"^(?:its|their)\s+capital$", re.IGNORECASE)


def _resolve_possessive_capital(propositions: list[Proposition], subject: str) -> list[Proposition]:
    return [
        Proposition(subject=subject, relation="capital", object=p.object,
                    source_id=p.source_id, source_text=p.source_text)
        if _POSSESSIVE_CAPITAL_SUBJECT_RE.match(p.subject) else p
        for p in propositions
    ]

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
_HAS_TEMPLATES = (
    "{subject} has {object}.",
    "{subject} have {object}.",
)
# Both variants match ConsolidationMemory's "capital" pattern
# (`^(?:the\s+)?capital\s+of\s+(.+?)\s+is\s+(.+)$`), so taught sentences
# round-trip to (subject=country, relation="capital", object=capital_city)
# -- NOT the inverted "capital_of" relation, and not the plain "is"
# catch-all (which would wrongly capture "france's capital" as the
# subject if a possessive phrasing were used instead).
_CAPITAL_TEMPLATES = (
    "The capital of {subject} is {object}.",
    "Capital of {subject} is {object}.",
)
_ANIMAL_RELATIONS = ("is", "in", "has")
_COUNTRY_RELATIONS = ("is", "in", "capital")


def _relations_for_kind(kind: str) -> tuple[str, ...]:
    return _COUNTRY_RELATIONS if kind == "country" else _ANIMAL_RELATIONS


def generate_lesson_content(model: str, index: int, kind: str = "animal") -> dict:
    """One LLM call per lesson: a subject plus a triple of facts about it
    -- enough to teach 3 propositions from a single generation call rather
    than 3. "animal" kind teaches is/in/has (category/place/property);
    "country" kind teaches is/in/capital (fixed "country" category/
    region/capital city) -- see module docstring for why."""
    if kind == "country":
        response = ollama.chat(model, messages=[
            {
                "role": "system",
                "content": (
                    "Output exactly one JSON object with keys \"subject\", "
                    "\"place\", \"capital_city\". \"subject\" is the plain "
                    "name of a real country (1-2 words, e.g. \"france\"). "
                    "\"place\" is a plain noun phrase (1-3 words) naming "
                    "the continent or region the country is in (e.g. "
                    "\"western europe\"). \"capital_city\" is that "
                    "country's real capital city, a plain 1-2 word name. "
                    "Every value must be a plain string with no "
                    "punctuation and none of the words \"is\", \"are\", "
                    "\"was\", \"were\", \"has\", \"have\", \"and\", "
                    "\"or\". Output only the JSON object, nothing else."
                ),
            },
            {"role": "user", "content": f"Item number: {index}."},
        ], options={"temperature": 0.9, "num_predict": 100})
        data = _extract_json(response["message"]["content"])
        return {
            "kind": "country",
            "subject": _as_plain_phrase(data["subject"], "subject"),
            "category": "country",
            "place": _as_plain_phrase(data["place"], "place"),
            "capital_city": _as_plain_phrase(data["capital_city"], "capital_city"),
        }

    response = ollama.chat(model, messages=[
        {
            "role": "system",
            "content": (
                "Output exactly one JSON object with keys \"subject\", "
                "\"category\", \"place\", \"property\". \"subject\" is a "
                "plain plural noun (1-2 words, e.g. \"wombats\") naming a "
                "real or invented kind of animal or object. \"category\" "
                "is a plain plural noun phrase (1-2 words) naming a "
                "broader class the subject belongs to. \"place\" is a "
                "plain noun phrase (1-3 words) naming a place associated "
                "with the subject. \"property\" is a plain noun phrase "
                "(1-3 words) naming a physical feature or trait the "
                "subject has (e.g. \"sharp teeth\"). Every value must be "
                "a plain string with no punctuation and none of the "
                "words \"is\", \"are\", \"was\", \"were\", \"has\", "
                "\"have\", \"and\", \"or\". Output only the JSON object, "
                "nothing else."
            ),
        },
        {"role": "user", "content": f"Item number: {index}."},
    ], options={"temperature": 0.9, "num_predict": 100})
    data = _extract_json(response["message"]["content"])
    return {
        "kind": "animal",
        "subject": _as_plain_phrase(data["subject"], "subject"),
        "category": _as_plain_phrase(data["category"], "category"),
        "place": _as_plain_phrase(data["place"], "place"),
        "property": _as_plain_phrase(data["property"], "property"),
    }


def _templates_for(relation: str) -> tuple[str, ...]:
    return {
        "is": _IS_TEMPLATES, "in": _IN_TEMPLATES, "has": _HAS_TEMPLATES,
        "capital": _CAPITAL_TEMPLATES,
    }[relation]


def _object_for(relation: str, content: dict) -> str:
    return {
        "is": content["category"], "in": content["place"],
        "has": content.get("property"), "capital": content.get("capital_city"),
    }[relation]


@dataclass
class AcquisitionRecord:
    index: int
    subject: str
    sentences: list[str] = field(default_factory=list)


def teach_lesson(
    agent: BioAIDialogueAgent, index: int, content: dict
) -> AcquisitionRecord:
    subject_text = content["subject"].capitalize()
    sentences = []
    for relation_index, relation in enumerate(_relations_for_kind(content["kind"])):
        templates = _templates_for(relation)
        template = templates[(index + relation_index) % len(templates)]
        obj = _object_for(relation, content)
        sentence = template.format(subject=subject_text, object=obj)
        prompt = f"Tell me about {content['subject']}."
        agent.learn_conversation(prompt, sentence)
        sentences.append(sentence)
    return AcquisitionRecord(
        index=index, subject=_norm(content["subject"]), sentences=sentences
    )


def run_acquisition(
    agent: BioAIDialogueAgent, contents: list[dict]
) -> list[AcquisitionRecord]:
    return [teach_lesson(agent, index, content) for index, content in enumerate(contents)]


def target_propositions(subject: str, content: dict) -> tuple[Proposition, ...]:
    return tuple(
        Proposition(
            subject=subject, relation=relation,
            object=_norm(_object_for(relation, content)),
            source_id=-1, source_text="",
        )
        for relation in _relations_for_kind(content["kind"])
    )


@dataclass
class HoldoutResult:
    index: int
    subject: str
    generated_text: str
    generation_success: bool
    no_library_overlap: bool
    propositions_match: bool
    propositions_recovered: int


def evaluate_holdout(
    contents: list[dict],
    frame_library,
    taught_sentences: list[str],
    taught_subjects: set[str],
) -> list[HoldoutResult]:
    taught_normalised = {" ".join(s.split()).lower() for s in taught_sentences}
    results: list[HoldoutResult] = []
    for index, content in enumerate(contents):
        subject = _norm(content["subject"])
        if subject in taught_subjects:
            print(f"  [holdout {index}] subject {subject!r} was also taught, skipping")
            continue
        targets = target_propositions(subject, content)

        # No LLM call anywhere below this line: pure deterministic
        # realisation from frames accumulated during acquisition.
        generated_text = PropositionRealiser.realise(targets, frame_library=frame_library)
        generation_success = bool(generated_text.strip())

        normalised = " ".join(generated_text.split()).lower()
        no_overlap = generation_success and not any(
            normalised == taught or normalised in taught or taught in normalised
            for taught in taught_normalised
        )

        recovered = extract_propositions([generated_text]) if generated_text else []
        recovered = _resolve_possessive_capital(recovered, subject)
        recovered_triples = {
            (p.subject, p.relation, _strip_article(p.object)) for p in recovered
        }
        target_triples = {
            (p.subject, p.relation, _strip_article(p.object)) for p in targets
        }
        matched = recovered_triples & target_triples

        results.append(HoldoutResult(
            index=index, subject=subject, generated_text=generated_text,
            generation_success=generation_success, no_library_overlap=no_overlap,
            propositions_match=(matched == target_triples),
            propositions_recovered=len(matched),
        ))
    return results


def measure_paraphrase_diversity(
    frame_library, relation: str, subject: str, obj: str, samples: int = 200,
    seed: int = 0,
) -> dict:
    """Sample `select_frame` many times for the same proposition and
    report the output distribution -- confirms weighted-random frame
    selection produces real variety, not just in one-shot manual checks,
    and that the weighting genuinely tracks evidence_count rather than
    always returning the same frame due to a `random` module state bug."""
    rng = random.Random(seed)
    target = Proposition(subject, relation, obj, source_id=-1, source_text="")
    outputs = Counter()
    for _ in range(samples):
        frame = frame_library.select_frame(relation, rng=rng)
        text = PropositionRealiser.realise((target,), frame_library=frame_library) \
            if frame is None else frame.template.replace(
                "[SUBJECT]", subject).replace("[OBJECT]", obj)
        outputs[text] += 1
    return {
        "relation": relation, "samples": samples,
        "distinct_outputs": len(outputs),
        "distribution": dict(outputs.most_common()),
    }


def run_adversarial_pairs(
    contents_a: list[dict], contents_b: list[dict], frame_library,
) -> list[dict]:
    """Two subjects sharing "is"/"in" facts but differing in "has" --
    confirm realising subject A's held-out "has" property never surfaces
    subject B's differing one. Same near-miss shape as
    llm_relational_benchmark.py's collision scenarios, applied to the
    realiser instead of RelationalMemory directly."""
    results = []
    for content_a, content_b in zip(contents_a, contents_b):
        subject_a = _norm(content_a["subject"])
        subject_b = _norm(content_b["subject"])
        if subject_a == subject_b:
            continue
        target_a = Proposition(
            subject_a, "has", _norm(content_a["property"]), source_id=-1, source_text="",
        )
        text_a = PropositionRealiser.realise((target_a,), frame_library=frame_library)
        contaminated = _norm(content_b["property"]) in _norm(text_a)
        results.append({
            "subject_a": subject_a, "subject_b": subject_b,
            "property_a": _norm(content_a["property"]),
            "property_b": _norm(content_b["property"]),
            "generated_text": text_a,
            "cross_contaminated": contaminated,
        })
    return results


def generate_contents(
    model: str, n: int, start_index: int, kind: str | None = None,
) -> list[dict]:
    """`kind=None` alternates roughly 1-in-3 "country" lessons among
    "animal" ones; pass an explicit kind to force every item to it (used
    for the adversarial-pairs content, which relies on the "has"
    relation existing on every item)."""
    contents = []
    for index in range(start_index, start_index + n):
        item_kind = kind if kind is not None else ("country" if index % 3 == 0 else "animal")
        try:
            contents.append(generate_lesson_content(model, index, kind=item_kind))
        except Exception as exc:  # noqa: BLE001 -- generation can fail many ways
            print(f"  [item {index}] generation failed: {exc}")
    return contents


def run(
    model: str, n_lessons: int, n_holdout: int, n_adversarial: int = 0,
    paraphrase_samples: int = 0, spec: dict | None = None,
) -> dict:
    if spec is not None:
        lesson_contents = spec["lessons"]
        holdout_contents = spec["holdout"]
        adversarial_a = spec.get("adversarial_a", [])
        adversarial_b = spec.get("adversarial_b", [])
    else:
        lesson_contents = generate_contents(model, n_lessons, start_index=0)
        holdout_contents = generate_contents(model, n_holdout, start_index=10_000)
        # Forced to "animal": run_adversarial_pairs compares "has"
        # properties, which only "animal"-kind content has.
        adversarial_a = generate_contents(model, n_adversarial, start_index=20_000, kind="animal")
        adversarial_b = generate_contents(model, n_adversarial, start_index=30_000, kind="animal")

    agent = BioAIDialogueAgent(vsa_dim=64)
    start = time.perf_counter()
    acquisition_records = run_acquisition(agent, lesson_contents)
    acquisition_latency = time.perf_counter() - start
    print(f"Acquisition: taught {len(acquisition_records)}/{len(lesson_contents)} lessons "
          f"({3 * len(acquisition_records)} sentences).")

    taught_sentences = [
        sentence for record in acquisition_records for sentence in record.sentences
    ]
    taught_subjects = {record.subject for record in acquisition_records}

    start = time.perf_counter()
    holdout_results = evaluate_holdout(
        holdout_contents, agent.frame_library, taught_sentences, taught_subjects,
    )
    holdout_latency = time.perf_counter() - start
    print(f"Holdout: evaluated {len(holdout_results)}/{len(holdout_contents)} targets.")

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
        for relation in ("is", "in", "has", "capital", "capital_of")
    }

    result = {
        "config": {
            "model": model, "lessons": len(lesson_contents),
            "holdout": len(holdout_contents),
        },
        "lessons_taught": len(acquisition_records),
        "acquisition_latency_seconds": acquisition_latency,
        "holdout_latency_seconds": holdout_latency,
        "frames_learned": len(agent.frame_library.frames),
        "distinct_frames_by_relation": frames_by_relation,
        "holdout_evaluated": len(holdout_results),
        "generation_success_rate": rate("generation_success"),
        "no_library_overlap_rate": rate("no_library_overlap"),
        "propositions_match_rate": rate("propositions_match"),
        "mean_propositions_recovered": (
            sum(r.propositions_recovered for r in holdout_results) / len(holdout_results)
            if holdout_results else 0.0
        ),
        "native_generation_proof_rate": all_three_rate,
        "acquisition_records": [asdict(r) for r in acquisition_records],
        "holdout_records": [asdict(r) for r in holdout_results],
    }

    animal_indices = [i for i, c in enumerate(lesson_contents) if c["kind"] == "animal"]
    if paraphrase_samples and animal_indices:
        richest_relation = max(
            ("is", "in", "has"),
            key=lambda r: len(agent.frame_library.frames_for_relation(r)),
        )
        sample_index = animal_indices[0]
        sample_record = acquisition_records[sample_index]
        sample_content = lesson_contents[sample_index]
        result["paraphrase_diversity"] = measure_paraphrase_diversity(
            agent.frame_library, richest_relation, sample_record.subject,
            _norm(_object_for(richest_relation, sample_content)),
            samples=paraphrase_samples,
        )

    if n_adversarial:
        result["adversarial_results"] = run_adversarial_pairs(
            adversarial_a, adversarial_b, agent.frame_library,
        )
        result["adversarial_contamination_rate"] = (
            sum(r["cross_contaminated"] for r in result["adversarial_results"])
            / len(result["adversarial_results"])
            if result["adversarial_results"] else 0.0
        )

    if spec is None:
        result["spec"] = {
            "lessons": lesson_contents, "holdout": holdout_contents,
            "adversarial_a": adversarial_a, "adversarial_b": adversarial_b,
        }

    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lessons", type=int, default=30)
    parser.add_argument("--holdout", type=int, default=15)
    parser.add_argument("--adversarial-pairs", type=int, default=0)
    parser.add_argument("--paraphrase-diversity-samples", type=int, default=0)
    parser.add_argument("--model", default="qwen2.5-coder:1.5b")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--save-spec", type=Path, default=None)
    parser.add_argument("--load-spec", type=Path, default=None)
    args = parser.parse_args()

    spec = None
    if args.load_spec is not None:
        spec = json.loads(args.load_spec.read_text())
        print(f"Loaded frozen spec from {args.load_spec} "
              f"({len(spec['lessons'])} lessons, {len(spec['holdout'])} holdout).")

    result = run(
        args.model, args.lessons, args.holdout,
        n_adversarial=args.adversarial_pairs,
        paraphrase_samples=args.paraphrase_diversity_samples,
        spec=spec,
    )

    if args.save_spec is not None and "spec" in result:
        args.save_spec.parent.mkdir(parents=True, exist_ok=True)
        args.save_spec.write_text(json.dumps(result.pop("spec"), indent=2) + "\n")
        print(f"Saved frozen spec to {args.save_spec}")
    else:
        result.pop("spec", None)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / (
        f"llm_teacher_curriculum_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    path.write_text(json.dumps(result, indent=2) + "\n")
    summary = {k: v for k, v in result.items() if not k.endswith("_records") and k != "adversarial_results"}
    print(json.dumps(summary, indent=2))
    print(f"Report: {path}")


if __name__ == "__main__":
    main()
