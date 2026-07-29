"""Phase 9 of the response-ecosystem plan: does exposure-acquired,
bidirectional frame learning generalize as a *parser*, not just a
generation template -- and can BioAI generate fluent text from known
facts independent of whether it can also parse them out of free text?

Every earlier phase's "native generation" proof conflated two questions:
propositions were always re-derived from evidence text via
`ConsolidationMemory._RELATION_PATTERNS` -- the same fixed regex family
used everywhere else in this plan -- so if a teacher's sentence used a
construction the regex couldn't recognize, the generator never got a
chance to prove anything, regardless of how good it was. This script
separates the two questions:

  1. Parsing: given a sentence using an unrestricted natural construction
     (not one of the 5 fixed copulas), can a *learned* frame -- taught
     bidirectionally via `FrameLibrary.observe_labelled` from a teacher's
     own structured label, never re-derived by parsing -- recover the
     correct (subject, relation, object) from an unseen sentence using
     the same construction with a novel filler? Compared against the
     existing fixed-regex parser and a hybrid of both
     (`extract_propositions`'s new `allow_fixed_patterns`/
     `allow_learned_frames` flags).
  2. Generation: given a structured fact directly (bypassing parsing
     entirely via `ResponseEcosystem`'s new `evidence_propositions`),
     can `PropositionRealiser` -- with `require_frame=True`, so no
     fixed-template fallback can inflate the result -- compose fluent
     text using only frames learned from *other* facts during
     acquisition?

Ground truth for both is the teacher's own frozen structured label,
never re-derived from any parser under test (the "independent target
triples" control) -- each example's `sentence`/`subject`/`relation`/
`object` come from one LLM call, validated by actually running
`FrameLibrary.observe_labelled` on them before acceptance (an
unlocatable label is a real alignment failure, retried then skipped, not
forced). Holdout is disjoint from acquisition by subject, guaranteed by
generate-and-retry rather than post-hoc filtering.

`capital_of` stays out of scope, consistent with every prior phase's
finding that it's untested-in-practice in this curriculum.

Phase 10 diagnosed *why* parsing was weak (`FrameLibrary.parse` had no
way to prefer a correct candidate over an incorrect one, and Phase 9's
naive fixed-first hybrid was actively worse, since the fixed catch-all
pattern confidently returns wrong triples instead of abstaining) and
built `PropositionParser.parse_best` to fix it via structural candidate
scoring and calibrated abstention. Every run now also evaluates two
more parsing conditions alongside Phase 9's original three -- passing
`--load-spec` on an already-frozen Phase 9 spec re-evaluates all five
against the same content with zero new LLM calls (the same content, a
different, better parser), which is how this phase's own results were
produced: `learned_scored`
(`PropositionParser.parse_best(include_fixed=False)`), and
`candidate_union` (`include_fixed=True, include_learned=True` --
replacing Phase 9's broken "hybrid"). Acceptance/margin thresholds are
`PropositionParser.parse_best`'s own frozen defaults (`0.5`/`0.1`,
exercised against synthetic examples in
`tests/test_text/test_proposition_parser.py`, not fit to this script's
own 40-item holdout) -- the risk-coverage sweep below reports how other
threshold choices would have performed, for diagnostic purposes, without
changing which single point is reported as the headline result.

Usage:
    python experiments/exposure_acquisition_benchmark.py --acquisition 60 --holdout 40 --model qwen2.5-coder:1.5b
    python experiments/exposure_acquisition_benchmark.py --acquisition 60 --holdout 40 --save-spec checkpoints/spec.json
    python experiments/exposure_acquisition_benchmark.py --load-spec checkpoints/spec.json
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import time
from datetime import datetime
from pathlib import Path

import lemminflect
import ollama

from src.text.consolidation import ConsolidationMemory
from src.text.ecology import Proposition, PropositionRealiser, extract_propositions
from src.text.ecology.frame_extractor import FrameLibrary
from src.text.ecology.proposition_parser import PropositionParser

from ecology_benchmark import _as_plain_phrase, _extract_json

_norm = ConsolidationMemory.normalise

# Local, benchmark-only comparison helper -- NOT a change to any
# production extractor. Same class of issue Phase 7's `_strip_article`
# (llm_teacher_curriculum.py) fixed locally: a parser recovers whatever
# is LITERALLY in the sentence text, which can differ cosmetically from
# the teacher's independently-supplied label (e.g. a sentence says "the
# marsupial family" -- singular -- while the label field is "marsupials"
# -- plural). Canonicalizing both sides (strip a leading article, reduce
# the head noun to its lemma) before comparing avoids penalizing correct
# parses for a surface difference that was never the thing being tested.
_LEADING_ARTICLE_RE = re.compile(r"^(?:a|an|the)\s+")


def _canonical_object(text: str) -> str:
    stripped = _LEADING_ARTICLE_RE.sub("", text)
    words = stripped.split()
    if not words:
        return stripped
    head = words[-1]
    lemma_result = lemminflect.getLemma(head, upos="NOUN")
    lemma = lemma_result[0] if lemma_result else head
    return " ".join(words[:-1] + [lemma])

_RELATIONS = ("is", "in", "has", "capital")

_RELATION_PROMPTS = {
    "is": (
        "Write one natural English sentence stating what broader category "
        "or class a subject belongs to. Use any natural phrasing for this "
        "-- 'belong to', 'are classified as', 'are a type of', 'are', or "
        "similar -- not always the same wording. Mention the subject "
        "before the category. Output exactly one JSON object with keys "
        "\"subject\", \"object\", \"sentence\". \"subject\" is a plain "
        "plural noun (1-2 words) naming a real or invented kind of animal "
        "or object. \"object\" is a plain plural noun phrase (1-2 words) "
        "naming the broader category. \"sentence\" is the natural "
        "sentence itself, ending in a period. Output only the JSON "
        "object, nothing else."
    ),
    "in": (
        "Write one natural English sentence stating where a subject is "
        "found or located. Use any natural phrasing -- 'is found in', "
        "'lives in', 'is native to', 'roams across', 'is in', or similar "
        "-- not always the same wording. Mention the subject before the "
        "place. Output exactly one JSON object with keys \"subject\", "
        "\"object\", \"sentence\". \"subject\" is a plain plural noun "
        "(1-2 words) naming a real or invented kind of animal or object. "
        "\"object\" is a plain noun phrase (1-3 words) naming a place. "
        "\"sentence\" is the natural sentence itself, ending in a period. "
        "Output only the JSON object, nothing else."
    ),
    "has": (
        "Write one natural English sentence stating a physical feature or "
        "trait a subject has. Use any natural phrasing -- 'has', 'is "
        "equipped with', 'possesses', 'is known for', or similar -- not "
        "always the same wording. Mention the subject before the "
        "feature. Output exactly one JSON object with keys \"subject\", "
        "\"object\", \"sentence\". \"subject\" is a plain plural noun "
        "(1-2 words) naming a real or invented kind of animal or object. "
        "\"object\" is a plain noun phrase (1-3 words) naming a physical "
        "feature or trait. \"sentence\" is the natural sentence itself, "
        "ending in a period. Output only the JSON object, nothing else."
    ),
    "capital": (
        "Write one natural English sentence stating a country's capital "
        "city. Use any natural phrasing -- \"'s capital is\", 'has its "
        "capital in', 'is home to the capital city of', or similar -- not "
        "always the same wording. Mention the country before the capital "
        "city. Output exactly one JSON object with keys \"subject\", "
        "\"object\", \"sentence\". \"subject\" is the plain name of a "
        "real country (1-2 words). \"object\" is that country's real "
        "capital city, a plain 1-2 word name. \"sentence\" is the "
        "natural sentence itself, ending in a period. Output only the "
        "JSON object, nothing else."
    ),
}


def _as_sentence(value) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"expected a non-empty sentence, got {value!r}")
    sentence = " ".join(value.split())
    if len(sentence) > 200:
        raise ValueError(f"sentence too long: {sentence!r}")
    return sentence


def generate_labelled_example(
    model: str, index: int, relation: str, max_attempts: int = 4,
) -> dict | None:
    """One LLM call per attempt: a natural sentence plus its own
    subject/object fields. Validated by actually running
    `FrameLibrary.observe_labelled` (a throwaway instance -- this is
    alignment validation, not accumulation) before acceptance; an
    unlocatable label or a malformed field retries with a fresh LLM call,
    then gives up and returns `None` after `max_attempts`."""
    for attempt in range(max_attempts):
        try:
            response = ollama.chat(model, messages=[
                {"role": "system", "content": _RELATION_PROMPTS[relation]},
                {"role": "user", "content": f"Generate example #{index}-{attempt}."},
            ], options={"temperature": 0.9, "num_predict": 120})
            data = _extract_json(response["message"]["content"])
            subject = _as_plain_phrase(data["subject"], "subject")
            obj = _as_plain_phrase(data["object"], "object")
            sentence = _as_sentence(data["sentence"])
        except (ValueError, KeyError) as exc:
            print(f"  [{relation} item {index} attempt {attempt}] invalid content: {exc}")
            continue
        frame = FrameLibrary().observe_labelled(sentence, subject, relation, obj)
        if frame is None:
            print(f"  [{relation} item {index} attempt {attempt}] alignment failed: {sentence!r}")
            continue
        return {
            "relation": relation, "subject": _norm(subject),
            "object": _norm(obj), "sentence": sentence,
        }
    return None


def generate_disjoint_examples(
    model: str, n: int, used_subjects: set[str], start_index: int = 0,
    max_attempts_per_item: int = 5,
) -> list[dict]:
    """Guarantees disjointness from `used_subjects` (mutated in place as
    examples are accepted) by generate-and-retry, not post-hoc filtering
    -- unlike the pre-Phase-9 curriculum scripts, which discarded roughly
    half of a naively-generated holdout set on subject collision."""
    examples = []
    index = start_index
    while len(examples) < n:
        relation = _RELATIONS[len(examples) % len(_RELATIONS)]
        accepted = None
        for _ in range(max_attempts_per_item):
            candidate = generate_labelled_example(model, index, relation)
            index += 1
            if candidate is None:
                continue
            if candidate["subject"] in used_subjects:
                print(f"  [{relation}] collision on subject {candidate['subject']!r}, retrying")
                continue
            accepted = candidate
            break
        if accepted is None:
            print(f"  [{relation}] giving up after {max_attempts_per_item} attempts")
            continue
        used_subjects.add(accepted["subject"])
        examples.append(accepted)
    return examples


def run_acquisition(frame_library: FrameLibrary, examples: list[dict]) -> None:
    for example in examples:
        frame_library.observe_labelled(
            example["sentence"], example["subject"], example["relation"], example["object"],
        )


_PARSE_CONDITIONS = {
    "fixed": dict(allow_fixed_patterns=True, allow_learned_frames=False),
    "learned": dict(allow_fixed_patterns=False, allow_learned_frames=True),
    "hybrid": dict(allow_fixed_patterns=True, allow_learned_frames=True),
}


def _acquisition_evidence_count(frame_library: FrameLibrary, example: dict) -> int | None:
    """How many times the construction *this specific holdout sentence*
    uses was actually observed during acquisition -- `None` if this exact
    template was never seen at all (the sentence still parsed via a
    different, compatible template). Cheap diagnostic for "how much
    evidence did this generalization need," without a second acquisition
    pass broken out by construction."""
    frame = FrameLibrary().observe_labelled(
        example["sentence"], example["subject"], example["relation"], example["object"],
    )
    if frame is None:
        return None
    existing = frame_library.frames.get(frame.template)
    return existing.evidence_count if existing else 0


def evaluate_parsing(holdout_examples: list[dict], frame_library: FrameLibrary) -> list[dict]:
    results = []
    for example in holdout_examples:
        target = (
            example["subject"], example["relation"],
            _canonical_object(example["object"]),
        )
        evidence_count = _acquisition_evidence_count(frame_library, example)
        for condition, kwargs in _PARSE_CONDITIONS.items():
            recovered = extract_propositions(
                [example["sentence"]], frame_library=frame_library, **kwargs
            )
            recovered_triples = [
                (p.subject, p.relation, _canonical_object(p.object)) for p in recovered
            ]
            results.append({
                "condition": condition, "relation": example["relation"],
                "subject": example["subject"], "target": target,
                "recovered": recovered_triples,
                "exact_match": target in recovered_triples,
                "abstained": len(recovered_triples) == 0,
                "acquisition_evidence_count": evidence_count,
            })
    return results


def summarize_parsing(parsing_results: list[dict]) -> dict:
    summaries = {}
    for condition in _PARSE_CONDITIONS:
        items = [r for r in parsing_results if r["condition"] == condition]
        total = len(items)
        asserted = sum(len(r["recovered"]) for r in items)
        supported = sum(1 for r in items if r["exact_match"])
        abstained = sum(1 for r in items if r["abstained"])
        summaries[condition] = {
            "exact_match_rate": supported / total if total else 0.0,
            "recall": supported / total if total else 0.0,
            "precision": supported / asserted if asserted else 0.0,
            "abstention_rate": abstained / total if total else 0.0,
        }
    return summaries


# ── Phase 10: scored parsing (PropositionParser) ────────────────────────

_SCORED_MODES = {
    "learned_scored": dict(include_fixed=False, include_learned=True),
    "candidate_union": dict(include_fixed=True, include_learned=True),
}
_DEFAULT_ACCEPTANCE_THRESHOLD = 0.5
_DEFAULT_MARGIN_THRESHOLD = 0.1
_ABSTAIN_REASONS = frozenset({"no_match", "low_score"})


def evaluate_scored_parsing(
    holdout_examples: list[dict], frame_library: FrameLibrary,
    acceptance_threshold: float = _DEFAULT_ACCEPTANCE_THRESHOLD,
    margin_threshold: float = _DEFAULT_MARGIN_THRESHOLD,
) -> list[dict]:
    results = []
    for example in holdout_examples:
        target = (
            example["subject"], example["relation"],
            _canonical_object(example["object"]),
        )
        for mode, kwargs in _SCORED_MODES.items():
            decision = PropositionParser.parse_best(
                example["sentence"], frame_library=frame_library,
                acceptance_threshold=acceptance_threshold,
                margin_threshold=margin_threshold, **kwargs,
            )
            candidate_triples = [
                (c.proposition.subject, c.proposition.relation,
                 _canonical_object(c.proposition.object))
                for c in decision.candidates
            ]
            selected_triple = (
                (decision.proposition.subject, decision.proposition.relation,
                 _canonical_object(decision.proposition.object))
                if decision.proposition is not None else None
            )
            exact_match = decision.accepted and selected_triple == target
            results.append({
                "mode": mode, "relation": example["relation"], "subject": example["subject"],
                "target": target, "accepted": decision.accepted, "ambiguous": decision.ambiguous,
                "reason": decision.reason, "selected_triple": selected_triple,
                "exact_match": exact_match,
                "confidently_wrong": decision.accepted and not exact_match,
                "candidate_count": len(decision.candidates),
                "correct_candidate_present": target in candidate_triples,
            })
    return results


def summarize_scored_parsing(scored_results: list[dict]) -> dict:
    summaries = {}
    for mode in _SCORED_MODES:
        items = [r for r in scored_results if r["mode"] == mode]
        total = len(items)
        accepted = [r for r in items if r["accepted"]]
        exact = sum(1 for r in items if r["exact_match"])
        summaries[mode] = {
            "exact_match_rate": exact / total if total else 0.0,
            "recall": exact / total if total else 0.0,
            "precision": exact / len(accepted) if accepted else 0.0,
            "conditional_accuracy_when_accepted": exact / len(accepted) if accepted else 0.0,
            "abstention_rate": (
                sum(1 for r in items if r["reason"] in _ABSTAIN_REASONS) / total
                if total else 0.0
            ),
            "ambiguity_rate": sum(1 for r in items if r["ambiguous"]) / total if total else 0.0,
            "confidently_wrong_rate": (
                sum(1 for r in items if r["confidently_wrong"]) / total if total else 0.0
            ),
            "correct_candidate_present_rate": (
                sum(1 for r in items if r["correct_candidate_present"]) / total if total else 0.0
            ),
            "correct_candidate_selected_rate": exact / total if total else 0.0,
            "mean_candidate_count": (
                sum(r["candidate_count"] for r in items) / total if total else 0.0
            ),
        }
    return summaries


def risk_coverage_sweep(
    holdout_examples: list[dict], frame_library: FrameLibrary,
    acceptance_thresholds: tuple[float, ...] = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7),
    margin_thresholds: tuple[float, ...] = (0.0, 0.05, 0.1, 0.2),
) -> list[dict]:
    """Reports precision/coverage at other threshold choices for
    diagnostic purposes -- the headline `candidate_union` number above
    always uses the frozen defaults, never one picked from this sweep."""
    points = []
    for acceptance_threshold in acceptance_thresholds:
        for margin_threshold in margin_thresholds:
            results = evaluate_scored_parsing(
                holdout_examples, frame_library,
                acceptance_threshold=acceptance_threshold,
                margin_threshold=margin_threshold,
            )
            union_items = [r for r in results if r["mode"] == "candidate_union"]
            accepted = [r for r in union_items if r["accepted"]]
            exact = sum(1 for r in union_items if r["exact_match"])
            total = len(union_items)
            points.append({
                "acceptance_threshold": acceptance_threshold,
                "margin_threshold": margin_threshold,
                "coverage": len(accepted) / total if total else 0.0,
                "precision": exact / len(accepted) if accepted else 0.0,
            })
    return points


def _mask_entity_tokens(text: str, subject: str, obj: str) -> list[str]:
    entity_tokens = set(subject.lower().split()) | set(obj.lower().split())
    tokens = text.lower().replace(".", "").split()
    return ["<ENT>" if token in entity_tokens else token for token in tokens]


def _longest_common_run(a: list[str], b: list[str]) -> int:
    matcher = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    return matcher.find_longest_match(0, len(a), 0, len(b)).size


def overlap_metrics(generated_text: str, taught_sentences: list[str], subject: str, obj: str) -> dict:
    """Longest common contiguous word run against every taught sentence,
    both raw and with the target's own subject/object tokens masked out
    -- raw overlap is expected (the fact's own words necessarily appear
    in both) and not itself concerning; a high *masked* overlap against
    some specific taught sentence is the real signal of copied
    connective phrasing, not just shared entity names."""
    gen_tokens = generated_text.lower().replace(".", "").split()
    gen_masked = _mask_entity_tokens(generated_text, subject, obj)
    raw_max = 0
    masked_max = 0
    for taught in taught_sentences:
        taught_tokens = taught.lower().replace(".", "").split()
        taught_masked = _mask_entity_tokens(taught, subject, obj)
        raw_max = max(raw_max, _longest_common_run(gen_tokens, taught_tokens))
        masked_max = max(masked_max, _longest_common_run(gen_masked, taught_masked))
    return {"raw_max_shared_ngram": raw_max, "masked_max_shared_ngram": masked_max}


def evaluate_generation(
    holdout_examples: list[dict], frame_library: FrameLibrary, taught_sentences: list[str],
) -> list[dict]:
    """Structured-evidence generation: bypasses parsing entirely by
    building the Proposition directly from the teacher's frozen label,
    then requires a genuinely acquired frame (`require_frame=True`) --
    isolating generation quality from the parsing question this script's
    other half measures. Success is checked without re-parsing the
    output with any of the three parsing conditions above -- that would
    just re-measure the parser's own limits, not generation quality."""
    results = []
    for example in holdout_examples:
        target = Proposition(
            example["subject"], example["relation"], example["object"], -1, "",
        )
        text = PropositionRealiser.realise(
            (target,), frame_library=frame_library, require_frame=True,
        )
        generation_success = bool(text.strip())
        overlap = (
            overlap_metrics(text, taught_sentences, example["subject"], example["object"])
            if generation_success else {"raw_max_shared_ngram": 0, "masked_max_shared_ngram": 0}
        )
        normalised = _norm(text)
        results.append({
            "subject": example["subject"], "relation": example["relation"],
            "target_object": example["object"], "generated_text": text,
            "generation_success": generation_success,
            "contains_subject": example["subject"] in normalised,
            "contains_object": example["object"] in normalised,
            **overlap,
        })
    return results


def run(
    model: str, n_acquisition: int, n_holdout: int, spec: dict | None = None,
) -> dict:
    if spec is not None:
        acquisition_examples = spec["acquisition"]
        holdout_examples = spec["holdout"]
    else:
        used_subjects: set[str] = set()
        acquisition_examples = generate_disjoint_examples(
            model, n_acquisition, used_subjects, start_index=0,
        )
        holdout_examples = generate_disjoint_examples(
            model, n_holdout, used_subjects, start_index=10_000,
        )

    frame_library = FrameLibrary()
    start = time.perf_counter()
    run_acquisition(frame_library, acquisition_examples)
    acquisition_latency = time.perf_counter() - start
    print(f"Acquisition: taught {len(acquisition_examples)} labelled examples "
          f"({len(frame_library.frames)} distinct frames).")

    start = time.perf_counter()
    parsing_results = evaluate_parsing(holdout_examples, frame_library)
    parsing_latency = time.perf_counter() - start

    start = time.perf_counter()
    scored_parsing_results = evaluate_scored_parsing(holdout_examples, frame_library)
    scored_parsing_latency = time.perf_counter() - start
    risk_coverage_points = risk_coverage_sweep(holdout_examples, frame_library)

    taught_sentences = [example["sentence"] for example in acquisition_examples]
    start = time.perf_counter()
    generation_results = evaluate_generation(holdout_examples, frame_library, taught_sentences)
    generation_latency = time.perf_counter() - start

    generation_successes = [r for r in generation_results if r["generation_success"]]
    grounded_rate = (
        sum(r["contains_subject"] and r["contains_object"] for r in generation_successes)
        / len(generation_successes) if generation_successes else 0.0
    )

    result = {
        "config": {
            "model": model, "acquisition": len(acquisition_examples),
            "holdout": len(holdout_examples),
        },
        "acquisition_latency_seconds": acquisition_latency,
        "parsing_latency_seconds": parsing_latency,
        "generation_latency_seconds": generation_latency,
        "frames_learned": len(frame_library.frames),
        "distinct_frames_by_relation": {
            r: len(frame_library.frames_for_relation(r)) for r in _RELATIONS
        },
        "scored_parsing_latency_seconds": scored_parsing_latency,
        "parsing_conditions": summarize_parsing(parsing_results),
        "scored_parsing_conditions": summarize_scored_parsing(scored_parsing_results),
        "risk_coverage_curve": risk_coverage_points,
        "generation_success_rate": (
            len(generation_successes) / len(generation_results) if generation_results else 0.0
        ),
        "generation_grounded_rate": grounded_rate,
        "generation_max_masked_overlap": max(
            (r["masked_max_shared_ngram"] for r in generation_results), default=0,
        ),
        "parsing_records": parsing_results,
        "scored_parsing_records": scored_parsing_results,
        "generation_records": generation_results,
    }
    if spec is None:
        result["spec"] = {"acquisition": acquisition_examples, "holdout": holdout_examples}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquisition", type=int, default=60)
    parser.add_argument("--holdout", type=int, default=40)
    parser.add_argument("--model", default="qwen2.5-coder:1.5b")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--save-spec", type=Path, default=None)
    parser.add_argument("--load-spec", type=Path, default=None)
    args = parser.parse_args()

    spec = None
    if args.load_spec is not None:
        spec = json.loads(args.load_spec.read_text())
        print(f"Loaded frozen spec from {args.load_spec} "
              f"({len(spec['acquisition'])} acquisition, {len(spec['holdout'])} holdout).")

    result = run(args.model, args.acquisition, args.holdout, spec=spec)

    if args.save_spec is not None and "spec" in result:
        args.save_spec.parent.mkdir(parents=True, exist_ok=True)
        args.save_spec.write_text(json.dumps(result.pop("spec"), indent=2) + "\n")
        print(f"Saved frozen spec to {args.save_spec}")
    else:
        result.pop("spec", None)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / (
        f"exposure_acquisition_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    path.write_text(json.dumps(result, indent=2) + "\n")
    summary = {k: v for k, v in result.items() if not k.endswith("_records")}
    print(json.dumps(summary, indent=2))
    print(f"Report: {path}")


if __name__ == "__main__":
    main()
