"""Extract linguistic frames -- a relation's literal surface phrasing, with
subject/object replaced by placeholders -- from observed sentences (Phase 4
of the response-ecosystem plan).

`ConsolidationMemory.extract_relations` (`src/text/consolidation.py`)
already regex-matches a sentence into `(subject, relation, object)`, but
then *discards* the matched wording in favor of one of 5 fixed canonical
labels: "lies within" and "sits in" both collapse to `"in"`, "is located
in" and "is in" both collapse to `"in"` too. This module reuses the exact
same pattern list (`ConsolidationMemory._RELATION_PATTERNS` -- not forked,
imported) but keeps the literal matched text around the captured spans as
the frame's template, so "France's capital is Paris" and "The capital of
France is Paris" become two distinct, separately-evidenced frames for the
same `capital_of`/`capital` relations, not one flattened label.

Phase 9 makes frame learning bidirectional. `observe`/`extract_frame`
above only ever recognize the 5 fixed copulas `ConsolidationMemory`
already knows -- they can't learn a template from "Wombats belong to the
marsupial family." (no matching pattern). `observe_labelled` accepts a
sentence *and* its already-known (subject, relation, object) -- supplied
independently, e.g. by an LLM teacher, not re-derived by parsing the
sentence -- and aligns the label's literal surface spans in the sentence
to build the same kind of `LinguisticFrame`, for constructions the fixed
regex could never recognize on its own. Every learned frame, from either
source, becomes usable as a *parser* too via `parse`: compiling the
template back into a regex is the literal inverse of `_templatize`, so a
frame is a bidirectional linguistic rule, not just a realisation
template -- see `src/text/ecology/proposition_extractor.py`'s
`allow_learned_frames` for where `parse` plugs into extraction.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, replace

import lemminflect

from src.text.consolidation import ConsolidationMemory
from .morphology import is_plural_noun

_PLACEHOLDER_SUBJECT = "[SUBJECT]"
_PLACEHOLDER_OBJECT = "[OBJECT]"


@dataclass(frozen=True)
class LinguisticFrame:
    template: str  # e.g. "[SUBJECT] is the capital of [OBJECT]"
    relation: str  # canonical label, same taxonomy as Proposition.relation
    evidence_count: int = 1


def _templatize(sentence: str, match: re.Match) -> str:
    """Replace the matched subject/object spans with placeholders, keeping
    every other character (the actual relation wording) verbatim. Group 1
    (subject) always precedes group 2 (object) in every pattern in
    `ConsolidationMemory._RELATION_PATTERNS`, so the spans never overlap
    and never need reordering."""
    start1, end1 = match.span(1)
    start2, end2 = match.span(2)
    return (
        sentence[:start1] + _PLACEHOLDER_SUBJECT + sentence[end1:start2]
        + _PLACEHOLDER_OBJECT + sentence[end2:]
    )


def _alt_number_form(phrase: str) -> str | None:
    """The opposite-number surface form of *phrase*'s head noun (last
    word) -- "marsupials" -> "marsupial", "mouse" -> "mice" -- via
    lemminflect, reusing `is_plural_noun`'s own lemma lookup rather than
    hand-rolled suffix rules. Used only as a fallback when a teacher
    label's exact surface form isn't found verbatim in its own sentence
    (free-form LLM phrasing may use a different number than the label).
    Returns `None` when lemminflect has no distinct inflection."""
    words = phrase.split()
    if not words:
        return None
    head = words[-1]
    lemma_result = lemminflect.getLemma(head, upos="NOUN")
    lemma = lemma_result[0] if lemma_result else head
    if is_plural_noun(phrase):
        alt = lemma
    else:
        plural_form = lemminflect.getInflection(lemma, tag="NNS")
        alt = plural_form[0] if plural_form else None
    if not alt or alt.lower() == head.lower():
        return None
    return " ".join(words[:-1] + [alt])


def _locate_phrase(sentence: str, phrase: str) -> tuple[int, int] | None:
    """Case-insensitive, word-boundary-aware span of *phrase* (or its
    opposite-number form) in *sentence*. Returns `None` rather than
    guessing when neither form appears -- an unlocatable label is a real
    alignment failure, not something to force-fit."""
    for candidate in (phrase, _alt_number_form(phrase)):
        if not candidate:
            continue
        match = re.search(r"\b" + re.escape(candidate) + r"\b", sentence, re.IGNORECASE)
        if match:
            return match.span()
    return None


def _compile_frame_pattern(template: str) -> re.Pattern:
    """The literal inverse of `_templatize`: turn a frame template back
    into a regex that recovers subject/object from new text. Non-greedy
    captures with literal text anchoring both ends handles placeholders
    followed by more literal text (e.g. "[SUBJECT] belong to the
    [OBJECT] family"), not just placeholders trailing to end of string."""
    parts = re.split(r"(\[SUBJECT\]|\[OBJECT\])", template)
    pattern_parts = []
    for part in parts:
        if part == _PLACEHOLDER_SUBJECT:
            pattern_parts.append(r"(?P<subject>.+?)")
        elif part == _PLACEHOLDER_OBJECT:
            pattern_parts.append(r"(?P<object>.+?)")
        else:
            pattern_parts.append(re.escape(part))
    return re.compile("^" + "".join(pattern_parts) + "$", re.IGNORECASE)


def extract_frame(text: str) -> LinguisticFrame | None:
    """Extract at most one frame per sentence, first-pattern-wins -- same
    limitation as `extract_relations`, since it reuses the identical
    pattern list. A sentence with no matching copula (e.g. "Marsupials
    carry their young in a pouch") yields `None`, exactly as it yields no
    propositions from `extract_propositions`."""
    sentence = text.strip().rstrip(".!?")
    for pattern, relation in ConsolidationMemory._RELATION_PATTERNS:
        match = re.match(pattern, sentence, flags=re.IGNORECASE)
        if match:
            subject = ConsolidationMemory._normalise(match.group(1))
            obj = ConsolidationMemory._normalise(match.group(2))
            if not subject or not obj:
                continue
            template = _templatize(sentence, match)
            return LinguisticFrame(template=template, relation=relation)
    return None


class FrameLibrary:
    """Counter-style evidence accumulation over observed frames, keyed by
    template text -- same learning style as `LearnedChunkComposer.chunk_counts`
    (`chunk_composer.py`), just for relation phrasings instead of response
    chunks.

    Frame *competition* here is frequency-weighted selection among the
    frames observed for a relation, not a VSA pattern-completion query:
    there is no semantic-context signal yet to condition a Hopfield-style
    competition on (that's Phase 4's later discourse-structure work, not
    built). Building that machinery now, with nothing real to query
    against, would repeat the exact mistake the removed NCA implementation
    made -- elaborate mechanism, no real signal wired to it. Per-fact
    phrasing recall (what template a *specific* stored triple was taught
    with) is instead handled by `RelationalEncoder`'s new `frame` role
    (`src/vsa/relational.py`), which does have a real per-triple query
    signal to condition on; see that module's docstring.
    """

    def __init__(self):
        self.frames: dict[str, LinguisticFrame] = {}
        self._compiled: dict[str, re.Pattern] = {}

    def _record_frame(self, relation: str, template: str) -> LinguisticFrame:
        existing = self.frames.get(template)
        if existing is not None:
            updated = replace(existing, evidence_count=existing.evidence_count + 1)
            self.frames[template] = updated
            return updated
        frame = LinguisticFrame(template=template, relation=relation)
        self.frames[template] = frame
        return frame

    def observe(self, text: str) -> LinguisticFrame | None:
        frame = extract_frame(text)
        if frame is None:
            return None
        return self._record_frame(frame.relation, frame.template)

    def observe_labelled(
        self, sentence: str, subject: str, relation: str, object_: str,
    ) -> LinguisticFrame | None:
        """Align a teacher-supplied (subject, relation, object) -- not
        re-derived by parsing *sentence* -- to its literal surface spans
        in *sentence*, producing a `LinguisticFrame` for constructions
        `extract_frame`'s fixed patterns could never recognize on their
        own (e.g. "Wombats belong to the marsupial family."). Returns
        `None`, without recording anything, when either label can't be
        located (`_locate_phrase`) or the object appears before the
        subject -- an honest alignment failure, not something to guess
        at (this codebase's own teacher-generated content is expected to
        mention the subject first; a sentence that doesn't is out of
        scope, not force-aligned)."""
        stripped = sentence.strip().rstrip(".!?")
        subject_span = _locate_phrase(stripped, subject)
        object_span = _locate_phrase(stripped, object_)
        if subject_span is None or object_span is None:
            return None
        if subject_span[0] >= object_span[0]:
            return None
        template = (
            stripped[:subject_span[0]] + _PLACEHOLDER_SUBJECT
            + stripped[subject_span[1]:object_span[0]] + _PLACEHOLDER_OBJECT
            + stripped[object_span[1]:]
        )
        return self._record_frame(relation, template)

    def _compiled_pattern(self, template: str) -> re.Pattern:
        pattern = self._compiled.get(template)
        if pattern is None:
            pattern = _compile_frame_pattern(template)
            self._compiled[template] = pattern
        return pattern

    def parse(self, sentence: str) -> list[tuple[str, str, str]]:
        """Try every learned frame (from either `observe` or
        `observe_labelled` -- one unified `self.frames`) as a parser
        against *sentence*, returning every match as (subject, relation,
        object) -- same list-of-triples shape as
        `ConsolidationMemory.extract_relations`, so it's a drop-in
        alternate/additional source for `extract_propositions`."""
        stripped = sentence.strip().rstrip(".!?")
        matches = []
        for template, frame in self.frames.items():
            match = self._compiled_pattern(template).match(stripped)
            if not match:
                continue
            subject = ConsolidationMemory._normalise(match.group("subject"))
            obj = ConsolidationMemory._normalise(match.group("object"))
            if subject and obj:
                matches.append((subject, frame.relation, obj))
        return matches

    def frames_for_relation(self, relation: str) -> list[LinguisticFrame]:
        return [
            frame for frame in self.frames.values() if frame.relation == relation
        ]

    def select_frame(
        self, relation: str, rng: random.Random | None = None
    ) -> LinguisticFrame | None:
        """Weighted-random pick among observed frames for *relation*,
        weighted by evidence_count -- favors the most common phrasing
        while still allowing less common ones through, rather than always
        rigidly picking the single most-observed template."""
        candidates = self.frames_for_relation(relation)
        if not candidates:
            return None
        rng = rng or random
        weights = [frame.evidence_count for frame in candidates]
        return rng.choices(candidates, weights=weights, k=1)[0]

    def get_state(self) -> dict:
        return {
            "frames": [
                (frame.template, frame.relation, frame.evidence_count)
                for frame in self.frames.values()
            ],
        }

    @classmethod
    def from_state(cls, state: dict) -> "FrameLibrary":
        library = cls()
        for template, relation, evidence_count in state.get("frames", []):
            library.frames[template] = LinguisticFrame(
                template=template, relation=relation,
                evidence_count=evidence_count,
            )
        return library
