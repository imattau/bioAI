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
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, replace

from src.text.consolidation import ConsolidationMemory

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

    def observe(self, text: str) -> LinguisticFrame | None:
        frame = extract_frame(text)
        if frame is None:
            return None
        existing = self.frames.get(frame.template)
        if existing is not None:
            updated = replace(existing, evidence_count=existing.evidence_count + 1)
            self.frames[frame.template] = updated
            return updated
        self.frames[frame.template] = frame
        return frame

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
