"""Calibrated parse-candidate scoring and abstention (Phase 10 of the
response-ecosystem plan).

Phase 9 established that BioAI's structured-evidence *generation* is
robust (100% success/grounded, `ecosystem.py`'s `evidence_propositions` +
`realiser.py`'s `require_frame=True`) while its exposure-acquired
*parsing* is real but weak: `FrameLibrary.parse` recovered the correct
triple for only 30% of unseen holdout sentences, with 0% abstention and
11% precision -- meaning it frequently produces the *correct* candidate
alongside incorrect ones, but has no way to prefer one over another.
Naively preferring `ConsolidationMemory`'s fixed patterns first (Phase
9's "hybrid") made this *worse*: the fixed catch-all pattern
(`^(.+?)\\s+(?:is|are|was|were)\\s+(.+)$`, the last of
`ConsolidationMemory._RELATION_PATTERNS`) confidently returns a *wrong*
triple instead of abstaining, and since hybrid only fell back to learned
frames when the fixed pass found nothing at all, that wrong-but-non-
abstaining match blocked the correct learned-frame fallback from ever
being tried.

This module treats parsing as a selection problem, not a coverage
problem: gather every candidate reading of a sentence (from learned
frames via `FrameLibrary.parse_candidates`, and from *every* fixed
pattern tried independently, not just the first-match-wins the
production `ConsolidationMemory.extract_relations` uses), score them
uniformly by structural features (how much of the sentence a candidate's
literal text explains, how specific/constrained its construction is, how
much evidence backs it), merge candidates that agree on the same triple,
and only accept the winner when it's confident and unambiguous --
abstaining or reporting ambiguity otherwise, rather than always guessing.

This is the smallest scored-selection slice, by design: structural
features only (anchor coverage, specificity, evidence count). Round-trip
reconstruction scoring, VSA semantic compatibility, relational-memory
agreement, discourse-context agreement, and a learned Go/NoGo selector
are named, deferred Tier 2B work -- see the response-ecosystem plan --
added one at a time against this module's structural baseline, not
attempted here. `ResponseEcosystem` is untouched: it already accepts
structured propositions directly via `evidence_propositions` (Phase 9),
and this module exists to produce better propositions to hand it, not to
change what it does with them.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace

from src.text.consolidation import ConsolidationMemory

from .frame_extractor import FrameLibrary, ParseCandidate, _anchor_coverage
from .proposition_extractor import Proposition

# `ConsolidationMemory._RELATION_PATTERNS`'s last entry is the generic
# catch-all ("[SUBJECT] is/are/was/were [OBJECT]") -- identified by
# position, not content, since it's the one pattern with (deliberately)
# almost no literal constraint. Everything before it is a "specific"
# fixed pattern (capital/capital_of/in/lies-within/has).
_CATCHALL_PATTERN_INDEX = len(ConsolidationMemory._RELATION_PATTERNS) - 1

# Hand-assigned, frozen per pattern index -- matches the qualitative
# ordering "capital city of [SUBJECT] is [OBJECT]" (very specific) vs.
# "[SUBJECT] is [OBJECT]" (bare) the same way `frame_extractor.py`'s
# `_template_specificity` scores learned frames, but computed by table
# lookup rather than an automatic regex-literal-word counter -- there
# are only 6 known fixed patterns, and a hand-assigned table is more
# robust than parsing regex syntax back into word counts.
_FIXED_PATTERN_SPECIFICITY = {
    0: 1.0,   # "(the) capital of [SUBJECT] is [OBJECT]"
    1: 1.0,   # "[SUBJECT] is/are the capital of [OBJECT]"
    2: 0.5,   # "[SUBJECT] is/are (located) in [OBJECT]"
    3: 0.6,   # "[SUBJECT] lies/sits/lie/sit within/in [OBJECT]"
    4: 0.5,   # "[SUBJECT] has/have [OBJECT]"
    5: 0.15,  # generic catch-all "[SUBJECT] is/are/was/were [OBJECT]"
}

# Fixed patterns don't accumulate evidence the way learned frames do --
# a small constant baseline, not zero (a zero-evidence learned frame is
# meaningfully different: it was never actually observed).
_FIXED_PATTERN_EVIDENCE = 1

# Frozen scoring weights (first cut) -- the exact values matter less
# than freezing them before touching the held-out evaluation set, per
# the plan's own instruction. `10` is the evidence count at which
# evidence_strength is considered saturated.
_ANCHOR_WEIGHT = 0.45
_SPECIFICITY_WEIGHT = 0.40
_EVIDENCE_WEIGHT = 0.15
_EVIDENCE_REFERENCE = 10


def _evidence_strength(evidence_count: int) -> float:
    return min(1.0, math.log1p(max(0, evidence_count)) / math.log1p(_EVIDENCE_REFERENCE))


def _score(candidate: ParseCandidate) -> float:
    return (
        _ANCHOR_WEIGHT * candidate.anchor_coverage
        + _SPECIFICITY_WEIGHT * candidate.specificity
        + _EVIDENCE_WEIGHT * _evidence_strength(candidate.frame_evidence)
    )


def _fixed_pattern_candidates(sentence: str) -> list[ParseCandidate]:
    """Tries *every* fixed pattern independently (unlike
    `ConsolidationMemory.extract_relations`, which stops at the first
    match) so each one becomes its own scoreable candidate rather than
    silently winning by pattern order."""
    stripped = sentence.strip().rstrip(".!?")
    candidates = []
    for index, (pattern, relation) in enumerate(ConsolidationMemory._RELATION_PATTERNS):
        match = re.match(pattern, stripped, flags=re.IGNORECASE)
        if not match:
            continue
        subject_text, object_text = match.group(1), match.group(2)
        subject = ConsolidationMemory._normalise(subject_text)
        obj = ConsolidationMemory._normalise(object_text)
        if not subject or not obj:
            continue
        is_catchall = index == _CATCHALL_PATTERN_INDEX
        candidates.append(ParseCandidate(
            proposition=Proposition(
                subject=subject, relation=relation, object=obj,
                source_id=-1, source_text=sentence,
            ),
            source="fixed_catchall" if is_catchall else "fixed_specific",
            frame_template=None,
            frame_evidence=_FIXED_PATTERN_EVIDENCE,
            anchor_coverage=_anchor_coverage(len(stripped), subject_text, object_text),
            specificity=_FIXED_PATTERN_SPECIFICITY[index],
        ))
    return candidates


def _triple(candidate: ParseCandidate) -> tuple[str, str, str]:
    prop = candidate.proposition
    return (prop.subject, prop.relation, prop.object)


def _merge_candidates(candidates: list[ParseCandidate]) -> list[ParseCandidate]:
    """Groups candidates by exact (subject, relation, object) -- string
    equality only, no lemmatization/canonicalization at this layer (that
    stays a benchmark-only comparison tool, not something production
    merging should do silently: two candidates producing "marsupial" vs.
    "marsupials" might genuinely disagree in a real ambiguous case).
    Combines each group's scores via noisy-OR (`1 - prod(1 - score)`),
    so two independent phrasings supporting the same reading reinforce
    each other rather than splitting the vote and looking like competing
    interpretations."""
    by_triple: dict[tuple[str, str, str], list[ParseCandidate]] = {}
    order: list[tuple[str, str, str]] = []
    for candidate in candidates:
        key = _triple(candidate)
        if key not in by_triple:
            by_triple[key] = []
            order.append(key)
        by_triple[key].append(candidate)

    merged = []
    for key in order:
        group = by_triple[key]
        combined = 1.0
        for candidate in group:
            combined *= (1.0 - candidate.score)
        combined_score = 1.0 - combined
        # Report the most-evidenced/most-specific member for provenance.
        representative = max(group, key=lambda c: (c.frame_evidence, c.specificity))
        merged.append(replace(representative, score=combined_score))
    return merged


@dataclass(frozen=True)
class ParseDecision:
    proposition: Proposition | None
    accepted: bool
    ambiguous: bool
    score: float
    margin: float
    candidates: tuple[ParseCandidate, ...]  # every candidate, post-merge
    reason: str


def _abstain(candidates: tuple[ParseCandidate, ...], reason: str) -> ParseDecision:
    return ParseDecision(
        proposition=None, accepted=False, ambiguous=False,
        score=0.0, margin=0.0, candidates=candidates, reason=reason,
    )


class PropositionParser:
    @staticmethod
    def parse_best(
        sentence: str,
        frame_library: FrameLibrary | None = None,
        include_fixed: bool = True,
        include_learned: bool = True,
        acceptance_threshold: float = 0.5,
        margin_threshold: float = 0.1,
    ) -> ParseDecision:
        raw_candidates: list[ParseCandidate] = []
        if include_fixed:
            raw_candidates.extend(_fixed_pattern_candidates(sentence))
        if include_learned and frame_library is not None:
            raw_candidates.extend(frame_library.parse_candidates(sentence))

        scored = [replace(c, score=_score(c)) for c in raw_candidates]
        candidates = tuple(
            sorted(_merge_candidates(scored), key=lambda c: c.score, reverse=True)
        )
        if not candidates:
            return _abstain(candidates, "no_match")

        best = candidates[0]
        if best.score < acceptance_threshold:
            return _abstain(candidates, "low_score")

        second_score = candidates[1].score if len(candidates) > 1 else -1.0
        margin = best.score - second_score
        if margin < margin_threshold:
            return ParseDecision(
                proposition=best.proposition, accepted=False, ambiguous=True,
                score=best.score, margin=margin, candidates=candidates,
                reason="low_margin",
            )

        return ParseDecision(
            proposition=best.proposition, accepted=True, ambiguous=False,
            score=best.score, margin=margin, candidates=candidates,
            reason="accepted",
        )
