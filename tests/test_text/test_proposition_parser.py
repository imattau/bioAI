"""Tests for Phase 10 of the response-ecosystem plan: calibrated parse
candidate scoring and abstention. See
src/text/ecology/proposition_parser.py's module docstring for why this
exists -- Phase 9 found FrameLibrary.parse frequently produces the
correct candidate alongside incorrect ones with no way to prefer one,
and that naively preferring ConsolidationMemory's fixed patterns first
makes this worse, since its catch-all pattern confidently returns a
wrong triple instead of abstaining.
"""

from dataclasses import replace

from src.text.ecology.frame_extractor import FrameLibrary, ParseCandidate
from src.text.ecology.proposition_extractor import Proposition
from src.text.ecology.proposition_parser import (
    PropositionParser,
    _evidence_strength,
    _fixed_pattern_candidates,
    _merge_candidates,
    _score,
)


def _candidate(source, subject, relation, obj, anchor_coverage, specificity, frame_evidence=1):
    return ParseCandidate(
        proposition=Proposition(subject, relation, obj, -1, ""),
        source=source, frame_template=None, frame_evidence=frame_evidence,
        anchor_coverage=anchor_coverage, specificity=specificity,
    )


def test_score_weights_anchor_coverage_and_specificity_and_evidence():
    low = _candidate("learned_frame", "x", "is", "y", anchor_coverage=0.1, specificity=0.1, frame_evidence=1)
    high = _candidate("learned_frame", "x", "is", "y", anchor_coverage=0.9, specificity=0.9, frame_evidence=10)
    assert _score(high) > _score(low)


def test_evidence_strength_saturates_and_is_monotonic():
    assert _evidence_strength(0) == 0.0
    assert 0 < _evidence_strength(1) < _evidence_strength(5) < _evidence_strength(10)
    assert _evidence_strength(10) == 1.0
    assert _evidence_strength(1000) == 1.0  # clamped, not unbounded


def test_fixed_pattern_candidates_tries_every_pattern_independently():
    """Unlike ConsolidationMemory.extract_relations (first-match-wins),
    every matching fixed pattern becomes its own candidate."""
    candidates = _fixed_pattern_candidates("Wombats are marsupials.")
    sources = {c.source for c in candidates}
    assert "fixed_catchall" in sources
    assert any(c.proposition.relation == "is" for c in candidates)


def test_fixed_catchall_has_low_specificity_relative_to_specific_patterns():
    capital = _fixed_pattern_candidates("The capital of France is Paris.")
    catchall_only = _fixed_pattern_candidates("Wombats are marsupials.")
    capital_candidate = next(c for c in capital if c.source == "fixed_specific")
    catchall_candidate = next(c for c in catchall_only if c.source == "fixed_catchall")
    assert capital_candidate.specificity > catchall_candidate.specificity


def test_merge_candidates_combines_agreeing_candidates_via_noisy_or():
    a = _candidate("learned_frame", "cats", "is", "mammals", 0.6, 0.6)
    b = _candidate("learned_frame", "cats", "is", "mammals", 0.6, 0.6)
    scored = [replace(a, score=_score(a)), replace(b, score=_score(b))]
    merged = _merge_candidates(scored)
    assert len(merged) == 1
    # Noisy-OR of two equal, non-trivial scores must exceed either alone.
    assert merged[0].score > scored[0].score


def test_merge_candidates_keeps_distinct_triples_separate():
    a = _candidate("learned_frame", "cats", "is", "mammals", 0.6, 0.6)
    b = _candidate("fixed_catchall", "cats", "is", "reptiles", 0.2, 0.15)
    scored = [replace(a, score=_score(a)), replace(b, score=_score(b))]
    merged = _merge_candidates(scored)
    assert len(merged) == 2


def test_parse_best_abstains_with_no_candidates():
    decision = PropositionParser.parse_best("Marsupials carry their young in a pouch.")
    assert decision.accepted is False
    assert decision.ambiguous is False
    assert decision.reason == "no_match"
    assert decision.proposition is None


def test_parse_best_abstains_below_acceptance_threshold():
    """The exact Phase 9 regression case: the fixed catch-all pattern
    confidently matches something for a sentence it can't really parse
    -- it must not be accepted just because it's the only candidate."""
    decision = PropositionParser.parse_best(
        "Echidnas are native to queensland.", include_learned=False,
    )
    assert decision.accepted is False
    assert decision.reason == "low_score"
    assert decision.candidates[0].source == "fixed_catchall"


def test_parse_best_prefers_correct_learned_frame_over_wrong_catchall():
    """The direct fix for Phase 9's diagnosed hybrid failure: given both
    a correct learned "in" frame and the wrong "is" catch-all match for
    the same sentence, the learned frame must score higher."""
    library = FrameLibrary()
    library.observe_labelled(
        "Wombats are native to australia.", "wombats", "in", "australia",
    )
    decision = PropositionParser.parse_best(
        "Echidnas are native to queensland.", frame_library=library,
    )
    by_source = {c.source: c for c in decision.candidates}
    assert by_source["learned_frame"].score > by_source["fixed_catchall"].score
    assert by_source["learned_frame"].proposition.relation == "in"


def test_parse_best_accepts_a_clear_unambiguous_winner():
    library = FrameLibrary()
    library.observe_labelled(
        "Wombats belong to the marsupial family.", "wombats", "is", "marsupials",
    )
    library.observe_labelled(
        "Wombats belong to the marsupial family.", "wombats", "is", "marsupials",
    )  # a second observation raises evidence_count, clearing the default threshold
    decision = PropositionParser.parse_best(
        "Koalas belong to the marsupial family.", frame_library=library,
        acceptance_threshold=0.3, margin_threshold=0.05,
    )
    assert decision.accepted is True
    assert decision.proposition == Proposition(
        "koalas", "is", "marsupial", -1, "Koalas belong to the marsupial family.",
    )


def test_parse_best_reports_ambiguous_when_two_candidates_are_close():
    """Two learned frames that both regex-match the same test sentence,
    disagreeing on where the object boundary falls -- genuine ambiguity,
    not a fixed-vs-learned mismatch."""
    library = FrameLibrary()
    library.observe_labelled(
        "Wombats belong to the marsupial family.", "wombats", "is", "marsupials",
    )
    library.observe_labelled(
        "Wombats belong to australia.", "wombats", "in", "australia",
    )
    decision = PropositionParser.parse_best(
        "Koalas belong to the marsupial family.", frame_library=library,
        include_fixed=False, acceptance_threshold=0.0, margin_threshold=0.99,
    )
    assert len(decision.candidates) >= 2
    assert decision.accepted is False
    assert decision.ambiguous is True
    assert decision.reason == "low_margin"


def test_parse_best_include_fixed_false_uses_only_learned_frames():
    library = FrameLibrary()
    library.observe_labelled(
        "Wombats are native to australia.", "wombats", "in", "australia",
    )
    decision = PropositionParser.parse_best(
        "Echidnas are native to queensland.", frame_library=library,
        include_fixed=False, acceptance_threshold=0.0,
    )
    assert all(c.source == "learned_frame" for c in decision.candidates)
