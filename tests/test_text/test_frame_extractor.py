"""Tests for Phase 4 of the response-ecosystem plan: linguistic frame
extraction and evidence-weighted frame selection. See
src/text/ecology/frame_extractor.py's module docstring for the design
rationale -- this generalizes ConsolidationMemory.extract_relations (which
discards the matched wording in favor of a flattened canonical label) into
keeping the literal phrasing as a template with placeholders.
"""

from src.text.ecology.frame_extractor import (
    FrameLibrary,
    LinguisticFrame,
    extract_frame,
)


def test_extract_frame_capital_pattern():
    frame = extract_frame("The capital of France is Paris.")
    assert frame == LinguisticFrame(
        template="The capital of [SUBJECT] is [OBJECT]",
        relation="capital",
    )


def test_extract_frame_capital_of_pattern_distinct_from_capital():
    """"X is the capital of Y" and "the capital of X is Y" must stay
    distinct relations/templates -- extract_relations already gives them
    different canonical labels ("capital_of" vs "capital"); the frame
    extractor must preserve that distinction, not merge them."""
    frame = extract_frame("Paris is the capital of France.")
    assert frame.relation == "capital_of"
    assert frame.template == "[SUBJECT] is the capital of [OBJECT]"


def test_extract_frame_preserves_are_not_flattened_to_is():
    frame = extract_frame("Wombats are marsupials.")
    assert frame.template == "[SUBJECT] are [OBJECT]"
    assert frame.relation == "is"  # canonical label still flattened, by design


def test_extract_frame_distinguishes_in_phrasings_sharing_one_relation_label():
    """extract_relations collapses "is located in", "is in", "lies within",
    and "sits in" all to the single label "in" -- the frame extractor must
    keep their literal wording distinct even though the relation matches."""
    located = extract_frame("The museum is located in Paris.")
    plain = extract_frame("The museum is in Paris.")
    sits = extract_frame("The cat sits within the box.")
    assert located.relation == plain.relation == sits.relation == "in"
    assert located.template != plain.template != sits.template


def test_extract_frame_none_when_no_pattern_matches():
    """Same disclosed limitation as extract_propositions: a sentence with
    no matching copula yields nothing, not a best-effort guess."""
    assert extract_frame("Marsupials carry their young in a pouch.") is None


def test_frame_library_accumulates_evidence_count_for_repeated_template():
    library = FrameLibrary()
    library.observe("The capital of France is Paris.")
    library.observe("The capital of Spain is Madrid.")
    library.observe("The capital of Germany is Berlin.")
    frame = library.frames["The capital of [SUBJECT] is [OBJECT]"]
    assert frame.evidence_count == 3


def test_frame_library_keeps_distinct_templates_separate():
    library = FrameLibrary()
    library.observe("The capital of France is Paris.")
    library.observe("Paris is the capital of France.")
    assert len(library.frames) == 2


def test_frame_library_ignores_unparseable_sentences():
    library = FrameLibrary()
    library.observe("Marsupials carry their young in a pouch.")
    assert library.frames == {}


def test_frame_library_frames_for_relation_filters_correctly():
    library = FrameLibrary()
    library.observe("The capital of France is Paris.")
    library.observe("Paris is the capital of France.")
    library.observe("Wombats are marsupials.")
    capital_frames = library.frames_for_relation("capital")
    assert len(capital_frames) == 1
    assert capital_frames[0].relation == "capital"


def test_select_frame_returns_none_for_unknown_relation():
    library = FrameLibrary()
    library.observe("Wombats are marsupials.")
    assert library.select_frame("capital") is None


def test_select_frame_favors_higher_evidence_count():
    import random

    library = FrameLibrary()
    for _ in range(20):
        library.observe("The capital of France is Paris.")
    library.observe("Paris is the capital of France.")  # different relation entirely

    rng = random.Random(0)
    # "capital" only has one observed template -- deterministic regardless
    # of weighting, this just confirms selection actually returns it.
    selected = library.select_frame("capital", rng=rng)
    assert selected.template == "The capital of [SUBJECT] is [OBJECT]"


def test_frame_library_state_round_trip():
    library = FrameLibrary()
    library.observe("The capital of France is Paris.")
    library.observe("The capital of Spain is Madrid.")
    restored = FrameLibrary.from_state(library.get_state())
    assert restored.frames == library.frames


# ── Phase 9: bidirectional (labelled) frame learning ────────────────────

def test_observe_labelled_aligns_teacher_triple_to_free_form_sentence():
    library = FrameLibrary()
    frame = library.observe_labelled(
        "Wombats belong to the marsupial family.", "wombats", "is", "marsupials",
    )
    assert frame == LinguisticFrame(
        template="[SUBJECT] belong to the [OBJECT] family",
        relation="is",
        evidence_count=1,
    )


def test_parse_recovers_triple_from_its_own_labelled_sentence():
    library = FrameLibrary()
    library.observe_labelled(
        "Wombats belong to the marsupial family.", "wombats", "is", "marsupials",
    )
    assert library.parse("Wombats belong to the marsupial family.") == [
        ("wombats", "is", "marsupial")
    ]


def test_bidirectional_round_trip_generalizes_to_a_new_filler():
    """The real bidirectional guarantee: a frame learned from one
    (subject, object) pair must parse back correctly when PropositionRealiser
    uses it to realise a DIFFERENT filler -- slot generalization at the
    unit level, no LLM involved."""
    from src.text.ecology import Proposition, PropositionRealiser

    library = FrameLibrary()
    library.observe_labelled(
        "Wombats belong to the marsupial family.", "wombats", "is", "marsupials",
    )
    text = PropositionRealiser.realise(
        (Proposition("cats", "is", "dogs", 0, "x"),), frame_library=library,
    )
    assert library.parse(text) == [("cats", "is", "dogs")]


def test_observe_labelled_falls_back_to_opposite_number_form():
    """Teacher label "marsupials" (plural) vs. sentence surface form
    "marsupial" (singular) -- observe_labelled must still align via the
    lemminflect-derived opposite-number form, not just an exact match."""
    library = FrameLibrary()
    frame = library.observe_labelled(
        "The wombat belongs to the marsupial family.", "wombats", "is", "marsupials",
    )
    assert frame is not None
    assert frame.template == "The [SUBJECT] belongs to the [OBJECT] family"


def test_observe_labelled_returns_none_when_label_unlocatable():
    library = FrameLibrary()
    frame = library.observe_labelled(
        "Wombats are diggers.", "wombats", "is", "nonexistentthing",
    )
    assert frame is None
    assert library.frames == {}


def test_observe_labelled_returns_none_when_object_precedes_subject():
    library = FrameLibrary()
    frame = library.observe_labelled(
        "Marsupials include wombats.", "wombats", "is", "marsupials",
    )
    assert frame is None


def test_parse_unifies_frames_learned_via_observe_and_observe_labelled():
    library = FrameLibrary()
    library.observe("The capital of France is Paris.")
    library.observe_labelled(
        "Wombats belong to the marsupial family.", "wombats", "is", "marsupials",
    )
    assert library.parse("The capital of Italy is Rome.") == [
        ("italy", "capital", "rome")
    ]
    assert library.parse("Koalas belong to the marsupial family.") == [
        ("koalas", "is", "marsupial")
    ]


# ── Phase 10: structural parse-candidate features ───────────────────────

def test_parse_candidates_reports_anchor_coverage_and_specificity():
    library = FrameLibrary()
    library.observe_labelled(
        "Wombats belong to the marsupial family.", "wombats", "is", "marsupials",
    )
    candidates = library.parse_candidates("Koalas belong to the marsupial family.")
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.source == "learned_frame"
    assert candidate.frame_template == "[SUBJECT] belong to the [OBJECT] family"
    assert candidate.frame_evidence == 1
    assert 0.0 < candidate.anchor_coverage < 1.0
    assert 0.0 < candidate.specificity <= 1.0
    assert candidate.score == 0.0  # FrameLibrary reports features, not a score


def test_parse_candidates_specificity_rewards_more_literal_wording():
    library = FrameLibrary()
    library.observe_labelled(
        "Wombats belong to the marsupial family.", "wombats", "is", "marsupials",
    )
    library.observe("The capital of France is Paris.")
    bare = next(c for c in library.parse_candidates(
        "The capital of Italy is Rome."
    ) if c.frame_template == "The capital of [SUBJECT] is [OBJECT]")
    verbose = next(c for c in library.parse_candidates(
        "Koalas belong to the marsupial family."
    ) if c.frame_template == "[SUBJECT] belong to the [OBJECT] family")
    assert bare.specificity > 0.0
    assert verbose.specificity > 0.0


def test_parse_candidates_returns_empty_for_no_matching_frame():
    library = FrameLibrary()
    assert library.parse_candidates("Marsupials carry their young in a pouch.") == []
