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
