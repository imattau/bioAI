"""Tests for Phase 5 of the response-ecosystem plan: subject-verb number
agreement in realised text, via lemminflect. See
src/text/ecology/morphology.py's module docstring for why a small
dictionary-based library was chosen over hand-rolled suffix rules for
this specific problem.
"""

from src.text.ecology.morphology import (
    agree_verb,
    apply_subject_verb_agreement,
    is_plural_noun,
)


def test_is_plural_noun_regular():
    assert is_plural_noun("wombats") is True
    assert is_plural_noun("wombat") is False


def test_is_plural_noun_irregular():
    assert is_plural_noun("mice") is True
    assert is_plural_noun("mouse") is False
    assert is_plural_noun("geese") is True


def test_is_plural_noun_checks_head_word_of_phrase():
    assert is_plural_noun("red wombats") is True
    assert is_plural_noun("a red wombat") is False


def test_is_plural_noun_empty_string():
    assert is_plural_noun("") is False


def test_agree_verb_be_present():
    assert agree_verb("is", True) == "are"
    assert agree_verb("is", False) == "is"
    assert agree_verb("are", True) == "are"
    assert agree_verb("are", False) == "is"


def test_agree_verb_be_past():
    assert agree_verb("was", True) == "were"
    assert agree_verb("was", False) == "was"
    assert agree_verb("were", True) == "were"
    assert agree_verb("were", False) == "was"


def test_agree_verb_regular_verb():
    assert agree_verb("sits", True) == "sit"
    assert agree_verb("sits", False) == "sits"
    assert agree_verb("sit", True) == "sit"
    assert agree_verb("sit", False) == "sits"


def test_agree_verb_preserves_capitalization():
    assert agree_verb("Is", True) == "Are"
    assert agree_verb("Sits", True) == "Sit"


def test_apply_subject_verb_agreement_when_subject_is_grammatical_subject():
    assert (
        apply_subject_verb_agreement("[SUBJECT] is [OBJECT]", "elephants")
        == "[SUBJECT] are [OBJECT]"
    )
    assert (
        apply_subject_verb_agreement("[SUBJECT] sits within [OBJECT]", "wombats")
        == "[SUBJECT] sit within [OBJECT]"
    )


def test_apply_subject_verb_agreement_leaves_singular_subject_unchanged():
    assert (
        apply_subject_verb_agreement("[SUBJECT] is [OBJECT]", "france")
        == "[SUBJECT] is [OBJECT]"
    )


def test_apply_subject_verb_agreement_skips_templates_where_placeholder_is_not_the_subject():
    """"The capital of [SUBJECT] is [OBJECT]" -- the grammatical subject
    is "the capital" (always singular), not [SUBJECT] itself. Applying
    agreement based on [SUBJECT]'s own plurality here would be wrong, not
    just unhelpful, so the template must be left untouched."""
    template = "The capital of [SUBJECT] is [OBJECT]"
    assert apply_subject_verb_agreement(template, "netherlands") == template
    assert apply_subject_verb_agreement(template, "wombats") == template


def test_apply_subject_verb_agreement_no_placeholder():
    assert apply_subject_verb_agreement("no placeholder here", "wombats") == (
        "no placeholder here"
    )
