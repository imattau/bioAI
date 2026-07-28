"""Lightweight English morphology for subject-verb agreement in realised
text (Phase 5 of the response-ecosystem plan).

Uses `lemminflect` (dictionary-based lemmatization/inflection; its only
dependency is numpy, already required by this project) rather than a
hand-rolled suffix-rule table, per an explicit decision made when this
phase started. This codebase's established preference is minimal
hand-written rules over heavy NLP dependencies (`ConsolidationMemory`'s
regex extractor, `TokenLibrary`'s 4-line plural stripper) -- but those
only need to *recognize* a handful of fixed suffixes, not correctly
*generate* an inflected form, and English verb conjugation has enough
irregularity ("go"/"went", "is"/"are"/"was"/"were", "mouse"/"mice") that a
small, well-tested dictionary was judged the better trade-off here.

Scope: subject-verb agreement only. A frame learned from one subject's
number should still read correctly when reused for a subject of the
*opposite* number -- e.g. a frame learned from "Elephants was herbivores"
should render "Elephants were herbivores" once its own subject is
correctly identified as plural. Not attempted: tense selection, aspect,
determiner agreement ("a"/"an"), or anything beyond number agreement on
the verb immediately following a template's [SUBJECT] placeholder.
"""

from __future__ import annotations

import lemminflect

_BE_PRESENT = {True: "are", False: "is"}
_BE_PAST = {True: "were", False: "was"}


def is_plural_noun(phrase: str) -> bool:
    """Best-effort plurality check on the head noun (last word) of a
    phrase -- "red wombats" checks "wombats". Handles irregular plurals
    (mice, geese) via lemminflect's dictionary, not suffix guessing."""
    words = phrase.split()
    if not words:
        return False
    word = words[-1]
    lemma_result = lemminflect.getLemma(word, upos="NOUN")
    lemma = lemma_result[0] if lemma_result else word
    if lemma.lower() == word.lower():
        return False
    plural_form = lemminflect.getInflection(lemma, tag="NNS")
    return bool(plural_form) and plural_form[0].lower() == word.lower()


def agree_verb(verb: str, plural: bool) -> str:
    """Re-inflect *verb* for number agreement with a subject that is (or
    isn't) plural, preserving the original's capitalization. "be" forms
    are special-cased -- lemminflect's regular VBZ/VBP tags don't cover
    "is"/"are"/"was"/"were" -- everything else goes through lemminflect's
    lemma -> VBZ/VBP inflection."""
    lower = verb.lower()
    if lower in ("is", "are"):
        result = _BE_PRESENT[plural]
    elif lower in ("was", "were"):
        result = _BE_PAST[plural]
    else:
        lemma_result = lemminflect.getLemma(lower, upos="VERB")
        lemma = lemma_result[0] if lemma_result else lower
        tag = "VBP" if plural else "VBZ"
        inflected = lemminflect.getInflection(lemma, tag=tag)
        result = inflected[0] if inflected else verb
    return result[:1].upper() + result[1:] if verb[:1].isupper() else result


def apply_subject_verb_agreement(template: str, subject: str) -> str:
    """If *template* (using "[SUBJECT]"/"[OBJECT]" placeholders) has
    "[SUBJECT]" in the grammatical-subject position -- i.e. as its very
    first token -- re-agree the verb immediately following it with
    *subject*'s number. Templates where the placeholder ISN'T the
    grammatical subject (e.g. "The capital of [SUBJECT] is [OBJECT]",
    where the real subject is "the capital", always singular) are left
    untouched -- applying subject-based agreement there would be wrong,
    not just unhelpful, regardless of whether [SUBJECT]'s own value looks
    plural."""
    prefix = "[SUBJECT] "
    if not template.startswith(prefix):
        return template
    remainder = template[len(prefix):]
    parts = remainder.split(" ", 1)
    verb, rest = parts[0], (parts[1] if len(parts) > 1 else "")
    agreed = agree_verb(verb, is_plural_noun(subject))
    return prefix + agreed + (" " + rest if rest else "")
