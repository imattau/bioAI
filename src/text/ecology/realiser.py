"""Proposition realiser: phenotype from genotype.

Phase 2 rendered every relation through a small, fixed table of English
templates. Phase 4 adds an optional `frame_library` (`frame_extractor.py`):
when given, a relation's phrasing is drawn from the frames actually
observed for it (`FrameLibrary.select_frame`), weighted by how often each
was seen, instead of always using the one fixed template -- "Wombats are
in Australia" and "The cat sits within the box" produce genuinely
different clause shapes for the same "in" relation once both have been
observed, rather than collapsing to one canonical phrasing. Falls back to
the fixed table when no `frame_library` is given or it has nothing for
that relation, so every existing caller (`PropositionRealiser.realise(propositions)`,
with no second argument) keeps working unchanged.

Phase 5 adds subject-verb number agreement (`morphology.py`): a frame
learned from one subject's number (e.g. "The cat sits within the box")
now re-inflects its verb when reused for a subject of the opposite number
(e.g. "Wombats sit within Australia", not "Wombats sits within
Australia") -- without this, a frame's usefulness was limited to subjects
sharing its originally-observed number. `_RELATION_TEMPLATES` moved to
the same "[SUBJECT]"/"[OBJECT]" bracket convention frame templates already
use so agreement applies uniformly to both, rather than needing two
separate code paths.

Phase 7 adds two more real generation improvements:

- **Compound-clause merging**: two propositions about the same subject
  but *different* relations used to always become two separate sentences
  ("Wombats are marsupials. Wombats are in Australia."). Relations whose
  template has "[SUBJECT]" in the true grammatical-subject position (the
  same distinction `apply_subject_verb_agreement` already draws) now
  merge into one compound clause ("Wombats are marsupials and are in
  Australia."). Relations whose template doesn't start with "[SUBJECT]"
  (e.g. "capital", whose real grammatical subject is "the capital") never
  merge -- doing so would be grammatically wrong, not just unhelpful.
- **Indefinite articles** ("a"/"an"), scoped to the "is"/"has" relations'
  *objects* only, for a singular object (`is_plural_noun`, Phase 5).
  `ConsolidationMemory._normalise` strips all casing before a Proposition
  ever exists, so there is no signal left to tell a common noun
  ("marsupial", needs "a") from a proper noun ("France", never takes
  one) -- rather than guess, article insertion is scoped to relations
  whose content is reliably common-noun in this curriculum ("is"
  categories, "has" properties) and excluded from "in"/"capital"/
  "capital_of", whose objects are reliably place/city/country names.
  Not applied to subjects: this curriculum's subjects are always taught
  as plural nouns (a real, disclosed scope limit, not an oversight).

Phase 8 adds possessive-pronoun substitution for "capital", scoped
narrowly: compound-clause merging (above) already means the same
subject's "is"/"in"/"has" facts always merge into one clause, so general
nominative "it"/"they" substitution has no use case in this curriculum --
the only pronoun ever needed is possessive ("its"/"their capital"), and
only when that subject was already introduced by an earlier compound
clause in the same `realise()` call (otherwise a pronoun would have no
antecedent, which is wrong, not just unhelpful -- the same category of
constraint every other exemption in this file already follows). When the
subject was NOT already introduced, "capital" keeps its full noun-phrase
template ("The capital of [SUBJECT] is [OBJECT]") exactly as before.
"capital_of" (subject is the city, not the country) is a structurally
different problem -- the pronoun would apply to the object, not the
subject -- and stays out of scope.

Every subject and object string still comes straight from a
`Proposition`'s fields, so all factual content stays traceable to a
source proposition regardless of which phrasing path rendered it -- only
grammatical connectives, learned frames, agreement, articles, pronouns,
and compound "and" joins are synthesized.

Phase 9 adds `require_frame`: when `True`, a relation with no available
frame makes the whole call return `""` instead of falling back to
`_RELATION_TEMPLATES` -- so a generation "success" can never be produced
by this file's own hand-authored scaffolding, only by a genuinely
acquired frame. Does not affect the possessive-pronoun or uncertainty-
qualifier branches, which aren't `_RELATION_TEMPLATES` fallbacks.
"""

from __future__ import annotations

from .morphology import apply_subject_verb_agreement, indefinite_article, is_plural_noun
from .proposition_extractor import Proposition, UNCERTAINTY_RELATION

_RELATION_TEMPLATES = {
    "is": "[SUBJECT] is [OBJECT]",
    "in": "[SUBJECT] is in [OBJECT]",
    "has": "[SUBJECT] has [OBJECT]",
    "capital": "The capital of [SUBJECT] is [OBJECT]",
    "capital_of": "[SUBJECT] is the capital of [OBJECT]",
}

# Relations whose object content is reliably a common noun in this
# curriculum ("is" categories, "has" properties) -- see the module
# docstring for why "in"/"capital"/"capital_of" are excluded rather than
# guessed at.
_ARTICLE_RELATIONS = frozenset({"is", "has"})

# "capital_of"'s subject is always a city name -- grammatically singular
# by definition, regardless of surface form. Found via real testing (not
# hypothetical): `is_plural_noun("paris")` returns True, because
# lemminflect's dictionary has no entry for the proper noun "Paris" and
# falls back to treating a trailing "-s" as a regular plural suffix (it
# infers a nonexistent singular "pari"). This is the same missing-
# casing-signal problem `_ARTICLE_RELATIONS` above is scoped around, not
# a new one -- rather than attempt general proper-noun detection, this
# one relation is exempted from agreement entirely, since its subject's
# number is already known from what the relation *means*, not from
# surface morphology.
_AGREEMENT_EXEMPT_RELATIONS = frozenset({"capital_of"})

_SUBJECT_PLACEHOLDER = "[SUBJECT] "

# Substituted for "The capital of [SUBJECT] is [OBJECT]" when [SUBJECT]
# was already introduced earlier in the same realise() call -- see the
# module docstring's Phase 8 note.
_CAPITAL_POSSESSIVE_TEMPLATE = "[POSSESSIVE] capital is [OBJECT]"


def _with_article(phrase: str) -> str:
    if is_plural_noun(phrase):
        return phrase
    words = phrase.split()
    if not words:
        return phrase
    return f"{indefinite_article(words[0])} {phrase}"


class PropositionRealiser:
    @staticmethod
    def realise(
        propositions: tuple[Proposition, ...],
        frame_library=None,
        require_frame: bool = False,
    ) -> str:
        if not propositions:
            return ""
        groups: dict[tuple[str, str], list[str]] = {}
        order: list[tuple[str, str]] = []
        for prop in propositions:
            key = (prop.subject, prop.relation)
            if key not in groups:
                groups[key] = []
                order.append(key)
            if prop.object not in groups[key]:
                groups[key].append(prop.object)

        # Clauses are built into `parts`, preserving `order`'s sequence.
        # A relation whose template puts [SUBJECT] in the true
        # grammatical-subject position doesn't get its own slot on repeat
        # occurrences of the same subject -- its verb phrase is appended
        # to that subject's existing compound slot instead (`None`
        # placeholder, filled in once all propositions are processed).
        parts: list[str | None] = []
        slot_subject: dict[int, str] = {}
        compound_slot_for_subject: dict[str, int] = {}
        compound_phrases: dict[str, list[str]] = {}

        for subject, relation in order:
            objects_list = groups[(subject, relation)]
            if relation == UNCERTAINTY_RELATION:
                objects = " and ".join(objects_list)
                parts.append(
                    f"there may be multiple possible answers for {subject}: {objects}"
                )
                continue
            if relation in _ARTICLE_RELATIONS:
                objects_list = [_with_article(obj) for obj in objects_list]
            objects = " and ".join(objects_list)

            if relation == "capital" and subject in compound_slot_for_subject:
                # Grammatical necessity, not a phrasing preference -- an
                # already-introduced subject takes a pronoun regardless
                # of whether frame_library has learned some other full
                # noun-phrase wording for "capital"; a learned frame here
                # would just repeat the noun phrase this branch exists to
                # avoid.
                possessive = "their" if is_plural_noun(subject) else "its"
                template = _CAPITAL_POSSESSIVE_TEMPLATE.replace(
                    "[POSSESSIVE]", possessive
                )
            else:
                frame = (
                    frame_library.select_frame(relation)
                    if frame_library is not None else None
                )
                if frame is not None:
                    template = frame.template
                elif require_frame:
                    # Phase 9: a "success" must never be produced by
                    # hand-authored scaffolding -- no frame available for
                    # this relation means the whole realisation is
                    # honestly impossible, not a reason to fall back.
                    return ""
                else:
                    template = _RELATION_TEMPLATES.get(relation, f"[SUBJECT] {relation} [OBJECT]")
            if relation not in _AGREEMENT_EXEMPT_RELATIONS:
                template = apply_subject_verb_agreement(template, subject)

            if template.startswith(_SUBJECT_PLACEHOLDER):
                verb_phrase = template[len(_SUBJECT_PLACEHOLDER):].replace(
                    "[OBJECT]", objects
                )
                if subject not in compound_slot_for_subject:
                    index = len(parts)
                    compound_slot_for_subject[subject] = index
                    slot_subject[index] = subject
                    compound_phrases[subject] = []
                    parts.append(None)
                compound_phrases[subject].append(verb_phrase)
            else:
                clause = template.replace("[SUBJECT]", subject).replace(
                    "[OBJECT]", objects
                )
                parts.append(clause)

        clauses = [
            part if part is not None
            else slot_subject[index] + " " + " and ".join(compound_phrases[slot_subject[index]])
            for index, part in enumerate(parts)
        ]

        clauses = [
            clause[:1].upper() + clause[1:] if clause else clause
            for clause in clauses
        ]
        text = ". ".join(clauses)
        return text + "." if text else ""
