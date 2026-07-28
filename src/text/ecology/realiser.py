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
with no second argument) keeps working unchanged. Every subject and object
string still comes straight from a `Proposition`'s fields, so all factual
content stays traceable to a source proposition regardless of which
phrasing path rendered it -- only grammatical connectives are synthesized
or drawn from a previously-observed frame.
"""

from __future__ import annotations

from .proposition_extractor import Proposition, UNCERTAINTY_RELATION

_RELATION_TEMPLATES = {
    "is": "{subject} is {object}",
    "in": "{subject} is in {object}",
    "capital": "the capital of {subject} is {object}",
    "capital_of": "{subject} is the capital of {object}",
}


class PropositionRealiser:
    @staticmethod
    def realise(
        propositions: tuple[Proposition, ...],
        frame_library=None,
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

        clauses = []
        for subject, relation in order:
            objects = " and ".join(groups[(subject, relation)])
            if relation == UNCERTAINTY_RELATION:
                clauses.append(
                    f"there may be multiple possible answers for {subject}: {objects}"
                )
                continue
            frame = (
                frame_library.select_frame(relation)
                if frame_library is not None else None
            )
            if frame is not None:
                clause = frame.template.replace(
                    "[SUBJECT]", subject).replace("[OBJECT]", objects)
            else:
                template = _RELATION_TEMPLATES.get(
                    relation, "{subject} {relation} {object}"
                )
                clause = template.format(
                    subject=subject, relation=relation, object=objects
                )
            clauses.append(clause)

        clauses = [
            clause[:1].upper() + clause[1:] if clause else clause
            for clause in clauses
        ]
        text = ". ".join(clauses)
        return text + "." if text else ""
