"""Minimal proposition realiser: phenotype from genotype (Phase 2).

Renders a tuple of propositions into text using a small, fixed set of
relation -> English templates -- no paraphrasing model. Every subject and
object string comes straight from a `Proposition`'s fields (which are
themselves `ConsolidationMemory._normalise`d substrings of the original
source sentence), so every piece of factual content in the realised text
is traceable to a source proposition; only grammatical connectives ("is",
"and", ". ") are synthesized. This intentionally stays this simple through
Phase 2 -- Phase 4 (deferred, see the response-ecosystem plan) is where a
learned clause-frame realiser would replace the fixed template table with
genuine grammatical recombination.
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
    def realise(propositions: tuple[Proposition, ...]) -> str:
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
            template = _RELATION_TEMPLATES.get(
                relation, "{subject} {relation} {object}"
            )
            clauses.append(
                template.format(subject=subject, relation=relation, object=objects)
            )

        clauses = [
            clause[:1].upper() + clause[1:] if clause else clause
            for clause in clauses
        ]
        text = ". ".join(clauses)
        return text + "." if text else ""
