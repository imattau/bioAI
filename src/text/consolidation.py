"""Evidence-backed consolidation of episodic memories into semantic structure."""

from __future__ import annotations

import heapq
import re
from array import array
from collections.abc import Iterable

import torch


class ConsolidationMemory:
    """Build compact concept prototypes and retain sourced relational claims."""

    def __init__(
        self,
        promotion_threshold: int = 3,
        max_prototypes: int = 50_000,
        max_sources_per_item: int = 16,
    ):
        self.promotion_threshold = promotion_threshold
        self.max_prototypes = max_prototypes
        self.max_sources_per_item = max_sources_per_item
        self.concepts: dict[str, dict] = {}
        self.relations: dict[tuple[str, str], dict[str, dict]] = {}
        self._prototype_terms: list[str] = []
        self._prototype_matrix: torch.Tensor | None = None
        self._prototype_lsh: dict[tuple[int, int], array] = {}
        self._prototype_index_dirty = True

    @staticmethod
    def normalise(value: str) -> str:
        return " ".join(re.findall(r"[a-z0-9]+", value.lower())).strip()

    # Single source of truth for the subject/object extraction patterns --
    # also reused, unmodified, by src/text/ecology/frame_extractor.py to
    # recover the literal matched wording (not just the flattened relation
    # label) for frame learning. Do not fork a second copy of this tuple.
    _RELATION_PATTERNS = (
        (r"^(?:the\s+)?capital\s+of\s+(.+?)\s+is\s+(.+)$", "capital"),
        # "is|are": "capital_of"'s template starts with [SUBJECT] (unlike
        # "capital"'s, which starts with "the capital of" and is never
        # touched by agreement), so it's in principle reachable by
        # subject-verb agreement if a city name ever looked plural --
        # hardened the same way as "in"/"has" even though real city names
        # essentially never trigger it.
        (r"^(.+?)\s+(?:is|are)\s+the\s+capital\s+of\s+(.+)$", "capital_of"),
        (r"^(.+?)\s+(?:is|are)\s+(?:located\s+)?in\s+(.+)$", "in"),
        # Both singular ("lies"/"sits") and plural ("lie"/"sit") verb
        # forms -- a real bug found via Phase 6 testing: Phase 5's
        # subject-verb agreement correctly re-inflects "lies within" to
        # "lie within" for a plural subject, but the original pattern
        # only recognized the singular form, so a grammatically-corrected
        # sentence could no longer be re-extracted by this same extractor.
        (r"^(.+?)\s+(?:lies|sits|lie|sit)\s+(?:within|in)\s+(.+)$", "in"),
        # "has"/"have" is a genuinely irregular present-tense pair (not a
        # simple -s suffix), unlike most of the fixed set above -- added
        # specifically so Phase 6's morphology tests have a non-"be" verb
        # to exercise subject-verb agreement on.
        (r"^(.+?)\s+(?:has|have)\s+(.+)$", "has"),
        (r"^(.+?)\s+(?:is|are|was|were)\s+(.+)$", "is"),
    )

    @classmethod
    def extract_relations(cls, text: str) -> list[tuple[str, str, str]]:
        sentence = text.strip().rstrip(".!?")
        for pattern, relation in cls._RELATION_PATTERNS:
            match = re.match(pattern, sentence, flags=re.IGNORECASE)
            if match:
                subject = cls.normalise(match.group(1))
                obj = cls.normalise(match.group(2))
                if subject and obj:
                    return [(subject, relation, obj)]
        return []

    def observe_relation(self, text: str, source_id: int) -> None:
        for subject, relation, obj in self.extract_relations(text):
            alternatives = self.relations.setdefault((subject, relation), {})
            evidence = alternatives.setdefault(
                obj, {"count": 0, "source_ids": []}
            )
            evidence["count"] += 1
            if source_id not in evidence["source_ids"]:
                evidence["source_ids"].append(source_id)
                del evidence["source_ids"][:-self.max_sources_per_item]

    def observe_concept(
        self,
        term: str,
        vector: torch.Tensor,
        source_id: int,
        initial_vectors: Iterable[torch.Tensor] | None = None,
        initial_source_ids: Iterable[int] | None = None,
    ) -> bool:
        """Create/update a prototype. Returns True when the term is promoted."""
        record = self.concepts.get(term)
        if record is None:
            if initial_vectors is None:
                return False
            vectors = [item.flatten().to(torch.int16) for item in initial_vectors]
            if len(vectors) < self.promotion_threshold:
                return False
            if len(self.concepts) >= self.max_prototypes:
                weakest = min(
                    self.concepts, key=lambda key: self.concepts[key]["count"]
                )
                if self.concepts[weakest]["count"] >= len(vectors):
                    return False
                del self.concepts[weakest]
            sources = list(dict.fromkeys(initial_source_ids or ()))
            self.concepts[term] = {
                "sum": torch.stack(vectors).sum(dim=0).to(torch.int16),
                "count": len(vectors),
                "source_ids": sources[-self.max_sources_per_item:],
            }
            self._prototype_index_dirty = True
            return True

        record["sum"] = torch.clamp(
            record["sum"].to(torch.int32) + vector.flatten().to(torch.int32),
            min=-32767,
            max=32767,
        ).to(torch.int16)
        record["count"] += 1
        if source_id not in record["source_ids"]:
            record["source_ids"].append(source_id)
            del record["source_ids"][:-self.max_sources_per_item]
        self._prototype_index_dirty = True
        return True

    @staticmethod
    def _prototype(record: dict) -> torch.Tensor:
        return torch.where(
            record["sum"] >= 0,
            torch.ones_like(record["sum"], dtype=torch.float32),
            -torch.ones_like(record["sum"], dtype=torch.float32),
        )

    @staticmethod
    def _pack(vector: torch.Tensor) -> bytes:
        bits = (vector.flatten() > 0).to(torch.uint8).tolist()
        packed = bytearray((len(bits) + 7) // 8)
        for index, bit in enumerate(bits):
            packed[index // 8] |= int(bit) << (index % 8)
        return bytes(packed)

    def _rebuild_prototype_index(self) -> None:
        self._prototype_terms = list(self.concepts)
        self._prototype_lsh = {}
        if not self._prototype_terms:
            self._prototype_matrix = None
            self._prototype_index_dirty = False
            return
        self._prototype_matrix = torch.stack([
            self._prototype(self.concepts[term])
            for term in self._prototype_terms
        ])
        for prototype_id, vector in enumerate(self._prototype_matrix):
            for band, code in enumerate(self._pack(vector)):
                self._prototype_lsh.setdefault(
                    (band, code), array("I")
                ).append(prototype_id)
        self._prototype_index_dirty = False

    def query_concepts(
        self,
        query_vector: torch.Tensor,
        limit: int = 8,
        max_candidates: int = 2048,
    ) -> list[dict]:
        if not self.concepts:
            return []
        if self._prototype_index_dirty:
            self._rebuild_prototype_index()
        query = query_vector.flatten().to(torch.float32)
        if (
            self._prototype_matrix is None
            or self._prototype_matrix.shape[1] != len(query)
        ):
            return []
        band_matches: dict[int, int] = {}
        for band, code in enumerate(self._pack(query)):
            for prototype_id in self._prototype_lsh.get((band, code), ()):
                band_matches[prototype_id] = (
                    band_matches.get(prototype_id, 0) + 1
                )
        if band_matches:
            candidate_ids = heapq.nlargest(
                min(max_candidates, len(band_matches)),
                band_matches,
                key=band_matches.get,
            )
        else:
            candidate_ids = list(range(len(self._prototype_terms)))
        candidate_tensor = torch.tensor(candidate_ids, dtype=torch.int64)
        scores = (
            self._prototype_matrix.index_select(0, candidate_tensor)
            @ query
        ) / len(query)
        top_count = min(limit, len(candidate_ids))
        values, positions = torch.topk(scores, k=top_count)
        best = [
            (
                values[index].item(),
                self._prototype_terms[candidate_ids[positions[index].item()]],
            )
            for index in range(top_count)
        ]
        return [
            {
                "term": term,
                "score": score,
                "count": self.concepts[term]["count"],
                "source_ids": list(self.concepts[term]["source_ids"]),
            }
            for score, term in best
        ]

    def relation_claims(self, subject: str, relation: str = "is") -> list[dict]:
        alternatives = self.relations.get(
            (self.normalise(subject), self.normalise(relation)), {}
        )
        return sorted(
            (
                {"object": obj, **evidence}
                for obj, evidence in alternatives.items()
            ),
            key=lambda item: item["count"],
            reverse=True,
        )

    @classmethod
    def parse_relation_query(cls, text: str) -> tuple[str, str] | None:
        query = text.strip().rstrip(".!?")
        patterns = (
            (r"^(?:what|which)\s+is\s+(?:the\s+)?capital\s+of\s+(.+)$",
             "capital"),
            (r"^(?:what|who)\s+is\s+(.+)$", "is"),
        )
        for pattern, relation in patterns:
            match = re.match(pattern, query, flags=re.IGNORECASE)
            if match:
                subject = cls.normalise(match.group(1))
                if subject:
                    return subject, relation
        return None

    def relation_candidates(self, query: str) -> list[tuple[int, float]]:
        parsed = self.parse_relation_query(query)
        if parsed is None:
            return []
        claims = self.relation_claims(*parsed)
        total = sum(claim["count"] for claim in claims)
        if total == 0:
            return []
        candidates = []
        for claim in claims:
            support = claim["count"] / total
            candidates.extend(
                (source_id, support) for source_id in claim["source_ids"]
            )
        return candidates

    @classmethod
    def parse_path_query(cls, text: str) -> str | None:
        query = text.strip().rstrip(".!?")
        patterns = (
            r"^where\s+is\s+(.+?)(?:\s+located)?$",
            r"^which\s+continent\s+(?:contains|is)\s+(.+?)(?:\s+in)?$",
            r"^what\s+continent\s+is\s+(.+?)\s+in$",
        )
        for pattern in patterns:
            match = re.match(pattern, query, flags=re.IGNORECASE)
            if match:
                return cls.normalise(match.group(1))
        return None

    @staticmethod
    def _supported_choice(alternatives: dict[str, dict]):
        ranked = sorted(
            alternatives.items(),
            key=lambda item: item[1]["count"],
            reverse=True,
        )
        if not ranked:
            return None
        if len(ranked) > 1 and ranked[0][1]["count"] == ranked[1][1]["count"]:
            return None
        return ranked[0]

    def reason_path(self, query: str, max_hops: int = 4) -> dict | None:
        """Follow supported containment/capital edges and retain path evidence."""
        origin = self.parse_path_query(query)
        if origin is None:
            return None
        node = origin
        source_ids = []
        path = [origin]
        for _ in range(max_hops):
            choice = self._supported_choice(
                self.relations.get((node, "in"), {})
            )
            if choice is None:
                reverse_capitals = {}
                for (subject, relation), alternatives in self.relations.items():
                    if relation != "capital":
                        continue
                    evidence = alternatives.get(node)
                    if evidence is not None:
                        reverse_capitals[subject] = evidence
                choice = self._supported_choice(reverse_capitals)
            if choice is None:
                choice = self._supported_choice(
                    self.relations.get((node, "capital_of"), {})
                )
            if choice is None:
                break
            destination, evidence = choice
            node = destination
            path.append(node)
            source_ids.extend(evidence["source_ids"])
        if len(path) < 2:
            return None
        return {
            "origin": origin,
            "answer": node,
            "path": path,
            "hops": len(path) - 1,
            "source_ids": list(dict.fromkeys(source_ids)),
            "text": f"{origin} is in {node}.",
        }

    def get_state(self) -> dict:
        return {
            "promotion_threshold": self.promotion_threshold,
            "max_prototypes": self.max_prototypes,
            "max_sources_per_item": self.max_sources_per_item,
            "concepts": self.concepts,
            "relation_keys": list(self.relations),
            "relation_values": list(self.relations.values()),
        }

    @classmethod
    def from_state(cls, state: dict) -> "ConsolidationMemory":
        memory = cls(
            promotion_threshold=state.get("promotion_threshold", 3),
            max_prototypes=state.get("max_prototypes", 50_000),
            max_sources_per_item=state.get("max_sources_per_item", 16),
        )
        memory.concepts = state.get("concepts", {})
        memory.relations = dict(zip(
            state.get("relation_keys", []),
            state.get("relation_values", []),
        ))
        memory._prototype_index_dirty = True
        return memory
