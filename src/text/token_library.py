"""Compact tokenised long-term text storage with indexed candidate retrieval."""

from __future__ import annotations

import hashlib
import heapq
import math
import re
from array import array
from collections import OrderedDict

import torch


class TokenLibrary:
    def __init__(self, vector_cache_size: int = 1024):
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: list[str] = []
        self.token_ids = array("I")
        self.offsets = array("Q", [0])
        self.texts: list[str] = []
        self.postings: dict[int, array] = {}
        self.semantic_postings: dict[str, array] = {}
        self.document_lengths = array("I")
        self._exact: dict[bytes, list[int]] = {}
        self.vector_cache_size = vector_cache_size
        self._vector_cache: OrderedDict[int, torch.Tensor] = OrderedDict()
        self.concept_aliases: dict[str, str] = {}
        self.semantic_vectors: list[bytes | None] = []
        self.semantic_vsa_dim: int | None = None
        self.semantic_band_bits = 8
        self.semantic_lsh: dict[tuple[int, int], array] = {}
        self._install_default_aliases()

    def _install_default_aliases(self):
        groups = {
            "capital": {"capital", "capital-city", "administrative-centre",
                        "administrative-center", "seat-of-government"},
            "automobile": {"automobile", "car", "vehicle"},
            "physician": {"physician", "doctor", "medic"},
            "purchase": {"purchase", "buy", "acquire"},
            "large": {"large", "big", "huge"},
            "small": {"small", "little", "tiny"},
            "fast": {"fast", "quick", "rapid"},
        }
        for canonical, aliases in groups.items():
            self.register_aliases(canonical, aliases)

    def register_aliases(self, canonical: str, aliases) -> None:
        canonical_term = self._normalise_term(canonical)
        self.concept_aliases[canonical_term] = canonical_term
        for alias in aliases:
            normalised = "-".join(
                self._normalise_term(token) for token in self.tokenize(alias)
            )
            self.concept_aliases[normalised] = canonical_term

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.lower())

    @staticmethod
    def _normalise_term(token: str) -> str:
        if token.endswith("'s"):
            token = token[:-2]
        if len(token) > 4 and token.endswith("ies"):
            return token[:-3] + "y"
        if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
            return token[:-1]
        return token

    def semantic_terms(self, text: str) -> set[str]:
        tokens = [self._normalise_term(token) for token in self.tokenize(text)]
        stop_words = {
            "a", "an", "the", "is", "are", "was", "were", "of", "to",
            "in", "on", "at", "for", "and", "or", "what", "who", "where",
            "when", "why", "how", "which", "do", "does", "did",
        }
        terms = {
            self.concept_aliases.get(token, token)
            for token in tokens if token not in stop_words
        }
        # Match registered multi-token concepts such as "administrative centre".
        for width in (2, 3):
            for start in range(len(tokens) - width + 1):
                phrase = "-".join(tokens[start:start + width])
                canonical = self.concept_aliases.get(phrase)
                if canonical:
                    terms.add(canonical)
        return terms

    @staticmethod
    def _digest(token_ids: list[int] | array) -> bytes:
        values = array("I", token_ids)
        return hashlib.blake2b(values.tobytes(), digest_size=16).digest()

    def _encode_tokens(self, tokens: list[str], add_new: bool) -> list[int]:
        ids = []
        for token in tokens:
            token_id = self.token_to_id.get(token)
            if token_id is None:
                if not add_new:
                    continue
                token_id = len(self.id_to_token)
                self.token_to_id[token] = token_id
                self.id_to_token.append(token)
            ids.append(token_id)
        return ids

    @staticmethod
    def _pack_semantic(vector: torch.Tensor) -> bytes:
        bits = (vector.flatten() > 0).to(torch.uint8).tolist()
        packed = bytearray((len(bits) + 7) // 8)
        for index, bit in enumerate(bits):
            packed[index // 8] |= int(bit) << (index % 8)
        return bytes(packed)

    def semantic_vector(self, sentence_id: int) -> torch.Tensor | None:
        """Restore a bit-packed semantic vector as a bipolar int8 tensor."""
        packed = self.semantic_vectors[sentence_id]
        if packed is None or self.semantic_vsa_dim is None:
            return None
        values = [
            1 if byte & (1 << bit) else -1
            for byte in packed
            for bit in range(8)
        ][:self.semantic_vsa_dim]
        return torch.tensor(values, dtype=torch.int8)

    def _index_semantic(self, sentence_id: int, packed: bytes):
        for band, code in enumerate(packed):
            self.semantic_lsh.setdefault((band, code), array("I")).append(
                sentence_id
            )

    def add(self, text: str, semantic_vector: torch.Tensor | None = None) -> int:
        ids = self._encode_tokens(self.tokenize(text), add_new=True)
        sentence_id = len(self.texts)
        self.texts.append(text)
        self.token_ids.extend(ids)
        self.offsets.append(len(self.token_ids))
        self.document_lengths.append(len(ids))
        for token_id in set(ids):
            self.postings.setdefault(token_id, array("I")).append(sentence_id)
        for term in self.semantic_terms(text):
            self.semantic_postings.setdefault(term, array("I")).append(sentence_id)
        self._exact.setdefault(self._digest(ids), []).append(sentence_id)
        if semantic_vector is None:
            self.semantic_vectors.append(None)
        else:
            semantic_vector = semantic_vector.flatten()
            if self.semantic_vsa_dim is None:
                self.semantic_vsa_dim = len(semantic_vector)
            elif len(semantic_vector) != self.semantic_vsa_dim:
                raise ValueError("Semantic VSA dimension changed")
            packed = self._pack_semantic(semantic_vector)
            self.semantic_vectors.append(packed)
            self._index_semantic(sentence_id, packed)
        return sentence_id

    def semantic_candidate_ids(
        self,
        query_vector: torch.Tensor,
        limit: int = 100,
        max_candidates: int = 20_000,
    ) -> list[tuple[int, float]]:
        if self.semantic_vsa_dim is None:
            return []
        if len(query_vector.flatten()) != self.semantic_vsa_dim:
            raise ValueError("Semantic query dimension does not match index")
        packed_query = self._pack_semantic(query_vector)
        band_matches: dict[int, int] = {}
        for band, code in enumerate(packed_query):
            for sentence_id in self.semantic_lsh.get((band, code), ()):
                band_matches[sentence_id] = band_matches.get(sentence_id, 0) + 1
                if len(band_matches) >= max_candidates:
                    break
            if len(band_matches) >= max_candidates:
                break
        shortlist = heapq.nlargest(
            min(max_candidates, len(band_matches)),
            band_matches,
            key=band_matches.get,
        )
        scores = []
        valid_bits = self.semantic_vsa_dim
        for sentence_id in shortlist:
            packed = self.semantic_vectors[sentence_id]
            if packed is None:
                continue
            differing_bits = sum(
                (left ^ right).bit_count()
                for left, right in zip(packed_query, packed)
            )
            similarity = 1.0 - 2.0 * differing_bits / valid_bits
            scores.append((sentence_id, similarity))
        return heapq.nlargest(limit, scores, key=lambda item: item[1])

    def sequence(self, sentence_id: int) -> array:
        start = self.offsets[sentence_id]
        end = self.offsets[sentence_id + 1]
        return self.token_ids[start:end]

    def exact_lookup(self, text: str) -> str | None:
        tokens = self.tokenize(text)
        if any(token not in self.token_to_id for token in tokens):
            return None
        ids = self._encode_tokens(tokens, add_new=False)
        for sentence_id in self._exact.get(self._digest(ids), []):
            if list(self.sequence(sentence_id)) == ids:
                return self.texts[sentence_id]
        return None

    def candidate_ids(
        self,
        query: str,
        limit: int = 100,
        max_candidates: int = 20_000,
    ) -> list[tuple[int, float]]:
        query_ids = set(self._encode_tokens(self.tokenize(query), add_new=False))
        semantic_terms = self.semantic_terms(query)
        if (not query_ids and not semantic_terms) or not self.texts:
            return []
        posting_sources = []
        for token_id in query_ids:
            posting = self.postings.get(token_id)
            if posting:
                posting_sources.append(("lexical", str(token_id), posting))
        for term in semantic_terms:
            posting = self.semantic_postings.get(term)
            if posting:
                posting_sources.append(("semantic", term, posting))
        if not posting_sources:
            return []

        # Seed from the rarest lists and cap work for all-common queries.
        posting_sources.sort(key=lambda item: len(item[2]))
        rarest_size = len(posting_sources[0][2])
        seed_sources = [
            source for source in posting_sources
            if len(source[2]) <= max(10, rarest_size * 10)
        ][:3]
        candidates = set()
        for _, _, posting in seed_sources:
            remaining = max_candidates - len(candidates)
            if remaining <= 0:
                break
            candidates.update(posting[-remaining:])

        average_length = len(self.token_ids) / max(1, len(self.texts))
        scores: dict[int, float] = {}
        for sentence_id in candidates:
            score = 0.0
            document_length = self.document_lengths[sentence_id]
            length_norm = 0.25 + 0.75 * document_length / average_length
            for source, _, posting in posting_sources:
                # Arrays are sorted by sentence ID, so membership can use binary search.
                import bisect
                position = bisect.bisect_left(posting, sentence_id)
                if position >= len(posting) or posting[position] != sentence_id:
                    continue
                inverse_frequency = math.log1p(
                    (len(self.texts) - len(posting) + 0.5) / (len(posting) + 0.5)
                )
                weight = 1.0 if source == "lexical" else 0.8
                score += weight * inverse_frequency / length_norm
            scores[sentence_id] = score
        return heapq.nlargest(limit, scores.items(), key=lambda item: item[1])

    def vector(self, sentence_id: int, encoder) -> torch.Tensor:
        cached = self._vector_cache.pop(sentence_id, None)
        if cached is None:
            cached = encoder.encode(self.texts[sentence_id])
        self._vector_cache[sentence_id] = cached
        while len(self._vector_cache) > self.vector_cache_size:
            self._vector_cache.popitem(last=False)
        return cached

    def get_state(self) -> dict:
        return {
            "token_to_id": self.token_to_id,
            "id_to_token": self.id_to_token,
            "token_ids": self.token_ids,
            "offsets": self.offsets,
            "texts": self.texts,
            "postings": self.postings,
            "semantic_postings": self.semantic_postings,
            "document_lengths": self.document_lengths,
            "concept_aliases": self.concept_aliases,
            "semantic_vectors": self.semantic_vectors,
            "semantic_vsa_dim": self.semantic_vsa_dim,
            "semantic_band_bits": self.semantic_band_bits,
            "semantic_lsh": self.semantic_lsh,
            "exact": self._exact,
            "vector_cache_size": self.vector_cache_size,
        }

    @classmethod
    def from_state(cls, state: dict) -> "TokenLibrary":
        library = cls(state["vector_cache_size"])
        library.token_to_id = state["token_to_id"]
        library.id_to_token = state["id_to_token"]
        library.token_ids = array("I", state["token_ids"])
        library.offsets = array("Q", state["offsets"])
        library.texts = state["texts"]
        library.postings = {
            int(token_id): array("I", posting)
            for token_id, posting in state["postings"].items()
        }
        library.semantic_postings = {
            term: array("I", posting)
            for term, posting in state.get("semantic_postings", {}).items()
        }
        library.document_lengths = array(
            "I", state.get("document_lengths", [
                library.offsets[index + 1] - library.offsets[index]
                for index in range(len(library.texts))
            ])
        )
        library.concept_aliases = state.get(
            "concept_aliases", library.concept_aliases
        )
        library.semantic_vectors = state.get(
            "semantic_vectors", [None] * len(library.texts)
        )
        library.semantic_vsa_dim = state.get("semantic_vsa_dim")
        library.semantic_band_bits = state.get("semantic_band_bits", 8)
        library.semantic_lsh = {
            tuple(key): array("I", posting)
            for key, posting in state.get("semantic_lsh", {}).items()
        }
        library._exact = state["exact"]
        return library

    @property
    def token_storage_bytes(self) -> int:
        return self.token_ids.buffer_info()[1] * self.token_ids.itemsize

    @property
    def semantic_storage_bytes(self) -> int:
        return sum(len(vector) for vector in self.semantic_vectors if vector)

    def __len__(self) -> int:
        return len(self.texts)
