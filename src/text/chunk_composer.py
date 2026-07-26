"""Learned, retrieval-conditioned response chunk composition."""

from __future__ import annotations

import math
import re
from collections import Counter

from src.text.token_library import TokenLibrary


class LearnedChunkComposer:
    """A compact associative policy over response chunks and transitions."""

    def __init__(
        self,
        max_postings_per_term: int = 20_000,
        max_query_terms: int = 24,
    ):
        self.chunks: list[str] = []
        self.chunk_to_id: dict[str, int] = {}
        self.term_chunks: dict[str, Counter] = {}
        self.start_chunks: dict[str, Counter] = {}
        self.transitions: dict[int, Counter] = {}
        self.end_counts: Counter = Counter()
        self.chunk_counts: Counter = Counter()
        self.max_postings_per_term = max_postings_per_term
        self.max_query_terms = max_query_terms
        self.pairs = 0

    @staticmethod
    def _terms(text: str) -> set[str]:
        stop = {
            "a", "an", "the", "is", "are", "was", "were", "of", "to",
            "in", "on", "at", "for", "and", "or", "what", "who", "where",
            "when", "why", "how", "which", "do", "does", "did", "i", "you",
            "it", "that", "this", "with", "be", "as", "by", "from",
        }
        return {
            token for token in TokenLibrary.tokenize(text)
            if token not in stop
        }

    @staticmethod
    def split_chunks(text: str) -> list[str]:
        chunks = [
            part.strip()
            for part in re.split(r"(?<=[.!?])\s+|(?<=[;:])\s+", text.strip())
            if part.strip()
        ]
        return chunks[:32]

    def _chunk_id(self, chunk: str) -> int:
        normalised = " ".join(chunk.split())
        chunk_id = self.chunk_to_id.get(normalised)
        if chunk_id is None:
            chunk_id = len(self.chunks)
            self.chunk_to_id[normalised] = chunk_id
            self.chunks.append(normalised)
        return chunk_id

    def learn(self, prompt: str, response: str) -> None:
        chunks = self.split_chunks(response)
        if not chunks:
            return
        ids = [self._chunk_id(chunk) for chunk in chunks]
        terms = self._terms(prompt)
        for chunk_id in ids:
            self.chunk_counts[chunk_id] += 1
            for term in terms:
                posting = self.term_chunks.setdefault(term, Counter())
                if (
                    chunk_id not in posting
                    and len(posting) >= self.max_postings_per_term
                ):
                    continue
                posting[chunk_id] += 1
        for term in terms:
            posting = self.start_chunks.setdefault(term, Counter())
            if (
                ids[0] in posting
                or len(posting) < self.max_postings_per_term
            ):
                posting[ids[0]] += 1
        for left, right in zip(ids, ids[1:]):
            self.transitions.setdefault(left, Counter())[right] += 1
        self.end_counts[ids[-1]] += 1
        self.pairs += 1

    def _candidate_scores(self, terms: set[str], starts: bool) -> Counter:
        index = self.start_chunks if starts else self.term_chunks
        scores = Counter()
        for term in terms:
            posting = index.get(term)
            if not posting:
                continue
            inverse_frequency = 1.0 / math.sqrt(len(posting))
            for chunk_id, count in posting.items():
                scores[chunk_id] += inverse_frequency * math.log1p(count)
        return scores

    def generate(
        self,
        prompt: str,
        evidence: list[str] | None = None,
        max_chunks: int = 4,
    ) -> str:
        if not self.chunks:
            return ""
        terms = self._terms(" ".join([prompt, *(evidence or [])]))
        terms = set(sorted(
            terms,
            key=lambda term: len(self.term_chunks.get(term, ())) or 10**9,
        )[:self.max_query_terms])
        selected = []
        previous = None
        topical_scores = self._candidate_scores(terms, starts=False)
        start_scores = self._candidate_scores(terms, starts=True)
        for step in range(max_chunks):
            if step == 0:
                scores = start_scores or topical_scores
            else:
                transitions = self.transitions.get(previous, {})
                scores = Counter({
                    chunk_id: topical_scores.get(chunk_id, 0.0)
                    + 2.0 * math.log1p(count)
                    for chunk_id, count in transitions.items()
                })
                if not scores:
                    break
            for chunk_id in selected:
                scores.pop(chunk_id, None)
            if not scores:
                break
            chosen, score = max(scores.items(), key=lambda item: item[1])
            if score <= 0:
                break
            selected.append(chosen)
            previous = chosen
            transition_total = sum(self.transitions.get(chosen, {}).values())
            if (
                step > 0
                and self.end_counts[chosen] > transition_total
            ):
                break
        return " ".join(self.chunks[chunk_id] for chunk_id in selected)

    def get_state(self) -> dict:
        def counters(mapping):
            return {
                key: dict(value) for key, value in mapping.items()
            }
        return {
            "chunks": self.chunks,
            "term_chunks": counters(self.term_chunks),
            "start_chunks": counters(self.start_chunks),
            "transitions": counters(self.transitions),
            "end_counts": dict(self.end_counts),
            "chunk_counts": dict(self.chunk_counts),
            "max_postings_per_term": self.max_postings_per_term,
            "max_query_terms": self.max_query_terms,
            "pairs": self.pairs,
        }

    @classmethod
    def from_state(cls, state: dict) -> "LearnedChunkComposer":
        composer = cls(
            state.get("max_postings_per_term", 20_000),
            state.get("max_query_terms", 24),
        )
        composer.chunks = state.get("chunks", [])
        composer.chunk_to_id = {
            chunk: index for index, chunk in enumerate(composer.chunks)
        }
        composer.term_chunks = {
            term: Counter(counts)
            for term, counts in state.get("term_chunks", {}).items()
        }
        composer.start_chunks = {
            term: Counter(counts)
            for term, counts in state.get("start_chunks", {}).items()
        }
        composer.transitions = {
            int(chunk_id): Counter(counts)
            for chunk_id, counts in state.get("transitions", {}).items()
        }
        composer.end_counts = Counter(state.get("end_counts", {}))
        composer.chunk_counts = Counter(state.get("chunk_counts", {}))
        composer.pairs = state.get("pairs", 0)
        return composer
