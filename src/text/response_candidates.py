"""Candidate generation and whole-response ranking for BioAI."""

from __future__ import annotations

import math
import re

from src.text.chunk_composer import LearnedChunkComposer
from src.text.token_library import TokenLibrary


class FixedSpliceCandidateGenerator:
    def __init__(self, max_candidates: int = 24):
        self.max_candidates = max_candidates

    @staticmethod
    def _sentences(text: str) -> list[str]:
        return [
            value.strip()
            for value in re.split(r"(?<=[.!?])\s+", text)
            if value.strip()
        ]

    def generate(
        self,
        prompt: str,
        responses: list[str],
        composer: LearnedChunkComposer | None = None,
    ) -> list[dict]:
        candidates = []
        for source_id, response in enumerate(responses):
            candidates.append({
                "text": response.strip(),
                "kind": "complete",
                "source_ids": [source_id],
            })
            sentences = self._sentences(response)
            for length in range(1, min(3, len(sentences)) + 1):
                candidates.append({
                    "text": " ".join(sentences[:length]),
                    "kind": "single_source_splice",
                    "source_ids": [source_id],
                })
        first_sentences = [
            self._sentences(response)[0]
            for response in responses if self._sentences(response)
        ]
        for length in range(2, min(4, len(first_sentences)) + 1):
            candidates.append({
                "text": " ".join(first_sentences[:length]),
                "kind": "cross_source_splice",
                "source_ids": list(range(length)),
            })
        if composer is not None:
            learned = composer.generate(prompt, evidence=responses)
            if learned:
                candidates.append({
                    "text": learned,
                    "kind": "learned_composition",
                    "source_ids": list(range(len(responses))),
                })
        unique = []
        seen = set()
        for candidate in candidates:
            text = " ".join(candidate["text"].split())
            if not text or text in seen:
                continue
            seen.add(text)
            candidate["text"] = text
            unique.append(candidate)
            if len(unique) == self.max_candidates:
                break
        return unique


class SequenceCandidateScorer:
    """Online pairwise scorer over features of complete response candidates."""

    FEATURE_NAMES = (
        "query_overlap",
        "evidence_coverage",
        "single_source",
        "complete_response",
        "retrieval_priority",
        "transition_coherence",
        "length_fit",
        "repetition",
    )

    def __init__(self, learning_rate: float = 0.05):
        self.weights = [0.8, 1.2, 0.6, 1.5, 1.0, 0.5, 0.2, -0.8]
        self.learning_rate = learning_rate
        self.updates = 0

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return TokenLibrary.tokenize(text)

    def features(self, prompt: str, candidate: dict, evidence: list[str]):
        prompt_terms = set(self._tokens(prompt))
        candidate_tokens = self._tokens(candidate["text"])
        candidate_terms = set(candidate_tokens)
        query_overlap = (
            len(prompt_terms & candidate_terms) / len(prompt_terms)
            if prompt_terms else 0.0
        )
        evidence_terms = set(self._tokens(" ".join(evidence)))
        evidence_coverage = (
            len(candidate_terms & evidence_terms) / len(candidate_terms)
            if candidate_terms and evidence_terms else 0.0
        )
        single_source = float(len(candidate["source_ids"]) <= 1)
        complete_response = float(
            candidate.get("kind") in {"complete", "human_preferred"}
        )
        retrieval_priority = (
            1.0 / (1 + min(candidate["source_ids"]))
            if candidate["source_ids"] else 0.0
        )
        sentences = FixedSpliceCandidateGenerator._sentences(candidate["text"])
        overlaps = []
        for left, right in zip(sentences, sentences[1:]):
            left_terms, right_terms = set(self._tokens(left)), set(self._tokens(right))
            overlaps.append(
                len(left_terms & right_terms) / max(1, len(left_terms | right_terms))
            )
        transition_coherence = (
            1.0 if len(sentences) <= 1 else sum(overlaps) / len(overlaps)
        )
        length_ratio = len(candidate_tokens) / max(1, len(self._tokens(prompt)))
        length_fit = math.exp(-abs(math.log(max(0.1, length_ratio / 2))))
        repetition = 1.0 - len(candidate_terms) / max(1, len(candidate_tokens))
        return [
            query_overlap, evidence_coverage, single_source,
            complete_response, retrieval_priority,
            transition_coherence, length_fit, repetition,
        ]

    def score(self, prompt: str, candidate: dict, evidence: list[str]) -> float:
        return sum(
            weight * value
            for weight, value in zip(
                self.weights, self.features(prompt, candidate, evidence)
            )
        )

    def rank(self, prompt: str, candidates: list[dict], evidence: list[str]):
        ranked = [
            {**candidate, "score": self.score(prompt, candidate, evidence)}
            for candidate in candidates
        ]
        return sorted(ranked, key=lambda item: item["score"], reverse=True)

    def learn_preference(
        self,
        prompt: str,
        preferred: dict,
        rejected: dict,
        evidence: list[str],
    ) -> None:
        positive = self.features(prompt, preferred, evidence)
        negative = self.features(prompt, rejected, evidence)
        margin = sum(
            weight * (left - right)
            for weight, left, right in zip(self.weights, positive, negative)
        )
        gradient = 1.0 / (1.0 + math.exp(min(30.0, margin)))
        for index, (left, right) in enumerate(zip(positive, negative)):
            self.weights[index] += (
                self.learning_rate * gradient * (left - right)
            )
        self.updates += 1

    def get_state(self) -> dict:
        return {
            "weights": self.weights,
            "learning_rate": self.learning_rate,
            "updates": self.updates,
        }

    @classmethod
    def from_state(cls, state: dict) -> "SequenceCandidateScorer":
        scorer = cls(state.get("learning_rate", 0.05))
        scorer.weights = state.get("weights", scorer.weights)
        scorer.updates = state.get("updates", 0)
        return scorer
