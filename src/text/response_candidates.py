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


class EpisodicPreferenceScorer:
    """Retrieves immutable preference episodes for context-local adaptation."""

    def __init__(
        self,
        base_scorer: SequenceCandidateScorer | None = None,
        adaptation_rate: float = 0.5,
        retrieval_limit: int = 32,
    ):
        self.base_scorer = base_scorer or SequenceCandidateScorer()
        self.adaptation_rate = adaptation_rate
        self.retrieval_limit = retrieval_limit
        self.prompt_memory = TokenLibrary(vector_cache_size=0)
        self.feature_deltas: list[list[float]] = []
        self.updates = 0

    def learn_preference(
        self,
        prompt: str,
        preferred: dict,
        rejected: dict,
        evidence: list[str],
    ) -> None:
        positive = self.base_scorer.features(prompt, preferred, evidence)
        negative = self.base_scorer.features(prompt, rejected, evidence)
        self.prompt_memory.add(prompt)
        self.feature_deltas.append([
            left - right for left, right in zip(positive, negative)
        ])
        self.updates += 1

    def contextual_weights(self, prompt: str) -> list[float]:
        weights = list(self.base_scorer.weights)
        matches = self.prompt_memory.candidate_ids(
            prompt, limit=self.retrieval_limit
        )
        if not matches:
            return weights
        query_terms = self.prompt_memory.semantic_terms(prompt)
        total_weight = 0.0
        accumulated = [0.0] * len(weights)
        for rank, (episode_id, _) in enumerate(matches):
            episode_terms = self.prompt_memory.semantic_terms(
                self.prompt_memory.texts[episode_id]
            )
            similarity = (
                len(query_terms & episode_terms)
                / max(1, len(query_terms | episode_terms))
            )
            relevance = max(0.05, similarity) / (1 + 0.1 * rank)
            total_weight += relevance
            for index, delta in enumerate(self.feature_deltas[episode_id]):
                accumulated[index] += relevance * delta
        for index in range(len(weights)):
            weights[index] += (
                self.adaptation_rate
                * accumulated[index] / max(total_weight, 1e-9)
            )
        return weights

    def score(
        self,
        prompt: str,
        candidate: dict,
        evidence: list[str],
        weights: list[float] | None = None,
    ) -> float:
        values = self.base_scorer.features(prompt, candidate, evidence)
        return sum(
            weight * value
            for weight, value in zip(
                weights or self.contextual_weights(prompt), values
            )
        )

    def rank(self, prompt: str, candidates: list[dict], evidence: list[str]):
        weights = self.contextual_weights(prompt)
        ranked = [
            {
                **candidate,
                "score": self.score(
                    prompt, candidate, evidence, weights=weights
                ),
            }
            for candidate in candidates
        ]
        return sorted(ranked, key=lambda item: item["score"], reverse=True)

    def get_state(self) -> dict:
        return {
            "base_scorer": self.base_scorer.get_state(),
            "adaptation_rate": self.adaptation_rate,
            "retrieval_limit": self.retrieval_limit,
            "prompt_memory": self.prompt_memory.get_state(),
            "feature_deltas": self.feature_deltas,
            "updates": self.updates,
        }

    @classmethod
    def from_state(cls, state: dict) -> "EpisodicPreferenceScorer":
        scorer = cls(
            SequenceCandidateScorer.from_state(state["base_scorer"]),
            state.get("adaptation_rate", 0.5),
            state.get("retrieval_limit", 32),
        )
        scorer.prompt_memory = TokenLibrary.from_state(state["prompt_memory"])
        scorer.feature_deltas = state.get("feature_deltas", [])
        scorer.updates = state.get("updates", len(scorer.feature_deltas))
        return scorer
