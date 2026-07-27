"""Offline LLM-teacher distillation for a teacher-free BioAI runtime."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Protocol

import torch

from src.text.chunk_composer import LearnedChunkComposer
from src.text.response_candidates import (
    EpisodicPreferenceScorer,
    SequenceCandidateScorer,
)
from src.text.token_library import TokenLibrary


@dataclass
class TeacherRecord:
    prompt: str
    evidence: list[str]
    candidates: list[dict]
    preferred_index: int
    rejection_reasons: list[str]
    required_facts: list[str]
    semantic_concepts: list[str]
    chunks: list[str]

    def validate(self) -> None:
        if not self.prompt.strip():
            raise ValueError("Teacher record requires a prompt")
        if len(self.candidates) < 2:
            raise ValueError("Teacher record requires at least two candidates")
        if not 0 <= self.preferred_index < len(self.candidates):
            raise ValueError("preferred_index is outside candidates")
        for candidate in self.candidates:
            if not isinstance(candidate, dict) or not str(
                candidate.get("text", "")
            ).strip():
                raise ValueError("Every candidate requires non-empty text")
        expected_rejections = len(self.candidates) - 1
        if len(self.rejection_reasons) != expected_rejections:
            raise ValueError(
                "A rejection reason is required for every rejected candidate"
            )
        if (
            not self.chunks
            or any(
                not isinstance(chunk, str) or not chunk.strip()
                for chunk in self.chunks
            )
        ):
            raise ValueError("Teacher record requires non-empty response chunks")

    @property
    def preferred(self) -> dict:
        return self.candidates[self.preferred_index]

    @classmethod
    def from_dict(cls, value: dict) -> "TeacherRecord":
        record = cls(
            prompt=value["prompt"],
            evidence=list(value.get("evidence", [])),
            candidates=list(value["candidates"]),
            preferred_index=int(value["preferred_index"]),
            rejection_reasons=list(value.get("rejection_reasons", [])),
            required_facts=list(value.get("required_facts", [])),
            semantic_concepts=list(value.get("semantic_concepts", [])),
            chunks=list(value.get("chunks", [])),
        )
        record.validate()
        return record


class BootstrapTeacher(Protocol):
    def label(
        self, prompt: str, response: str, evidence: list[str]
    ) -> TeacherRecord:
        ...


class TeacherRefusal(ValueError):
    """The teacher declined a source record; no supervision should be made."""


class OllamaBootstrapTeacher:
    """Requests structured supervision from Ollama during offline bootstrap."""

    SYSTEM_PROMPT = """You create training labels for a retrieval-based AI.
Return one JSON object only, with these keys:
alternatives: 2-5 objects with text and reason. They must be plausible but
inferior answer attempts, including an incomplete answer and a hard negative.
required_facts: atomic facts needed by the preferred response.
semantic_concepts: short canonical concepts.
chunks: ordered reusable chunks that reproduce the preferred response.
Do not repeat the supplied human response as an alternative. Keep alternatives
concise and do not add unsupported facts."""

    def __init__(
        self,
        model: str,
        retries: int = 3,
        timeout_seconds: float = 60.0,
    ):
        self.model = model
        self.retries = retries
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def _json_object(text: str) -> dict:
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
            text = text.rsplit("```", 1)[0]
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end < start:
            raise ValueError("Teacher did not return a JSON object")
        return json.loads(text[start:end + 1])

    def label(
        self, prompt: str, response: str, evidence: list[str]
    ) -> TeacherRecord:
        import ollama

        client = (
            ollama.Client(timeout=self.timeout_seconds)
            if hasattr(ollama, "Client")
            else ollama
        )
        payload = {
            "prompt": prompt,
            "human_response": response,
            "retrieved_evidence": evidence,
        }
        error = None
        for _ in range(self.retries):
            try:
                result = client.chat(
                    self.model,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": json.dumps(payload)},
                    ],
                    format="json",
                    options={"temperature": 0},
                )
                labels = self._json_object(result["message"]["content"])
                response_status = labels.get("response")
                if (
                    isinstance(response_status, dict)
                    and response_status.get("status") == "refusal"
                ):
                    raise TeacherRefusal(
                        response_status.get("message", "Teacher refused")
                    )
                alternatives = labels.get("alternatives")
                if alternatives is not None:
                    candidates = [{
                        "text": response,
                        "kind": "human_preferred",
                        "source_ids": [0],
                    }]
                    rejection_reasons = []
                    for index, alternative in enumerate(alternatives):
                        candidates.append({
                            "text": alternative["text"],
                            "kind": (
                                "incomplete" if index == 0
                                else "hard_negative"
                            ),
                            "source_ids": [],
                        })
                        rejection_reasons.append(alternative["reason"])
                    labels = {
                        **labels,
                        "candidates": candidates,
                        "preferred_index": 0,
                        "rejection_reasons": rejection_reasons,
                    }
                raw_chunks = labels.get("chunks", [])
                labels["chunks"] = [
                    (
                        chunk if isinstance(chunk, str)
                        else chunk.get("text")
                        or chunk.get("chunk")
                        or chunk.get("content")
                        or ""
                    )
                    for chunk in raw_chunks
                    if isinstance(chunk, (str, dict))
                ]
                labels["chunks"] = [
                    chunk for chunk in labels["chunks"] if chunk.strip()
                ]
                if not labels["chunks"]:
                    labels["chunks"] = LearnedChunkComposer.split_chunks(
                        response
                    )
                for field, keys in (
                    ("required_facts", ("fact", "text", "content")),
                    ("semantic_concepts", ("concept", "text", "name")),
                ):
                    normalised = []
                    for value in labels.get(field, []):
                        if isinstance(value, str):
                            normalised.append(value)
                        elif isinstance(value, dict):
                            text = next(
                                (
                                    value[key] for key in keys
                                    if value.get(key)
                                ),
                                "",
                            )
                            if text:
                                normalised.append(str(text))
                    labels[field] = normalised
                record = TeacherRecord.from_dict({
                    "prompt": prompt,
                    "evidence": evidence,
                    **labels,
                })
                if response not in {
                    candidate["text"] for candidate in record.candidates
                }:
                    raise ValueError(
                        "Teacher omitted the supplied human response"
                    )
                return record
            except TeacherRefusal:
                raise
            except Exception as exc:
                error = exc
        raise ValueError(
            f"Teacher failed after {self.retries} attempts: {error}"
        ) from error


def label_conversations(
    teacher: BootstrapTeacher,
    conversations: Iterable[tuple[str, str, list[str]]],
) -> list[TeacherRecord]:
    records = []
    for prompt, response, evidence in conversations:
        record = teacher.label(prompt, response, evidence)
        record.validate()
        if record.prompt != prompt:
            raise ValueError("Teacher changed the source prompt")
        records.append(record)
    return records


def write_teacher_records(
    path: Path, records: Iterable[TeacherRecord]
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as destination:
        for record in records:
            record.validate()
            destination.write(json.dumps(asdict(record)) + "\n")
            count += 1
    return count


def build_teacher_records(
    teacher: BootstrapTeacher,
    conversations: Iterable[tuple[str, str, list[str]]],
    path: Path,
    target_count: int | None = None,
    resume: bool = False,
    progress=None,
    skipped=None,
    skip_failures: bool = False,
) -> list[TeacherRecord]:
    """Label and durably append each record, optionally resuming a prior run."""
    if target_count is not None and target_count < 1:
        raise ValueError("target_count must be at least 1")
    existing = (
        read_teacher_records(path)
        if resume and path.exists()
        else []
    )
    if target_count is not None and len(existing) > target_count:
        raise ValueError("Existing records exceed requested target")
    path.parent.mkdir(parents=True, exist_ok=True)
    iterator = iter(conversations)
    for index, record in enumerate(existing):
        for prompt, _, _ in iterator:
            if prompt == record.prompt:
                break
        else:
            raise ValueError(
                f"Resume source mismatch at record {index + 1}"
            )

    mode = "a" if existing else "w"
    records = list(existing)
    with path.open(mode, encoding="utf-8") as destination:
        for prompt, response, evidence in iterator:
            if target_count is not None and len(records) >= target_count:
                break
            try:
                record = teacher.label(prompt, response, evidence)
            except TeacherRefusal as exc:
                if skipped is not None:
                    skipped(prompt, str(exc))
                continue
            except ValueError as exc:
                if not skip_failures:
                    raise
                if skipped is not None:
                    skipped(prompt, f"invalid teacher output: {exc}")
                continue
            record.validate()
            if record.prompt != prompt:
                raise ValueError("Teacher changed the source prompt")
            destination.write(json.dumps(asdict(record)) + "\n")
            destination.flush()
            os.fsync(destination.fileno())
            records.append(record)
            if progress is not None:
                progress(len(records), target_count)
    if target_count is not None and len(records) < target_count:
        raise ValueError(
            f"Conversation source produced only {len(records)} usable records"
        )
    return records


def read_teacher_records(path: Path) -> list[TeacherRecord]:
    records = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                records.append(TeacherRecord.from_dict(json.loads(line)))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"Invalid teacher record at line {line_number}: {exc}"
                ) from exc
    return records


def train_bootstrap(
    records: Iterable[TeacherRecord],
    epochs: int = 5,
) -> dict:
    """Distil records into BioAI state without retaining the teacher."""
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    records = list(records)
    if not records:
        raise ValueError("At least one teacher record is required")
    composer = LearnedChunkComposer()
    scorer = SequenceCandidateScorer()
    prompts = TokenLibrary(vector_cache_size=0)
    responses = TokenLibrary(vector_cache_size=0)

    for record in records:
        record.validate()
        preferred_text = record.preferred["text"]
        prompts.add(record.prompt)
        responses.add(preferred_text)
        composer.learn(record.prompt, " ".join(record.chunks))

    for _ in range(epochs):
        for record in records:
            preferred = {
                **record.preferred,
                "kind": "complete",
                "source_ids": [0],
            }
            for index, rejected in enumerate(record.candidates):
                if index == record.preferred_index:
                    continue
                rejected = {
                    **rejected,
                    "kind": "complete",
                    "source_ids": [0],
                }
                scorer.learn_preference(
                    record.prompt,
                    preferred,
                    rejected,
                    record.evidence,
                )

    # Episodic adaptation receives its own immutable copy of the trained base.
    episodic_base = SequenceCandidateScorer.from_state(scorer.get_state())
    return {
        "format": "bioai_teacher_bootstrap_v1",
        "records": len(records),
        "epochs": epochs,
        "teacher_free": True,
        "prompt_library": prompts.get_state(),
        "response_library": responses.get_state(),
        "chunk_composer": composer.get_state(),
        "candidate_scorer": scorer.get_state(),
        "preference_scorer": EpisodicPreferenceScorer(
            base_scorer=episodic_base
        ).get_state(),
    }


def save_bootstrap(path: Path, state: dict) -> None:
    if state.get("format") != "bioai_teacher_bootstrap_v1":
        raise ValueError("Unsupported bootstrap state")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_bootstrap(path: Path) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("format") != "bioai_teacher_bootstrap_v1":
        raise ValueError("Unsupported bootstrap state")
    return state


def apply_bootstrap(agent, state: dict) -> None:
    """Install distilled generation policy into an existing BioAI agent."""
    if state.get("format") != "bioai_teacher_bootstrap_v1":
        raise ValueError("Unsupported bootstrap state")
    agent.chunk_composer = LearnedChunkComposer.from_state(
        state["chunk_composer"]
    )
    agent.candidate_scorer = SequenceCandidateScorer.from_state(
        state["candidate_scorer"]
    )
    agent.preference_scorer = EpisodicPreferenceScorer.from_state(
        state["preference_scorer"]
    )
