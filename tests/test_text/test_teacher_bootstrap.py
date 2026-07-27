import json
import sys
from types import SimpleNamespace

import torch

from experiments.evaluate_teacher_bootstrap import paired_bootstrap_interval
from experiments.teacher_bootstrap import learning_state_rows
from src.text import (
    BioAIDialogueAgent,
    OllamaBootstrapTeacher,
    SequenceCandidateScorer,
    TeacherRefusal,
    TeacherRecord,
    apply_bootstrap,
    build_teacher_records,
    label_conversations,
    load_bootstrap,
    read_teacher_records,
    save_bootstrap,
    train_bootstrap,
    write_teacher_records,
)


def record():
    return TeacherRecord(
        prompt="How is bread baked?",
        evidence=["Mix flour and water. Bake the dough until golden."],
        candidates=[
            {
                "text": "Mix flour and water. Bake the dough until golden.",
                "kind": "human_preferred",
                "source_ids": [0],
            },
            {
                "text": "Bread is always cooked by freezing it.",
                "kind": "hard_negative",
                "source_ids": [],
            },
            {
                "text": "Mix flour and water.",
                "kind": "incomplete",
                "source_ids": [0],
            },
        ],
        preferred_index=0,
        rejection_reasons=[
            "Contradicts the evidence.",
            "Omits the baking step.",
        ],
        required_facts=["Flour and water are mixed.", "The dough is baked."],
        semantic_concepts=["bread", "baking"],
        chunks=["Mix flour and water.", "Bake the dough until golden."],
    )


def test_teacher_records_round_trip(tmp_path):
    path = tmp_path / "records.jsonl"
    assert write_teacher_records(path, [record()]) == 1

    restored = read_teacher_records(path)

    assert restored == [record()]


def test_fake_teacher_labels_conversations_end_to_end():
    class FakeTeacher:
        def label(self, prompt, response, evidence):
            value = record()
            value.prompt = prompt
            value.candidates[0]["text"] = response
            value.chunks = [response]
            value.evidence = evidence
            return value

    records = label_conversations(FakeTeacher(), [
        ("Question", "Grounded answer.", ["Grounded evidence."])
    ])

    assert records[0].prompt == "Question"
    assert records[0].preferred["text"] == "Grounded answer."


def test_teacher_record_build_resumes_and_preserves_prior_work(tmp_path):
    class FakeTeacher:
        def label(self, prompt, response, evidence):
            value = record()
            value.prompt = prompt
            value.candidates[0]["text"] = response
            value.chunks = [response]
            value.evidence = evidence
            return value

    conversations = [
        (f"Question {index}", f"Answer {index}.", [f"Evidence {index}."])
        for index in range(3)
    ]
    path = tmp_path / "records.jsonl"
    first = build_teacher_records(
        FakeTeacher(), conversations, path, target_count=2
    )
    resumed = build_teacher_records(
        FakeTeacher(), conversations, path, target_count=3, resume=True
    )

    assert len(first) == 2
    assert [item.prompt for item in resumed] == [
        "Question 0", "Question 1", "Question 2"
    ]
    assert len(read_teacher_records(path)) == 3


def test_teacher_failure_keeps_each_completed_record(tmp_path):
    class FailingTeacher:
        def label(self, prompt, response, evidence):
            if prompt == "Question 1":
                raise ValueError("teacher failure")
            value = record()
            value.prompt = prompt
            return value

    path = tmp_path / "records.jsonl"
    conversations = [
        ("Question 0", "Answer 0.", []),
        ("Question 1", "Answer 1.", []),
    ]
    try:
        build_teacher_records(
            FailingTeacher(), conversations, path, target_count=2
        )
    except ValueError as exc:
        assert str(exc) == "teacher failure"
    else:
        raise AssertionError("teacher failure was not propagated")

    assert [item.prompt for item in read_teacher_records(path)] == [
        "Question 0"
    ]


def test_exhausted_teacher_failure_can_use_replacement_sample(tmp_path):
    class SometimesFailingTeacher:
        def label(self, prompt, response, evidence):
            if prompt == "Question 0":
                raise ValueError("invalid output")
            value = record()
            value.prompt = prompt
            return value

    records = build_teacher_records(
        SometimesFailingTeacher(),
        [
            ("Question 0", "Answer 0.", []),
            ("Question 1", "Answer 1.", []),
        ],
        tmp_path / "records.jsonl",
        target_count=1,
        skip_failures=True,
    )

    assert [item.prompt for item in records] == ["Question 1"]


def test_teacher_refusal_uses_later_source_and_resume_crosses_gap(tmp_path):
    class RefusingTeacher:
        def label(self, prompt, response, evidence):
            if prompt == "Question 1":
                raise TeacherRefusal("declined")
            value = record()
            value.prompt = prompt
            return value

    conversations = [
        (f"Question {index}", f"Answer {index}.", [])
        for index in range(3)
    ]
    path = tmp_path / "records.jsonl"
    records = build_teacher_records(
        RefusingTeacher(), conversations, path, target_count=2
    )
    resumed = build_teacher_records(
        RefusingTeacher(),
        conversations,
        path,
        target_count=2,
        resume=True,
    )

    assert [item.prompt for item in records] == [
        "Question 0", "Question 2"
    ]
    assert resumed == records


def test_learning_state_can_supply_teacher_conversations(tmp_path):
    path = tmp_path / "learning.pt"
    torch.save({
        "prompts": {"texts": ["First prompt", "Second prompt"]},
        "responses": ["First response", "Second response"],
    }, path)

    rows = list(learning_state_rows(path, limit=1))

    assert rows == [
        ("First prompt", "First response", ["First response"])
    ]


def test_learning_state_skips_empty_language_records(tmp_path):
    path = tmp_path / "learning.pt"
    torch.save({
        "prompts": {"texts": ["Empty answer", "Usable prompt"]},
        "responses": ["   ", "Usable response"],
    }, path)

    assert list(learning_state_rows(path, limit=1)) == [
        ("Usable prompt", "Usable response", ["Usable response"])
    ]


def test_learning_state_evenly_samples_full_eligible_range(tmp_path):
    path = tmp_path / "learning.pt"
    torch.save({
        "prompts": {"texts": [f"Prompt {index}" for index in range(5)]},
        "responses": [f"Response {index}" for index in range(5)],
    }, path)

    rows = list(learning_state_rows(path, limit=3, sampling="even"))

    assert [prompt for prompt, _, _ in rows] == [
        "Prompt 0", "Prompt 2", "Prompt 4"
    ]


def test_paired_bootstrap_confirms_consistent_improvement():
    result = paired_bootstrap_interval([0.1] * 20, samples=500)

    assert result["mean_points"] == 10.0
    assert result["ci95_low_points"] > 0
    assert result["probability_improvement"] == 1.0


def test_bootstrap_trains_and_exports_teacher_free_policy(tmp_path):
    state = train_bootstrap([record()], epochs=3)
    path = tmp_path / "bootstrap.pt"
    save_bootstrap(path, state)
    restored = load_bootstrap(path)
    scorer = SequenceCandidateScorer.from_state(
        restored["candidate_scorer"]
    )

    ranked = scorer.rank(
        record().prompt, record().candidates, record().evidence
    )
    assert ranked[0]["text"] == record().preferred["text"]
    assert restored["candidate_scorer"]["updates"] == 6
    assert restored["teacher_free"] is True
    assert "model" not in restored
    initial = SequenceCandidateScorer().weights
    trained = restored["candidate_scorer"]["weights"]
    assert trained[2:5] == initial[2:5]


def test_bootstrap_installs_frozen_base_and_runs_without_teacher():
    state = train_bootstrap([record()])
    agent = BioAIDialogueAgent(vsa_dim=64)

    apply_bootstrap(agent, state)

    assert agent.chunk_composer.generate(record().prompt) == (
        "Mix flour and water. Bake the dough until golden."
    )
    assert agent.preference_scorer.updates == 0
    assert (
        agent.preference_scorer.base_scorer.weights
        == agent.candidate_scorer.weights
    )
    assert agent.response_model is None


def test_teacher_json_parser_accepts_fenced_json():
    parsed = OllamaBootstrapTeacher._json_object(
        '```json\n{"preferred_index": 0}\n```'
    )
    assert parsed == {"preferred_index": 0}


def test_ollama_teacher_falls_back_to_deterministic_chunks(monkeypatch):
    labels = {
        "alternatives": [
            {"text": "Incomplete.", "reason": "Incomplete answer."},
            {"text": "Incorrect.", "reason": "Contradicts evidence."},
        ],
        "required_facts": ["A fact."],
        "semantic_concepts": ["concept"],
        "chunks": [],
    }
    monkeypatch.setitem(
        sys.modules,
        "ollama",
        SimpleNamespace(chat=lambda *args, **kwargs: {
            "message": {"content": json.dumps(labels)}
        }),
    )

    result = OllamaBootstrapTeacher("fake").label(
        "Question?", "First sentence. Second sentence.", ["Evidence."]
    )

    assert result.chunks == ["First sentence.", "Second sentence."]


def test_ollama_teacher_normalises_structured_chunk_objects(monkeypatch):
    labels = {
        "alternatives": [
            {"text": "Incomplete.", "reason": "Incomplete answer."},
            {"text": "Incorrect.", "reason": "Contradicts evidence."},
        ],
        "required_facts": [{"fact": "A fact."}],
        "semantic_concepts": [{"concept": "memory"}],
        "chunks": [{"text": "First chunk."}, {"content": "Second chunk."}],
    }
    monkeypatch.setitem(
        sys.modules,
        "ollama",
        SimpleNamespace(chat=lambda *args, **kwargs: {
            "message": {"content": json.dumps(labels)}
        }),
    )

    result = OllamaBootstrapTeacher("fake").label(
        "Question?", "Human answer.", ["Evidence."]
    )

    assert result.chunks == ["First chunk.", "Second chunk."]
    assert result.required_facts == ["A fact."]
    assert result.semantic_concepts == ["memory"]


def test_ollama_teacher_exposes_refusal_without_retrying(monkeypatch):
    calls = []
    monkeypatch.setitem(
        sys.modules,
        "ollama",
        SimpleNamespace(chat=lambda *args, **kwargs: (
            calls.append(1)
            or {
                "message": {
                    "content": json.dumps({
                        "response": {
                            "status": "refusal",
                            "message": "declined",
                        }
                    })
                }
            }
        )),
    )

    try:
        OllamaBootstrapTeacher("fake").label(
            "Question?", "Answer.", ["Evidence."]
        )
    except TeacherRefusal as exc:
        assert str(exc) == "declined"
    else:
        raise AssertionError("teacher refusal was not exposed")
    assert len(calls) == 1


def test_invalid_teacher_record_is_rejected():
    value = record()
    value.rejection_reasons = []

    try:
        value.validate()
    except ValueError as exc:
        assert "rejection reason" in str(exc)
    else:
        raise AssertionError("invalid teacher record was accepted")
