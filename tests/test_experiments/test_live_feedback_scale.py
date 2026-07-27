import sys
from types import SimpleNamespace

from experiments.live_feedback_scale_benchmark import (
    evaluate_transfer,
    load_learning_state,
    proposed_responses,
    run,
    save_learning_state,
)
from src.text import (
    EpisodicPreferenceScorer,
    LearnedChunkComposer,
    SequenceCandidateScorer,
    TokenLibrary,
)


def test_live_proposal_uses_learned_candidate_pipeline():
    prompts = TokenLibrary()
    responses = ["Mix flour and water. Bake until golden."]
    prompts.add("How do I bake bread?")
    composer = LearnedChunkComposer()
    composer.learn("How do I bake bread?", responses[0])
    static, frozen, live, episodic, evidence, candidates, ranked = proposed_responses(
        "How can bread be baked?",
        prompts,
        responses,
        composer,
        SequenceCandidateScorer(),
    )
    assert static
    assert frozen
    assert live
    assert episodic
    assert evidence == responses
    assert candidates
    assert ranked


def test_parallel_transfer_matches_serial_transfer():
    prompts = TokenLibrary()
    responses = [
        "Mix flour and water. Bake until golden.",
        "Water freezes at zero degrees Celsius.",
    ]
    prompts.add("How do I bake bread?")
    prompts.add("When does water freeze?")
    composer = LearnedChunkComposer()
    for prompt, response in zip(prompts.texts, responses):
        composer.learn(prompt, response)
    pairs = [
        ("How can bread be baked?", responses[0]),
        ("What is the freezing point of water?", responses[1]),
    ]
    scorer = SequenceCandidateScorer()
    episodic = EpisodicPreferenceScorer()
    frozen = SequenceCandidateScorer()

    serial = evaluate_transfer(
        pairs, prompts, responses, composer, scorer, episodic, frozen,
        workers=1,
    )
    parallel = evaluate_transfer(
        pairs, prompts, responses, composer, scorer, episodic, frozen,
        workers=2,
    )

    assert parallel == serial


def test_learning_state_is_atomically_replaced(tmp_path):
    path = tmp_path / "state.pt"
    save_learning_state(path, {"interactions": 1})
    save_learning_state(path, {"interactions": 2})

    assert load_learning_state(path) == {"interactions": 2}
    assert list(tmp_path.iterdir()) == [path]


def test_benchmark_resumes_without_relearning_prior_turns(
    tmp_path, monkeypatch
):
    held_out = [
        {
            "message_id": "prompt",
            "parent_id": None,
            "role": "prompter",
            "text": "What is memory?",
            "deleted": False,
            "lang": "en",
            "rank": None,
        },
        {
            "message_id": "answer",
            "parent_id": "prompt",
            "role": "assistant",
            "text": "Memory retains information.",
            "deleted": False,
            "lang": "en",
            "rank": 0,
        },
    ]
    stream = [
        {
            "language": "English",
            "toxic": False,
            "redacted": False,
            "conversation": [
                {"role": "user", "content": f"Prompt {index}"},
                {"role": "assistant", "content": f"Response {index}"},
            ],
        }
        for index in range(3)
    ]

    def load_dataset(name, split=None, streaming=False):
        if name == "OpenAssistant/oasst1":
            return {"validation": held_out}
        assert name == "allenai/WildChat-1M"
        return stream

    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(load_dataset=load_dataset),
    )
    path = tmp_path / "learning.pt"
    run(2, held_out_count=1, state_path=path)
    first = load_learning_state(path)
    assert first["responses"] == ["Response 0", "Response 1"]

    report = run(
        3,
        held_out_count=1,
        state_path=path,
        resume=True,
    )
    resumed = load_learning_state(path)

    assert resumed["responses"] == [
        "Response 0", "Response 1", "Response 2"
    ]
    assert resumed["composer"]["pairs"] == 3
    assert report["resumed_from_interactions"] == 2
