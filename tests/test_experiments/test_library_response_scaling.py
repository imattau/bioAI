from experiments.library_response_scaling import (
    compose_response,
    conversation_pairs,
    token_f1,
    wildchat_pairs,
)
from src.text import TokenLibrary


def test_token_f1():
    assert token_f1("the blue sky", "the sky is blue") > 0.8
    assert token_f1("cats", "aircraft") == 0


def test_composition_uses_retrieved_sentences():
    result = compose_response(
        "How do I bake bread?",
        [0, 1],
        [
            "Mix flour and water. Let the dough rise.",
            "Preheat the oven before baking.",
        ],
    )
    assert result
    assert all(
        sentence in " ".join([
            "Mix flour and water. Let the dough rise.",
            "Preheat the oven before baking.",
        ])
        for sentence in [part + "." for part in result.split(". ")[:-1]]
    )


def test_extracts_parent_assistant_pairs():
    rows = [
        {
            "message_id": "p", "parent_id": None, "role": "prompter",
            "text": "Question", "lang": "en", "deleted": False, "rank": None,
        },
        {
            "message_id": "a", "parent_id": "p", "role": "assistant",
            "text": "Answer", "lang": "en", "deleted": False, "rank": 0,
        },
    ]
    assert conversation_pairs(rows) == [("Question", "Answer")]


def test_library_size_includes_prompts_and_responses():
    from experiments.library_response_scaling import evaluate

    prompts = TokenLibrary()
    replies = TokenLibrary()
    prompts.add("how are you")
    replies.add("I am doing very well today")
    result = evaluate(
        prompts, replies, ["I am doing very well today"],
        [("how are you", "I am doing very well today")],
    )
    assert result["library_tokens"] == 9


def test_extracts_safe_wildchat_turns():
    row = {
        "language": "English", "toxic": False, "redacted": False,
        "conversation": [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer"},
            {"role": "user", "content": "Follow-up"},
            {"role": "assistant", "content": "Second answer"},
        ],
    }
    assert wildchat_pairs(row) == [
        ("Question", "Answer"),
        ("Follow-up", "Second answer"),
    ]
    row["toxic"] = True
    assert wildchat_pairs(row) == []
