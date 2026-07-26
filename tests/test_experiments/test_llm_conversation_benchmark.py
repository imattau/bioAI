import pytest

from experiments.llm_conversation_benchmark import parse_pair


def test_parse_pair_accepts_either_line_order():
    assert parse_pair(
        "QUESTION: What is the capital?\nFACT: Paris is the capital."
    ) == ("Paris is the capital.", "What is the capital?")


def test_parse_pair_rejects_incomplete_output():
    with pytest.raises(ValueError, match="Could not parse"):
        parse_pair("FACT: An isolated fact")
