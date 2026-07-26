from experiments.replay_conversation_benchmark import extract_pairs, summarise


def test_extract_pairs():
    report = {"records": [
        {"phase": "learn", "user": "fact one"},
        {"phase": "question", "user": "question one", "expected": "fact one"},
        {"phase": "learn", "user": "fact two"},
        {"phase": "question", "user": "question two", "expected": "fact two"},
    ]}
    assert extract_pairs(report) == [
        ("fact one", "question one", "fact one"),
        ("fact two", "question two", "fact two"),
    ]
