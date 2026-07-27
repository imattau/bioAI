from src.text import (
    EpisodicPreferenceScorer,
    SequenceCandidateScorer,
    TokenLibrary,
)


def item(text, kind="complete", sources=None):
    return {
        "text": text,
        "kind": kind,
        "source_ids": [0] if sources is None else sources,
    }


def test_exact_sentence_ids_distinguish_raw_text_collisions():
    library = TokenLibrary()
    first = library.add("Hello, world!")
    second = library.add("hello world")
    assert library.exact_sentence_ids("Hello, world!") == [first]
    assert library.exact_sentence_ids("hello world") == [second]
    assert library.exact_sentence_ids(
        "HELLO WORLD", require_identical_text=False
    ) == [first, second]


def test_preference_episodes_are_retained_not_overwritten():
    scorer = EpisodicPreferenceScorer()
    preferred = item("Detailed astronomy answer.")
    rejected = item("Short answer.", "single_source_splice")
    scorer.learn_preference(
        "Explain a star", preferred, rejected, ["Astronomy evidence."]
    )
    original = list(scorer.feature_deltas[0])
    for index in range(100):
        scorer.learn_preference(
            f"Cooking question {index}",
            item("Cooking response."),
            item("Brief.", "single_source_splice"),
            ["Cooking evidence."],
        )
    assert scorer.feature_deltas[0] == original
    assert scorer.prompt_memory.exact_sentence_ids("Explain a star") == [0]


def test_context_retrieves_relevant_preference():
    base = SequenceCandidateScorer()
    scorer = EpisodicPreferenceScorer(base, adaptation_rate=2.0)
    complete = item("A detailed response about stars.")
    short = item("Stars.", "single_source_splice")
    scorer.learn_preference(
        "Explain stars", complete, short, ["Stars are luminous."]
    )
    ranked = scorer.rank(
        "Please explain stars", [short, complete], ["Stars are luminous."]
    )
    assert ranked[0]["text"] == complete["text"]
    restored = EpisodicPreferenceScorer.from_state(scorer.get_state())
    assert restored.updates == 1
