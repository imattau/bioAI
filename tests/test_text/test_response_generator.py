from src.text import OllamaResponseGenerator


MEMORIES = [{
    "id": 2,
    "text": "The optic nerve carries visual signals from the retina to the brain.",
}]


def test_grounding_guard_accepts_supported_composition():
    assert OllamaResponseGenerator.is_grounded(
        "Visual signals travel through the optic nerve [memory:2].",
        "How do visual signals travel?",
        MEMORIES,
    )


def test_grounding_guard_rejects_unsupported_details():
    assert not OllamaResponseGenerator.is_grounded(
        "The optic nerve is a bundle of fibers connected to the visual cortex.",
        "How does information reach the brain?",
        MEMORIES,
    )


def test_extractive_fallback_is_cited():
    assert OllamaResponseGenerator.extractive_fallback(MEMORIES) == (
        "According to [memory:2], The optic nerve carries visual signals "
        "from the retina to the brain."
    )
