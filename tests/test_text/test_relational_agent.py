"""Tests for RelationalMemory wired into BioAIDialogueAgent.process_turn.

Covers src/text/agent.py's _answer_relational_query / resolve() integration —
see RELATIONAL_MEMORY.md for the design this implements.
"""

import tempfile
from pathlib import Path

from src.text import BioAIDialogueAgent


def test_confident_single_relation_answer():
    agent = BioAIDialogueAgent(vsa_dim=64)
    agent.process_turn("The capital of France is Paris")
    result = agent.process_turn("What is the capital of France?")

    assert result["retrieval_accepted"]
    assert result["response"] == "The capital of France is Paris"
    assert result["response_mode"] == "relational_reasoning"
    assert not result["ambiguous"]


def test_genuine_tie_is_surfaced_honestly_not_guessed():
    # parse_relation_query's "what/who is X" reads X as the SUBJECT being
    # described (relation "is"), so the tie needs to be on that vocabulary —
    # not "in" (that's parse_path_query/reason_path's separate phrasing,
    # which already has its own tie-abstention, untouched by this wiring).
    agent = BioAIDialogueAgent(vsa_dim=64)
    agent.process_turn("Pluto is a planet")
    agent.process_turn("Pluto is a dwarf planet")
    result = agent.process_turn("What is Pluto?")

    assert result["response_mode"] == "relational_ambiguous"
    assert result["ambiguous"]
    assert set(result["ambiguous_candidates"]) >= {"a planet", "a dwarf planet"}
    assert "a planet" in result["response"].lower()
    assert "a dwarf planet" in result["response"].lower()


def test_multi_clue_resolve_narrows_a_tie():
    agent = BioAIDialogueAgent(vsa_dim=64)
    agent.process_turn("Cat is a mammal")
    agent.process_turn("Dog is a mammal")
    agent.process_turn("Cat is furry")
    agent.process_turn("Dog is furry")
    agent.process_turn("Dog is loyal")

    # Two clues that both cat and dog share: still genuinely tied, no way
    # to pick between them from this evidence alone.
    tied = agent.process_turn("Who is a mammal and is furry?")
    assert tied["response_mode"] == "relational_ambiguous"
    assert set(tied["ambiguous_candidates"]) >= {"cat", "dog"}

    # A clue only ONE of them shares narrows it via resolve()'s candidate
    # intersection — not just the last clue winning on its own.
    result = agent.process_turn("Who is a mammal and is loyal?")
    assert result["response_mode"] == "relational_reasoning"
    assert result["response"] == "Dog is loyal"
    assert not result["ambiguous"]


def test_never_asserted_subject_does_not_hallucinate():
    """'Who is X' is genuinely ambiguous between 'describe X' and 'which
    entity has property X' — parse_relation_query always takes the former
    reading. If X ('a mammal') was never actually asserted as a subject,
    Hopfield recall would otherwise return a soft nearest-match guess
    instead of admitting it has no data; the agent must decline.
    """
    agent = BioAIDialogueAgent(vsa_dim=64)
    agent.process_turn("Cat is a mammal")
    agent.process_turn("Dog is a mammal")
    result = agent.process_turn("Who is a mammal?")

    assert result["response_mode"] == "abstention"
    assert not result["retrieval_accepted"]


def test_relational_survives_agent_save_load():
    agent = BioAIDialogueAgent(vsa_dim=64)
    agent.process_turn("The capital of France is Paris")
    path = Path(tempfile.mktemp(suffix=".pt"))
    try:
        agent.save(path)
        restored = BioAIDialogueAgent.load(path)
        result = restored.process_turn("What is the capital of France?")
        assert result["response"] == "The capital of France is Paris"
        assert result["response_mode"] == "relational_reasoning"
    finally:
        path.unlink(missing_ok=True)


def test_relational_survives_save_load_and_accepts_new_facts():
    agent = BioAIDialogueAgent(vsa_dim=64)
    agent.process_turn("Cat is a mammal")
    path = Path(tempfile.mktemp(suffix=".pt"))
    try:
        agent.save(path)
        restored = BioAIDialogueAgent.load(path)
        restored.process_turn("Dog is a mammal")
        result = restored.process_turn("What is Dog?")
        assert result["response"] == "Dog is a mammal"
    finally:
        path.unlink(missing_ok=True)


def test_ambiguous_candidates_exclude_unrelated_entities():
    """Regression test: complete_detailed's top-k candidates are drawn from
    the whole shared entity-vector namespace (every entity ever seen, across
    every relation), so an unrelated low-score entity from a completely
    different relation could ride along into the top-k in a small
    vocabulary and get named as a plausible answer. The candidates shown to
    the user must come from ground_truth_ambiguity (the literal stored
    triples), which can't contain that noise.
    """
    agent = BioAIDialogueAgent(vsa_dim=64)
    agent.process_turn("The capital of France is Paris")
    agent.process_turn("The capital of Japan is Tokyo")
    agent.process_turn("Pluto is a planet")
    agent.process_turn("Pluto is a dwarf planet")

    result = agent.process_turn("What is Pluto?")

    assert result["response_mode"] == "relational_ambiguous"
    assert set(result["ambiguous_candidates"]) == {"a planet", "a dwarf planet"}
