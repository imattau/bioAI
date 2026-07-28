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


def test_multi_clue_resolve_narrows_a_tie_via_explicit_clue():
    agent = BioAIDialogueAgent(vsa_dim=64)
    agent.process_turn("Cat is a mammal")
    agent.process_turn("Dog is a mammal")
    agent.process_turn("Cat is furry")
    agent.process_turn("Dog is furry")

    # Nothing else is known yet about cat/dog beyond mammal+furry (both
    # share both) -- genuinely tied, nothing to escalate to either.
    tied = agent.process_turn("Who is a mammal and is furry?")
    assert tied["response_mode"] == "relational_ambiguous"
    assert set(tied["ambiguous_candidates"]) >= {"cat", "dog"}

    agent.process_turn("Dog is loyal")

    # A clue only ONE of them shares narrows it via resolve()'s explicit
    # candidate intersection — not just the last clue winning on its own.
    result = agent.process_turn("Who is a mammal and is loyal?")
    assert result["response_mode"] == "relational_reasoning"
    assert result["response"] == "Dog is loyal"
    assert not result["ambiguous"]


def test_ambiguous_multi_clue_escalates_to_other_known_facts():
    """When the explicitly stated clues alone leave it tied, the agent
    should escalate via resolve_auto to other already-known facts about
    the tied candidates instead of giving up at "ambiguous" -- the
    basal-ganglia relevance-selection wiring (RELATIONAL_MEMORY.md SS2.7).
    Same setup as the "still tied" case above, except "Dog is loyal" is
    already known (just not mentioned in this question) -- the agent must
    find and use it on its own.
    """
    agent = BioAIDialogueAgent(vsa_dim=64)
    agent.process_turn("Cat is a mammal")
    agent.process_turn("Dog is a mammal")
    agent.process_turn("Cat is furry")
    agent.process_turn("Dog is furry")
    agent.process_turn("Dog is loyal")

    result = agent.process_turn("Who is a mammal and is furry?")
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


def test_multi_clue_property_containing_and_is_not_split():
    """Regression test (found via experiments/llm_relational_benchmark.py,
    not a hand-crafted example): a property whose own text contains the
    word "and" (e.g. "loyal and affectionate") must not be split into two
    separate clues just because naive "and"-splitting can't tell a
    conjunction from the word "and" occurring inside a value. The known-
    value merge pass in _parse_multi_clue_identity_question should re-join
    the over-split pieces using the actual stored vocabulary.
    """
    agent = BioAIDialogueAgent(vsa_dim=64)
    agent.process_turn("Dog is four legs.")
    agent.process_turn("Cat is four legs.")
    agent.process_turn("Dog is loyal and affectionate.")

    clues = agent._parse_multi_clue_identity_question(
        "Who is four legs and is loyal and affectionate?"
    )
    assert clues == [("is", "four legs"), ("is", "loyal and affectionate")]

    result = agent.process_turn(
        "Who is four legs and is loyal and affectionate?"
    )
    assert result["response_mode"] == "relational_reasoning"
    assert result["response"] == "Dog is loyal and affectionate."


def test_multi_clue_unknown_property_falls_back_to_word_split():
    """When the property text isn't a previously-stored value at all (so
    there's no known vocabulary to merge against), splitting still falls
    back to the naive word-level behavior rather than failing outright --
    the merge pass can only correct segmentation using what's actually
    recorded, it can't discover a novel multi-word property it's never
    seen (documented limitation, not a bug).
    """
    agent = BioAIDialogueAgent(vsa_dim=64)
    clues = agent._parse_multi_clue_identity_question(
        "Who is brand new and is never seen before?"
    )
    assert clues == [("is", "brand new"), ("is", "never seen before")]
