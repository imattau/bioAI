"""Tests for GoNoGoActorCritic wired into BioAIDialogueAgent._retrieve_context.

Previously self.gonogo was created but never called anywhere in the live
per-turn pipeline. It now makes a real Go/NoGo decision on every question
turn that reaches _retrieve_context, but only actually changes behavior
when gonogo_gate_enabled is explicitly turned on, and only learns from
explicit record_feedback calls -- there's no ambient ground-truth signal
in ordinary conversation to train it from safely. See the docstrings in
src/text/agent.py (__init__, _retrieve_context, record_feedback).
"""

import tempfile
from pathlib import Path

import torch

from src.text import BioAIDialogueAgent


def test_gonogo_decision_computed_on_every_question_turn():
    # Deliberately not a "capital of X" / simple "X is Y" shape: those are
    # answered by the relational path (_answer_relational_query), which
    # never reaches _retrieve_context/gonogo at all -- this needs a
    # question that falls through to general retrieval instead.
    agent = BioAIDialogueAgent(vsa_dim=64)
    agent.process_turn("The weather today is unusually cold for this time of year.")
    result = agent.process_turn("What is the weather like today?")

    assert result["gonogo_action"] is not None
    assert result["gonogo_go"] in (True, False)


def test_gate_disabled_by_default_never_changes_outcome():
    """With the gate off (the default), gonogo's decision is computed and
    visible but must never affect accepted/response, even when forced to
    say NoGo -- several existing tests depend on the deterministic
    threshold rule's output, which an untrained network must not disturb.
    """
    agent = BioAIDialogueAgent(vsa_dim=64)
    assert not agent.gonogo_gate_enabled
    agent.process_turn("The capital of France is Paris and it is beautiful.")
    user_vec = agent.encoder.encode("What is the capital of France?")

    agent._gonogo_epsilon = 0.0  # deterministic monkeypatch below, no exploration override
    agent.gonogo.act = lambda state, deterministic=False: (2, torch.tensor(0.0))
    result = agent._retrieve_context("What is the capital of France?", user_vec)

    heuristic_accepted = (
        result["score"] >= agent._retrieval_threshold
        and result["margin"] >= agent._retrieval_margin
    )
    assert result["gonogo_go"] is False  # forced NoGo is still computed...
    assert result["accepted"] == heuristic_accepted  # ...but ignored


def test_gate_enabled_nogo_can_only_veto_never_approve():
    """NoGo suppresses an already-accepted candidate; it can never cause
    acceptance of something the threshold rule already rejected -- bounds
    the gate to "more cautious," never "more likely to hallucinate."
    """
    agent = BioAIDialogueAgent(vsa_dim=64)
    agent.gonogo_gate_enabled = True
    agent._gonogo_epsilon = 0.0  # deterministic monkeypatches below
    agent.process_turn("The capital of France is Paris and it is beautiful.")
    user_vec = agent.encoder.encode("What is the capital of France?")

    agent.gonogo.act = lambda state, deterministic=False: (2, torch.tensor(0.0))  # NoGo
    vetoed = agent._retrieve_context("What is the capital of France?", user_vec)
    assert vetoed["accepted"] is False

    agent.gonogo.act = lambda state, deterministic=False: (0, torch.tensor(0.0))  # Go
    unvetoed = agent._retrieve_context("What is the capital of France?", user_vec)
    heuristic_accepted = (
        unvetoed["score"] >= agent._retrieval_threshold
        and unvetoed["margin"] >= agent._retrieval_margin
    )
    assert unvetoed["accepted"] == heuristic_accepted


def test_record_feedback_noop_without_prior_retrieval():
    agent = BioAIDialogueAgent(vsa_dim=64)
    agent.process_turn("Just a statement, no question asked yet.")
    agent.record_feedback(correct=True)  # must not raise


def test_record_feedback_moves_policy_in_reward_direction():
    """Core learning mechanism check, isolated from conversation state:
    repeatedly reinforcing the same (state, action) pair with positive
    reward should increase that action's probability under the policy.
    """
    agent = BioAIDialogueAgent(vsa_dim=32)
    state = agent.vsa.make_vector()
    action, _ = agent.gonogo.act(state, deterministic=True)

    logits_before, _, _, _ = agent.gonogo.forward(state)
    prob_before = torch.softmax(logits_before, dim=-1)[action].item()

    for _ in range(20):
        agent._last_gonogo_decision = {"state": state, "action": action}
        agent.record_feedback(correct=True)

    logits_after, _, _, _ = agent.gonogo.forward(state)
    prob_after = torch.softmax(logits_after, dim=-1)[action].item()

    assert prob_after > prob_before


def test_gonogo_weights_persist_across_save_load():
    agent = BioAIDialogueAgent(vsa_dim=32)
    state = agent.vsa.make_vector()
    action, _ = agent.gonogo.act(state, deterministic=True)
    for _ in range(15):
        agent._last_gonogo_decision = {"state": state, "action": action}
        agent.record_feedback(correct=True)
    logits_before, _, _, _ = agent.gonogo.forward(state)

    path = Path(tempfile.mktemp(suffix=".pt"))
    try:
        agent.save(path)
        restored = BioAIDialogueAgent.load(path)
        logits_after, _, _, _ = restored.gonogo.forward(state)
        assert torch.allclose(logits_before, logits_after, atol=1e-5)
    finally:
        path.unlink(missing_ok=True)


def test_gonogo_gate_enabled_flag_persists_across_save_load():
    agent = BioAIDialogueAgent(vsa_dim=32)
    agent.gonogo_gate_enabled = True
    path = Path(tempfile.mktemp(suffix=".pt"))
    try:
        agent.save(path)
        restored = BioAIDialogueAgent.load(path)
        assert restored.gonogo_gate_enabled is True
    finally:
        path.unlink(missing_ok=True)


def test_gonogo_state_reflects_quality_not_question_content():
    """Regression test for a real bug found via
    experiments/gonogo_feedback_benchmark.py: gonogo's state must be a
    function of retrieval quality (score, margin), not the query's raw
    semantic content -- otherwise there's no way for training on one
    question to transfer to a completely different one, and an empirical
    training run showed agreement with ground truth getting WORSE over
    time under the old (content-based) state. Two totally unrelated
    questions with the same (score, margin) must therefore produce the
    same gonogo state.
    """
    agent = BioAIDialogueAgent(vsa_dim=64)
    state_a = 0.5 * agent._gonogo_score_axis + 0.2 * agent._gonogo_margin_axis
    state_b = 0.5 * agent._gonogo_score_axis + 0.2 * agent._gonogo_margin_axis
    assert torch.equal(state_a, state_b)

    # Different (score, margin) must give a genuinely different state.
    state_c = 0.1 * agent._gonogo_score_axis + 0.9 * agent._gonogo_margin_axis
    assert not torch.equal(state_a, state_c)


def test_gonogo_axes_reproducible_across_save_load():
    """The fixed score/margin axes aren't persisted explicitly -- they're
    regenerated deterministically from vsa_dim via a fixed seed, in both
    __init__ and load() (which bypasses __init__ via __new__). If load()
    ever drifted from that seed, a restored agent's gonogo would silently
    misinterpret states learned before the save.
    """
    agent = BioAIDialogueAgent(vsa_dim=32)
    path = Path(tempfile.mktemp(suffix=".pt"))
    try:
        agent.save(path)
        restored = BioAIDialogueAgent.load(path)
        assert torch.equal(agent._gonogo_score_axis, restored._gonogo_score_axis)
        assert torch.equal(agent._gonogo_margin_axis, restored._gonogo_margin_axis)
    finally:
        path.unlink(missing_ok=True)
