"""Tests for RelevanceSelector — the basal-ganglia relevance-selection
wrapper around GoNoGoActorCritic used by RelationalMemory.resolve_auto.

See RELATIONAL_MEMORY.md SS2.7 for the design and the exploration-failure
finding that led to the -0.2 (not 0) uninformative-reward and epsilon-greedy
exploration in the actual implementation.
"""

import torch

from src.basal.relevance import RelevanceSelector


def test_masking_restricts_to_valid_range():
    selector = RelevanceSelector(state_dim=16, max_candidates=5)
    state = torch.randn(16)
    for _ in range(50):
        action = selector.select(state, n_valid=2, deterministic=False)
        assert 0 <= action < 2


def test_masking_rejects_zero_valid_candidates():
    selector = RelevanceSelector(state_dim=16, max_candidates=5)
    state = torch.randn(16)
    try:
        selector.select(state, n_valid=0)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_learns_two_armed_bandit_preference():
    """Core mechanism check, isolated from RelationalMemory's noisier VSA
    setup: a fixed state, two actions, one always rewarded and one always
    penalized -- across repeated calls on the same selector instance, the
    learned policy should shift toward the rewarded action.

    This is inherently stochastic RL, not something that converges
    identically every run within a small trial budget -- confirmed
    empirically during development that even with epsilon-greedy
    exploration, a small fraction of random initializations take longer to
    escape an unlucky early policy. So this asserts the aggregate,
    statistical claim ("usually learns, comfortably above chance, within a
    reasonable budget") across several independent selectors, not
    "converges deterministically every time" -- the latter would be a
    dishonest claim about a real bandit-RL mechanism.
    """
    state = torch.randn(20)
    good_action, bad_action = 1, 0
    trials = 60
    successes = 0

    for _ in range(10):
        selector = RelevanceSelector(state_dim=20, max_candidates=2)
        picks = []
        for _ in range(trials):
            action = selector.select(state, n_valid=2)
            reward = 1.0 if action == good_action else -0.2
            selector.learn(state, action, reward)
            picks.append(action == good_action)
        # Converged for this run if it prefers the good action in the
        # final stretch, well above the 50% chance baseline.
        if sum(picks[-15:]) >= 11:
            successes += 1

    # Not 10/10 -- see docstring -- but comfortably better than chance
    # across independent selectors.
    assert successes >= 7, f"only {successes}/10 selectors learned the preference"
