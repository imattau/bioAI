"""Basal-ganglia-style relevance selection over candidate memory queries.

ARCHITECTURE.md SS3.4: "use this same competitive selection mechanism over
candidate memories retrieved from the VSA store, not just over motor
actions, treating 'which memory to attend to' as an action-selection
problem." RelevanceSelector is that: it wraps GoNoGoActorCritic to pick
which of several candidate follow-up queries is worth trying next, rather
than a fixed heuristic or trying all of them, and learns online from
reward-prediction-error -- did the chosen query actually help.

This is deliberately generic (a state vector in, an index out), not
specific to RelationalMemory, so it can be reused anywhere "which candidate
memory operation is worth trying" is a competitive-selection problem over a
bounded, positionally-ordered set of options.
"""

from __future__ import annotations

import torch

from src.basal.actor_critic import GoNoGoActorCritic


class RelevanceSelector:
    """Picks an index in [0, n_valid) from a caller-supplied, positionally
    ordered candidate list, given a fixed-size state vector describing the
    current situation. Learns online via `learn()` -- there is no
    pretraining step; an unused selector picks close to uniformly at
    random and improves with repeated use on the same instance.

    The action space size (`max_candidates`) is fixed at construction, per
    GoNoGoActorCritic's fixed-n_actions design -- only the first
    `max_candidates` entries of any candidate list are ever selectable.
    Actions are positional, not identity-based: action 0 always means
    "whichever candidate is currently first in the list the caller
    passed," so the caller's ordering matters and should be consistent
    across calls for what's meant to be "the same kind of situation" if
    the selector is expected to learn a stable preference.
    """

    def __init__(
        self,
        state_dim: int,
        max_candidates: int = 8,
        lr: float = 1e-3,
        device: str | None = None,
        epsilon: float = 0.15,
    ):
        if max_candidates < 1:
            raise ValueError("max_candidates must be >= 1")
        self.max_candidates = max_candidates
        self.model = GoNoGoActorCritic(input_dim=state_dim, n_actions=max_candidates)
        if device is not None:
            self.model = self.model.to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        # Epsilon-greedy on top of the learned policy: reward shaping alone
        # (a small negative penalty for uninformative picks — see
        # RelationalMemory.resolve_auto) reduces but does not eliminate
        # getting permanently stuck on a bad early choice, confirmed
        # empirically — once softmax drives a losing action's probability
        # near zero, multinomial sampling can go arbitrarily long without
        # ever trying the alternative, and advantage collapses toward zero
        # once the critic learns to expect the (bad but consistent) reward
        # for the only action actually being tried, so there's nothing left
        # to push it out. A fixed floor of random exploration guarantees
        # every valid action keeps getting sampled regardless of how
        # confident (and wrong) the learned policy currently is.
        self.epsilon = epsilon

    def select(self, state: torch.Tensor, n_valid: int, deterministic: bool = False) -> int:
        """Pick an index in [0, min(n_valid, max_candidates))."""
        if n_valid <= 0:
            raise ValueError("select() requires at least one valid candidate")
        n_valid = min(n_valid, self.max_candidates)
        if not deterministic and torch.rand(1).item() < self.epsilon:
            return torch.randint(0, n_valid, (1,)).item()
        logits, _, _, _ = self.model.forward(state)
        masked = logits.clone()
        if n_valid < self.max_candidates:
            masked[n_valid:] = float("-inf")
        probs = torch.softmax(masked, dim=-1)
        if deterministic:
            return probs.argmax().item()
        return torch.multinomial(probs, 1).item()

    def learn(self, state: torch.Tensor, action: int, reward: float) -> None:
        """Single-step (bandit-style) update: no next_state, since picking
        one candidate query doesn't lead to a meaningfully different
        future "state" in this use case -- each pick is scored on its own
        immediate reward (did it narrow the pool), not a return over time.
        """
        loss = self.model.compute_loss(state, action, reward, next_state=None)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
