# Phase 4: Action Selection — Complete

## What was built
**Track A continuation:** Basal-ganglia-style Go/NoGo actor-critic

### `src/basal/actor_critic.py`
- `ThreeFactorHebbian`: local learning rule module (Δw ∝ pre × post × RPE)
- `GoNoGoActorCritic`: dual-pathway architecture
  - Go path promotes actions, NoGo path suppresses them
  - Action logits = Go - NoGo
  - Critic estimates state value
  - `act()`: sample from softmax over action logits
  - `compute_loss()`: advantage-based actor + critic loss

### `src/basal/environment.py`
- `MemoryRetrievalEnv`: Gymnasium environment where agent must select which stored memory to retrieve
- Observation: VSA query vector; Action: memory index; Reward: +1 for correct, -0.1 for wrong

## Test results
```
tests/test_basal/test_actor_critic.py .... PASSED  (4 tests)
```

## Key design decisions
- Go/NoGo architecture implemented as two separate MLPs whose outputs are subtracted
- The three-factor Hebbian rule is provided as a separate module (can be swapped for the standard backprop loss when needed)
- Environment is deliberately simple for unit testing; complexity to be added in stress testing
