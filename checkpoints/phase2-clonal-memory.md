# Phase 2: Clonal Memory — Complete

## What was built
**Track A continuation:** Immune-inspired clonal memory for continual learning

### `src/clonal/module.py`
- `ClonalModule`: small autoencoder adapter with a receptor vector for similarity matching
- `affinity()`: cosine similarity between input and receptor
- `clone()`: creates a child module with slightly mutated weights
- `local_update()`: fine-tunes on a single input via MSE reconstruction

### `src/clonal/pool.py`
- `ClonalPool`: manages a pool of ClonalModule instances
- On novel input:
  - If best affinity > threshold + margin: clone best match + mutate + fine-tune
  - If best affinity > threshold (but below margin): fine-tune existing module
  - Otherwise: create new module
- Pruning: when pool exceeds `max_modules`, oldest unused modules are removed

## Test results
```
tests/test_clonal/test_pool.py ..... PASSED  (5 tests)
```
## Key design decisions
- Reconstruction-based learning (autoencoder): module learns to reproduce its input, capturing the pattern
- Cloning threshold margin (threshold + 0.2) avoids excessive proliferation
- Pruning keeps the bottom 75% of max_modules by last_used timestamp
