# Phase 5: NCA Generation — Complete

## What was built
**Track B (Generation):** Neural Cellular Automata for coarse-to-fine unfolding

### `src/nca/cell.py`
- `NCACell`: small conv net update rule (conv → relu → conv → relu → update)
- Stochastic fire rate gating for biologically plausible sparse updates
- Target conditioning: optional coarse target added as residual after projection

### `src/nca/coarse.py`
- `CoarseConditioner`: linear projection from VSA vector (1000-dim) to 2D seed grid (28×28)
- Sigmoid output for normalized [0,1] initial states

### `src/nca/__init__.py`
- Exports: `NCACell`, `NCA`, `CoarseConditioner`

## Test results
```
tests/test_nca/test_cell.py ..... PASSED  (4 tests)
tests/test_nca/test_coarse.py ... PASSED  (1 test)
```
All 5 tests passing.

## Key design decisions
- Cell hidden state = `channels × H × W` (target projected and added as residual)
- Fire mask computed per spatial location, broadcast across channels
- `NCA.generate()` with optional `consistency_fn` callback for in-process checks
- Grid size and channels configurable; defaults 16×16 with 16 channels
