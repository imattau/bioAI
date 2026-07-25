# Phase 1: VSA Core — Complete

## What was built
**Track A (Reasoning):** VSA primitives, associative store, Hopfield network, toy benchmark

### `src/vsa/primitives.py`
- `VSA` class wrapping `torchhd` operations with 10000-dim default
- Methods: `make_vector`, `make_vectors`, `bind`, `unbind`, `bundle`, `permute`, `similarity`
- Role/filler encoding for structured representations
- Fractional power encoding for continuous values

### `src/vsa/store.py`
- `AssociativeStore`: insert key-value pairs, lookup by cosine similarity
- FIFO eviction at capacity limit

### `src/vsa/hopfield.py`
- `HopfieldNet`: modern continuous Hopfield network
- Hebbian-style weight updates (local, no backprop)
- `store`, `store_batch`, `recall` (iterative), `energy` functions
- Supports arbitrary pattern shapes via flattening

### `src/vsa/benchmark.py`
- `RavenLikeTask`: constructs structured relational analogies using VSA

## Test results (VSA + SNN)
```
tests/test_vsa/test_primitives.py ...... PASSED  (5 tests)
tests/test_vsa/test_store.py ........... PASSED  (2 tests)
tests/test_vsa/test_hopfield.py ........ PASSED  (2 tests)
tests/test_vsa/test_benchmark.py ....... PASSED  (2 tests)
tests/test_vsa/test_snn.py ............. PASSED  (2 tests)
```
13 tests, all passing.

## Key design decisions
- MAP (bipolar) VSA encoding used by default — components in {-1, +1}, norm = sqrt(dim)
- Hopfield network uses outer-product Hebbian learning, not backprop
- Associative store uses FIFO eviction (simplest); can be swapped for LRU later
- All computations run on CUDA by default when available
