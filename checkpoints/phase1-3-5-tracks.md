# All Parallel Tracks — First Pass Complete

## Summary
All three parallel tracks from Phase 0 plan are now implemented and tested:

| Track | Phase | Description | Tests |
|---|---|---|---|
| A (Reasoning) | Phase 1 | VSA primitives, associative store, Hopfield net, benchmark | 13 |
| B (Generation) | Phase 5 | NCA cell, coarse conditioning | 5 |
| C (Safety) | Phase 3 | Negative selection detectors, self-monitor | 5 |
| — | — | SNN smoke test | 2 |
| _Total_ | | | _25_ |

## Next steps
- **Phase 2** (Track A continuation): Clonal memory for continual learning on top of VSA store
- **Phase 4** (Track A continuation): Basal-ganglia action selection via actor-critic
- **Phase 6** (Integration): Wire all subsystems together, end-to-end test

## Current project structure
```
src/
├── core/          (empty, ready for shared types)
├── vsa/           VSA primitives, store, Hopfield, benchmark
├── nca/           NCA cell, coarse conditioner
├── immune/        Detector, detector ensemble, self-monitor
├── clonal/        (Phase 2 — pending)
└── basal/         (Phase 4 — pending)
```
