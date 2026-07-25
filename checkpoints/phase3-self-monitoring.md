# Phase 3: Self-Monitoring — Complete

## What was built
**Track C (Safety):** Immune-inspired negative selection for OOD detection

### `src/immune/detector.py`
- `NegativeSelectionDetector`: wraps `sklearn.svm.OneClassSVM` for per-detector anomaly detection
- `DetectorEnsemble`: multiple detectors trained on different activation batches, ensemble scoring

### `src/immune/monitor.py`
- `SelfMonitor`: high-level monitor that calibrates on a baseline of "normal" activations
- Computes drift (z-score norm) and anomaly score (ensemble score) per activation
- Returns structured dict with `anomaly_score`, `drift`, and `is_anomaly` flag

## Test results
```
tests/test_immune/test_detector.py ... PASSED  (3 tests)
tests/test_immune/test_monitor.py .... PASSED  (2 tests)
```
All 5 tests passing.

## Key design decisions
- Detectors operate on CPU (scikit-learn) while activations come from CUDA — explicit `.cpu()` transfers
- Default threshold: anomaly_score < -0.5 or drift > 3.0 flags anomaly (conservative)
- Ensemble approach reduces false-positive rate vs. single detector
- Designed to observe internal activations from VSA store/clonal memory in later integration
