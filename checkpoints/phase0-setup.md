# Phase 0: Project Setup — Complete

## What was built
- Virtual environment (`.venv/`) with `--system-site-packages`
- `pyproject.toml` with all dependencies
- `.gitignore`
- Directory structure under `src/` and `tests/`
- Smoke test (`tests/test_imports.py`)

## Dependencies installed
| Package | Version | Notes |
|---|---|---|
| torch | 2.13.0+cu130 | System-installed, inherited via `--system-site-packages` |
| torchhd | 5.8.4 | PyPI name `torch-hd`, import name `torchhd` |
| snntorch | 1.0.0 | Spiking neural networks |
| stable-baselines3 | 2.9.0 | RL scaffolding |
| gymnasium | 1.3.0 | RL environments |
| numpy, scipy, scikit-learn, pandas, networkx, matplotlib, tqdm | latest | Standard ML stack |
| pytest | 9.1.1 | Test runner |
| jupyter, ipywidgets | latest | Notebooks for visualisation |

## Test results
```
tests/test_imports.py::test_imports PASSED
tests/test_imports.py::test_cuda PASSED
```
CUDA available: RTX 5060 Ti

## Git
Initial commit `8c40366` — 4 files, 221 insertions.

## Notes
- `torchhd` is imported as `torchhd` (not `torch_hd`); the pyproject.toml dependency is on `torch-hd` (PyPI name).
- System pip is PEP 668 protected; `--system-site-packages` was required to avoid re-downloading the 526 MB torch wheel.
