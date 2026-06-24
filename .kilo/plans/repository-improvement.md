# SCIO-Bench Repository Improvement Plan

## Executive Summary

SCIO-Bench is a well-structured research benchmark for anomaly detection in off-grid renewable energy IoT. The codebase has **182 passing tests** across 13 test files covering all pipeline phases. However, several code quality, architecture, and correctness issues need addressing before this can be considered production-grade open-source software.

---

## Priority 1: Bug Fixes & Correctness

### 1.1 Fix LSTM-AE alignment inconsistency in `statistical_tests.py`

**Problem:** `src/evaluation/statistical_tests.py:106-109` uses a "hacky" alignment for LSTM-AE predictions that differs from the proper `sequences_to_row_scores()` used in Phase 7 (`src/models/lstm_autoencoder.py:94-131`). The current code pads with zeros and takes the last sequence error per row, which is incorrect — it doesn't account for rows being covered by multiple overlapping sequences.

```python
# CURRENT (wrong): src/evaluation/statistical_tests.py:106-109
pad = np.zeros(seq_len - 1)
row_err = np.concatenate([pad, seq_err])  # wrong alignment
results["LSTM-AE"] = (row_err > threshold)
```

**Fix:** Import and use `sequences_to_row_scores()` from `lstm_autoencoder.py` (which is already imported elsewhere via `_get_l1_test_predictions`).

### 1.2 Fix bare `except: pass` in `statistical_tests.py`

**Problem:** Lines 212, 218, 228 use bare `except: pass` which silently swallows all errors including `KeyError`, `TypeError`, `FileNotFoundError`, making debugging impossible.

**Fix:** Replace with `except (KeyError, ValueError, IndexError):` or log the exception.

### 1.3 Fix docstring/code mismatch in `preprocess.py`

**Problem:** `src/data/preprocess.py:7-8` docstring says "Output: data/processed/plant1_clean.parquet" but line 221-222 actually saves `.csv`.

**Fix:** Update docstring to match actual output format.

---

## Priority 2: Code Deduplication (DRY)

### 2.1 Centralize `_get_feature_cols()` 

**Problem:** The function `_get_feature_cols()` is duplicated 4 times with slightly different implementations:
- `src/models/classical_ml.py:45`
- `src/models/lstm_autoencoder.py:63`
- `src/models/l2_classifier.py:69`
- `src/xai/shap_analysis.py:37`

Additionally, `src/evaluation/edge_profiling.py:204` and `src/evaluation/statistical_tests.py:61` have inline equivalents.

**Fix:** Add a single `get_feature_cols(df, label_cols=None)` function to `src/config.py` or `src/evaluation/metrics.py`. All modules import from there.

### 2.2 Centralize `LABEL_COLS` definitions

**Problem:** `LABEL_COLS` is defined in 6 places with inconsistent values:
- `src/config.py:125` — 6 items (includes `timestamp`, `device_id`, `protocol`)
- `src/data/feature_engineering.py:74` — 3 items (only boolean labels)
- `src/models/l2_classifier.py:65` — 6 items (duplicate of config)
- `src/xai/shap_analysis.py:33` — 6 items (duplicate of config)

**Fix:** Use `config.LABEL_COLS` everywhere. The `feature_engineering.py` subset should be derived: `BOOLEAN_LABEL_COLS = config.LABEL_COLS[:3]`.

### 2.3 Deduplicate `build_sequences()` and `sequences_to_row_scores()`

**Problem:** Both functions are duplicated between:
- `src/models/lstm_autoencoder.py:72-131`
- `src/xai/reconstruction_analysis.py:26-60`

The `reconstruction_analysis.py` version of `sequences_to_row_scores()` differs — it returns a 2D array `(n_rows, n_features)` instead of 1D, which is correct for per-feature analysis. However, `build_sequences()` is identical.

**Fix:** Keep `build_sequences()` only in `lstm_autoencoder.py` and import from there. Keep both versions of `sequences_to_row_scores()` but rename them to clarify: `sequences_to_row_scores_1d()` and `sequences_to_row_scores_per_feature()`.

---

## Priority 3: Code Cleanup

### 3.1 Remove `sys.path` hacks from all 13 test files

**Problem:** Every test file has `sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))` which is unnecessary since `pyproject.toml` already sets `pythonpath = ["."]` for pytest.

**Fix:** Remove these 3 lines from all 13 test files:
```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
```

### 3.2 Remove `requirements.txt` (or generate from pyproject.toml)

**Problem:** `requirements.txt` duplicates `[project.dependencies]` in `pyproject.toml`. This creates a maintenance burden (two places to update).

**Fix:** Either delete `requirements.txt` and add a note in README to use `pip install -e .`, or make `requirements.txt` a generated file (e.g., `pip freeze > requirements.txt` from the lockfile).

### 3.3 Consolidate imports in test files

**Problem:** Several test files import `pathlib` and `sys` only for the `sys.path` hack. After removing that, cleanup the remaining imports.

---

## Priority 4: Missing Tests

### 4.1 Add tests for model save/load roundtrip

**Problem:** `RuleBasedDetector.save()/load()` and `L2Classifier.save()/load()` are not tested. A serialization bug would only be caught at runtime.

**Fix:** Add roundtrip tests:
- `test_rule_based.py`: save → load → verify thresholds and predictions match
- `test_l2_classifier.py`: save → load → verify predictions match

### 4.2 Add basic test for `visualization/plots.py`

**Problem:** `src/visualization/plots.py` is the only module with zero test coverage. It generates 5 publication figures but has no smoke tests.

**Fix:** Add a smoke test that calls each `plot_figure_*` function with mock data (no actual model files needed) and verifies the PDF files are created. Mark as `@pytest.mark.slow` or use a `tmp_path` fixture.

---

## Priority 5: Developer Experience & CI

### 5.1 Add `Makefile` or `scripts/` for common operations

**Problem:** Users must remember individual `python -m src.data.*` commands. No single entrypoint for lint, test, or full pipeline.

**Fix:** Add a `Makefile` with targets:
```makefile
test:        ## Run all tests
lint:        ## Run ruff linter
typecheck:   ## Run mypy
pipeline:    ## Run full pipeline (Phases 1-12)
clean:       ## Remove generated outputs
```

### 5.2 Add GitHub Actions CI

**Problem:** No CI pipeline. Tests must be run manually.

**Fix:** Add `.github/workflows/ci.yml` with:
- `ruff check` linting
- `pytest` test suite
- Triggered on push/PR to main

### 5.3 Add `ruff` to dev dependencies

**Problem:** `ruff` is listed in `pyproject.toml` optional deps but not installed in the venv, making lint non-functional.

**Fix:** Ensure dev deps are installable: `pip install -e ".[dev]"`

---

## Priority 6: Type Safety

### 6.1 Add `from __future__ import annotations` and fix type hints

**Problem:** Type hints are inconsistent — some functions have full annotations, others have none. `mypy` is configured but likely produces many errors.

**Fix:** Add `from __future__ import annotations` to all source files for PEP 604 union syntax support. Then run `mypy src/` and fix critical type errors.

### 6.2 Add return type annotations to all public functions

**Problem:** Functions like `inject_all_anomalies()`, `preprocess_plant()`, `engineer_all_features()` lack return type annotations.

**Fix:** Add return types to all public functions (functions without leading underscore).

---

## Priority 7: Documentation & Repo Hygiene

### 7.1 Create `AGENTS.md` with coding guidelines

**Problem:** No contributor guidelines exist for coding style, test patterns, or module conventions.

**Fix:** Create `AGENTS.md` documenting:
- How to run tests: `.venv/bin/python -m pytest tests/ -v`
- Code style: ruff rules, type hints policy
- Module structure: `src/data/`, `src/models/`, `src/evaluation/`, `src/xai/`, `src/visualization/`
- Label column conventions

### 7.2 Clean up `github_key.md` from disk

**Problem:** `github_key.md` contains a GitHub PAT and exists on disk (even though gitignored). This is a security risk if `.gitignore` rules are ever relaxed.

**Fix:** Delete the file from disk: `rm github_key.md`

### 7.3 Add `.editorconfig`

**Problem:** No editor configuration for consistent formatting across contributors.

**Fix:** Add `.editorconfig` with Python 4-space indent, UTF-8, LF line endings.

---

## Implementation Order

| Phase | Items | Estimated Effort |
|-------|-------|-----------------|
| **Phase A** | 1.1, 1.2, 1.3 (Bug fixes) | ~30 min |
| **Phase B** | 2.1, 2.2, 2.3 (DRY) | ~1 hour |
| **Phase C** | 3.1, 3.2, 3.3 (Cleanup) | ~30 min |
| **Phase D** | 4.1, 4.2 (Tests) | ~45 min |
| **Phase E** | 5.1, 5.2, 5.3 (DX) | ~30 min |
| **Phase F** | 6.1, 6.2 (Types) | ~45 min |
| **Phase G** | 7.1, 7.2, 7.3 (Docs) | ~15 min |

**Total estimated effort: ~4 hours**

---

## Files to Modify

| File | Changes |
|------|---------|
| `src/evaluation/statistical_tests.py` | Fix LSTM alignment, fix bare excepts |
| `src/data/preprocess.py` | Fix docstring |
| `src/config.py` | Add `get_feature_cols()`, consolidate `LABEL_COLS` |
| `src/models/classical_ml.py` | Import centralized `get_feature_cols()` |
| `src/models/lstm_autoencoder.py` | Remove duplicate helpers, import from centralized |
| `src/models/l2_classifier.py` | Import centralized `get_feature_cols()`, `LABEL_COLS` |
| `src/xai/shap_analysis.py` | Import centralized `get_feature_cols()` |
| `src/xai/reconstruction_analysis.py` | Import `build_sequences` from lstm_autoencoder |
| `src/data/feature_engineering.py` | Use derived label cols from config |
| All 13 test files | Remove `sys.path` hacks |
| `tests/test_rule_based.py` | Add save/load roundtrip test |
| `tests/test_l2_classifier.py` | Add save/load roundtrip test |
| `tests/test_plots.py` (new) | Smoke tests for visualization |
| `Makefile` (new) | Developer commands |
| `.github/workflows/ci.yml` (new) | CI pipeline |
| `AGENTS.md` (new) | Coding guidelines |
| `.editorconfig` (new) | Editor config |
| `requirements.txt` | Remove or regenerate |
