# Upstream Reconciliation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `VolRegime-Engine` the tested upstream authority for the eight duplicated detector files and the existing shared market-clock implementation without expanding its BTCUSDT/OHLCV scope.

**Architecture:** Port the two substantive certification-side safety fixes into the engine under regression tests; preserve the six semantically identical detector files and the already-identical `gauge_bars.py`. Verify both repositories from clean isolated branches. Defer the final certification snapshot manifest until the engine branch is merged and tagged.

**Tech Stack:** Python 3.11+, NumPy, SciPy, statsmodels, pytest, uv, Git

---

## File Map

- Modify `VolRegime-Engine/src/strategies/vol_regime_switch/hmm_detector.py`: anchored-fit scope and fail-loud restart/API behavior.
- Modify `VolRegime-Engine/tests/vol_regime_switch/test_hmm_detector.py`: focused HMM safety regressions.
- Modify `VolRegime-Engine/src/strategies/vol_regime_switch/regime_population.py`: preserve genuine percentile intervals and warn on bootstrap bias.
- Modify `VolRegime-Engine/tests/wp1/test_gate_analysis.py`: assert unclamped interval behavior.
- Verify `VolRegime-Engine/scripts/wp1/gauge_bars.py`: upstream-owned shared market-clock implementation; no EUR/USD concepts.
- Verify the six semantically identical detector files without rewriting line endings.
- Update `m5-detector-certification/docs/superpowers/specs/2026-08-14-jfds-remediation-design.md`: remove Markdown trailing whitespace found during commit verification.

### Task 1: HMM safety behavior

**Files:**
- Modify: `VolRegime-Engine/tests/vol_regime_switch/test_hmm_detector.py`
- Modify: `VolRegime-Engine/src/strategies/vol_regime_switch/hmm_detector.py`

- [ ] **Step 1: Add failing HMM regression tests**

Add `SimpleNamespace` and a module import:

```python
from types import SimpleNamespace

import strategies.vol_regime_switch.hmm_detector as hmm_module
```

Add these tests after the frozen-parameter test:

```python
def test_module_documents_anchored_fit_scope() -> None:
    doc = hmm_module.__doc__ or ""
    assert "anchored full-sample" in doc
    assert "NOT a real-time causal detector" in doc


def test_all_nonfinite_restart_likelihoods_fail_loud(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeMarkovRegression:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def fit(self, **kwargs):
            return SimpleNamespace(llf=float("nan"))

    monkeypatch.setattr(hmm_module, "MarkovRegression", FakeMarkovRegression)
    detector = HMMDetector(rv_window=2, k_regimes=2, em_iter=1, search_reps=2)

    with pytest.raises(RuntimeError, match="non-finite log-likelihood"):
        detector.fit(np.array([100.0, 101.0, 102.0, 103.0]))


def test_missing_filtered_probability_api_fails_under_optimization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeMarkovRegression:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def fit(self, **kwargs):
            return SimpleNamespace(llf=1.0)

    monkeypatch.setattr(hmm_module, "MarkovRegression", FakeMarkovRegression)
    detector = HMMDetector(rv_window=2, k_regimes=2, em_iter=1, search_reps=2)

    with pytest.raises(RuntimeError, match="filtered_marginal_probabilities"):
        detector.fit(np.array([100.0, 101.0, 102.0, 103.0]))
```

- [ ] **Step 2: Run the focused tests and verify failure**

Run:

```powershell
$env:UV_PROJECT_ENVIRONMENT = 'D:\Coding\M5 Research\.venvs\VolRegime-jfds-upstream-2'
uv run pytest tests/vol_regime_switch/test_hmm_detector.py -k 'anchored_fit_scope or nonfinite_restart or missing_filtered_probability' -q
```

Expected: three failures against the pre-reconciliation engine copy.

- [ ] **Step 3: Port the certification HMM safety changes**

Replace the module title and add the causality scope:

```python
"""HMM Markov-switching volatility-regime detector (anchored full-sample fit).

CAUSALITY SCOPE — read before citing this detector as causal.
  Inference is causal GIVEN THE PARAMETERS: labels come from
  `filtered_marginal_probabilities` and never from the Kim smoother. The
  parameters, state ordering, and log-domain floor are estimated once over the
  full sample, so a label at t depends on future data through fitted parameters.

  This is an anchored measurement instrument, NOT a real-time causal detector.
"""
```

Before selecting the best restart, normalize and validate likelihoods:

```python
ll_arr = np.where(np.isnan(ll_arr), -np.inf, ll_arr)
if not np.any(np.isfinite(ll_arr)):
    raise RuntimeError(
        "All HMM EM restarts returned a non-finite log-likelihood. "
        "The input series may be too short, flat, or degenerate."
    )
best_idx = int(np.argmax(ll_arr))
```

Replace the runtime assertion:

```python
if not hasattr(best_res, "filtered_marginal_probabilities"):
    raise RuntimeError(
        "statsmodels API: filtered_marginal_probabilities not found on "
        "the result object. statsmodels has flagged this module as "
        "'not guaranteed stable' (§7.2)."
    )
```

- [ ] **Step 4: Run the focused HMM tests**

Run the Step 2 command.

Expected: three passed.

- [ ] **Step 5: Run the complete HMM module suite**

Run:

```powershell
$env:UV_PROJECT_ENVIRONMENT = 'D:\Coding\M5 Research\.venvs\VolRegime-jfds-upstream-2'
uv run pytest tests/vol_regime_switch/test_hmm_detector.py -q
```

Expected: all synthetic tests pass; real-data artifact tests may skip when validation JSON is absent.

- [ ] **Step 6: Commit the HMM reconciliation**

```powershell
git add src/strategies/vol_regime_switch/hmm_detector.py tests/vol_regime_switch/test_hmm_detector.py
git commit -m "fix(detector): harden HMM fit failures"
```

### Task 2: Preserve genuine bootstrap intervals

**Files:**
- Modify: `VolRegime-Engine/tests/wp1/test_gate_analysis.py`
- Modify: `VolRegime-Engine/src/strategies/vol_regime_switch/regime_population.py`

- [ ] **Step 1: Replace the clamping-dependent test with a failing bias regression**

Wrap the bootstrap call and replace the ordering assertion:

```python
with pytest.warns(RuntimeWarning, match="excludes its point estimate"):
    point, lo, hi = epsilon_sq_boot_ci(
        pred_nl,
        regime_nl,
        block=10,
        n_boot=2000,
        seed=43,
    )

assert np.isfinite(point)
assert np.isfinite(lo)
assert np.isfinite(hi)
assert lo < hi
assert point > hi
assert (point - hi) / point < 0.01
```

Keep the existing exact `_epsilon_sq_kw` equality assertion.

- [ ] **Step 2: Run the focused test and verify failure**

Run:

```powershell
$env:UV_PROJECT_ENVIRONMENT = 'D:\Coding\M5 Research\.venvs\VolRegime-jfds-upstream-2'
uv run pytest tests/wp1/test_gate_analysis.py::test_epsilon_sq_boot_ci -q
```

Expected: failure because the current engine clamps `hi` to the point and emits no warning.

- [ ] **Step 3: Remove interval clamping and add the diagnostic warning**

Add `import warnings` and replace the clamp with:

```python
if not (lo <= point <= hi):
    warnings.warn(
        f"percentile CI [{lo:.6g}, {hi:.6g}] excludes its point estimate "
        f"{point:.6g}; epsilon^2 is bounded below at 0, so this is "
        "expected near the floor and signals bootstrap bias rather than "
        "an error",
        RuntimeWarning,
        stacklevel=2,
    )
```

- [ ] **Step 4: Run the gate-analysis suite**

Run:

```powershell
$env:UV_PROJECT_ENVIRONMENT = 'D:\Coding\M5 Research\.venvs\VolRegime-jfds-upstream-2'
uv run pytest tests/wp1/test_gate_analysis.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit the interval reconciliation**

```powershell
git add src/strategies/vol_regime_switch/regime_population.py tests/wp1/test_gate_analysis.py
git commit -m "fix(stats): preserve bootstrap interval bias"
```

### Task 3: Prove semantic reconciliation and ownership boundaries

**Files:**
- Verify: the eight duplicated detector files in both repositories
- Verify: `scripts/wp1/gauge_bars.py` in both repositories
- Do not create: an upstream snapshot manifest

- [ ] **Step 1: Compare all eight detector files while ignoring line-ending whitespace**

Run `git diff --no-index --ignore-space-at-eol` for each of:

```text
defaults.py
hmm_detector.py
realized_vol.py
regime_detector.py
regime_engine.py
regime_population.py
rolling_quantile_detector.py
strategy_modules.py
```

Expected: no semantic differences after Tasks 1–2.

- [ ] **Step 2: Compare the shared market-clock implementation**

Run:

```powershell
git diff --no-index --ignore-space-at-eol -- `
  'D:\Coding\M5 Research\VolRegime-Engine\scripts\wp1\gauge_bars.py' `
  'D:\Coding\M5 Research\m5-detector-certification\scripts\wp1\gauge_bars.py'
```

Expected: no semantic differences. Verify manually that the module contains no EUR/USD, bid/ask, certificate, or submission concepts.

- [ ] **Step 3: Verify the engine branch boundary**

Run:

```powershell
git diff --name-only d7b0df9..HEAD
```

Expected: only the four detector/statistics source and test files from Tasks 1–2. No EUR/USD or certificate files.

### Task 4: Run both repositories before authority handoff

**Files:**
- Test only

- [ ] **Step 1: Run engine lint and default tests**

```powershell
$env:UV_PROJECT_ENVIRONMENT = 'D:\Coding\M5 Research\.venvs\VolRegime-jfds-upstream-2'
uv run ruff check src tests scripts
uv run pytest -q
```

Expected: lint passes and the default suite passes.

- [ ] **Step 2: Run engine artifact and slow suites**

```powershell
$env:UV_PROJECT_ENVIRONMENT = 'D:\Coding\M5 Research\.venvs\VolRegime-jfds-upstream-2'
uv run pytest -m artifact -q
uv run pytest -m slow -q
```

Expected: all runnable tests pass; missing external data/build artifacts skip explicitly.

- [ ] **Step 3: Run certification default, artifact, slow, and path-hygiene suites**

```powershell
python -m pytest -q
python -m pytest -m artifact -q
python -m pytest -m slow -q
python scripts/check_no_absolute_paths.py
```

Expected: every runnable test passes.

- [ ] **Step 4: Record the authority gate**

The engine branch is ready for review, merge, and tag only if Tasks 1–4 pass. Do not create the certification snapshot manifest yet. The next plan begins only after the upstream release tag exists.
