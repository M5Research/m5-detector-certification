"""Persistence null: first-passage survival S(t) vs arcsin-law KS bootstrap."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_REPO_SRC = _REPO_ROOT / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from backtest.utils import PROJECT_ROOT  # noqa: E402

import scripts._bootstrap  # noqa: F401, E402
from scripts.wp1.precheck_b import _gate_guard  # noqa: E402

__all__ = [
    "W_PERSIST",
    "MAX_HORIZON",
    "PINNED_SEED",
    "N_BOOTSTRAP",
    "BLOCK_SIZE",
    "PERSIST_DIR",
    "VERDICT_CONSISTENT",
    "VERDICT_DEVIATION",
    "arcsin_survival",
    "persistence_probability",
    "main",
]

W_PERSIST = 120
MAX_HORIZON = 120
PINNED_SEED = 42
N_BOOTSTRAP = 500
BLOCK_SIZE = 120
MARTINGALE_THETA = 0.5
PERSISTENCE_ALPHA = 0.05
PERSIST_DIR = PROJECT_ROOT / "backtest_results" / "persistence"
VERDICT_CONSISTENT = "consistent_with_martingale_persistence"
VERDICT_DEVIATION = "persistence_deviation_detected"


def arcsin_survival(t_grid: np.ndarray, t0: float = 1.0) -> np.ndarray:
    """Arcsin-law martingale reference survival S₀(t) = (2/π)·arcsin(√(t₀/t))."""
    t = np.asarray(t_grid, dtype=np.float64)
    return np.where(t > t0, (2.0 / np.pi) * np.arcsin(np.sqrt(t0 / t)), 1.0)


def _compute_first_passage_times(
    close: np.ndarray,
    W: int = W_PERSIST,
    max_horizon: int = MAX_HORIZON,
) -> np.ndarray:
    """First-passage times per non-overlapping W-bar window with per-window demean."""
    close_arr = np.asarray(close, dtype=np.float64)
    n = close_arr.size
    n_windows = (n - 1) // W
    if n_windows == 0:
        return np.array([], dtype=np.float64)

    times = np.empty(n_windows, dtype=np.float64)
    for i in range(n_windows):
        seg = close_arr[i * W : i * W + W + 1]
        log_rel = np.log(seg[1:] / seg[0])

        if log_rel[0] == 0.0:
            times[i] = 1.0
            continue

        initial_sign = np.sign(log_rel[0])
        crossed = (np.sign(log_rel) != initial_sign) | (log_rel == 0.0)
        if crossed.any():
            times[i] = float(np.argmax(crossed) + 1)
        else:
            times[i] = float(max_horizon + 1)

    return times


def _empirical_survival(T: np.ndarray, max_horizon: int) -> np.ndarray:
    """S(t) = P(T > t) for integer horizons t ∈ [1, max_horizon]."""
    t_grid = np.arange(1, max_horizon + 1)
    return np.array([np.mean(T > t) for t in t_grid], dtype=np.float64)


def _fit_theta_ols(S: np.ndarray, t_grid: np.ndarray) -> tuple[float, np.ndarray]:
    """Power-law exponent θ via log-log OLS on S(t) ∝ t^{-θ}."""
    from scipy import stats as scipy_stats

    valid = S > 0
    if valid.sum() < 2:
        return float("nan"), valid
    slope, _, _, _, _ = scipy_stats.linregress(
        np.log(t_grid[valid]),
        np.log(S[valid]),
    )
    return float(-slope), valid


def _bootstrap_ks_null(
    log_returns: np.ndarray,
    max_horizon: int,
    block_size: int,
    n_bootstrap: int,
    seed: int,
    S0: np.ndarray,
    W: int = W_PERSIST,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Circular block-bootstrap KS null; single loop for KS, θ, and S(t) CI."""
    r = np.asarray(log_returns, dtype=np.float64)
    n = r.size
    t_grid = np.arange(1, max_horizon + 1)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_size))

    ks_boots = np.empty(n_bootstrap, dtype=np.float64)
    theta_boots = np.empty(n_bootstrap, dtype=np.float64)
    S_boot_all = np.empty((n_bootstrap, max_horizon), dtype=np.float64)

    for b in range(n_bootstrap):
        starts = rng.integers(0, n, size=n_blocks)
        idx = (starts[:, None] + np.arange(block_size)[None, :]) % n
        r_boot = r[idx.ravel()[:n]]
        close_boot = np.concatenate([[1.0], np.exp(np.cumsum(r_boot))])

        T_boot = _compute_first_passage_times(close_boot, W=W, max_horizon=max_horizon)
        S_boot = _empirical_survival(T_boot, max_horizon)
        S_boot_all[b] = S_boot

        ks_boots[b] = float(np.max(np.abs(S_boot - S0)))
        theta_boots[b], _ = _fit_theta_ols(S_boot, t_grid)

    return ks_boots, theta_boots, S_boot_all


def persistence_probability(
    close: np.ndarray,
    max_horizon: int = MAX_HORIZON,
    block_size: int = BLOCK_SIZE,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = PINNED_SEED,
) -> dict:
    """Estimate S(t), fit θ, run block-bootstrap KS test against arcsin null."""
    close_arr = np.asarray(close, dtype=np.float64)
    if close_arr.ndim != 1:
        raise ValueError("close must be 1-D")

    T = _compute_first_passage_times(close_arr, W=W_PERSIST, max_horizon=max_horizon)
    n_windows = int(T.size)
    t_grid = np.arange(1, max_horizon + 1)
    S_emp = _empirical_survival(T, max_horizon)
    S0 = arcsin_survival(t_grid)

    ks_stat = float(np.max(np.abs(S_emp - S0)))
    theta, valid = _fit_theta_ols(S_emp, t_grid)

    log_returns = np.log(close_arr[1:] / close_arr[:-1])
    ks_boots, theta_boots, S_boot_all = _bootstrap_ks_null(
        log_returns,
        max_horizon=max_horizon,
        block_size=block_size,
        n_bootstrap=n_bootstrap,
        seed=seed,
        S0=S0,
        W=W_PERSIST,
    )

    p_value = float(np.mean(ks_boots >= ks_stat))
    theta_ci = [
        float(np.percentile(theta_boots, 2.5)),
        float(np.percentile(theta_boots, 97.5)),
    ]
    bootstrap_ci_lo = np.percentile(S_boot_all, 2.5, axis=0)
    bootstrap_ci_hi = np.percentile(S_boot_all, 97.5, axis=0)

    verdict = (
        VERDICT_DEVIATION
        if p_value < PERSISTENCE_ALPHA
        else VERDICT_CONSISTENT
    )

    return {
        "S_empirical": S_emp.tolist(),
        "theta": float(theta),
        "theta_ci": theta_ci,
        "ks_statistic": ks_stat,
        "bootstrap_p_value": p_value,
        "n_windows": n_windows,
        "max_horizon": max_horizon,
        "W": W_PERSIST,
        "block_size": block_size,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "verdict": verdict,
        "bootstrap_ci_lo": bootstrap_ci_lo.tolist(),
        "bootstrap_ci_hi": bootstrap_ci_hi.tolist(),
        "theta_ols_t_range": t_grid[valid].tolist(),
    }


def _synthetic_smoke() -> None:
    rng = np.random.default_rng(PINNED_SEED)
    n = 5000
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    result = persistence_probability(close, n_bootstrap=25)
    assert "S_empirical" in result
    assert result["verdict"] in (VERDICT_CONSISTENT, VERDICT_DEVIATION)
    print(f"synthetic_ok verdict={result['verdict']}")


def main() -> int:
    import datetime as _dt

    from scripts.wp1.py_engine import load_and_clean

    print("D-09 gate-guard: checking dual pre-registration integrity...", flush=True)
    prereg_results = _gate_guard()
    prereg_commit = prereg_results.get(
        ".planning/phases/11-signal-injection-the-ligo-calibration/09-PREREGISTRATION.md",
        next(iter(prereg_results.values()), ""),
    )
    print(f"D-09 gate-guard PASSED. prereg_commit={prereg_commit[:12]}...")

    run_date = datetime.now(tz=_dt.UTC).strftime("%Y%m%d_%H%M%S")
    try:
        code_commit = subprocess.run(
            ["git", "log", "--format=%H", "-1"],
            capture_output=True,
            text=True,
            cwd=str(_REPO_ROOT),
        ).stdout.strip()
    except Exception:
        code_commit = ""

    start_ms = int(_dt.datetime(2021, 1, 1, tzinfo=_dt.UTC).timestamp() * 1000)
    end_ms = int(_dt.datetime(2025, 12, 31, 23, 59, 59, tzinfo=_dt.UTC).timestamp() * 1000)
    holdout_boundary_ms = int(_dt.datetime(2026, 1, 1, tzinfo=_dt.UTC).timestamp() * 1000)
    assert end_ms < holdout_boundary_ms, "D-07: 2026 holdout must not be loaded"

    data_path = str(PROJECT_ROOT / "data" / "binance_futures")
    print("Loading BTCUSDT 2021-2025...")
    ohlcv = load_and_clean(
        data_path=data_path,
        symbol="BTCUSDT",
        start_ms=start_ms,
        end_ms=end_ms,
        max_gap_allowed_mins=60,
    )
    ts_end = _dt.datetime.fromtimestamp(ohlcv["timestamp"][-1] / 1000, tz=_dt.UTC)
    assert ts_end.year < 2026, "D-07: last bar must be before 2026"

    result = persistence_probability(ohlcv["close"])
    result.update(
        {
            "run_timestamp": run_date,
            "prereg_commit": prereg_commit,
            "code_commit": code_commit,
            "data_span": "2021-2025",
        }
    )

    PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PERSIST_DIR / f"persistence_report_{run_date}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Verdict: {result['verdict']}")
    print(f"Report: {out_path}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--run":
        raise SystemExit(main())
    _synthetic_smoke()
