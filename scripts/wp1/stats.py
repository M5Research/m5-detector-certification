"""Offline performance & significance statistics for WP1 (do NOT gate on the engine's C++ Sharpe)."""
from __future__ import annotations

import numpy as np
from scipy import stats as scipy_stats

DAY_MS = 86_400_000


def daily_returns_from_equity(equity: np.ndarray, timestamps_ms: np.ndarray) -> np.ndarray:
    """Last-equity-per-UTC-day -> day-over-day simple returns (sample, not seeded by capital)."""
    equity = np.asarray(equity, dtype=np.float64)
    ts = np.asarray(timestamps_ms, dtype=np.int64)
    if equity.size == 0:
        return np.array([], dtype=np.float64)
    day = ts // DAY_MS
    # last equity per day, in day order
    idx: dict[int, float] = {}
    for d, e in zip(day, equity, strict=True):
        idx[int(d)] = float(e)
    days = sorted(idx)
    eq_daily = np.array([idx[d] for d in days], dtype=np.float64)
    if eq_daily.size < 2:
        return np.array([], dtype=np.float64)
    return eq_daily[1:] / eq_daily[:-1] - 1.0


def sharpe_ratio(equity: np.ndarray, timestamps_ms: np.ndarray, periods_per_year: float = 365.0) -> float:
    """Annualized Sharpe from daily returns with SAMPLE std (ddof=1). Diagnostic; report PSR alongside."""
    r = daily_returns_from_equity(equity, timestamps_ms)
    if r.size < 2:
        return 0.0
    sd = r.std(ddof=1)
    if sd == 0.0:
        return 0.0
    return float(np.sqrt(periods_per_year) * r.mean() / sd)


def sortino_ratio(equity: np.ndarray, timestamps_ms: np.ndarray, periods_per_year: float = 365.0) -> float:
    r = daily_returns_from_equity(equity, timestamps_ms)
    if r.size < 2:
        return 0.0
    downside = r[r < 0]
    dd = downside.std(ddof=1) if downside.size >= 2 else 0.0
    if dd == 0.0:
        return 0.0
    return float(np.sqrt(periods_per_year) * r.mean() / dd)


def max_drawdown(equity: np.ndarray) -> float:
    equity = np.asarray(equity, dtype=np.float64)
    if equity.size == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = np.where(peak > 0, (peak - equity) / peak, 0.0)
    return float(dd.max())


def calmar_ratio(total_return: float, mdd: float, years: float) -> float:
    if mdd <= 0.0 or years <= 0.0:
        return 0.0
    annualized = (1.0 + total_return) ** (1.0 / years) - 1.0
    return float(annualized / mdd)


def probabilistic_sharpe_ratio(sr: float, n_obs: int, skew: float, kurt: float, sr_benchmark: float = 0.0) -> float:
    """Bailey & Lopez de Prado PSR: P(true SR > benchmark) given skew/kurtosis of returns."""
    if n_obs < 2:
        return 0.0
    denom = np.sqrt(max(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr, 1e-12))
    z = (sr - sr_benchmark) * np.sqrt(n_obs - 1) / denom
    return float(scipy_stats.norm.cdf(z))


def deflated_sharpe_ratio(observed_sr: float, n_obs: int, n_trials: int, sr_variance: float,
                          skew: float = 0.0, kurt: float = 3.0) -> float:
    """Deflated Sharpe: PSR against a benchmark inflated by the expected max of N trials' Sharpe."""
    if n_trials < 1 or n_obs < 2 or sr_variance <= 0.0:
        return 0.0
    emc = 0.5772156649  # Euler-Mascheroni
    e_max_z = (1 - emc) * scipy_stats.norm.ppf(1 - 1.0 / n_trials) + emc * scipy_stats.norm.ppf(
        1 - 1.0 / (n_trials * np.e)
    )
    sr_benchmark = np.sqrt(sr_variance) * e_max_z
    return probabilistic_sharpe_ratio(observed_sr, n_obs, skew, kurt, sr_benchmark=float(sr_benchmark))


def spearman_ic(signal: np.ndarray, fwd_return: np.ndarray) -> tuple[float, int]:
    """Spearman rank IC over finite, non-NaN pairs. Returns (ic, n_used)."""
    s = np.asarray(signal, dtype=np.float64)
    f = np.asarray(fwd_return, dtype=np.float64)
    mask = np.isfinite(s) & np.isfinite(f)
    if mask.sum() < 10:
        return float("nan"), int(mask.sum())
    ic, _ = scipy_stats.spearmanr(s[mask], f[mask])
    return float(ic), int(mask.sum())


def block_bootstrap_ci(x: np.ndarray, stat=np.mean, block: int = 20, n_boot: int = 2000,
                       alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """Circular block-bootstrap CI for a statistic of a (possibly autocorrelated) series.

    Returns (point, lo, hi). `seed` is explicit for determinism (no Math.random / wall clock).
    """
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = x.size
    if n < block * 2:
        p = float(stat(x)) if n else float("nan")
        return p, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    boots = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]) % n
        sample = x[idx.ravel()][:n]
        boots[b] = stat(sample)
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return float(stat(x)), lo, hi
