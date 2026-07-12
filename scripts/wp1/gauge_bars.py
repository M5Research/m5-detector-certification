"""Volume-time and intrinsic-time bar builders for gauge-invariance tests."""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numba
import numpy as np

__all__ = [
    "VOLUME_THRESHOLDS_BTC",
    "PRIMARY_VOLUME_THRESHOLD_BTC",
    "RV_THRESHOLDS",
    "PRIMARY_RV_THRESHOLD",
    "DEFAULT_RV_WINDOW",
    "build_volume_bars",
    "build_intrinsic_time_bars",
    "compute_bar_duration_stats",
    "summarize_overshoot",
]

VOLUME_THRESHOLDS_BTC = (500, 1000, 2000)
PRIMARY_VOLUME_THRESHOLD_BTC = 1000
RV_THRESHOLDS = (0.0001, 0.0005, 0.001)
PRIMARY_RV_THRESHOLD = 0.0005
DEFAULT_RV_WINDOW = 20


@numba.njit
def _accumulate_volume_bars(
    open_arr: np.ndarray,
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    close_arr: np.ndarray,
    volume_arr: np.ndarray,
    timestamp_arr: np.ndarray,
    volume_threshold: float,
    out_open: np.ndarray,
    out_high: np.ndarray,
    out_low: np.ndarray,
    out_close: np.ndarray,
    out_volume: np.ndarray,
    out_timestamp: np.ndarray,
) -> int:
    n = len(open_arr)
    if n == 0:
        return 0

    out_idx = 0
    cum_vol = 0.0
    bar_open = 0.0
    bar_high = -np.inf
    bar_low = np.inf
    bar_close = 0.0
    bar_ts = 0
    in_bar = False

    for i in range(n):
        vol = volume_arr[i]
        if vol <= 0.0:
            continue

        if not in_bar:
            in_bar = True
            cum_vol = 0.0
            bar_open = open_arr[i]
            bar_high = high_arr[i]
            bar_low = low_arr[i]

        bar_high = max(bar_high, high_arr[i])
        bar_low = min(bar_low, low_arr[i])
        bar_close = close_arr[i]
        bar_ts = timestamp_arr[i]
        cum_vol += vol

        if cum_vol >= volume_threshold:
            out_open[out_idx] = bar_open
            out_high[out_idx] = bar_high
            out_low[out_idx] = bar_low
            out_close[out_idx] = bar_close
            out_volume[out_idx] = cum_vol
            out_timestamp[out_idx] = bar_ts
            out_idx += 1
            in_bar = False
            cum_vol = 0.0

    return out_idx


@numba.njit
def _rolling_rv_sum(sq_returns: np.ndarray, rv_window: int) -> np.ndarray:
    n = len(sq_returns)
    rv = np.zeros(n, dtype=np.float64)
    if n == 0:
        return rv

    for i in range(n):
        start = max(0, i - rv_window + 1)
        total = 0.0
        for j in range(start, i + 1):
            total += sq_returns[j]
        rv[i] = total
    return rv


@numba.njit
def _accumulate_intrinsic_bars(
    open_arr: np.ndarray,
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    close_arr: np.ndarray,
    volume_arr: np.ndarray,
    timestamp_arr: np.ndarray,
    rv_contrib: np.ndarray,
    rv_threshold: float,
    out_open: np.ndarray,
    out_high: np.ndarray,
    out_low: np.ndarray,
    out_close: np.ndarray,
    out_volume: np.ndarray,
    out_timestamp: np.ndarray,
) -> int:
    n = len(open_arr)
    if n == 0:
        return 0

    out_idx = 0
    cum_rv = 0.0
    bar_open = 0.0
    bar_high = -np.inf
    bar_low = np.inf
    bar_close = 0.0
    bar_vol = 0.0
    bar_ts = 0
    in_bar = False

    for i in range(n):
        contrib = rv_contrib[i]
        if contrib <= 0.0:
            continue

        if not in_bar:
            in_bar = True
            cum_rv = 0.0
            bar_open = open_arr[i]
            bar_high = high_arr[i]
            bar_low = low_arr[i]
            bar_vol = 0.0

        bar_high = max(bar_high, high_arr[i])
        bar_low = min(bar_low, low_arr[i])
        bar_close = close_arr[i]
        bar_vol += volume_arr[i]
        bar_ts = timestamp_arr[i]
        cum_rv += contrib

        if cum_rv >= rv_threshold:
            out_open[out_idx] = bar_open
            out_high[out_idx] = bar_high
            out_low[out_idx] = bar_low
            out_close[out_idx] = bar_close
            out_volume[out_idx] = bar_vol
            out_timestamp[out_idx] = bar_ts
            out_idx += 1
            in_bar = False
            cum_rv = 0.0

    return out_idx


def _pack_bars(
    n_bars: int,
    out_open: np.ndarray,
    out_high: np.ndarray,
    out_low: np.ndarray,
    out_close: np.ndarray,
    out_volume: np.ndarray,
    out_timestamp: np.ndarray,
) -> dict:
    return {
        "open": out_open[:n_bars].copy(),
        "high": out_high[:n_bars].copy(),
        "low": out_low[:n_bars].copy(),
        "close": out_close[:n_bars].copy(),
        "volume": out_volume[:n_bars].copy(),
        "timestamp": out_timestamp[:n_bars].copy(),
    }


def build_volume_bars(ohlcv: dict, volume_threshold: float) -> dict:
    """Aggregate 1-min OHLCV into volume bars at ``volume_threshold`` BTC."""
    open_arr = np.asarray(ohlcv["open"], dtype=np.float64)
    high_arr = np.asarray(ohlcv["high"], dtype=np.float64)
    low_arr = np.asarray(ohlcv["low"], dtype=np.float64)
    close_arr = np.asarray(ohlcv["close"], dtype=np.float64)
    volume_arr = np.asarray(ohlcv["volume"], dtype=np.float64)
    timestamp_arr = np.asarray(ohlcv["timestamp"], dtype=np.int64)

    n = len(open_arr)
    out_open = np.empty(n, dtype=np.float64)
    out_high = np.empty(n, dtype=np.float64)
    out_low = np.empty(n, dtype=np.float64)
    out_close = np.empty(n, dtype=np.float64)
    out_volume = np.empty(n, dtype=np.float64)
    out_timestamp = np.empty(n, dtype=np.int64)

    n_bars = _accumulate_volume_bars(
        open_arr,
        high_arr,
        low_arr,
        close_arr,
        volume_arr,
        timestamp_arr,
        float(volume_threshold),
        out_open,
        out_high,
        out_low,
        out_close,
        out_volume,
        out_timestamp,
    )
    return _pack_bars(n_bars, out_open, out_high, out_low, out_close, out_volume, out_timestamp)


def build_intrinsic_time_bars(
    ohlcv: dict,
    rv_threshold: float,
    rv_window: int = DEFAULT_RV_WINDOW,
) -> dict:
    """Aggregate 1-min OHLCV into intrinsic-time bars at ``rv_threshold`` RV."""
    open_arr = np.asarray(ohlcv["open"], dtype=np.float64)
    high_arr = np.asarray(ohlcv["high"], dtype=np.float64)
    low_arr = np.asarray(ohlcv["low"], dtype=np.float64)
    close_arr = np.asarray(ohlcv["close"], dtype=np.float64)
    volume_arr = np.asarray(ohlcv["volume"], dtype=np.float64)
    timestamp_arr = np.asarray(ohlcv["timestamp"], dtype=np.int64)

    n = len(close_arr)
    log_returns = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if close_arr[i - 1] > 0.0 and close_arr[i] > 0.0:
            log_returns[i] = np.log(close_arr[i] / close_arr[i - 1])

    sq_returns = log_returns * log_returns
    rv_contrib = _rolling_rv_sum(sq_returns, int(rv_window))

    out_open = np.empty(n, dtype=np.float64)
    out_high = np.empty(n, dtype=np.float64)
    out_low = np.empty(n, dtype=np.float64)
    out_close = np.empty(n, dtype=np.float64)
    out_volume = np.empty(n, dtype=np.float64)
    out_timestamp = np.empty(n, dtype=np.int64)

    n_bars = _accumulate_intrinsic_bars(
        open_arr,
        high_arr,
        low_arr,
        close_arr,
        volume_arr,
        timestamp_arr,
        rv_contrib,
        float(rv_threshold),
        out_open,
        out_high,
        out_low,
        out_close,
        out_volume,
        out_timestamp,
    )
    return _pack_bars(n_bars, out_open, out_high, out_low, out_close, out_volume, out_timestamp)


def compute_bar_duration_stats(timestamps: np.ndarray) -> dict:
    """Median/mean/p90 bar duration in minutes from bar timestamps."""
    ts = np.asarray(timestamps, dtype=np.int64)
    if len(ts) < 2:
        return {"median_min": 0.0, "mean_min": 0.0, "p90_min": 0.0, "n_bars": len(ts)}

    durations_min = np.diff(ts).astype(np.float64) / 60_000.0
    return {
        "median_min": float(np.median(durations_min)),
        "mean_min": float(np.mean(durations_min)),
        "p90_min": float(np.percentile(durations_min, 90)),
        "n_bars": len(ts),
    }


def summarize_overshoot(volumes: np.ndarray, threshold: float) -> dict:
    """Summarize volume overshoot relative to ``threshold``."""
    vol = np.asarray(volumes, dtype=np.float64)
    if len(vol) == 0:
        return {
            "count": 0,
            "median_pct": 0.0,
            "max_pct": 0.0,
            "fraction_overshoot": 0.0,
        }

    overshoot_mask = vol > threshold
    count = int(np.sum(overshoot_mask))
    if count == 0:
        return {
            "count": 0,
            "median_pct": 0.0,
            "max_pct": 0.0,
            "fraction_overshoot": 0.0,
        }

    pct = (vol[overshoot_mask] - threshold) / threshold * 100.0
    return {
        "count": count,
        "median_pct": float(np.median(pct)),
        "max_pct": float(np.max(pct)),
        "fraction_overshoot": float(count / len(vol)),
    }
