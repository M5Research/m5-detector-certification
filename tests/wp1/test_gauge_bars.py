import numpy as np
import pytest

from scripts.wp1.gauge_bars import (
    PRIMARY_RV_THRESHOLD,
    PRIMARY_VOLUME_THRESHOLD_BTC,
    build_intrinsic_time_bars,
    build_volume_bars,
    compute_bar_duration_stats,
    summarize_overshoot,
)


def _make_ohlcv(
    n: int,
    *,
    volume: float = 100.0,
    start_ts: int = 1_700_000_000_000,
    step_ms: int = 60_000,
    volumes: np.ndarray | None = None,
) -> dict:
    idx = np.arange(n, dtype=np.float64)
    ts = start_ts + idx.astype(np.int64) * step_ms
    vol = np.full(n, volume, dtype=np.float64) if volumes is None else volumes.astype(np.float64)
    close = 100.0 + idx * 0.01
    return {
        "open": close.copy(),
        "high": close + 0.5,
        "low": close - 0.5,
        "close": close,
        "volume": vol,
        "timestamp": ts,
    }


def test_volume_bars_constant_volume():
    ohlcv = _make_ohlcv(50, volume=100.0)
    bars = build_volume_bars(ohlcv, volume_threshold=500.0)
    assert len(bars["close"]) == 10
    median_vol = float(np.median(bars["volume"]))
    assert abs(median_vol - 500.0) / 500.0 <= 0.15


def test_volume_threshold_inverse_scaling():
    ohlcv = _make_ohlcv(40, volume=100.0)
    bars_500 = build_volume_bars(ohlcv, volume_threshold=500.0)
    bars_1000 = build_volume_bars(ohlcv, volume_threshold=1000.0)
    assert abs(len(bars_1000["close"]) * 2 - len(bars_500["close"])) <= 1


def test_single_minute_overshoot():
    volumes = np.full(10, 100.0)
    volumes[3] = 1500.0
    ohlcv = _make_ohlcv(10, volumes=volumes)
    bars = build_volume_bars(ohlcv, volume_threshold=1000.0)
    overshoot = summarize_overshoot(bars["volume"], 1000.0)
    assert overshoot["count"] >= 1
    assert len(bars["close"]) >= 1


def test_intrinsic_bars_constant_rv():
    rng = np.random.default_rng(43)
    n = 200
    returns = rng.normal(0.0, 0.01, n)
    close = 100.0 * np.exp(np.cumsum(returns))
    ohlcv = {
        "open": close.copy(),
        "high": close * 1.001,
        "low": close * 0.999,
        "close": close,
        "volume": np.full(n, 100.0),
        "timestamp": 1_700_000_000_000 + np.arange(n, dtype=np.int64) * 60_000,
    }
    bars = build_intrinsic_time_bars(ohlcv, rv_threshold=PRIMARY_RV_THRESHOLD, rv_window=20)
    assert len(bars["close"]) >= 5
    log_returns = np.diff(np.log(bars["close"]))
    assert np.all(np.isfinite(log_returns))


def test_no_lookahead_timestamp():
    ohlcv = _make_ohlcv(12, volume=100.0)
    bars = build_volume_bars(ohlcv, volume_threshold=500.0)
    # First bar closes at index 4 (5 bars * 100 vol)
    assert bars["timestamp"][0] == ohlcv["timestamp"][4]


def test_incomplete_tail_discarded():
    ohlcv = _make_ohlcv(13, volume=100.0)
    bars = build_volume_bars(ohlcv, volume_threshold=500.0)
    # 13 bars -> 2 complete 500-vol bars, tail of 3 bars discarded
    assert len(bars["close"]) == 2


def test_output_contract():
    ohlcv = _make_ohlcv(20, volume=100.0)
    bars = build_volume_bars(ohlcv, volume_threshold=PRIMARY_VOLUME_THRESHOLD_BTC)
    keys = {"open", "high", "low", "close", "volume", "timestamp"}
    assert set(bars.keys()) == keys
    lengths = {len(bars[k]) for k in keys}
    assert len(lengths) == 1
    assert bars["open"].dtype == np.float64
    assert bars["timestamp"].dtype == np.int64


def test_compute_bar_duration_stats():
    ts = np.array([0, 60_000, 180_000, 300_000], dtype=np.int64)
    stats = compute_bar_duration_stats(ts)
    assert stats["median_min"] == 2.0
    assert stats["n_bars"] == 4
