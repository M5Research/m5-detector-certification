"""EUR/USD rolling-quantile certificate unit tests."""
from __future__ import annotations

import numpy as np


def _minutes() -> dict[str, np.ndarray]:
    return {
        "timestamp": np.array([0, 60, 60, 120, 180], dtype=np.int64),
        "bid": np.array([1.0, 1.01, 1.02, 1.03, 1.04]),
        "ask": np.array([1.001, 1.011, 1.021, 1.031, 1.041]),
        "mid": np.array([1.0005, 1.0105, 1.0205, 1.0305, 1.0405]),
        "n_ticks": np.array([2, 3, 4, 2, 1], dtype=np.int64),
    }


def test_duplicate_minutes_keep_last_quote_and_sum_tick_counts() -> None:
    from scripts.wp1.eurusd_certificate import consolidate_minutes

    clean, report = consolidate_minutes(_minutes())

    assert clean["timestamp"].tolist() == [0, 60, 120, 180]
    assert clean["bid"][1] == 1.02
    assert clean["n_ticks"][1] == 7
    assert report["duplicate_rows"] == 1


def test_tick_and_intrinsic_clocks_are_causal_and_nonempty() -> None:
    from scripts.wp1.eurusd_certificate import build_intrinsic_clock, build_tick_clock

    data, _ = consolidate_fixture()
    tick = build_tick_clock(data, threshold=5)
    intrinsic = build_intrinsic_clock(data, threshold=1e-5)

    assert np.all(np.diff(tick["timestamp"]) > 0)
    assert np.all(np.diff(intrinsic["timestamp"]) > 0)
    assert tick["timestamp"][-1] <= data["timestamp"][-1]
    assert intrinsic["timestamp"][-1] <= data["timestamp"][-1]


def consolidate_fixture() -> tuple[dict[str, np.ndarray], dict[str, int]]:
    from scripts.wp1.eurusd_certificate import consolidate_minutes

    return consolidate_minutes(_minutes())


def test_episode_response_detects_large_burst_more_often() -> None:
    from scripts.wp1.eurusd_certificate import episode_successes

    rng = np.random.default_rng(4)
    segments = rng.normal(0.0, 0.001, size=(200, 419))
    threshold = np.full((200, 360), 0.0012)

    low = episode_successes(segments, threshold, scale=1.0)
    high = episode_successes(segments, threshold, scale=4.0)

    assert int(np.sum(high)) > int(np.sum(low))


def test_value_utility_applies_half_spread_slippage_and_turnover() -> None:
    from scripts.wp1.eurusd_certificate import paired_utility

    forward = np.array([0.01, -0.01, 0.02])
    exposure = np.array([1.0, 0.5, 0.0])
    spread_fraction = np.array([0.0002, 0.0002, 0.0002])
    difference, costs = paired_utility(
        forward,
        exposure,
        spread_fraction,
        gamma=100.0,
    )

    assert costs[0] == 0.0
    assert costs[1] == 0.5 * (0.0001 + 0.0001)
    assert costs[2] == 0.5 * (0.0001 + 0.0001)
    assert difference.shape == forward.shape
