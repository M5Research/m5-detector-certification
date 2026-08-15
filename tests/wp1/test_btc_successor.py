"""Protocol-v5 BTC q=2 successor tests."""
from __future__ import annotations

import numpy as np


def test_vr2_draw_statistic_is_positive_for_positive_ar1() -> None:
    from scripts.wp1.btc_successor import simulate_recentered_vr_draws

    null = simulate_recentered_vr_draws(phi=0.0, n_draws=100, n_windows=20, seed=9)
    signal = simulate_recentered_vr_draws(phi=0.35, n_draws=100, n_windows=20, seed=9)

    assert float(np.median(signal)) > float(np.median(null))


def test_calibration_uses_declared_draw_counts_and_exact_bounds() -> None:
    from scripts.wp1.btc_successor import calibrate_successor

    result = calibrate_successor(
        amplitudes=[0.0005, 0.02],
        phi_by_amplitude={0.0005: 0.001, 0.02: 0.2},
        n_null=120,
        n_injection=80,
        n_windows=12,
        alpha=0.025,
        seed=11,
    )

    assert result["size"]["draws"] == 120
    assert [row["draws"] for row in result["power"]] == [80, 80]
    assert all(row["bounds"]["lower"] <= row["rate"] <= row["bounds"]["upper"] for row in result["power"])
    assert result == calibrate_successor(
        amplitudes=[0.0005, 0.02],
        phi_by_amplitude={0.0005: 0.001, 0.02: 0.2},
        n_null=120,
        n_injection=80,
        n_windows=12,
        alpha=0.025,
        seed=11,
    )


def test_transport_tost_rejects_difference_outside_margin() -> None:
    from scripts.wp1.btc_successor import transport_tost

    calendar = np.linspace(0.0, 0.01, 200)
    close = transport_tost(calendar, calendar + 0.001, margin=0.02, alpha=0.025, n_boot=199, seed=3)
    far = transport_tost(calendar, calendar + 0.05, margin=0.02, alpha=0.025, n_boot=199, seed=3)

    assert close["state"] == "pass"
    assert far["state"] == "fail"
