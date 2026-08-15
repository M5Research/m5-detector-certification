"""Protocol-v5 BTCUSDT q=2 successor calibration and evidence runner."""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import norm

import scripts._bootstrap  # noqa: F401

from certificate.information import primary_block_length, stationary_block_indices
from certificate.statistics import exact_binomial_bounds

W = 120
Q = 2
RECENTER_CENTER = -0.004886325945105274
RECENTER_SCALE = 0.010302944723275254
AMPLITUDES = (0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.15)
TRANSPORT_MARGIN = 0.019748
N_NULL = 10_000
N_INJECTION = 2_000
N_WINDOWS = 100


def _median_vr2(returns: np.ndarray) -> np.ndarray:
    """Median q=2 variance-ratio departure across each draw's windows."""
    one_var = np.var(returns, axis=-1, ddof=1)
    two = returns[..., :-1] + returns[..., 1:]
    two_var = np.var(two, axis=-1, ddof=1)
    vr_departure = np.divide(
        two_var,
        2.0 * one_var,
        out=np.zeros_like(two_var),
        where=one_var > 0.0,
    ) - 1.0
    return np.median(vr_departure, axis=1)


def simulate_recentered_vr_draws(
    *,
    phi: float,
    n_draws: int,
    n_windows: int = N_WINDOWS,
    seed: int,
    batch_size: int = 128,
) -> np.ndarray:
    """Simulate the existing Gaussian AR(1) family and return recentered statistics."""
    if not 0.0 <= phi < 1.0:
        raise ValueError("phi must be in [0, 1)")
    rng = np.random.default_rng(seed)
    output = np.empty(n_draws, dtype=np.float64)
    scale = float(np.sqrt(1.0 - phi * phi))
    for start in range(0, n_draws, batch_size):
        stop = min(n_draws, start + batch_size)
        innovation = rng.normal(size=(stop - start, n_windows, W))
        series = np.empty_like(innovation)
        series[..., 0] = innovation[..., 0]
        for index in range(1, W):
            series[..., index] = phi * series[..., index - 1] + scale * innovation[..., index]
        raw = _median_vr2(series)
        output[start:stop] = (raw - RECENTER_CENTER) / RECENTER_SCALE
    return output


def calibrate_successor(
    *,
    amplitudes: list[float],
    phi_by_amplitude: dict[float, float],
    n_null: int = N_NULL,
    n_injection: int = N_INJECTION,
    n_windows: int = N_WINDOWS,
    alpha: float = 0.025,
    seed: int = 43,
) -> dict[str, Any]:
    """Run size and power calibration at the revision-specific alpha."""
    critical = float(norm.ppf(1.0 - alpha))
    null = simulate_recentered_vr_draws(
        phi=0.0,
        n_draws=n_null,
        n_windows=n_windows,
        seed=seed,
    )
    null_fires = int(np.sum(null > critical))
    size_bounds = exact_binomial_bounds(null_fires, n_null, alpha=alpha)
    size = {
        "draws": n_null,
        "fires": null_fires,
        "rate": null_fires / n_null,
        "bounds": size_bounds,
        "threshold": critical,
        "state": "pass" if size_bounds["upper"] <= alpha else "fail",
    }
    power: list[dict[str, Any]] = []
    for index, amplitude in enumerate(amplitudes):
        draws = simulate_recentered_vr_draws(
            phi=float(phi_by_amplitude[amplitude]),
            n_draws=n_injection,
            n_windows=n_windows,
            seed=seed + 10_000 * (index + 1),
        )
        fires = int(np.sum(draws > critical))
        bounds = exact_binomial_bounds(fires, n_injection, alpha=alpha)
        power.append(
            {
                "amplitude": float(amplitude),
                "phi": float(phi_by_amplitude[amplitude]),
                "draws": n_injection,
                "fires": fires,
                "rate": fires / n_injection,
                "bounds": bounds,
            }
        )
    cmde = next((row["amplitude"] for row in power if row["bounds"]["lower"] >= 0.90), None)
    return {
        "method": "Gaussian AR(1), 100 independent W=120 windows per draw, median q=2 VR departure",
        "recenter_center": RECENTER_CENTER,
        "recenter_scale": RECENTER_SCALE,
        "alpha": alpha,
        "size": size,
        "power": power,
        "cmde_90": cmde,
        "power_state": "pass" if cmde is not None else "fail",
    }


def transport_tost(
    calendar: np.ndarray,
    volume: np.ndarray,
    *,
    margin: float,
    alpha: float,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    """Dependence-aware TOST for the single frozen calendar--volume comparison."""
    left = np.asarray(calendar, dtype=np.float64)
    right = np.asarray(volume, dtype=np.float64)
    left = left[np.isfinite(left)]
    right = right[np.isfinite(right)]
    if left.size < 2 or right.size < 2:
        raise ValueError("transport samples are too small")
    rng = np.random.default_rng(seed)
    left_block = primary_block_length(left.size)
    right_block = primary_block_length(right.size)
    boot = np.empty(n_boot, dtype=np.float64)
    for index in range(n_boot):
        left_idx = stationary_block_indices(left.size, left_block, rng)
        right_idx = stationary_block_indices(right.size, right_block, rng)
        boot[index] = float(np.median(left[left_idx]) - np.median(right[right_idx]))
    point = float(np.median(left) - np.median(right))
    lower, upper = np.quantile(boot, [alpha, 1.0 - alpha])
    p_lower = float((np.sum(boot <= -margin) + 1) / (n_boot + 1))
    p_upper = float((np.sum(boot >= margin) + 1) / (n_boot + 1))
    passed = float(lower) > -margin and float(upper) < margin
    return {
        "estimand": point,
        "margin": margin,
        "ci": [float(lower), float(upper)],
        "tost_p_value": max(p_lower, p_upper),
        "n_calendar": int(left.size),
        "n_volume": int(right.size),
        "n_boot": n_boot,
        "state": "pass" if passed else "fail",
    }
