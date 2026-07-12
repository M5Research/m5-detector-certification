"""Fast probe: median z_m2 vs delta at primary cell (VR path only, no full cascade)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import scripts._bootstrap  # noqa: F401
from scripts.wp1.signal_injector import inject_ar1_autocorrelation, invert_phi_delta_mapping
from scripts.wp1.vr_significance import (
    apply_holm_b,
    compute_rolling_vr_and_z_strided,
    compute_vr_significance,
)
from strategies.vol_regime_switch.regime_population import non_overlapping_samples

W = 120
Q = 5
Q_GRID = (2, 5, 15, 60)
DELTAS = [0.10, 0.15, 0.20, 0.30, 0.50, 0.70, 1.0, 1.5, 2.0]
N_BARS = 50_000


def cascade_fires(close: np.ndarray, regime: np.ndarray) -> tuple[bool, float, list[float]]:
    holm_p: list[float] = []
    z_by_q: dict[int, float] = {}
    for q in Q_GRID:
        vr, z = compute_rolling_vr_and_z_strided(close, W=W, q=q, stride=W)
        pred, _, idx = non_overlapping_samples(vr, regime, stride=W)
        sig = compute_vr_significance(pred, z[idx])
        holm_p.append(sig["p_twotailed"])
        z_by_q[q] = sig["median_z_m2"]
    adj = apply_holm_b(holm_p)
    fired = any(adj[i] < 0.05 and z_by_q[Q_GRID[i]] > 0 for i in range(len(Q_GRID)))
    return fired, z_by_q[Q], holm_p


def main() -> int:
    print("Loading precomputed returns ...", flush=True)
    npz = _REPO_ROOT / "data" / "injection_runs" / "precomputed.npz"
    r_real = np.load(npz)["r_real"][:N_BARS]
    print(f"Loaded {len(r_real):,} bars.", flush=True)
    sigma = np.ones_like(r_real)
    regime = np.zeros(len(r_real), dtype=np.int8)

    print(f"Probe on {len(r_real):,} bars, primary cell W={W}, q={Q}")
    print(f"{'delta':>8} {'z_q5':>8} {'fire':>6} {'min_holm_p':>12}")
    for delta in DELTAS:
        phi = invert_phi_delta_mapping(delta, Q)
        r_inj = inject_ar1_autocorrelation(r_real, phi, seed=43, sigma_t=sigma)
        close = np.exp(np.cumsum(r_inj))
        fired, z5, holm_p = cascade_fires(close, regime)
        print(f"{delta:8g} {z5:8.3f} {str(fired):>6} {min(apply_holm_b(holm_p)):12.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
