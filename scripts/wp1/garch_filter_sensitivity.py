"""GARCH pre-filter sensitivity for injection calibration (exploratory).

Compares P_det at primary cell under GARCH-scaled vs constant-sigma injection.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backtest.utils import PROJECT_ROOT
import scripts._bootstrap  # noqa: F401
from scripts.wp1.precheck_b import run_precheck_b_cascade
from scripts.wp1.signal_injection import resolve_phi
from scripts.wp1.signal_injector import (
    clopper_pearson_ci,
    hash_combine,
    inject_ar1_autocorrelation,
)

INJECTION_DIR = PROJECT_ROOT / "data" / "injection_runs"
PRECOMPUTED_NPZ = INJECTION_DIR / "precomputed.npz"
BOOT_SEED = 43
N_MC = 50  # lighter probe; sensitivity diagnostic only
DELTA = 0.15
Q = 5
W = 120


def _run_variant(r_real, sigma_t, timestamps, label: str) -> dict:
    phi = resolve_phi(DELTA, Q)
    grid_hash = hash_combine(BOOT_SEED, W, Q * 1000 + int(DELTA * 100000))
    n_fires = 0
    for mc_idx in range(N_MC):
        mc_seed = hash_combine(BOOT_SEED, mc_idx, grid_hash)
        r_inj = inject_ar1_autocorrelation(r_real, phi, mc_seed, sigma_t)
        close_inj = np.exp(np.cumsum(r_inj))
        result = run_precheck_b_cascade(close_inj, timestamps, mc_seed, diagnostics=False)
        n_fires += int(result["cascade_fired"])
    p_det = n_fires / N_MC
    ci_lo, ci_hi = clopper_pearson_ci(n_fires, N_MC)
    return {
        "variant": label,
        "sigma_mode": label,
        "delta": DELTA,
        "q": Q,
        "W": W,
        "phi": float(phi),
        "N_mc": N_MC,
        "P_det": p_det,
        "ci_95_lo": ci_lo,
        "ci_95_hi": ci_hi,
    }


def main() -> int:
    data = np.load(PRECOMPUTED_NPZ)
    r_real = data["r_real"]
    sigma_garch = data["sigma_t"]
    timestamps = data["timestamps"]

    sigma_const = np.full_like(r_real, float(np.median(sigma_garch)))
    variants = [
        _run_variant(r_real, sigma_garch, timestamps, "garch"),
        _run_variant(r_real, sigma_const, timestamps, "constant_median"),
    ]

    out = {
        "probe": "garch_filter_sensitivity",
        "primary_cell": {"delta": DELTA, "q": Q, "W": W},
        "variants": variants,
        "provenance": {
            "exploratory_addendum": True,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        },
    }
    out_path = INJECTION_DIR / "garch_filter_sensitivity.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    for v in variants:
        print(f"  {v['variant']}: P_det={v['P_det']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
