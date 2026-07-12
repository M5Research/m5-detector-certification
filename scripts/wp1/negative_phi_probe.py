"""Exploratory negative-AR(1) injection probe (post-freeze addendum).

Tests direction-specificity of the one-sided sign gate: at matched positive
phi, negative phi should yield Z_q < 0 and P_det = 0 regardless of |VR-1|.
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
N_MC = 50  # diagnostic: sign-gate directionality (P_det expected 0)

# Primary cell + δ values used in exploratory bracket
PROBE_CELLS = [
    {"delta": 0.10, "q": 5, "W": 120},
    {"delta": 0.15, "q": 5, "W": 120},
    {"delta": 0.20, "q": 5, "W": 120},
]


def run_negative_phi_probe() -> dict:
    data = np.load(PRECOMPUTED_NPZ)
    r_real = data["r_real"]
    sigma_t = data["sigma_t"]
    timestamps = data["timestamps"]

    results = []
    for cell in PROBE_CELLS:
        delta = cell["delta"]
        q = cell["q"]
        W = cell["W"]
        phi_pos = resolve_phi(delta, q)
        phi_neg = -phi_pos
        grid_hash = hash_combine(BOOT_SEED, int(W), int(q) * 1000 + int(delta * 100000))

        n_fires = 0
        z_samples = []
        for mc_idx in range(N_MC):
            mc_seed = hash_combine(BOOT_SEED, mc_idx, grid_hash)
            r_inj = inject_ar1_autocorrelation(
                r_real, phi_neg, mc_seed, sigma_t, allow_negative_phi=True
            )
            close_inj = np.exp(np.cumsum(r_inj))
            cascade = run_precheck_b_cascade(close_inj, timestamps, mc_seed, diagnostics=False)
            n_fires += int(cascade["cascade_fired"])
            for row in cascade["holm_correction"]["ordered"]:
                if row["q"] == q:
                    z_samples.append(row["median_z_m2"])
                    break

        p_det = n_fires / N_MC
        ci_lo, ci_hi = clopper_pearson_ci(n_fires, N_MC)
        results.append({
            "delta_target": delta,
            "q": q,
            "W": W,
            "phi_positive_reference": float(phi_pos),
            "phi_injected": float(phi_neg),
            "injection_sign": "negative",
            "N_mc": N_MC,
            "n_fires": n_fires,
            "P_det": p_det,
            "ci_95_lo": ci_lo,
            "ci_95_hi": ci_hi,
            "median_z_q": float(np.median(z_samples)) if z_samples else None,
            "provenance": {
                "exploratory_addendum": True,
                "negative_phi_probe": True,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            },
        })

    out = {
        "probe": "negative_ar1_exploratory",
        "description": "Direction-specific exclusion bound diagnostic",
        "cells": results,
    }
    out_path = INJECTION_DIR / "negative_phi_probe_primary.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    for row in results:
        print(
            f"  delta={row['delta_target']} phi={row['phi_injected']:.4f} "
            f"P_det={row['P_det']} median_z={row['median_z_q']}"
        )
    return out


if __name__ == "__main__":
    run_negative_phi_probe()
