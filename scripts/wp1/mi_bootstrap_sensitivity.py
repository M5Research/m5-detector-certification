"""MI bootstrap resampling sensitivity (exploratory diagnostic).

Compares sign-pair MI stderr under i.i.d. pair resampling vs circular block
resampling at block lengths {60, 120, 240}.  Uses the same discrete sign-pair
estimator as the production thermodynamic bound.
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
from scripts.wp1.mutual_information import (
    BOOTSTRAP_PAIR_SAMPLE_SIZE,
    BOOT_SEED,
    build_sign_pairs,
    estimate_mi_from_returns,
)

OUT_PATH = PROJECT_ROOT / "backtest_results" / "thermodynamic_bound" / "mi_bootstrap_sensitivity.json"
Q_GRID = (5,)
BLOCK_LENGTHS = (0, 60, 120, 240)
N_BOOT_PROBE = 100


def _block_bootstrap_stderr_signs(
    x: np.ndarray,
    y: np.ndarray,
    *,
    block_len: int,
    n_boot: int,
    seed: int,
) -> tuple[float, float]:
    from scripts.wp1.mutual_information import _bootstrap_stderr_signs, _discrete_mi_from_signs

    n = len(x)
    rng = np.random.default_rng(seed)
    sample_n = min(n, BOOTSTRAP_PAIR_SAMPLE_SIZE)
    idx_est = rng.choice(n, size=sample_n, replace=False) if n > sample_n else np.arange(n)
    mi = _discrete_mi_from_signs(x[idx_est], y[idx_est])

    if block_len <= 0:
        se = _bootstrap_stderr_signs(x, y, n_boot=n_boot, seed=seed)
        return float(mi), float(se)

    boots = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        n_blocks = int(np.ceil(sample_n / block_len))
        starts = rng.integers(0, max(n - block_len, 1), size=n_blocks)
        idx = np.concatenate([np.arange(s, min(s + block_len, n)) for s in starts])[:sample_n]
        idx = np.clip(idx, 0, n - 1)
        boots[b] = _discrete_mi_from_signs(x[idx], y[idx])
    return float(mi), float(np.std(boots, ddof=1))


def main() -> int:
    npz = PROJECT_ROOT / "data" / "injection_runs" / "precomputed.npz"
    print("Loading precomputed returns ...", flush=True)
    r = np.load(npz)["r_real"]
    print(f"Loaded {len(r):,} bars.", flush=True)

    rows = []
    for q in Q_GRID:
        sx, sy = build_sign_pairs(r, q)
        for bl in BLOCK_LENGTHS:
            mi, se = _block_bootstrap_stderr_signs(
                sx, sy, block_len=bl, n_boot=N_BOOT_PROBE, seed=BOOT_SEED + q + bl
            )
            rows.append({
                "q": q,
                "block_length": bl,
                "resampling": "iid_pairs" if bl == 0 else f"block_{bl}",
                "mi_nats": mi,
                "mi_stderr": se,
                "gmax_bps": mi * 10_000.0,
                "gmax_stderr_bps": se * 10_000.0,
            })

    # Production reference (thermo report)
    prod_mi, prod_se = estimate_mi_from_returns(r, q=5, method="signs")
    out = {
        "probe": "mi_bootstrap_sensitivity",
        "n_boot": N_BOOT_PROBE,
        "production_reference": {"q": 5, "mi_nats": prod_mi, "mi_stderr": prod_se},
        "rows": rows,
        "provenance": {"timestamp": datetime.utcnow().isoformat() + "Z"},
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_PATH}")
    for row in rows:
        print(
            f"  q=5 {row['resampling']}: G_max={row['gmax_bps']:.2f} "
            f"± {row['gmax_stderr_bps']:.2f} bps"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
