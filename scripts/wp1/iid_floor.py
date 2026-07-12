"""Synthetic i.i.d. VR sampling floors for the signal-injection scale check."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backtest.utils import PROJECT_ROOT  # noqa: E402

import scripts._bootstrap  # noqa: F401, E402
from scripts.wp1 import vr_significance  # noqa: E402

DEFAULT_W_GRID = (60, 120, 240)
DEFAULT_Q_GRID = (2, 5, 15, 60)
DEFAULT_SEED = 43
DEFAULT_N_PATHS = 250
DEFAULT_N_OBS = 50_000
DEFAULT_SIGMA = 0.001
OUT_DIR = PROJECT_ROOT / "backtest_results" / "iid_floor"

__all__ = [
    "DEFAULT_Q_GRID",
    "DEFAULT_W_GRID",
    "simulate_iid_vr_floor",
]


def _non_overlapping_finite(vr_arr: np.ndarray, stride: int) -> np.ndarray:
    idx = np.arange(len(vr_arr))
    mask = ((idx - (stride - 1)) % stride == 0) & (idx >= stride - 1)
    return np.asarray(vr_arr[mask & np.isfinite(vr_arr)], dtype=np.float64)


def simulate_iid_vr_floor(
    *,
    n_paths: int = DEFAULT_N_PATHS,
    n_obs: int = DEFAULT_N_OBS,
    W_grid: tuple[int, ...] = DEFAULT_W_GRID,
    q_grid: tuple[int, ...] = DEFAULT_Q_GRID,
    seed: int = DEFAULT_SEED,
    sigma: float = DEFAULT_SIGMA,
) -> list[dict]:
    """Return per-window i.i.d. VR floor quantiles for each (W, q)."""
    if n_paths <= 0:
        raise ValueError("n_paths must be positive")
    if n_obs <= max(W_grid):
        raise ValueError("n_obs must exceed the largest W")
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")

    rng = np.random.default_rng(seed)
    samples: dict[tuple[int, int], list[np.ndarray]] = {
        (int(W), int(q)): [] for W in W_grid for q in q_grid
    }

    for _ in range(n_paths):
        returns = rng.normal(0.0, sigma, n_obs)
        close = np.exp(np.cumsum(returns))
        for W in W_grid:
            for q in q_grid:
                vr_arr, _ = vr_significance.compute_rolling_vr_and_z(close, W=int(W), q=int(q))
                samples[(int(W), int(q))].append(_non_overlapping_finite(vr_arr, stride=int(W)))

    rows: list[dict] = []
    for W in W_grid:
        for q in q_grid:
            pooled = np.concatenate(samples[(int(W), int(q))])
            rows.append(
                {
                    "W": int(W),
                    "q": int(q),
                    "n_paths": int(n_paths),
                    "n_obs": int(n_obs),
                    "n_samples_total": int(len(pooled)),
                    "delta_iid_p50": float(np.quantile(pooled, 0.50)),
                    "delta_iid_p95": float(np.quantile(pooled, 0.95)),
                }
            )
    return rows


def _parse_int_tuple(raw: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in raw.split(",") if x.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-paths", type=int, default=DEFAULT_N_PATHS)
    parser.add_argument("--n-obs", type=int, default=DEFAULT_N_OBS)
    parser.add_argument("--W-grid", type=_parse_int_tuple, default=DEFAULT_W_GRID)
    parser.add_argument("--q-grid", type=_parse_int_tuple, default=DEFAULT_Q_GRID)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    rows = simulate_iid_vr_floor(
        n_paths=args.n_paths,
        n_obs=args.n_obs,
        W_grid=args.W_grid,
        q_grid=args.q_grid,
        seed=args.seed,
    )
    payload = {
        "description": "Synthetic Gaussian i.i.d. per-window |VR(q)-1| floor.",
        "seed": args.seed,
        "rows": rows,
    }

    out_path = args.out
    if out_path is None:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUT_DIR / f"iid_floor_seed{args.seed}_n{args.n_paths}_obs{args.n_obs}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Wrote {out_path}")
    for row in rows:
        print(
            f"W={row['W']:>3} q={row['q']:>2} "
            f"p50={row['delta_iid_p50']:.4f} p95={row['delta_iid_p95']:.4f} "
            f"n={row['n_samples_total']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
