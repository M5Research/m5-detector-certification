"""Estimate detector-contingent information for the frozen VR cascade trigger.

The artifact separates two objects reviewers often conflate:

* path-level Monte Carlo cascade-firing cells, whose trigger entropy can be zero
  inside a saturated cell; and
* time-local, non-overlapping Holm-corrected VR trigger labels, which can be
  paired with forward return signs for detector-contingent MI.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.wp1.harmonized_benchmark import (  # noqa: E402
    DEFAULT_COST_BPS,
    _load_btc_window,
    _to_serializable,
    discrete_entropy,
    vr_holm_trigger_information,
)

DEFAULT_INJECTION_CELLS = (
    Path("data/injection_runs/inj_d0.15_q2_W120.json"),
    Path("data/injection_runs/inj_d0.15_q5_W120.json"),
    Path("data/injection_runs/inj_d0.2_q5_W120.json"),
    Path("data/injection_runs/inj_d0.3_q5_W120.json"),
)


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    return "unavailable"


def injection_cell_trigger_summary(path: Path) -> dict[str, Any]:
    """Summarize trigger entropy for one precomputed injection cell."""
    data = json.loads(path.read_text(encoding="utf-8"))
    per_draw = data.get("per_draw") or []
    labels = np.asarray([bool(row.get("cascade_fired")) for row in per_draw], dtype=np.int8)
    n_draws = int(len(labels))
    n_fires = int(np.sum(labels))
    entropy = discrete_entropy(labels)
    return {
        "path": str(path.as_posix()),
        "W": int(data.get("W", 0)),
        "q": int(data.get("q", 0)),
        "delta_target": float(data.get("delta_target", math.nan)),
        "N_mc": int(data.get("N_mc", n_draws)),
        "n_draws": n_draws,
        "n_fires": n_fires,
        "P_det": float(data.get("P_det", n_fires / n_draws if n_draws else 0.0)),
        "trigger_entropy_nats": entropy,
        "mi_upper_bound_nats": entropy,
        "mi_nats_if_conditioning_within_saturated_cell": 0.0 if entropy == 0.0 else None,
        "note": (
            "Within-cell path-level trigger MI is bounded by H(D). "
            "A saturated 0/200 or 200/200 cell has H(D)=0 and therefore MI=0."
        ),
    }


def build_artifact(
    *,
    symbol: str,
    start_date: str,
    end_date: str,
    W: int,
    q_values: list[int],
    injection_cells: list[Path],
    cost_bps: float,
    n_boot: int,
    n_perm: int,
    seed: int,
) -> dict[str, Any]:
    close, timestamps, span = _load_btc_window(symbol, start_date, end_date)
    time_local: dict[str, Any] = {}
    for q_val in q_values:
        time_local[str(q_val)] = vr_holm_trigger_information(
            close,
            timestamps,
            W=W,
            target_q=int(q_val),
            stride=W,
            cost_bps=cost_bps,
            n_boot=n_boot,
            n_perm=n_perm,
            seed=seed + int(q_val),
        )

    cell_summaries = [
        injection_cell_trigger_summary((_REPO_ROOT / path).resolve() if not path.is_absolute() else path)
        for path in injection_cells
    ]
    return {
        "schema_version": 1,
        "artifact": "vr_detector_contingent_mi",
        "data_span": span,
        "detector": {
            "name": "holm_corrected_vr_cascade_trigger",
            "W": int(W),
            "q_grid": [2, 5, 15, 60],
            "label": (
                "D_t(q)=1 iff the q-specific M2 z-statistic is positive and "
                "its two-sided p-value survives the frozen Holm four-q family."
            ),
            "decision_sampling": "non_overlapping_stride_W",
            "soft_outputs_used": False,
        },
        "time_local_detector_mi": time_local,
        "monte_carlo_cell_trigger_entropy": cell_summaries,
        "interpretation": (
            "The Monte Carlo cell trigger is a path-level saturated label in the "
            "cited q=2 and q=5 cells, so within-cell MI is zero by H(D)=0. "
            "The time-local rows report the detector-contingent MI estimand used "
            "for economic work checks."
        ),
        "provenance": {
            "code_commit": _git_commit(),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS", ""),
            "seed": int(seed),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--W", type=int, default=120)
    parser.add_argument("--q-values", type=int, nargs="+", default=[2, 5])
    parser.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS)
    parser.add_argument("--n-boot", type=int, default=500)
    parser.add_argument("--n-perm", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument(
        "--injection-cells",
        type=Path,
        nargs="+",
        default=list(DEFAULT_INJECTION_CELLS),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("backtest_results/thermodynamic_bound/vr_detector_mi.json"),
    )
    args = parser.parse_args(argv)

    artifact = build_artifact(
        symbol=args.symbol,
        start_date=args.start,
        end_date=args.end,
        W=args.W,
        q_values=args.q_values,
        injection_cells=args.injection_cells,
        cost_bps=args.cost_bps,
        n_boot=args.n_boot,
        n_perm=args.n_perm,
        seed=args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(_to_serializable(artifact), indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
