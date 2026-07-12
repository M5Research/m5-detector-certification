"""Thermodynamic profit bound: VR → MI → Kelly → transaction cost chain."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_REPO_SRC = _REPO_ROOT / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from backtest.utils import PROJECT_ROOT  # noqa: E402

import scripts._bootstrap  # noqa: F401, E402
from scripts.wp1.mutual_information import (  # noqa: E402
    KSG_BOOTSTRAP_RESAMPLES,
    estimate_mi_from_returns,
)
from scripts.wp1.precheck_b import Q_GRID, _gate_guard  # noqa: E402
from scripts.wp1.vr_significance import compute_vr_horizon_profile  # noqa: E402

__all__ = [
    "W_THERMO",
    "PRIMARY_Q",
    "DELTA_GRID_MAX",
    "TAKER_FILL_BPS",
    "MAKER_FILL_BPS",
    "TAKER_RT_BPS",
    "MAKER_RT_BPS",
    "BOOT_SEED",
    "vr_departure_to_gaussian_mi",
    "mi_to_gmax_bps",
    "vr_to_mutual_information",
    "load_delta_star_primary",
    "compute_thermodynamic_bound",
    "main",
]

W_THERMO = 120
PRIMARY_Q = 5
DELTA_GRID_MAX = 0.10
TAKER_FILL_BPS = 4.5
MAKER_FILL_BPS = 1.8
TAKER_RT_BPS = (9.0, 10.0)
MAKER_RT_BPS = (3.6, 5.6)
BOOT_SEED = 43
THERMO_DIR = PROJECT_ROOT / "backtest_results" / "thermodynamic_bound"
VERDICT_DEMON = "demon_runs_at_loss"
VERDICT_EXCEEDS = "bound_exceeds_costs"


def _default_cost_schedule() -> dict:
    return {
        "taker_fill_bps": TAKER_FILL_BPS,
        "maker_fill_bps": MAKER_FILL_BPS,
        "taker_rt_bps": list(TAKER_RT_BPS),
        "maker_rt_bps": list(MAKER_RT_BPS),
    }


def vr_departure_to_gaussian_mi(vr_dep: float, q: int) -> float:
    """Convert |VR(q)-1| departure to Gaussian MI (nats) per pre-reg §5.2."""
    rho = min(abs(vr_dep) / (2.0 * (1.0 - 1.0 / q)), 1.0 - 1e-12)
    mi = -0.5 * np.log(1.0 - rho * rho)
    return float(max(mi, 0.0))


def mi_to_gmax_bps(mi_nats: float) -> float:
    """Kelly growth bound in bps per pre-reg §5.4."""
    return float(mi_nats * 10_000.0)


def vr_to_mutual_information(
    vr_departure: float,
    q: int,
    method: str = "gaussian",
    close: np.ndarray | None = None,
    seed: int = BOOT_SEED,
    n_boot: int | None = None,
) -> dict:
    """VR departure → MI via Gaussian formula or KSG on sign pairs."""
    if method == "gaussian":
        rho = min(abs(vr_departure) / (2.0 * (1.0 - 1.0 / q)), 1.0 - 1e-12)
        mi = vr_departure_to_gaussian_mi(vr_departure, q)
        return {"mi_nats": mi, "mi_stderr": None, "method": "gaussian", "rho": float(rho)}

    if method == "ksg":
        if close is None:
            raise ValueError("close is required for method='ksg'")
        close_arr = np.asarray(close, dtype=np.float64)
        if close_arr.ndim != 1 or len(close_arr) < 2:
            raise ValueError("close must be a 1-D array with length >= 2")
        returns = np.log(close_arr[1:] / close_arr[:-1])
        boot = KSG_BOOTSTRAP_RESAMPLES if n_boot is None else n_boot
        mi, stderr = estimate_mi_from_returns(
            returns,
            q,
            method="signs",
            orientation="forward",
            seed=seed,
            n_boot=boot,
        )
        return {
            "mi_nats": float(mi),
            "mi_stderr": float(stderr),
            "method": "ksg_signs_forward",
            "orientation": "signal_to_forward_payoff",
        }

    raise ValueError(f"method must be 'gaussian' or 'ksg', got {method!r}")


def _resolve_injection_dir(injection_dir: Path) -> Path:
    raw = Path(injection_dir)
    if ".." in raw.parts:
        raise ValueError("injection_dir must not contain '..'")
    resolved = raw.resolve()
    root = PROJECT_ROOT.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("injection_dir must resolve under PROJECT_ROOT") from exc
    return resolved


def load_delta_star_primary(injection_dir: Path) -> dict:
    """Load δ*₉₀ from Phase 11 injection JSON at PRIMARY_WQ (q=5, W=120)."""
    from scripts.wp1.exclusion_plot import (
        compute_delta_star_90,
        compute_pdet_per_grid_point,
        load_grid_results,
    )

    if not injection_dir.exists() or not any(injection_dir.glob("inj_*.json")):
        return {"delta_star_90": None, "status": "injection_dir_missing"}

    try:
        resolved = _resolve_injection_dir(injection_dir)
    except ValueError:
        return {"delta_star_90": None, "status": "injection_dir_invalid"}

    results = load_grid_results(resolved)
    cells = compute_pdet_per_grid_point(results)
    cell_key = (PRIMARY_Q, W_THERMO)
    if cell_key not in cells:
        return {"delta_star_90": None, "status": "primary_cell_missing"}

    cell = cells[cell_key]
    ds = compute_delta_star_90(cell["deltas"], cell["P_det"])
    if not np.isfinite(ds):
        return {
            "delta_star_90": DELTA_GRID_MAX,
            "status": "above_grid",
            "proxy_delta": DELTA_GRID_MAX,
        }
    return {"delta_star_90": float(ds), "status": "ok"}


def _verdict_for_gmax(gmax_bps: float) -> str:
    if gmax_bps < min(TAKER_RT_BPS):
        return VERDICT_DEMON
    return VERDICT_EXCEEDS


def compute_thermodynamic_bound(
    close: np.ndarray,
    q_grid: tuple[int, ...],
    cost_schedule: dict | None = None,
    injection_dir: Path | None = None,
    seed: int = BOOT_SEED,
    n_boot: int | None = None,
) -> dict:
    """Full VR→MI→Kelly→cost chain with optional δ*₉₀ envelope."""
    close_arr = np.asarray(close, dtype=np.float64)
    if close_arr.ndim != 1:
        raise ValueError("close must be 1-D")
    if cost_schedule is None:
        cost_schedule = _default_cost_schedule()

    returns = np.log(close_arr[1:] / close_arr[:-1])
    horizon = compute_vr_horizon_profile(close_arr, W=W_THERMO, q_grid=q_grid)

    delta_info: dict
    if injection_dir is not None:
        delta_info = load_delta_star_primary(Path(injection_dir))
    else:
        delta_info = {"delta_star_90": None, "status": "injection_dir_missing"}

    delta_star = delta_info.get("delta_star_90")
    envelope_available = delta_star is not None and np.isfinite(delta_star)

    per_q: list[dict] = []
    for q in q_grid:
        observed_dep = float(horizon.get(q, float("nan")))
        gauss_mi = vr_departure_to_gaussian_mi(observed_dep, q)
        gauss_gmax = mi_to_gmax_bps(gauss_mi)

        envelope_mi = None
        envelope_gmax = None
        exclusion_gap = None
        if envelope_available:
            envelope_mi = vr_departure_to_gaussian_mi(float(delta_star), q)
            envelope_gmax = mi_to_gmax_bps(envelope_mi)
            exclusion_gap = float(envelope_gmax - gauss_gmax)

        ksg = vr_to_mutual_information(
            0.0, q, method="ksg", close=close_arr, seed=seed, n_boot=n_boot
        )
        ksg_gmax = mi_to_gmax_bps(ksg["mi_nats"])
        verdict_gaussian = _verdict_for_gmax(gauss_gmax)
        verdict_sign_pair = _verdict_for_gmax(ksg_gmax)

        per_q.append(
            {
                "q": int(q),
                "observed_vr_dep": observed_dep,
                "gaussian_mi_nats": gauss_mi,
                "gaussian_gmax_bps": gauss_gmax,
                "envelope_delta_star_90": float(delta_star) if envelope_available else None,
                "envelope_gaussian_mi_nats": envelope_mi,
                "envelope_gaussian_gmax_bps": envelope_gmax,
                "ksg_mi_nats": ksg["mi_nats"],
                "ksg_mi_stderr": ksg["mi_stderr"],
                "ksg_gmax_bps": ksg_gmax,
                "exclusion_gap_bps": exclusion_gap,
                "verdict_gaussian_vs_taker": verdict_gaussian,
                "verdict_sign_pair_vs_taker": verdict_sign_pair,
                "verdict_vs_taker": verdict_sign_pair,
            }
        )

    primary = next(row for row in per_q if row["q"] == PRIMARY_Q)
    headline_gap = primary.get("exclusion_gap_bps")

    return {
        "per_q_results": per_q,
        "headline_verdict": _verdict_for_gmax(primary["ksg_gmax_bps"]),
        "headline_q": PRIMARY_Q,
        "delta_star_info": delta_info,
        "exclusion_gap_summary": {
            "headline_gap_bps": headline_gap,
            "headline_q": PRIMARY_Q,
        },
        "frozen_params": {
            "W": W_THERMO,
            "q_grid": list(q_grid),
            "primary_q": PRIMARY_Q,
            "delta_grid_max": DELTA_GRID_MAX,
        },
        "cost_schedule": cost_schedule,
        "boot_seed": seed,
    }


def _synthetic_smoke() -> None:
    rng = np.random.default_rng(BOOT_SEED)
    n = 5000
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    report = compute_thermodynamic_bound(
        close, Q_GRID, _default_cost_schedule(), injection_dir=None, seed=BOOT_SEED, n_boot=25
    )
    assert len(report["per_q_results"]) == len(Q_GRID)
    print(f"synthetic_ok verdict={report['headline_verdict']}")


def main() -> int:
    import datetime as _dt

    from scripts.wp1.py_engine import load_and_clean

    print("D-09 gate-guard: checking dual pre-registration integrity...", flush=True)
    prereg_results = _gate_guard()
    prereg_commit = prereg_results.get(
        ".planning/phases/11-signal-injection-the-ligo-calibration/09-PREREGISTRATION.md",
        next(iter(prereg_results.values()), ""),
    )
    print(f"D-09 gate-guard PASSED. prereg_commit={prereg_commit[:12]}...")

    run_date = datetime.now(tz=_dt.UTC).strftime("%Y%m%d_%H%M%S")
    try:
        code_commit = subprocess.run(
            ["git", "log", "--format=%H", "-1"],
            capture_output=True,
            text=True,
            cwd=str(_REPO_ROOT),
        ).stdout.strip()
    except Exception:
        code_commit = ""

    start_ms = int(_dt.datetime(2021, 1, 1, tzinfo=_dt.UTC).timestamp() * 1000)
    end_ms = int(_dt.datetime(2025, 12, 31, 23, 59, 59, tzinfo=_dt.UTC).timestamp() * 1000)
    holdout_boundary_ms = int(_dt.datetime(2026, 1, 1, tzinfo=_dt.UTC).timestamp() * 1000)
    assert end_ms < holdout_boundary_ms, "D-07: 2026 holdout must not be loaded"

    data_path = str(PROJECT_ROOT / "data" / "binance_futures")
    print("Loading BTCUSDT 2021-2025...")
    ohlcv = load_and_clean(
        data_path=data_path,
        symbol="BTCUSDT",
        start_ms=start_ms,
        end_ms=end_ms,
        max_gap_allowed_mins=60,
    )
    ts_end = _dt.datetime.fromtimestamp(ohlcv["timestamp"][-1] / 1000, tz=_dt.UTC)
    assert ts_end.year < 2026, "D-07: last bar must be before 2026"

    injection_dir = PROJECT_ROOT / "data" / "injection_runs"
    report = compute_thermodynamic_bound(
        ohlcv["close"],
        Q_GRID,
        _default_cost_schedule(),
        injection_dir=injection_dir,
        seed=BOOT_SEED,
    )
    report.update(
        {
            "run_id": run_date,
            "prereg_commit": prereg_commit,
            "code_commit": code_commit,
            "run_date": run_date,
        }
    )

    THERMO_DIR.mkdir(parents=True, exist_ok=True)
    out_path = THERMO_DIR / f"thermo_report_{run_date}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Verdict: {report['headline_verdict']}")
    print(f"Report: {out_path}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--run":
        raise SystemExit(main())
    _synthetic_smoke()
