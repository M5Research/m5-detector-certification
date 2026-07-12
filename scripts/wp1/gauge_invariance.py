"""Time-gauge invariance VR pipeline, TOST equivalence test, and orchestrator."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from itertools import combinations
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
import scripts.wp1.vr_significance as vr_significance  # noqa: E402
from scripts.wp1.gate_analysis import WQ_GRID  # noqa: E402
from scripts.wp1.gauge_bars import (  # noqa: E402
    PRIMARY_RV_THRESHOLD,
    PRIMARY_VOLUME_THRESHOLD_BTC,
    RV_THRESHOLDS,
    VOLUME_THRESHOLDS_BTC,
    build_intrinsic_time_bars,
    build_volume_bars,
    compute_bar_duration_stats,
    summarize_overshoot,
)
from scripts.wp1.precheck_b import Q_GRID, _gate_guard  # noqa: E402
from strategies.vol_regime_switch.rolling_quantile_detector import (  # noqa: E402
    RollingQuantileDetector,
)

__all__ = [
    "all_bars_non_overlapping",
    "calibrate_wq_calendar",
    "build_gauge_series",
    "run_vr_pipeline_for_gauge",
    "fit_gauge_garch_noise_floor",
    "compute_regime_population",
    "run_sensitivity_diagnostics",
    "test_gauge_invariance",
    "compute_tost_epsilon",
    "PRACTICAL_TOST_EPSILON",
]

GAUGE_DIR = PROJECT_ROOT / "backtest_results" / "gauge_invariance"
BOOT_SEED = 43
W_PRIMARY = 120
PRACTICAL_TOST_EPSILON = 0.02
_NORM_Z_90 = 1.6448536269514722
_NORM_Z_95 = 1.959963984540054
VERDICT_INVARIANT = "gauge-invariant within margin"
VERDICT_VIOLATION = "gauge equivalence not certified"
PAIR_LABELS = ("clock-volume", "clock-intrinsic", "volume-intrinsic")


def all_bars_non_overlapping(
    series: np.ndarray,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    """All-bars non-overlapping sampler (no regime conditioning)."""
    series = np.asarray(series, dtype=np.float64)
    idx = np.arange(len(series))
    stride_mask = ((idx - (stride - 1)) % stride == 0) & (idx >= stride - 1)
    sample_mask = stride_mask & np.isfinite(series)
    retained_idx = idx[sample_mask]
    return series[sample_mask], retained_idx


def calibrate_wq_calendar(
    wq_grid: list[tuple[int, int]],
    median_duration_min: float,
) -> list[dict]:
    """Pre-calibrate W/q for fixed-calendar-span view."""
    if median_duration_min <= 0.0:
        raise ValueError("median_duration_min must be positive")
    scale = 1.0 / median_duration_min
    rows: list[dict] = []
    for w_clock, q_clock in wq_grid:
        q_raw = round(q_clock * scale)
        w_raw = round(w_clock * scale)
        q_adj = max(2, q_raw)
        w_adj = max(q_adj + 10, w_raw)
        truncated = bool(q_raw < 2 or w_raw < q_adj + 10)
        rows.append(
            {
                "W_clock": w_clock,
                "q_clock": q_clock,
                "W_adj": w_adj,
                "q_adj": q_adj,
                "scale_factor": scale,
                "truncated": truncated,
            }
        )
    return rows


def build_gauge_series(
    ohlcv: dict,
    gauge: str,
    volume_threshold: float | None = None,
    rv_threshold: float | None = None,
) -> dict:
    """Build close series for clock, volume, or intrinsic gauge."""
    if gauge == "clock":
        bars = {k: np.asarray(ohlcv[k]).copy() for k in ("open", "high", "low", "close", "volume", "timestamp")}
    elif gauge == "volume":
        if volume_threshold is None:
            raise ValueError("volume_threshold required for volume gauge")
        bars = build_volume_bars(ohlcv, volume_threshold)
    elif gauge == "intrinsic":
        if rv_threshold is None:
            raise ValueError("rv_threshold required for intrinsic gauge")
        bars = build_intrinsic_time_bars(ohlcv, rv_threshold)
    else:
        raise ValueError(f"Unknown gauge: {gauge}")
    bars["duration_stats"] = compute_bar_duration_stats(bars["timestamp"])
    bars["gauge"] = gauge
    return bars


def _tost_pvalue(diff: float, epsilon: float, se: float) -> float:
    """TOST p-value for H0: diff <= -epsilon or diff >= epsilon."""
    from scipy.stats import norm  # noqa: PLC0415

    if not np.isfinite(diff) or epsilon <= 0.0:
        return 1.0
    if se <= 0.0 or not np.isfinite(se):
        return 0.0 if abs(diff) < epsilon else 1.0

    p_lower = 1.0 - norm.cdf((diff + epsilon) / se)
    p_upper = norm.cdf((diff - epsilon) / se)
    return float(max(p_lower, p_upper))


def _median_row_for_q(pipeline: dict, q: int) -> dict | None:
    for cell in pipeline.get("per_wq_primary_q", []):
        if int(cell["q"]) == q:
            return cell
    for cell in pipeline.get("per_wq", []):
        if int(cell.get("W", W_PRIMARY)) == W_PRIMARY and int(cell["q"]) == q:
            return cell
    return None


def _median_for_q(pipeline: dict, q: int) -> tuple[float, int]:
    cell = _median_row_for_q(pipeline, q)
    if cell is None:
        return float("nan"), 0
    return float(cell["median_vr_dep"]), int(cell.get("n_nl", 0))


def _median_se_for_q(pipeline: dict, q: int) -> tuple[float, int, float]:
    cell = _median_row_for_q(pipeline, q)
    if cell is None:
        return float("nan"), 0, float("inf")

    med = float(cell["median_vr_dep"])
    n_nl = int(cell.get("n_nl", 0))
    ci_lo = cell.get("ci_95_lo")
    ci_hi = cell.get("ci_95_hi")
    if ci_lo is not None and ci_hi is not None:
        ci_lo_f = float(ci_lo)
        ci_hi_f = float(ci_hi)
        if np.isfinite(ci_lo_f) and np.isfinite(ci_hi_f) and ci_hi_f >= ci_lo_f:
            return med, n_nl, float((ci_hi_f - ci_lo_f) / (2.0 * _NORM_Z_95))

    if n_nl > 0:
        return med, n_nl, float(1.0 / np.sqrt(n_nl))
        return med, n_nl, float("inf")


def _holm_adjust(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values for the supplied family."""
    if not pvalues:
        return []
    sanitized = [float(p) if np.isfinite(p) else 1.0 for p in pvalues]
    m = len(sanitized)
    order = sorted(range(m), key=lambda i: sanitized[i])
    adjusted = [1.0] * m
    prev = 0.0
    for rank, idx in enumerate(order):
        adj = min(1.0, sanitized[idx] * (m - rank))
        adj = max(adj, prev)
        adjusted[idx] = adj
        prev = adj
    return adjusted


def test_gauge_invariance(
    gauge_results: dict,
    epsilon: float,
    alpha: float = 0.05,
) -> dict:
    """TOST equivalence test across gauge pairwise comparisons."""
    gauges = ("clock", "volume", "intrinsic")
    per_q_comparisons: list[dict] = []
    raw_pvalues: list[float] = []

    for q in Q_GRID:
        medians: dict[str, float] = {}
        n_eff: dict[str, int] = {}
        median_se: dict[str, float] = {}
        for g in gauges:
            med, n_nl, se = _median_se_for_q(gauge_results[g], q)
            medians[g] = med
            n_eff[g] = n_nl
            median_se[g] = se

        for ga, gb in combinations(gauges, 2):
            diff = medians[ga] - medians[gb]
            n_pair = max(1, min(n_eff[ga], n_eff[gb]))
            se = float(np.hypot(median_se[ga], median_se[gb]))
            p_equiv = _tost_pvalue(diff, epsilon, se)
            within_margin = bool(abs(diff) < epsilon)
            if np.isfinite(diff) and np.isfinite(se):
                ci_90_lo = float(diff - _NORM_Z_90 * se)
                ci_90_hi = float(diff + _NORM_Z_90 * se)
            else:
                ci_90_lo = float("-inf")
                ci_90_hi = float("inf")
            ci_within_margin = bool(ci_90_lo >= -epsilon and ci_90_hi <= epsilon)
            raw_pvalues.append(p_equiv)
            per_q_comparisons.append(
                {
                    "q": q,
                    "pair": f"{ga}-{gb}",
                    "median_a": medians[ga],
                    "median_b": medians[gb],
                    "diff": diff,
                    "epsilon": epsilon,
                    "se_diff": se,
                    "ci_90_lo": ci_90_lo,
                    "ci_90_hi": ci_90_hi,
                    "ci_within_margin": ci_within_margin,
                    "p_equiv": p_equiv,
                    "within_margin": within_margin,
                    "n_eff": n_pair,
                }
            )

    holm_adjusted = _holm_adjust(raw_pvalues)
    for i, row in enumerate(per_q_comparisons):
        row["holm_adjusted"] = holm_adjusted[i]
        row["equivalent_holm"] = bool(row["ci_within_margin"] and holm_adjusted[i] < alpha)

    all_pass = all(row["equivalent_holm"] for row in per_q_comparisons)
    verdict = VERDICT_INVARIANT if all_pass else VERDICT_VIOLATION

    return {
        "epsilon": float(epsilon),
        "alpha": alpha,
        "per_q_comparisons": per_q_comparisons,
        "holm_adjusted": holm_adjusted,
        "verdict": verdict,
    }


def compute_tost_epsilon(clock_pipeline_result: dict) -> float:
    """Practical equivalence margin for median |VR-1| clock differences."""
    _ = clock_pipeline_result
    return float(PRACTICAL_TOST_EPSILON)


def run_vr_pipeline_for_gauge(
    close: np.ndarray,
    timestamps: np.ndarray,
    wq_grid: list[tuple[int, int]],
    q_grid: tuple[int, ...],
    seed: int = BOOT_SEED,
    view: str = "fixed_bar",
) -> dict:
    """Run VR pipeline on a gauge close series with all-bars sampling."""
    close = np.asarray(close, dtype=np.float64)
    timestamps = np.asarray(timestamps, dtype=np.int64)

    needed_wq = set(wq_grid)
    needed_wq.update((W_PRIMARY, q) for q in q_grid)
    vr_series: dict[tuple[int, int], np.ndarray] = {}
    z_series: dict[tuple[int, int], np.ndarray] = {}
    for w, q in sorted(needed_wq):
        vr_arr, z_arr = vr_significance.compute_rolling_vr_and_z(close, W=w, q=q)
        vr_series[(w, q)] = vr_arr
        z_series[(w, q)] = z_arr

    per_wq: list[dict] = []
    for w, q in wq_grid:
        vr_arr = vr_series[(w, q)]
        z_arr = z_series[(w, q)]
        pred_nl, retained_idx = all_bars_non_overlapping(vr_arr, stride=w)
        z_nl = z_arr[retained_idx]
        timestamps_nl = timestamps[retained_idx]
        sig = vr_significance.compute_vr_significance(pred_nl, z_nl)
        median_point, ci_lo, ci_hi = vr_significance.median_vr_dep_boot_ci(
            pred_nl, block=w, n_boot=2000, seed=seed
        )
        per_year = vr_significance.compute_per_year_breakdown(pred_nl, timestamps_nl, z_nl)
        per_wq.append(
            {
                "W": w,
                "q": q,
                "is_primary": bool(w == W_PRIMARY and q == 5),
                "n_nl": int(len(pred_nl)),
                "median_vr_dep": sig["median_vr_dep"],
                "mean_vr_dep": sig["mean_vr_dep"],
                "ci_95_lo": ci_lo,
                "ci_95_hi": ci_hi,
                "closed": sig["closed"],
                "p_twotailed": sig["p_twotailed"],
                "median_z_m2": sig["median_z_m2"],
                "per_year": per_year,
            }
        )

    per_wq_primary_q: list[dict] = []
    holm_pvalues: list[float] = []
    for q in q_grid:
        vr_arr = vr_series[(W_PRIMARY, q)]
        z_arr = z_series[(W_PRIMARY, q)]
        pred_nl, retained_idx = all_bars_non_overlapping(vr_arr, stride=W_PRIMARY)
        z_nl = z_arr[retained_idx]
        timestamps_nl = timestamps[retained_idx]
        sig = vr_significance.compute_vr_significance(pred_nl, z_nl)
        median_q, ci_lo_q, ci_hi_q = vr_significance.median_vr_dep_boot_ci(
            pred_nl, block=W_PRIMARY, n_boot=2000, seed=seed
        )
        holm_pvalues.append(sig["p_twotailed"])
        per_wq_primary_q.append(
            {
                "W": W_PRIMARY,
                "q": q,
                "is_primary": bool(q == 5),
                "n_nl": int(len(pred_nl)),
                "median_vr_dep": sig["median_vr_dep"],
                "mean_vr_dep": sig["mean_vr_dep"],
                "ci_95_lo": ci_lo_q,
                "ci_95_hi": ci_hi_q,
                "closed": sig["closed"],
                "p_twotailed": sig["p_twotailed"],
                "median_z_m2": sig["median_z_m2"],
            }
        )

    holm_adj = vr_significance.apply_holm_b(holm_pvalues)
    for i, row in enumerate(per_wq_primary_q):
        row["holm_adjusted"] = holm_adj[i]

    horizon_profile = vr_significance.compute_vr_horizon_profile(
        close, W=W_PRIMARY, q_grid=q_grid
    )

    return {
        "view": view,
        "per_wq": per_wq,
        "per_wq_primary_q": per_wq_primary_q,
        "horizon_profile": {str(k): v for k, v in horizon_profile.items()},
        "n_bars": int(len(close)),
    }


def fit_gauge_garch_noise_floor(close: np.ndarray) -> dict:
    """Fit GARCH(1,1) noise floor per gauge with EWMA fallback."""
    close = np.asarray(close, dtype=np.float64)
    returns = np.diff(np.log(close[np.isfinite(close) & (close > 0)]))
    if len(returns) < 50:
        return {"source": "ewma", "alpha": None, "beta": None, "lambda": 0.94}

    try:
        from arch import arch_model  # noqa: PLC0415

        am = arch_model(
            returns * 100.0,
            mean="Constant",
            vol="GARCH",
            p=1,
            q=1,
            dist="normal",
            rescale=True,
        )
        res = am.fit(disp="off")
        alpha = float(res.params.get("alpha[1]", 0.0))
        beta = float(res.params.get("beta[1]", 0.0))
        if res.convergence_flag != 0 or (alpha + beta >= 0.999):
            raise ValueError("GARCH near-integrated or failed convergence")
        return {
            "source": "garch",
            "alpha": alpha,
            "beta": beta,
            "omega": float(res.params.get("omega", 0.0)),
        }
    except Exception:
        return {"source": "ewma", "alpha": None, "beta": None, "lambda": 0.94}


def compute_regime_population(close: np.ndarray) -> dict:
    """Regime population % per gauge for appendix diagnostics."""
    regime = RollingQuantileDetector().fit(close)
    n = len(regime)
    if n == 0:
        return {"LOW_pct": 0.0, "ELEVATED_pct": 0.0, "EXTREME_pct": 0.0}
    active = regime >= 0
    denom = max(1, int(np.sum(active)))
    return {
        "LOW_pct": float(100.0 * np.sum(regime == 0) / denom),
        "ELEVATED_pct": float(100.0 * np.sum(regime == 1) / denom),
        "EXTREME_pct": float(100.0 * np.sum(regime == 2) / denom),
        "n_bars": n,
    }


def run_sensitivity_diagnostics(
    ohlcv: dict,
    thresholds: tuple[float, ...],
    gauge_type: str,
) -> list[dict]:
    """Bar-count and duration diagnostics for non-primary thresholds."""
    rows: list[dict] = []
    for thr in thresholds:
        if gauge_type == "volume":
            bars = build_volume_bars(ohlcv, thr)
            overshoot = summarize_overshoot(bars["volume"], thr)
        elif gauge_type == "intrinsic":
            bars = build_intrinsic_time_bars(ohlcv, thr)
            overshoot = None
        else:
            raise ValueError(f"Unknown gauge_type: {gauge_type}")
        rows.append(
            {
                "threshold": thr,
                "n_bars": len(bars["close"]),
                "duration_stats": compute_bar_duration_stats(bars["timestamp"]),
                "overshoot": overshoot,
            }
        )
    return rows


def _synthetic_smoke() -> None:
    rng = np.random.default_rng(BOOT_SEED)
    n = 5000
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    ts = np.arange(n, dtype=np.int64) * 60_000 + 1_700_000_000_000
    result = run_vr_pipeline_for_gauge(close, ts, list(WQ_GRID), Q_GRID, seed=BOOT_SEED)
    assert result["n_bars"] == n
    print("synthetic_ok")


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

    gauge_specs = {
        "clock": build_gauge_series(ohlcv, "clock"),
        "volume": build_gauge_series(
            ohlcv, "volume", volume_threshold=PRIMARY_VOLUME_THRESHOLD_BTC
        ),
        "intrinsic": build_gauge_series(
            ohlcv, "intrinsic", rv_threshold=PRIMARY_RV_THRESHOLD
        ),
    }

    median_durations = {
        g: spec["duration_stats"]["median_min"] if g != "clock" else 1.0
        for g, spec in gauge_specs.items()
    }
    wq_calibration: dict[str, list[dict]] = {}
    fixed_calendar_grids: dict[str, list[tuple[int, int]]] = {}
    for g, med in median_durations.items():
        table = calibrate_wq_calendar(list(WQ_GRID), med)
        wq_calibration[g] = table
        fixed_calendar_grids[g] = [(row["W_adj"], row["q_adj"]) for row in table]

    fixed_bar_results: dict[str, dict] = {}
    fixed_calendar_results: dict[str, dict] = {}
    diagnostics: dict[str, dict] = {}

    for g, spec in gauge_specs.items():
        close = spec["close"]
        ts = spec["timestamp"]
        print(f"VR pipeline fixed_bar: {g} ({len(close):,} bars)...", flush=True)
        fixed_bar_results[g] = run_vr_pipeline_for_gauge(
            close, ts, list(WQ_GRID), Q_GRID, seed=BOOT_SEED, view="fixed_bar"
        )
        print(f"VR pipeline fixed_calendar: {g}...", flush=True)
        fixed_calendar_results[g] = run_vr_pipeline_for_gauge(
            close,
            ts,
            fixed_calendar_grids[g],
            Q_GRID,
            seed=BOOT_SEED,
            view="fixed_calendar",
        )
        diagnostics[g] = {
            "duration_stats": spec["duration_stats"],
            "garch_noise_floor": fit_gauge_garch_noise_floor(close),
            "regime_population": compute_regime_population(close),
        }

    sensitivity = {
        "volume": run_sensitivity_diagnostics(
            ohlcv,
            tuple(t for t in VOLUME_THRESHOLDS_BTC if t != PRIMARY_VOLUME_THRESHOLD_BTC),
            "volume",
        ),
        "intrinsic": run_sensitivity_diagnostics(
            ohlcv,
            tuple(t for t in RV_THRESHOLDS if t != PRIMARY_RV_THRESHOLD),
            "intrinsic",
        ),
    }

    epsilon = compute_tost_epsilon(fixed_bar_results["clock"])
    tost = test_gauge_invariance(fixed_bar_results, epsilon)
    verdict = tost["verdict"]

    report = {
        "run_id": run_date,
        "prereg_commit": prereg_commit,
        "code_commit": code_commit,
        "boot_seed": BOOT_SEED,
        "frozen_params": {
            "wq_grid": [list(wq) for wq in WQ_GRID],
            "q_grid": list(Q_GRID),
            "primary_volume_btc": PRIMARY_VOLUME_THRESHOLD_BTC,
            "primary_rv": PRIMARY_RV_THRESHOLD,
        },
        "wq_calibration_table": wq_calibration,
        "median_durations_min": median_durations,
        "gauges": {
            g: {
                "fixed_bar": fixed_bar_results[g],
                "fixed_calendar": fixed_calendar_results[g],
                "diagnostics": diagnostics[g],
            }
            for g in gauge_specs
        },
        "tost": tost,
        "verdict": verdict,
        "sensitivity_thresholds": sensitivity,
        "data_span": {
            "start": str(
                _dt.datetime.fromtimestamp(ohlcv["timestamp"][0] / 1000, tz=_dt.UTC).date()
            ),
            "end": str(ts_end.date()),
            "n_bars_clock": len(ohlcv["close"]),
        },
    }

    GAUGE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = GAUGE_DIR / f"gauge_report_{run_date}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Verdict: {verdict}")
    print(f"Report: {out_path}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--run":
        raise SystemExit(main())
    _synthetic_smoke()
