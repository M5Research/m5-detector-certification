"""ETHUSDT external-asset replication for the calibrated-detector paper.

This driver keeps BTCUSDT as the paper's primary asset and runs a symbol-swapped
ETHUSDT replication through the existing WP1 analysis functions. It does not
reuse the BTC signal-injection grid as an ETH-specific calibration.
"""
# ruff: noqa: E402,I001

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
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
from scripts.wp1 import vr_significance  # noqa: E402
from scripts.wp1.gate_analysis import WQ_GRID  # noqa: E402
from scripts.wp1.gauge_bars import (  # noqa: E402
    PRIMARY_RV_THRESHOLD,
    PRIMARY_VOLUME_THRESHOLD_BTC,
    RV_THRESHOLDS,
    VOLUME_THRESHOLDS_BTC,
)
from scripts.wp1.gauge_invariance import (  # noqa: E402
    BOOT_SEED as GAUGE_BOOT_SEED,
    build_gauge_series,
    calibrate_wq_calendar,
    compute_regime_population,
    compute_tost_epsilon,
    fit_gauge_garch_noise_floor,
    run_sensitivity_diagnostics,
    run_vr_pipeline_for_gauge,
    test_gauge_invariance,
)
from scripts.wp1.holdout_confirmatory import (  # noqa: E402
    HOLDOUT_END_MS as HOLDOUT_FULL_END_MS,
    compute_holdout_primary,
)
from scripts.wp1.persistence_null import (  # noqa: E402
    N_BOOTSTRAP as PERSISTENCE_BOOTSTRAP_DEFAULT,
    persistence_probability,
)
from scripts.wp1.precheck_b import (  # noqa: E402
    BOOT_SEED,
    CLOSURE_THRESHOLD,
    FREEZE_COMMIT,
    Q_GRID,
    _gate_guard,
    run_precheck_b_cascade,
)
from scripts.wp1.py_engine import load_and_clean  # noqa: E402
from scripts.wp1.thermodynamic_bound import (  # noqa: E402
    _default_cost_schedule,
    compute_thermodynamic_bound,
)

PRIMARY_ASSET = "BTCUSDT"
DEFAULT_SYMBOL = "ETHUSDT"
OUTPUT_DIR = PROJECT_ROOT / "backtest_results" / "asset_replication"

IN_SAMPLE_START_MS = int(datetime(2021, 5, 29, 19, 32, tzinfo=UTC).timestamp() * 1000)
IN_SAMPLE_END_MS = int(datetime(2025, 12, 31, 23, 59, 59, tzinfo=UTC).timestamp() * 1000)
HOLDOUT_START_MS = int(datetime(2026, 1, 1, tzinfo=UTC).timestamp() * 1000)
ETH_HOLDOUT_END_MS = int(datetime(2026, 5, 29, 20, 34, tzinfo=UTC).timestamp() * 1000)
MAX_GAP_ALLOWED_MINS = 60


def _dt(ms: int) -> datetime:
    return datetime.fromtimestamp(int(ms) / 1000, tz=UTC)


def _code_commit() -> str:
    try:
        return subprocess.run(
            ["git", "log", "--format=%H", "-1"],
            capture_output=True,
            text=True,
            cwd=str(_REPO_ROOT),
            check=False,
        ).stdout.strip()
    except Exception:
        return ""


def _span(timestamps: np.ndarray, *, year_2026_loaded: bool, partial_holdout: bool = False) -> dict:
    ts = np.asarray(timestamps, dtype=np.int64)
    if ts.ndim != 1 or len(ts) == 0:
        raise ValueError("timestamps must be a nonempty 1-D array")
    start = _dt(int(ts[0]))
    end = _dt(int(ts[-1]))
    return {
        "start": str(start.date()),
        "end": str(end.date()),
        "n_bars": int(len(ts)),
        "year_2026_loaded": year_2026_loaded,
        "partial_holdout": partial_holdout,
        "first_ts": start.isoformat(),
        "last_ts": end.isoformat(),
    }


def validate_in_sample_span(timestamps: np.ndarray) -> dict:
    """Validate that an in-sample replication span does not load 2026 holdout."""
    ts = np.asarray(timestamps, dtype=np.int64)
    if ts.ndim != 1 or len(ts) == 0:
        raise ValueError("timestamps must be a nonempty 1-D array")
    if int(np.max(ts)) >= HOLDOUT_START_MS:
        raise ValueError("in-sample replication includes 2026 holdout data")
    return _span(ts, year_2026_loaded=False)


def validate_holdout_span(timestamps: np.ndarray) -> dict:
    """Validate a 2026 holdout span and mark whether it is a partial-year block."""
    ts = np.asarray(timestamps, dtype=np.int64)
    if ts.ndim != 1 or len(ts) == 0:
        raise ValueError("timestamps must be a nonempty 1-D array")
    if int(np.min(ts)) < HOLDOUT_START_MS:
        raise ValueError("holdout replication received pre-2026 data")
    if int(np.max(ts)) > HOLDOUT_FULL_END_MS:
        raise ValueError("holdout replication received post-2026 data")
    partial = int(np.max(ts)) < HOLDOUT_FULL_END_MS
    return _span(ts, year_2026_loaded=True, partial_holdout=partial)


def _load_window(symbol: str, start_ms: int, end_ms: int) -> dict:
    return load_and_clean(
        data_path=str(PROJECT_ROOT / "data" / "binance_futures"),
        symbol=symbol,
        start_ms=start_ms,
        end_ms=end_ms,
        max_gap_allowed_mins=MAX_GAP_ALLOWED_MINS,
    )


def _primary_cell(cascade: dict) -> dict:
    for row in cascade.get("per_wq", []):
        if int(row.get("W", -1)) == 120 and int(row.get("q", -1)) == 5:
            return row
    for row in cascade.get("per_wq_primary_q", []):
        if int(row.get("q", -1)) == 5:
            return row
    raise ValueError("primary cell (W=120, q=5) not found in cascade report")


def _append_spike_guard(close: np.ndarray) -> dict:
    per_wq: dict[str, bool] = {}
    for w, q in WQ_GRID:
        vr_arr, _ = vr_significance.compute_rolling_vr_and_z(close, W=w, q=q)
        close_extended = np.append(close, close[-1] * 1.5)
        vr_arr_ext, _ = vr_significance.compute_rolling_vr_and_z(close_extended, W=w, q=q)
        try:
            np.testing.assert_array_equal(vr_arr, vr_arr_ext[:-1])
            passed = True
        except AssertionError:
            passed = False
        per_wq[f"W{w}_q{q}"] = passed
    return {"passed": bool(all(per_wq.values())), "per_wq": per_wq}


def run_precheck_section(close: np.ndarray, timestamps: np.ndarray) -> dict:
    cascade = run_precheck_b_cascade(close, timestamps, seed=BOOT_SEED)
    return {
        "primary_cell": _primary_cell(cascade),
        "cascade_fired": bool(cascade["cascade_fired"]),
        "append_spike_sc1": _append_spike_guard(close),
        "cascade": cascade,
        "frozen_params": {
            "wq_grid": [list(wq) for wq in WQ_GRID],
            "primary_wq": [120, 5],
            "q_grid": list(Q_GRID),
            "closure_threshold": CLOSURE_THRESHOLD,
            "holm_family_size": 4,
            "central_statistic": "median",
            "boot_block": "W",
        },
    }


def run_gauge_section(ohlcv: dict) -> dict:
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
        gauge: spec["duration_stats"]["median_min"] if gauge != "clock" else 1.0
        for gauge, spec in gauge_specs.items()
    }
    wq_calibration: dict[str, list[dict]] = {}
    fixed_calendar_grids: dict[str, list[tuple[int, int]]] = {}
    for gauge, median_duration in median_durations.items():
        table = calibrate_wq_calendar(list(WQ_GRID), median_duration)
        wq_calibration[gauge] = table
        fixed_calendar_grids[gauge] = [(row["W_adj"], row["q_adj"]) for row in table]

    fixed_bar_results: dict[str, dict] = {}
    fixed_calendar_results: dict[str, dict] = {}
    diagnostics: dict[str, dict] = {}
    for gauge, spec in gauge_specs.items():
        close = spec["close"]
        timestamps = spec["timestamp"]
        fixed_bar_results[gauge] = run_vr_pipeline_for_gauge(
            close,
            timestamps,
            list(WQ_GRID),
            Q_GRID,
            seed=GAUGE_BOOT_SEED,
            view="fixed_bar",
        )
        fixed_calendar_results[gauge] = run_vr_pipeline_for_gauge(
            close,
            timestamps,
            fixed_calendar_grids[gauge],
            Q_GRID,
            seed=GAUGE_BOOT_SEED,
            view="fixed_calendar",
        )
        diagnostics[gauge] = {
            "duration_stats": spec["duration_stats"],
            "garch_noise_floor": fit_gauge_garch_noise_floor(close),
            "regime_population": compute_regime_population(close),
        }

    tost_epsilon = compute_tost_epsilon(fixed_bar_results["clock"])
    tost = test_gauge_invariance(fixed_bar_results, tost_epsilon)
    return {
        "verdict": tost["verdict"],
        "tost": tost,
        "wq_calibration_table": wq_calibration,
        "median_durations_min": median_durations,
        "gauges": {
            gauge: {
                "fixed_bar": fixed_bar_results[gauge],
                "fixed_calendar": fixed_calendar_results[gauge],
                "diagnostics": diagnostics[gauge],
            }
            for gauge in gauge_specs
        },
        "sensitivity_thresholds": {
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
        },
        "frozen_params": {
            "primary_volume_threshold": PRIMARY_VOLUME_THRESHOLD_BTC,
            "primary_volume_threshold_units": "base-asset volume; numeric BTC protocol value reused",
            "primary_rv": PRIMARY_RV_THRESHOLD,
        },
    }


def run_information_section(close: np.ndarray, mi_bootstrap: int) -> dict:
    report = compute_thermodynamic_bound(
        close,
        Q_GRID,
        _default_cost_schedule(),
        injection_dir=None,
        seed=BOOT_SEED,
        n_boot=mi_bootstrap,
    )
    report["eth_specific_injection_grid"] = False
    report["btc_injection_grid_policy"] = (
        "BTC injection-grid artifacts are not reused as ETH calibration; "
        "this ETH section reports observed VR and sign-pair information only."
    )
    return report


def run_persistence_section(close: np.ndarray, n_bootstrap: int) -> dict:
    return persistence_probability(close, n_bootstrap=n_bootstrap)


def run_holdout_section(symbol: str) -> dict:
    data = _load_window(symbol, HOLDOUT_START_MS, ETH_HOLDOUT_END_MS)
    data_span = validate_holdout_span(data["timestamp"])
    return {
        "data_span": data_span,
        "primary_cell": compute_holdout_primary(data["close"], data["timestamp"]),
    }


def build_eth_replication_report(
    *,
    symbol: str,
    in_sample: dict,
    holdout: dict,
    run_id: str,
    code_commit: str,
    prereg_commit: str,
) -> dict:
    return {
        "run_id": run_id,
        "scope": "external_asset_replication",
        "primary_asset": PRIMARY_ASSET,
        "symbol": symbol,
        "venue": "Binance USD-M perpetual futures",
        "frequency": "1-minute OHLCV",
        "data_span": in_sample["data_span"],
        "holdout_span": holdout["data_span"],
        "in_sample": in_sample,
        "holdout": holdout,
        "injection_calibration": {
            "eth_specific_injection_grid": False,
            "btc_injection_grid_policy": (
                "BTC injection-grid artifacts are not reused as ETH calibration; "
                "an ETH-specific injection grid would require rerunning "
                "scripts/wp1/signal_injection.py on ETH returns."
            ),
        },
        "provenance": {
            "freeze_commit": FREEZE_COMMIT,
            "prereg_commit": prereg_commit,
            "code_commit": code_commit,
            "run_utc": datetime.now(tz=UTC).isoformat(),
        },
    }


def run_eth_replication(
    *,
    symbol: str = DEFAULT_SYMBOL,
    output_dir: Path = OUTPUT_DIR,
    persistence_bootstrap: int = PERSISTENCE_BOOTSTRAP_DEFAULT,
    mi_bootstrap: int = 500,
) -> Path:
    print(f"Gate guard for {symbol} external replication ...", flush=True)
    prereg_results = _gate_guard()
    prereg_commit = next(iter(prereg_results.values()), "")
    code_commit = _code_commit()

    print(f"Loading {symbol} in-sample window ...", flush=True)
    in_sample_data = _load_window(symbol, IN_SAMPLE_START_MS, IN_SAMPLE_END_MS)
    data_span = validate_in_sample_span(in_sample_data["timestamp"])

    close = in_sample_data["close"]
    timestamps = in_sample_data["timestamp"]
    print(f"Running {symbol} Pre-Check B cascade ...", flush=True)
    precheck = run_precheck_section(close, timestamps)
    print(f"Running {symbol} gauge-invariance section ...", flush=True)
    gauge = run_gauge_section(in_sample_data)
    print(f"Running {symbol} information-cost section ...", flush=True)
    information = run_information_section(close, mi_bootstrap=mi_bootstrap)
    print(f"Running {symbol} persistence section ...", flush=True)
    persistence = run_persistence_section(close, n_bootstrap=persistence_bootstrap)
    in_sample = {
        "data_span": data_span,
        "precheck": precheck,
        "gauge": gauge,
        "information": information,
        "persistence": persistence,
    }
    print(f"Running {symbol} holdout section ...", flush=True)
    holdout = run_holdout_section(symbol)

    run_id = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    report = build_eth_replication_report(
        symbol=symbol,
        in_sample=in_sample,
        holdout=holdout,
        run_id=run_id,
        code_commit=code_commit,
        prereg_commit=prereg_commit,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"eth_replication_{run_id}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report: {out_path}")
    print(
        "ETH primary cell: "
        f"median |VR(5)-1|={in_sample['precheck']['primary_cell']['median_vr_dep']:.4f}, "
        f"gauge={in_sample['gauge']['verdict']}, "
        f"info={in_sample['information']['headline_verdict']}, "
        f"holdout |VR(5)-1|={holdout['primary_cell']['observed_vr_dep']:.4f}"
    )
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--persistence-bootstrap", type=int, default=PERSISTENCE_BOOTSTRAP_DEFAULT)
    parser.add_argument("--mi-bootstrap", type=int, default=500)
    args = parser.parse_args(argv)

    run_eth_replication(
        symbol=args.symbol,
        output_dir=args.output_dir,
        persistence_bootstrap=args.persistence_bootstrap,
        mi_bootstrap=args.mi_bootstrap,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
