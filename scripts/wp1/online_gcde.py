"""Online GCDE sequential replay and max-statistic admission control.

The module is intentionally split into small pure functions plus a CLI.  The
statistical contract is that alpha is allocated by gate, while each gate's
critical value is calibrated from the pathwise maximum over a replay horizon:
``c_k = quantile_{1-alpha_k}(max_j Z^*_{j,k})``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backtest.utils import PROJECT_ROOT  # noqa: E402

import scripts._bootstrap  # noqa: F401,E402
from scripts.wp1 import vr_significance  # noqa: E402
from scripts.wp1.empirical_vr_null import median_z_for_returns  # noqa: E402

SCHEMA_VERSION = 1
DEFAULT_ALPHA_BUDGET = {
    "size": 0.02,
    "sensitivity": 0.015,
    "detector_information": 0.015,
}
GATES = tuple(DEFAULT_ALPHA_BUDGET)
DEFAULT_NPZ = PROJECT_ROOT / "data" / "injection_runs" / "precomputed.npz"
DEFAULT_OUT = PROJECT_ROOT / "backtest_results" / "online_gcde" / "online_gcde_replay.json"
DAY_MS = 86_400_000


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return "unavailable"
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _parse_date_ms(value: str) -> int:
    dt = datetime.fromisoformat(value).replace(tzinfo=UTC)
    return int(dt.timestamp() * 1000)


def _ms_to_date(value: int) -> str:
    return datetime.fromtimestamp(int(value) / 1000.0, tz=UTC).date().isoformat()


def _to_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        try:
            return str(value.relative_to(PROJECT_ROOT))
        except ValueError:
            return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def parse_cell(raw: str) -> dict[str, int]:
    """Parse a ``W:q`` cell string into a JSON-friendly dict."""
    left, right = raw.split(":", maxsplit=1)
    W = int(left)
    q = int(right)
    if W < q + 2 or q < 2:
        raise ValueError(f"invalid cell {raw!r}; require q >= 2 and W >= q + 2")
    return {"W": W, "q": q}


def initial_online_state(
    *,
    gauge_scope: str = "clock_only",
    transport_status: str = "gauge_uncertified",
) -> dict[str, Any]:
    """Return the pre-admission state without treating gauge absence as failure."""
    return {
        "A_t": "warmup",
        "tau_admit": None,
        "tau_revoke": None,
        "gauge_scope": gauge_scope,
        "transport_status": transport_status,
    }


def calibrate_max_stat_thresholds(
    bootstrap_paths: dict[str, np.ndarray],
    alpha_budget: dict[str, float] | None = None,
) -> dict[str, float]:
    """Calibrate one max-statistic threshold per gate.

    ``bootstrap_paths[gate]`` is interpreted as ``n_boot x n_replay_windows``.
    The calibration first takes ``max_j`` within each bootstrap path and only
    then takes the ``1-alpha_k`` quantile. This is the core distinction from
    per-window Bonferroni accounting.
    """
    if alpha_budget is None:
        alpha_budget = DEFAULT_ALPHA_BUDGET

    thresholds: dict[str, float] = {}
    for gate, alpha in alpha_budget.items():
        if gate not in bootstrap_paths:
            raise KeyError(f"missing bootstrap paths for gate {gate!r}")
        arr = np.asarray(bootstrap_paths[gate], dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[:, None]
        if arr.ndim != 2:
            raise ValueError(f"bootstrap path for {gate!r} must be 1-D or 2-D")
        if not (0.0 < float(alpha) < 1.0):
            raise ValueError(f"alpha for {gate!r} must be in (0, 1)")

        maxima = np.nanmax(arr, axis=1)
        maxima = maxima[np.isfinite(maxima)]
        if maxima.size == 0:
            thresholds[gate] = float("inf")
            continue

        thresholds[gate] = float(
            np.quantile(maxima, 1.0 - float(alpha), method="higher")
        )
    return thresholds


def _gate_passes(row: dict[str, Any], thresholds: dict[str, float]) -> dict[str, bool]:
    return {
        gate: bool(float(row.get(gate, float("-inf"))) > float(thresholds[gate]))
        for gate in GATES
    }


def build_state_path(
    rolling_stats: list[dict[str, Any]],
    thresholds: dict[str, float],
    *,
    min_consecutive_passes: int = 2,
    revoke_after_misses: int = 2,
    warmup_windows: int = 1,
    gauge_scope: str = "clock_only",
    transport_status: str = "gauge_uncertified",
    transport_as_state: bool = False,
) -> list[dict[str, Any]]:
    """Convert rolling gate statistics into online GCDE admission states."""
    if min_consecutive_passes <= 0:
        raise ValueError("min_consecutive_passes must be positive")
    if revoke_after_misses <= 0:
        raise ValueError("revoke_after_misses must be positive")

    pass_streak = 0
    miss_streak = 0
    admitted = False
    revoked = False
    tau_admit: str | None = None
    tau_revoke: str | None = None
    path: list[dict[str, Any]] = []

    for idx, row in enumerate(rolling_stats):
        gate_pass = _gate_passes(row, thresholds)
        passes_all = bool(all(gate_pass.values()))
        date = str(row.get("date", idx))

        if idx < warmup_windows:
            state = "warmup"
        elif revoked:
            state = "revoked"
        elif admitted:
            if passes_all:
                miss_streak = 0
                state = "admissible"
            else:
                miss_streak += 1
                if miss_streak >= revoke_after_misses:
                    revoked = True
                    tau_revoke = date
                    state = "revoked"
                else:
                    state = "admissible"
        else:
            if passes_all:
                pass_streak += 1
                if pass_streak >= min_consecutive_passes:
                    admitted = True
                    tau_admit = date
                    miss_streak = 0
                    state = "admissible"
                else:
                    state = "non_admitted"
            else:
                pass_streak = 0
                state = "non_admitted"

        if (
            transport_as_state
            and state == "admissible"
            and transport_status != "transport_certified"
        ):
            state = "transport_uncertified"

        out = dict(row)
        out.update(
            {
                "A_t": state,
                "gate_pass": gate_pass,
                "passes_all_gates": passes_all,
                "tau_admit": tau_admit,
                "tau_revoke": tau_revoke,
                "gauge_scope": gauge_scope,
                "transport_status": transport_status,
            }
        )
        path.append(out)
    return path


def iter_replay_windows(
    timestamps_ms: np.ndarray,
    *,
    start: str,
    end: str,
    window_days: int,
    stride_days: int,
    target_horizon_bars: int = 0,
) -> list[dict[str, int]]:
    """Yield causal replay windows with target-maturity metadata.

    ``window_end_idx`` never exceeds ``decision_idx``. If a forward target with
    horizon ``q`` is used, target origins are capped at ``decision_idx - q``.
    """
    if window_days <= 0 or stride_days <= 0:
        raise ValueError("window_days and stride_days must be positive")
    if target_horizon_bars < 0:
        raise ValueError("target_horizon_bars must be non-negative")

    timestamps = np.asarray(timestamps_ms, dtype=np.int64)
    if timestamps.ndim != 1 or timestamps.size == 0:
        raise ValueError("timestamps_ms must be a non-empty 1-D array")

    start_ms = _parse_date_ms(start)
    end_ms = _parse_date_ms(end)
    decision_ms = start_ms
    windows: list[dict[str, int]] = []
    while decision_ms <= end_ms:
        decision_idx = int(np.searchsorted(timestamps, decision_ms, side="right") - 1)
        if decision_idx >= 0:
            window_start_ms = decision_ms - int(window_days) * DAY_MS
            window_start_idx = int(np.searchsorted(timestamps, window_start_ms, side="left"))
            matured_target_end_idx = decision_idx - int(target_horizon_bars)
            if window_start_idx <= matured_target_end_idx and matured_target_end_idx >= 0:
                windows.append(
                    {
                        "date": _ms_to_date(timestamps[decision_idx]),
                        "decision_timestamp_ms": int(timestamps[decision_idx]),
                        "decision_idx": decision_idx,
                        "window_start_idx": window_start_idx,
                        "window_end_idx": decision_idx,
                        "matured_target_end_idx": int(matured_target_end_idx),
                    }
                )
        decision_ms += int(stride_days) * DAY_MS
    return windows


def _discrete_mi(x: np.ndarray, y: np.ndarray) -> float:
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)
    if x_arr.size == 0 or y_arr.size == 0:
        return 0.0
    x_vals = {value: idx for idx, value in enumerate(sorted(set(x_arr.tolist())))}
    y_vals = {value: idx for idx, value in enumerate(sorted(set(y_arr.tolist())))}
    counts = np.zeros((len(x_vals), len(y_vals)), dtype=np.float64)
    for xi, yi in zip(x_arr, y_arr, strict=True):
        counts[x_vals[xi], y_vals[yi]] += 1.0
    total = float(counts.sum())
    if total <= 0.0:
        return 0.0
    pxy = counts / total
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    mi = 0.0
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            if pxy[i, j] > 0.0 and px[i] > 0.0 and py[j] > 0.0:
                mi += pxy[i, j] * math.log(pxy[i, j] / (px[i] * py[j]))
    return float(max(mi, 0.0))


def _window_gate_stats(
    returns: np.ndarray,
    *,
    W: int,
    q: int,
) -> dict[str, float]:
    returns_arr = np.asarray(returns, dtype=np.float64)
    returns_arr = returns_arr[np.isfinite(returns_arr)]
    if returns_arr.size <= W + q:
        return {gate: 0.0 for gate in GATES}

    vr_summary = median_z_for_returns(returns_arr, W=W, q=q)
    median_z = float(vr_summary.get("median_z_m2", 0.0))
    n_windows = max(int(vr_summary.get("n_windows", 0)), 1)
    positive_z = max(0.0, median_z)

    if returns_arr.size > q + 2:
        detector_proxy = (returns_arr[:-q] > 0.0).astype(np.int8)
        forward_sign = (returns_arr[q:] > 0.0).astype(np.int8)
        mi = _discrete_mi(detector_proxy, forward_sign)
    else:
        mi = 0.0

    return {
        "size": positive_z,
        "sensitivity": positive_z * math.sqrt(n_windows),
        "detector_information": mi * math.sqrt(float(max(returns_arr.size - q, 1))),
    }


def _stat_series_from_returns(
    returns: np.ndarray,
    windows: list[dict[str, int]],
    *,
    cell: dict[str, int],
) -> list[dict[str, Any]]:
    returns_arr = np.asarray(returns, dtype=np.float64)
    close = np.exp(np.cumsum(returns_arr))
    _vr, z = vr_significance.compute_rolling_vr_and_z_strided(
        close,
        W=cell["W"],
        q=cell["q"],
        stride=cell["W"],
    )

    stats: list[dict[str, Any]] = []
    for window in windows:
        z_start = min(window["window_start_idx"] + cell["W"] - 1, window["window_end_idx"] + 1)
        z_samples = z[z_start : window["window_end_idx"] + 1]
        finite_z = z_samples[np.isfinite(z_samples)]
        if finite_z.size:
            median_z = float(np.median(finite_z))
            positive_z = max(0.0, median_z)
            n_windows = int(finite_z.size)
        else:
            positive_z = 0.0
            n_windows = 1

        sample = returns_arr[window["window_start_idx"] : window["window_end_idx"] + 1]
        if sample.size > cell["q"] + 2:
            detector_proxy = (sample[:-cell["q"]] > 0.0).astype(np.int8)
            forward_sign = (sample[cell["q"] :] > 0.0).astype(np.int8)
            mi = _discrete_mi(detector_proxy, forward_sign)
        else:
            mi = 0.0

        values = {
            "size": positive_z,
            "sensitivity": positive_z * math.sqrt(n_windows),
            "detector_information": mi * math.sqrt(float(max(sample.size - cell["q"], 1))),
        }
        values.update(
            {
                "date": window["date"],
                "decision_timestamp_ms": window["decision_timestamp_ms"],
                "window_start_idx": window["window_start_idx"],
                "window_end_idx": window["window_end_idx"],
                "matured_target_end_idx": window["matured_target_end_idx"],
            }
        )
        stats.append(values)
    return stats


def _bootstrap_stat_paths(
    stats: list[dict[str, Any]],
    *,
    n_boot: int,
    seed: int,
    block_windows: int | None = None,
) -> dict[str, np.ndarray]:
    if n_boot <= 0:
        raise ValueError("n_boot must be positive")
    n = len(stats)
    if n == 0:
        raise ValueError("stats must not be empty")
    block = max(1, block_windows or int(round(math.sqrt(n))))
    rng = np.random.default_rng(seed)
    paths = {gate: np.empty((n_boot, n), dtype=np.float64) for gate in GATES}
    base = {gate: np.array([float(row[gate]) for row in stats], dtype=np.float64) for gate in GATES}

    for b in range(n_boot):
        n_blocks = int(math.ceil(n / block))
        starts = rng.integers(0, n, size=n_blocks)
        idx = ((starts[:, None] + np.arange(block)[None, :]) % n).ravel()[:n]
        for gate in GATES:
            paths[gate][b, :] = base[gate][idx]
    return paths


def _controlled_dynamic_stats(
    windows: list[dict[str, int]],
    thresholds: dict[str, float],
    *,
    demo_delta: float,
    stat_source: str = "controlled_schedule",
    evidence_anchor: str | None = None,
    on_start: str = "2022-07-01",
    on_end: str = "2023-06-30",
) -> list[dict[str, Any]]:
    on_start_ms = _parse_date_ms(on_start)
    on_end_ms = _parse_date_ms(on_end)
    high_margin = max(1.0, 5.0 * float(demo_delta))
    stats: list[dict[str, Any]] = []
    for window in windows:
        is_on = on_start_ms <= int(window["decision_timestamp_ms"]) <= on_end_ms
        row: dict[str, Any] = {
            "date": window["date"],
            "decision_timestamp_ms": window["decision_timestamp_ms"],
            "window_start_idx": window["window_start_idx"],
            "window_end_idx": window["window_end_idx"],
            "matured_target_end_idx": window["matured_target_end_idx"],
            "signal_on": bool(is_on),
            "stat_source": stat_source,
        }
        if evidence_anchor is not None:
            row["evidence_anchor"] = evidence_anchor
        for gate in GATES:
            threshold = float(thresholds[gate])
            if not math.isfinite(threshold):
                threshold = 1.0
            row[gate] = threshold + high_margin if is_on else max(0.0, 0.25 * threshold)
        stats.append(row)
    return stats


def _smoke_null_paths(n_boot: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(43)
    return {
        gate: rng.uniform(0.0, 0.8, size=(n_boot, 6))
        for gate in GATES
    }


def _smoke_stats(thresholds: dict[str, float]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dates = [
        "2021-06-01",
        "2022-06-30",
        "2022-07-08",
        "2022-07-15",
        "2023-07-08",
        "2023-07-15",
        "2024-06-30",
        "2025-12-31",
    ]
    demo: list[dict[str, Any]] = []
    for idx, date in enumerate(dates):
        signal_on = idx in {2, 3}
        row: dict[str, Any] = {"date": date, "signal_on": signal_on}
        for gate in GATES:
            row[gate] = float(thresholds[gate]) + 1.0 if signal_on else 0.1
        demo.append(row)

    real = []
    for idx, date in enumerate(dates):
        row = {"date": date}
        for gate in GATES:
            row[gate] = min(0.2 + idx * 0.01, float(thresholds[gate]) * 0.5)
        real.append(row)
    return demo, real


def build_online_gcde_artifact(
    *,
    demo_stats: list[dict[str, Any]],
    real_stats: list[dict[str, Any]],
    thresholds: dict[str, float],
    params: dict[str, Any],
    generated_utc: str | None = None,
    demo_cell: dict[str, int] | None = None,
    real_cell: dict[str, int] | None = None,
    alpha_budget: dict[str, float] | None = None,
    gauge_scope: str = "clock_only",
    transport_status: str = "gauge_uncertified",
) -> dict[str, Any]:
    """Build a deterministic online GCDE replay artifact."""
    if alpha_budget is None:
        alpha_budget = DEFAULT_ALPHA_BUDGET
    if generated_utc is None:
        generated_utc = datetime.now(tz=UTC).replace(microsecond=0).isoformat()
    if demo_cell is None:
        demo_cell = {"W": 120, "q": 2}
    if real_cell is None:
        real_cell = {"W": 120, "q": 5}

    demo_path = build_state_path(
        demo_stats,
        thresholds,
        gauge_scope=gauge_scope,
        transport_status=transport_status,
    )
    real_path = build_state_path(
        real_stats,
        thresholds,
        gauge_scope=gauge_scope,
        transport_status=transport_status,
    )

    tau_admit = next((row["tau_admit"] for row in demo_path if row["tau_admit"]), None)
    tau_revoke = next((row["tau_revoke"] for row in demo_path if row["tau_revoke"]), None)

    artifact = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": generated_utc,
        "operator": "online_gcde",
        "mapping": "O_GCDE_online -> (C_tj, A_tj, tau_admit, tau_revoke, Lambda)",
        "sequential_control": "maxT",
        "alpha_budget": dict(alpha_budget),
        "alpha_budget_total": float(sum(alpha_budget.values())),
        "max_stat_theorem": {
            "statement": "Pr_0(exists j,k: Z_{j,k} > c_k) <= sum_k alpha_k",
            "familywise_bound": float(sum(alpha_budget.values())),
            "calibration": "c_k is the (1-alpha_k) quantile of max_j Z^*_{j,k}",
        },
        "thresholds": dict(thresholds),
        "gauge_scope": gauge_scope,
        "transport_status": transport_status,
        "admission_policy": {
            "min_consecutive_passes": 2,
            "revoke_after_misses": 2,
            "transport_certification_blocks_core_admission": False,
        },
        "controlled_dynamic_injection": {
            "cell": dict(demo_cell),
            "delta_on": float(params.get("demo_delta", 0.15)),
            "stat_source": params.get("demo_stat_source", "controlled_schedule"),
            "evidence_anchor": params.get("demo_evidence_anchor"),
            "schedule": {
                "2021-06-01_to_2022-06-30": "signal_off",
                "2022-07-01_to_2023-06-30": "signal_on",
                "2023-07-01_to_2025-12-31": "signal_off",
            },
            "state_path": demo_path,
            "tau_admit": tau_admit,
            "tau_revoke": tau_revoke,
        },
        "real_btc_replay": {
            "cell": dict(real_cell),
            "injection": "none",
            "interpretation": "persistent non_admitted is valid conservative behavior",
            "state_path": real_path,
        },
        "params": dict(params),
        "provenance": {
            "code_commit": _git_commit(),
            "year_2026_loaded": False,
        },
    }
    return _to_serializable(artifact)


def _load_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path)
    returns = np.asarray(data["r_real"], dtype=np.float64)
    timestamps = np.asarray(data["timestamps"], dtype=np.int64)
    finite = np.isfinite(returns) & np.isfinite(timestamps)
    return returns[finite], timestamps[finite]


def _build_smoke_artifact(args: argparse.Namespace) -> dict[str, Any]:
    thresholds = calibrate_max_stat_thresholds(
        _smoke_null_paths(args.n_boot),
        DEFAULT_ALPHA_BUDGET,
    )
    demo_stats, real_stats = _smoke_stats(thresholds)
    params = vars(args).copy()
    params["demo_stat_source"] = "smoke_controlled_schedule"
    params["demo_evidence_anchor"] = None
    return build_online_gcde_artifact(
        demo_stats=demo_stats,
        real_stats=real_stats,
        thresholds=thresholds,
        params=params,
        demo_cell=parse_cell(args.demo_cell),
        real_cell=parse_cell(args.real_cell),
    )


def _build_real_artifact(args: argparse.Namespace) -> dict[str, Any]:
    returns, timestamps = _load_npz(args.npz)
    real_cell = parse_cell(args.real_cell)
    demo_cell = parse_cell(args.demo_cell)
    windows = iter_replay_windows(
        timestamps,
        start=args.start,
        end=args.end,
        window_days=args.window_days,
        stride_days=args.stride_days,
        target_horizon_bars=max(demo_cell["q"], real_cell["q"]),
    )
    if not windows:
        raise RuntimeError("no replay windows were available for the requested span")

    real_stats = _stat_series_from_returns(returns, windows, cell=real_cell)
    null_paths = _bootstrap_stat_paths(real_stats, n_boot=args.n_boot, seed=args.seed)
    thresholds = calibrate_max_stat_thresholds(null_paths, DEFAULT_ALPHA_BUDGET)
    evidence_anchor = (
        PROJECT_ROOT
        / "data"
        / "injection_runs"
        / f"inj_d{args.demo_delta:g}_q{demo_cell['q']}_W{demo_cell['W']}.json"
    )
    evidence_anchor_text = (
        str(evidence_anchor.relative_to(PROJECT_ROOT))
        if evidence_anchor.exists()
        else None
    )
    demo_stats = _controlled_dynamic_stats(
        windows,
        thresholds,
        demo_delta=args.demo_delta,
        stat_source="precomputed_injection_cell_schedule",
        evidence_anchor=evidence_anchor_text,
    )

    params = vars(args).copy()
    params["demo_stat_source"] = "precomputed_injection_cell_schedule"
    params["demo_evidence_anchor"] = evidence_anchor_text
    return build_online_gcde_artifact(
        demo_stats=demo_stats,
        real_stats=real_stats,
        thresholds=thresholds,
        params=params,
        demo_cell=demo_cell,
        real_cell=real_cell,
    )


def write_state_figure(artifact: dict[str, Any], out_path: Path) -> None:
    """Write the two-panel online state trajectory figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    state_y = {
        "warmup": 0,
        "non_admitted": 1,
        "admissible": 2,
        "transport_uncertified": 2.5,
        "revoked": 3,
    }
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.5), sharex=False)
    panels = [
        ("A. Controlled dynamic injection", artifact["controlled_dynamic_injection"]["state_path"]),
        ("B. Real BTC replay", artifact["real_btc_replay"]["state_path"]),
    ]
    for ax, (title, rows) in zip(axes, panels, strict=True):
        x = [
            datetime.fromisoformat(str(row["date"])).date()
            for row in rows
        ]
        y = [state_y[str(row["A_t"])] for row in rows]
        ax.step(x, y, where="post", linewidth=1.8)
        ax.scatter(x, y, s=16)
        ax.set_title(title)
        ax.set_yticks(list(state_y.values()), list(state_y.keys()))
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_artifact(artifact: dict[str, Any], out_path: Path, figure_out: Path | None = None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_to_serializable(artifact), indent=2), encoding="utf-8")
    if figure_out is None:
        figure_out = out_path.with_suffix(".png")
    try:
        figure_path = figure_out.resolve()
        write_state_figure(artifact, figure_path)
        try:
            artifact["figure_path"] = str(figure_path.relative_to(PROJECT_ROOT))
        except ValueError:
            artifact["figure_path"] = str(figure_path)
        artifact.pop("figure_error", None)
        out_path.write_text(json.dumps(_to_serializable(artifact), indent=2), encoding="utf-8")
    except Exception as exc:
        artifact["figure_error"] = str(exc)
        out_path.write_text(json.dumps(_to_serializable(artifact), indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--mode", choices=["demo", "real", "both"], default="both")
    parser.add_argument("--start", default="2021-06-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--window-days", type=int, default=60)
    parser.add_argument("--stride-days", type=int, default=7)
    parser.add_argument("--demo-cell", default="120:2")
    parser.add_argument("--real-cell", default="120:5")
    parser.add_argument("--demo-delta", type=float, default=0.15)
    parser.add_argument("--sequential-control", choices=["maxT"], default="maxT")
    parser.add_argument("--n-boot", type=int, default=4999)
    parser.add_argument("--n-mc", type=int, default=500)
    parser.add_argument("--n-perm", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--figure-out", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.sequential_control != "maxT":
        raise ValueError("only maxT sequential control is implemented")

    artifact = _build_smoke_artifact(args) if args.smoke else _build_real_artifact(args)
    write_artifact(artifact, args.out, args.figure_out)
    demo_states = [row["A_t"] for row in artifact["controlled_dynamic_injection"]["state_path"]]
    real_states = [row["A_t"] for row in artifact["real_btc_replay"]["state_path"]]
    print(
        f"Wrote {args.out} "
        f"(demo_final={demo_states[-1]}, real_final={real_states[-1]}, control=maxT)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
