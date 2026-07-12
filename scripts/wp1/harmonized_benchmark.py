"""Harmonized regime-detector benchmark.

This module implements the benchmark layer for asking whether regime detectors
measure the same market state after controlling obvious sources of disagreement:
state vocabulary, timestamp intersection, mapping, validity gates, and economic
side-information tests.

The default CLI path used in tests is ``--smoke``. Full BTCUSDT runs are wired
through the same artifact builder but are intentionally explicit because HMM/MS
fits can be slow on the full 2021-2025 span.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime, time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

os.environ.setdefault("OMP_NUM_THREADS", "1")

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_REPO_SRC = _REPO_ROOT / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

import scripts._bootstrap  # noqa: F401,E402
from scripts.wp1.predictability import rolling_vr_m2_z_arrays  # noqa: E402
from strategies.vol_regime_switch.hmm_detector import HMMDetector  # noqa: E402
from strategies.vol_regime_switch.realized_vol import compute_rv  # noqa: E402
from strategies.vol_regime_switch.rolling_quantile_detector import (  # noqa: E402
    RollingQuantileDetector,
)

PROTOCOL_ID = "harmonized-regime-benchmark-mvp-v1"
SCHEMA_VERSION = 1

STATUS_VALID = "valid"
STATUS_INSTRUMENT_FAILURE = "instrument_failure"
STATUS_EXCLUDED = "excluded_from_agreement_headline"

CLASS_HARMONIZED_CONVERGENCE = "harmonized_convergence"
CLASS_PERSISTENT_DISAGREEMENT = "persistent_disagreement"
CLASS_MIXED_FAMILY_STRUCTURE = "mixed_family_structure"
CLASS_INSTRUMENT_FAILURE = "instrument_failure"

DEFAULT_MIN_NONOVERLAP = 50
DEFAULT_STATE_STRIDE = 60
DEFAULT_VARIANCE_SEPARATION = 1.10
DEFAULT_COST_BPS = 10.0
DEFAULT_VR_Q_GRID = (2, 5, 15, 60)


@dataclass(frozen=True)
class DetectorOutput:
    """Common detector output contract for the harmonized benchmark."""

    name: str
    timestamps: np.ndarray
    labels: np.ndarray
    soft_probabilities: np.ndarray | None = None
    target_statistic: np.ndarray | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    common_state_eligible: bool = True


def _to_serializable(value: Any) -> Any:
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


def collapse_to_two_state(labels: np.ndarray) -> np.ndarray:
    """Map LOW/ELEVATED/EXTREME labels to LOW/HIGH while preserving warmup."""
    labels_arr = np.asarray(labels, dtype=np.int8)
    mapped = np.full(labels_arr.shape, -1, dtype=np.int8)
    mapped[labels_arr == 0] = 0
    mapped[(labels_arr == 1) | (labels_arr == 2)] = 1
    return mapped


def _parse_utc_ms(value: str, end_of_day: bool = False) -> int:
    """Parse a date or timestamp string as UTC epoch milliseconds."""
    normalized = value.strip().replace("Z", "+00:00")
    if "T" in normalized:
        dt = datetime.fromisoformat(normalized)
    else:
        day = datetime.fromisoformat(normalized).date()
        dt = datetime.combine(
            day,
            time(23, 59, 59, 999000) if end_of_day else time.min,
            tzinfo=UTC,
        )
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    else:
        dt = dt.astimezone(UTC)
    return int(dt.timestamp() * 1000)


def _format_utc(ms: int) -> str:
    return datetime.fromtimestamp(int(ms) / 1000, tz=UTC).isoformat()


def slice_1m_window(
    close: np.ndarray,
    timestamps: np.ndarray,
    start_date: str,
    end_date: str,
    symbol: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Select a contiguous time-domain window while preserving native 1-minute bars."""
    close_arr = np.asarray(close, dtype=np.float64)
    ts_arr = np.asarray(timestamps, dtype=np.int64)
    if close_arr.shape != ts_arr.shape:
        raise ValueError("close and timestamps must have the same shape")

    start_ms = _parse_utc_ms(start_date)
    end_ms = _parse_utc_ms(end_date, end_of_day="T" not in end_date)
    if end_ms < start_ms:
        raise ValueError("end_date must be >= start_date")

    mask = (ts_arr >= start_ms) & (ts_arr <= end_ms)
    close_w = close_arr[mask].copy()
    ts_w = ts_arr[mask].copy()
    if len(close_w) == 0:
        raise ValueError(f"no {symbol} bars in requested window {start_date} to {end_date}")
    if np.any(np.diff(ts_w) <= 0):
        raise ValueError("timestamps must be strictly increasing after slicing")

    span = {
        "symbol": symbol,
        "start": str(datetime.fromtimestamp(int(ts_w[0]) / 1000, tz=UTC).date()),
        "end": str(datetime.fromtimestamp(int(ts_w[-1]) / 1000, tz=UTC).date()),
        "n_bars": int(len(close_w)),
        "first_ts": _format_utc(int(ts_w[0])),
        "last_ts": _format_utc(int(ts_w[-1])),
        "bar_frequency": "1min",
        "downsampled": False,
        "window_policy": "contiguous_time_domain_truncation_no_downsampling",
        "q_units": "bars_at_native_1min_frequency",
        "requested_start": start_date,
        "requested_end": end_date,
        "year_2026_loaded": bool(
            datetime.fromtimestamp(int(ts_w[-1]) / 1000, tz=UTC).year >= 2026
        ),
    }
    if span["year_2026_loaded"]:
        raise RuntimeError("benchmark loaded 2026 holdout data")
    return close_w, ts_w, span


def map_score_to_three_states(score: np.ndarray, valid_mask: np.ndarray | None = None) -> np.ndarray:
    """Frozen base-rate 3-state mapping: bottom 75%, next 20%, top 5%."""
    score_arr = np.asarray(score, dtype=np.float64)
    if valid_mask is None:
        valid_mask = np.isfinite(score_arr)
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool) & np.isfinite(score_arr)

    labels = np.full(score_arr.shape, -1, dtype=np.int8)
    if int(np.sum(valid_mask)) == 0:
        return labels

    q75, q95 = np.quantile(score_arr[valid_mask], [0.75, 0.95])
    labels[valid_mask] = np.where(
        score_arr[valid_mask] > q95,
        2,
        np.where(score_arr[valid_mask] > q75, 1, 0),
    ).astype(np.int8)
    return labels


def state_occupancy(labels: np.ndarray, n_states: int) -> dict[str, Any]:
    valid = np.asarray(labels) >= 0
    total = int(np.sum(valid))
    counts = {str(s): int(np.sum(np.asarray(labels)[valid] == s)) for s in range(n_states)}
    fractions = {
        str(s): (float(counts[str(s)] / total) if total else 0.0)
        for s in range(n_states)
    }
    return {"n_valid": total, "counts": counts, "fractions": fractions}


def non_overlapping_state_counts(
    labels: np.ndarray,
    n_states: int,
    stride: int = DEFAULT_STATE_STRIDE,
) -> dict[str, int]:
    labels_arr = np.asarray(labels, dtype=np.int8)
    valid_idx = np.flatnonzero(labels_arr >= 0)
    if len(valid_idx) == 0:
        sampled = np.array([], dtype=np.int8)
    else:
        sampled = labels_arr[valid_idx[0] :: max(int(stride), 1)]
        sampled = sampled[sampled >= 0]
    return {str(s): int(np.sum(sampled == s)) for s in range(n_states)}


def transition_matrix(labels: np.ndarray, n_states: int) -> list[list[int]]:
    labels_arr = np.asarray(labels, dtype=np.int8)
    valid = labels_arr[labels_arr >= 0]
    matrix = np.zeros((n_states, n_states), dtype=int)
    for prev, curr in zip(valid[:-1], valid[1:], strict=False):
        if 0 <= prev < n_states and 0 <= curr < n_states:
            matrix[int(prev), int(curr)] += 1
    return matrix.tolist()


def run_length_summary(labels: np.ndarray) -> dict[str, Any]:
    labels_arr = np.asarray(labels, dtype=np.int8)
    valid = labels_arr[labels_arr >= 0]
    if len(valid) == 0:
        return {"n_runs": 0, "mean": 0.0, "median": 0.0, "max": 0}

    lengths: list[int] = []
    current = int(valid[0])
    run_len = 1
    for value in valid[1:]:
        if int(value) == current:
            run_len += 1
        else:
            lengths.append(run_len)
            current = int(value)
            run_len = 1
    lengths.append(run_len)
    arr = np.asarray(lengths, dtype=np.float64)
    return {
        "n_runs": int(len(lengths)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "max": int(np.max(arr)),
    }


def validate_label_stream(
    detector_name: str,
    labels: np.ndarray,
    n_states: int,
    stride: int = DEFAULT_STATE_STRIDE,
    min_nonoverlap: int = DEFAULT_MIN_NONOVERLAP,
    variance_separation: tuple[float, ...] | list[float] | None = None,
    convergence_ok: bool = True,
    deterministic: bool = True,
    common_state_eligible: bool = True,
) -> dict[str, Any]:
    """Classify a detector stream as valid, failed, or excluded by frozen gates."""
    labels_arr = np.asarray(labels, dtype=np.int8)
    nonoverlap = non_overlapping_state_counts(labels_arr, n_states=n_states, stride=stride)
    reasons: list[str] = []

    if not common_state_eligible:
        return {
            "status": STATUS_EXCLUDED,
            "reasons": ["target_not_common_market_state"],
            "nonoverlap_counts": nonoverlap,
            "occupancy": state_occupancy(labels_arr, n_states),
        }

    if not convergence_ok:
        reasons.append("convergence_failed")
    if not deterministic:
        reasons.append("determinism_failed")
    for state in range(n_states):
        if nonoverlap[str(state)] < min_nonoverlap:
            reasons.append(
                f"state_{state}_nonoverlap_below_{min_nonoverlap}"
            )
    if variance_separation is not None:
        min_sep = min((float(v) for v in variance_separation), default=float("inf"))
        if min_sep < DEFAULT_VARIANCE_SEPARATION:
            reasons.append(f"variance_separation_below_{DEFAULT_VARIANCE_SEPARATION:.2f}")

    return {
        "status": STATUS_INSTRUMENT_FAILURE if reasons else STATUS_VALID,
        "reasons": reasons,
        "nonoverlap_counts": nonoverlap,
        "occupancy": state_occupancy(labels_arr, n_states),
    }


def confusion_matrix(labels_a: np.ndarray, labels_b: np.ndarray, n_states: int) -> list[list[int]]:
    a = np.asarray(labels_a, dtype=np.int64)
    b = np.asarray(labels_b, dtype=np.int64)
    matrix = np.zeros((n_states, n_states), dtype=int)
    valid = (a >= 0) & (a < n_states) & (b >= 0) & (b < n_states)
    for ai, bi in zip(a[valid], b[valid], strict=False):
        matrix[int(ai), int(bi)] += 1
    return matrix.tolist()


def cohen_kappa(labels_a: np.ndarray, labels_b: np.ndarray, n_states: int) -> float:
    matrix = np.asarray(confusion_matrix(labels_a, labels_b, n_states), dtype=np.float64)
    total = matrix.sum()
    if total == 0.0:
        return float("nan")
    p_o = float(np.trace(matrix) / total)
    row = matrix.sum(axis=1) / total
    col = matrix.sum(axis=0) / total
    p_e = float(np.dot(row, col))
    if abs(1.0 - p_e) < 1e-12:
        return float("nan")
    return float((p_o - p_e) / (1.0 - p_e))


def _entropy_from_counts(counts: np.ndarray) -> float:
    total = float(np.sum(counts))
    if total <= 0.0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-np.sum(p * np.log(p)))


def variation_of_information(labels_a: np.ndarray, labels_b: np.ndarray, n_states: int) -> float:
    matrix = np.asarray(confusion_matrix(labels_a, labels_b, n_states), dtype=np.float64)
    total = float(matrix.sum())
    if total <= 0.0:
        return 0.0
    hx = _entropy_from_counts(matrix.sum(axis=1))
    hy = _entropy_from_counts(matrix.sum(axis=0))
    pxy = matrix / total
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    mi = 0.0
    for i in range(n_states):
        for j in range(n_states):
            if pxy[i, j] > 0.0 and px[i] > 0.0 and py[j] > 0.0:
                mi += pxy[i, j] * math.log(pxy[i, j] / (px[i] * py[j]))
    vi = hx + hy - 2.0 * mi
    return float(max(vi, 0.0))


def pairwise_agreement(labels_a: np.ndarray, labels_b: np.ndarray, n_states: int) -> dict[str, Any]:
    a = np.asarray(labels_a, dtype=np.int8)
    b = np.asarray(labels_b, dtype=np.int8)
    valid = (a >= 0) & (b >= 0)
    if int(np.sum(valid)) == 0:
        return {
            "n": 0,
            "confusion_matrix": confusion_matrix(a, b, n_states),
            "cohens_kappa": None,
            "ari": None,
            "nmi": None,
            "variation_of_information": None,
        }
    av = a[valid]
    bv = b[valid]
    return {
        "n": int(len(av)),
        "confusion_matrix": confusion_matrix(av, bv, n_states),
        "cohens_kappa": cohen_kappa(av, bv, n_states),
        "ari": float(adjusted_rand_score(av, bv)),
        "nmi": float(normalized_mutual_info_score(av, bv)),
        "variation_of_information": variation_of_information(av, bv, n_states),
    }


def shared_intersection(detectors: list[DetectorOutput]) -> dict[str, Any]:
    """Return labels aligned to timestamps where every detector has a valid label."""
    if not detectors:
        return {"timestamps": np.array([], dtype=np.int64), "labels": {}}
    common = set(np.asarray(detectors[0].timestamps, dtype=np.int64).tolist())
    for det in detectors[1:]:
        common &= set(np.asarray(det.timestamps, dtype=np.int64).tolist())
    ordered = np.array(sorted(common), dtype=np.int64)

    aligned: dict[str, np.ndarray] = {}
    keep = np.ones(len(ordered), dtype=bool)
    for det in detectors:
        idx = {int(ts): i for i, ts in enumerate(np.asarray(det.timestamps, dtype=np.int64))}
        labels = np.array([det.labels[idx[int(ts)]] for ts in ordered], dtype=np.int8)
        aligned[det.name] = labels
        keep &= labels >= 0

    return {
        "timestamps": ordered[keep],
        "labels": {name: labels[keep] for name, labels in aligned.items()},
    }


def _discrete_mi(x: np.ndarray, y: np.ndarray) -> float:
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)
    if len(x_arr) != len(y_arr):
        raise ValueError("x and y must have the same length")
    if len(x_arr) == 0:
        return 0.0
    _, x_inv = np.unique(x_arr, return_inverse=True)
    _, y_inv = np.unique(y_arr, return_inverse=True)
    n_x = int(np.max(x_inv)) + 1
    n_y = int(np.max(y_inv)) + 1
    flat = x_inv * n_y + y_inv
    counts = np.bincount(flat, minlength=n_x * n_y).reshape(n_x, n_y).astype(np.float64)
    total = counts.sum()
    pxy = counts / total
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    mi = 0.0
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            if pxy[i, j] > 0.0 and px[i] > 0.0 and py[j] > 0.0:
                mi += pxy[i, j] * math.log(pxy[i, j] / (px[i] * py[j]))
    return float(max(mi, 0.0))


def discrete_entropy(labels: np.ndarray) -> float:
    """Return empirical entropy in nats for a discrete label vector."""
    labels_arr = np.asarray(labels)
    if len(labels_arr) == 0:
        return 0.0
    _, counts = np.unique(labels_arr, return_counts=True)
    probs = counts.astype(np.float64) / float(np.sum(counts))
    entropy = float(-np.sum(probs * np.log(probs)))
    return 0.0 if abs(entropy) < 1e-15 else entropy


def detector_contingent_information(
    labels: np.ndarray,
    forward_returns: np.ndarray,
    n_states: int,
    cost_bps: float = DEFAULT_COST_BPS,
    n_boot: int = 500,
    n_perm: int = 1000,
    seed: int = 43,
) -> dict[str, Any]:
    """Estimate detector-contingent MI against forward return sign."""
    labels_arr = np.asarray(labels, dtype=np.int8)
    returns = np.asarray(forward_returns, dtype=np.float64)
    if labels_arr.shape != returns.shape:
        raise ValueError("labels and forward_returns must have the same shape")
    valid = (labels_arr >= 0) & (labels_arr < n_states) & np.isfinite(returns)
    x = labels_arr[valid]
    y = np.sign(returns[valid]).astype(np.int8)
    mi = _discrete_mi(x, y)
    rng = np.random.default_rng(seed)

    boot_values: list[float] = []
    if len(x) > 0 and n_boot > 0:
        for _ in range(n_boot):
            idx = rng.choice(len(x), size=len(x), replace=True)
            boot_values.append(_discrete_mi(x[idx], y[idx]))
    if boot_values:
        ci = np.quantile(np.asarray(boot_values), [0.025, 0.975]).tolist()
    else:
        ci = [mi, mi]

    perm_values: list[float] = []
    if len(x) > 0 and n_perm > 0:
        for _ in range(n_perm):
            perm_values.append(_discrete_mi(rng.permutation(x), y))
    perm_p = (
        float((np.sum(np.asarray(perm_values) >= mi) + 1) / (len(perm_values) + 1))
        if perm_values
        else None
    )

    conditional: dict[str, dict[str, float | int | None]] = {}
    for state in range(n_states):
        state_returns = returns[valid & (labels_arr == state)]
        conditional[str(state)] = {
            "n": int(len(state_returns)),
            "mean": float(np.mean(state_returns)) if len(state_returns) else None,
            "median": float(np.median(state_returns)) if len(state_returns) else None,
            "positive_fraction": (
                float(np.mean(state_returns > 0.0)) if len(state_returns) else None
            ),
        }

    gross_bps = 10_000.0 * mi
    return {
        "n": int(len(x)),
        "mi_nats": mi,
        "bootstrap_ci_nats": [float(ci[0]), float(ci[1])],
        "permutation_p": perm_p,
        "gross_bound_bps": float(gross_bps),
        "cost_bps": float(cost_bps),
        "net_bound_bps": float(gross_bps - cost_bps),
        "conditional_forward_return": conditional,
    }


def forward_returns_from_close(close: np.ndarray, horizon: int) -> np.ndarray:
    close_arr = np.asarray(close, dtype=np.float64)
    out = np.full(len(close_arr), np.nan, dtype=np.float64)
    if horizon <= 0 or horizon >= len(close_arr):
        return out
    out[:-horizon] = np.log(close_arr[horizon:] / close_arr[:-horizon])
    return out


def vr_holm_trigger_labels(
    close: np.ndarray,
    W: int,
    target_q: int,
    q_grid: tuple[int, ...] = DEFAULT_VR_Q_GRID,
    stride: int | None = None,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Return non-overlapping binary labels for one Holm-corrected VR trigger.

    The label is one when the target horizon's window-level M2 statistic is
    positive and its two-sided p-value survives the frozen four-horizon Holm
    family used by the VR cascade.
    """
    if target_q not in q_grid:
        raise ValueError("target_q must be included in q_grid")
    if len(q_grid) != len(DEFAULT_VR_Q_GRID):
        raise ValueError("VR cascade Holm labels require the frozen four-q family")
    close_arr = np.asarray(close, dtype=np.float64)
    if W < max(q_grid) + 2:
        raise ValueError("W must be large enough for every q in the frozen family")

    from scipy.stats import norm  # noqa: PLC0415

    import scripts.wp1.vr_significance as vr_significance  # noqa: PLC0415

    step = W if stride is None else int(stride)
    z_by_q: list[np.ndarray] = []
    for q_val in q_grid:
        _vr, z_arr = vr_significance.compute_rolling_vr_and_z_strided(
            close_arr,
            W=W,
            q=int(q_val),
            stride=step,
        )
        z_by_q.append(z_arr)

    z_matrix = np.vstack(z_by_q).T
    finite = np.all(np.isfinite(z_matrix), axis=1)
    indices = np.flatnonzero(finite)
    labels = np.zeros(len(indices), dtype=np.int8)
    target_idx = q_grid.index(target_q)

    for out_idx, row_idx in enumerate(indices):
        z_row = z_matrix[row_idx]
        pvalues = [2.0 * float(norm.sf(abs(float(z_val)))) for z_val in z_row]
        adjusted = vr_significance.apply_holm_b(pvalues)
        labels[out_idx] = int(adjusted[target_idx] < alpha and z_row[target_idx] > 0.0)

    return {
        "indices": indices,
        "labels": labels,
        "target_q": int(target_q),
        "q_grid": list(q_grid),
        "W": int(W),
        "stride": int(step),
        "alpha": float(alpha),
    }


def vr_holm_trigger_information(
    close: np.ndarray,
    timestamps: np.ndarray,
    W: int,
    target_q: int,
    q_grid: tuple[int, ...] = DEFAULT_VR_Q_GRID,
    stride: int | None = None,
    alpha: float = 0.05,
    forward_horizon: int | None = None,
    cost_bps: float = DEFAULT_COST_BPS,
    n_boot: int = 500,
    n_perm: int = 1000,
    seed: int = 43,
) -> dict[str, Any]:
    """Estimate detector-contingent MI for a binary Holm-corrected VR trigger."""
    close_arr = np.asarray(close, dtype=np.float64)
    ts_arr = np.asarray(timestamps, dtype=np.int64)
    if close_arr.shape != ts_arr.shape:
        raise ValueError("close and timestamps must have the same shape")

    trigger = vr_holm_trigger_labels(
        close_arr,
        W=W,
        target_q=target_q,
        q_grid=q_grid,
        stride=stride,
        alpha=alpha,
    )
    indices = np.asarray(trigger["indices"], dtype=int)
    labels = np.asarray(trigger["labels"], dtype=np.int8)
    horizon = int(forward_horizon or target_q)
    forward = forward_returns_from_close(close_arr, horizon=horizon)
    forward_at_labels = forward[indices] if len(indices) else np.array([], dtype=np.float64)
    info = detector_contingent_information(
        labels,
        forward_at_labels,
        n_states=2,
        cost_bps=cost_bps,
        n_boot=n_boot,
        n_perm=n_perm,
        seed=seed,
    )
    n_labels = int(len(labels))
    n_triggers = int(np.sum(labels == 1))
    entropy = discrete_entropy(labels)
    info.update(
        {
            "target_q": int(target_q),
            "forward_horizon": horizon,
            "W": int(W),
            "stride": int(trigger["stride"]),
            "q_grid": list(q_grid),
            "alpha": float(alpha),
            "n_labels": n_labels,
            "n_triggers": n_triggers,
            "trigger_rate": float(n_triggers / n_labels) if n_labels else 0.0,
            "label_entropy_nats": entropy,
            "first_label_ts": _format_utc(int(ts_arr[indices[0]])) if n_labels else None,
            "last_label_ts": _format_utc(int(ts_arr[indices[-1]])) if n_labels else None,
        }
    )
    return info


def discrete_labels_for_task(det: DetectorOutput, task: str) -> np.ndarray:
    """Return the discrete label vector used for agreement and detector MI."""
    if det.soft_probabilities is not None:
        probs = np.asarray(det.soft_probabilities, dtype=np.float64)
        labels = np.full(len(probs), -1, dtype=np.int8)
        if probs.ndim == 1:
            finite = np.isfinite(probs)
            labels[finite] = (probs[finite] > 0.5).astype(np.int8)
        elif probs.ndim == 2:
            finite = np.all(np.isfinite(probs), axis=1)
            labels[finite] = np.argmax(probs[finite], axis=1).astype(np.int8)
        else:
            raise ValueError("soft_probabilities must be a 1-D or 2-D array")
    else:
        labels = np.asarray(det.labels, dtype=np.int8)

    if task == "2-state":
        return collapse_to_two_state(labels)
    if task == "3-state":
        return labels
    raise ValueError(f"unknown task {task!r}")


def _summarize_detector(
    det: DetectorOutput,
    labels: np.ndarray,
    n_states: int,
    validity: dict[str, Any],
) -> dict[str, Any]:
    return {
        "name": det.name,
        "common_state_eligible": bool(det.common_state_eligible),
        "validity": validity,
        "occupancy": state_occupancy(labels, n_states),
        "transition_matrix": transition_matrix(labels, n_states),
        "run_lengths": run_length_summary(labels),
        "diagnostics": det.diagnostics,
    }


def _task_labels(detectors: list[DetectorOutput], task: str) -> dict[str, np.ndarray]:
    return {det.name: discrete_labels_for_task(det, task) for det in detectors}


def build_task_report(
    detectors: list[DetectorOutput],
    task: str,
    n_states: int,
    forward_returns_by_timestamp: dict[int, float] | None = None,
    min_nonoverlap: int = DEFAULT_MIN_NONOVERLAP,
    stride: int = DEFAULT_STATE_STRIDE,
    cost_bps: float = DEFAULT_COST_BPS,
) -> dict[str, Any]:
    labels_by_name = _task_labels(detectors, task)
    remapped = [
        DetectorOutput(
            name=det.name,
            timestamps=det.timestamps,
            labels=labels_by_name[det.name],
            diagnostics=det.diagnostics,
            common_state_eligible=det.common_state_eligible,
        )
        for det in detectors
    ]
    intersection = shared_intersection(remapped)

    detector_reports: dict[str, Any] = {}
    for det in remapped:
        aligned_labels = intersection["labels"].get(det.name, np.array([], dtype=np.int8))
        diagnostics = det.diagnostics
        validity = validate_label_stream(
            det.name,
            aligned_labels,
            n_states=n_states,
            stride=stride,
            min_nonoverlap=min_nonoverlap,
            variance_separation=diagnostics.get("variance_separation"),
            convergence_ok=bool(diagnostics.get("convergence_ok", True)),
            deterministic=bool(diagnostics.get("deterministic", True)),
            common_state_eligible=bool(det.common_state_eligible),
        )
        detector_reports[det.name] = _summarize_detector(det, aligned_labels, n_states, validity)

    pairwise: dict[str, Any] = {}
    eligible_names = [
        name
        for name, report in detector_reports.items()
        if report["validity"]["status"] == STATUS_VALID
    ]
    for i, left in enumerate(eligible_names):
        for right in eligible_names[i + 1 :]:
            pairwise[f"{left}__{right}"] = pairwise_agreement(
                intersection["labels"][left],
                intersection["labels"][right],
                n_states=n_states,
            )

    economic: dict[str, Any] = {}
    if forward_returns_by_timestamp is not None and len(intersection["timestamps"]) > 0:
        returns = np.array(
            [
                forward_returns_by_timestamp.get(int(ts), float("nan"))
                for ts in intersection["timestamps"]
            ],
            dtype=np.float64,
        )
        for name, labels in intersection["labels"].items():
            economic[name] = detector_contingent_information(
                labels,
                returns,
                n_states=n_states,
                cost_bps=cost_bps,
                n_boot=100,
                n_perm=200,
            )

    return {
        "task": task,
        "n_states": n_states,
        "n_intersection": int(len(intersection["timestamps"])),
        "detectors": detector_reports,
        "pairwise": pairwise,
        "economic": economic,
    }


def classify_disposition(tasks: dict[str, Any]) -> dict[str, Any]:
    statuses = [
        det["validity"]["status"]
        for task in tasks.values()
        for det in task["detectors"].values()
    ]
    if STATUS_INSTRUMENT_FAILURE in statuses:
        return {
            "classification": CLASS_INSTRUMENT_FAILURE,
            "reason": "At least one detector failed frozen validity gates; report instrument failure before interpreting disagreement.",
        }

    agreement_values = [
        pair["nmi"]
        for task in tasks.values()
        for pair in task["pairwise"].values()
        if pair["nmi"] is not None
    ]
    if not agreement_values:
        return {
            "classification": CLASS_MIXED_FAMILY_STRUCTURE,
            "reason": "No valid pairwise common-state comparisons were available.",
        }
    if min(agreement_values) >= 0.60:
        return {
            "classification": CLASS_HARMONIZED_CONVERGENCE,
            "reason": "All valid pairwise normalized mutual information values are at least 0.60.",
        }
    if max(agreement_values) <= 0.20:
        return {
            "classification": CLASS_PERSISTENT_DISAGREEMENT,
            "reason": "All valid pairwise normalized mutual information values are at most 0.20.",
        }
    return {
        "classification": CLASS_MIXED_FAMILY_STRUCTURE,
        "reason": "Agreement is heterogeneous across detector pairs or label tasks.",
    }


def build_artifact(
    detectors: list[DetectorOutput],
    data_span: dict[str, Any],
    forward_returns_by_timestamp: dict[int, float] | None = None,
    min_nonoverlap: int = DEFAULT_MIN_NONOVERLAP,
    stride: int = DEFAULT_STATE_STRIDE,
    cost_bps: float = DEFAULT_COST_BPS,
    run_mode: str = "smoke",
) -> dict[str, Any]:
    tasks = {
        "2-state": build_task_report(
            detectors,
            task="2-state",
            n_states=2,
            forward_returns_by_timestamp=forward_returns_by_timestamp,
            min_nonoverlap=min_nonoverlap,
            stride=stride,
            cost_bps=cost_bps,
        ),
        "3-state": build_task_report(
            detectors,
            task="3-state",
            n_states=3,
            forward_returns_by_timestamp=forward_returns_by_timestamp,
            min_nonoverlap=min_nonoverlap,
            stride=stride,
            cost_bps=cost_bps,
        ),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "run_date": datetime.now(tz=UTC).isoformat(),
        "benchmark": {
            "protocol_id": PROTOCOL_ID,
            "run_mode": run_mode,
            "state_tasks": ["2-state", "3-state"],
            "status_policy": {
                STATUS_INSTRUMENT_FAILURE: "Detector outputs are not admissible as labels for agreement interpretation.",
                STATUS_EXCLUDED: "Detector has a different measurement target for this common-state headline.",
                STATUS_VALID: "Detector stream passes frozen benchmark validity gates.",
            },
            "metrics": ["confusion_matrix", "cohens_kappa", "ari", "nmi", "variation_of_information"],
            "discrete_output_policy": (
                "Agreement and detector-contingent MI use discrete detector states; "
                "soft HMM probabilities are converted by argmax before task remapping."
            ),
        },
        "data_span": data_span,
        "tasks": tasks,
        "disposition": classify_disposition(tasks),
        "provenance": {
            "code_commit": _git_commit(),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS", ""),
        },
    }


def synthetic_smoke_detectors() -> tuple[list[DetectorOutput], dict[int, float]]:
    timestamps = np.arange(100, 220, dtype=np.int64)
    base = np.tile(np.array([0, 0, 1, 1, 2, 2], dtype=np.int8), 20)
    hmm = collapse_to_two_state(base)
    ms = np.tile(np.array([0, 0, 1, 1, 2, 2], dtype=np.int8), 20)
    vr = np.where(base == 0, 0, np.where(base == 1, 2, 1)).astype(np.int8)
    forward = np.where(base == 0, -0.001, np.where(base == 1, 0.001, 0.002))
    forward_by_ts = {int(ts): float(ret) for ts, ret in zip(timestamps, forward, strict=True)}
    detectors = [
        DetectorOutput(
            name="rolling_quantile",
            timestamps=timestamps,
            labels=base,
            diagnostics={"convergence_ok": True, "deterministic": True},
        ),
        DetectorOutput(
            name="hmm",
            timestamps=timestamps,
            labels=hmm,
            diagnostics={
                "convergence_ok": True,
                "deterministic": True,
                "variance_separation": (1.20,),
                "note": "2-state harmonized smoke stream; 3-state task should fail because EXTREME is absent.",
            },
        ),
        DetectorOutput(
            name="ms_variance",
            timestamps=timestamps,
            labels=ms,
            diagnostics={
                "convergence_ok": True,
                "deterministic": True,
                "variance_separation": (1.25, 1.30),
            },
        ),
        DetectorOutput(
            name="vr_cascade",
            timestamps=timestamps,
            labels=vr,
            diagnostics={
                "convergence_ok": True,
                "deterministic": True,
                "target": "serial_dependence",
            },
            common_state_eligible=False,
        ),
    ]
    return detectors, forward_by_ts


def run_smoke(out: Path) -> dict[str, Any]:
    detectors, forward_by_ts = synthetic_smoke_detectors()
    artifact = build_artifact(
        detectors,
        data_span={
            "symbol": "SYNTHETIC",
            "start": "smoke",
            "end": "smoke",
            "n_bars": len(detectors[0].timestamps),
            "year_2026_loaded": False,
        },
        forward_returns_by_timestamp=forward_by_ts,
        min_nonoverlap=3,
        stride=3,
        cost_bps=1.0,
        run_mode="smoke",
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_to_serializable(artifact), indent=2), encoding="utf-8")
    return artifact


def _load_btc_window(
    symbol: str,
    start_date: str,
    end_date: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    import datetime as _dt

    from scripts.wp1.py_engine import load_and_clean

    start_ms = _parse_utc_ms(start_date)
    end_ms = _parse_utc_ms(end_date, end_of_day="T" not in end_date)
    holdout_boundary_ms = int(_dt.datetime(2026, 1, 1, tzinfo=_dt.UTC).timestamp() * 1000)
    if end_ms >= holdout_boundary_ms:
        raise RuntimeError("benchmark in-sample window must end before 2026 holdout")
    data = load_and_clean(
        data_path=str(_REPO_ROOT / "data" / "binance_futures"),
        symbol=symbol,
        start_ms=start_ms,
        end_ms=end_ms,
        max_gap_allowed_mins=60,
    )
    close = np.asarray(data["close"], dtype=np.float64)
    timestamps = np.asarray(data["timestamp"], dtype=np.int64)
    return slice_1m_window(close, timestamps, start_date, end_date, symbol=symbol)


def _fit_ms_variance_labels(close: np.ndarray, k_regimes: int = 3) -> tuple[np.ndarray, dict[str, Any]]:
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

    rv = compute_rv(close, rv_window=60, ewma=False)
    warmup = int(np.sum(np.isnan(rv)))
    finite = rv[warmup:]
    positive = finite[finite > 0.0]
    floor = float(np.min(positive)) if len(positive) else 1e-10
    endog = np.log(np.where(finite > 0.0, finite, floor))
    np.random.seed(42)
    model = MarkovRegression(
        endog,
        k_regimes=k_regimes,
        trend="c",
        switching_variance=True,
        switching_trend=False,
    )
    res = model.fit(em_iter=100, search_reps=3, disp=0)
    sigma_idx = [i for i, name in enumerate(res.model.param_names) if name.startswith("sigma2")]
    sigmas = res.params[sigma_idx]
    order = np.argsort(sigmas)
    probs = res.filtered_marginal_probabilities[:, order]
    finite_labels = np.argmax(probs, axis=1).astype(np.int8)
    labels = np.full(len(close), -1, dtype=np.int8)
    labels[warmup:] = finite_labels
    sorted_sigmas = sigmas[order]
    ratios = tuple(
        float(sorted_sigmas[i + 1] / sorted_sigmas[i])
        for i in range(len(sorted_sigmas) - 1)
        if sorted_sigmas[i] > 0.0
    )
    return labels, {
        "convergence_ok": bool(getattr(res, "mle_retvals", {}).get("converged", True)),
        "deterministic": True,
        "variance_separation": ratios,
        "sigma2_sorted": sorted_sigmas.tolist(),
        "llf": float(res.llf),
    }


def build_real_detectors(
    close: np.ndarray,
    timestamps: np.ndarray,
    detectors: list[str],
    W: int,
    q: int,
    hmm_regimes: int = 2,
    hmm_em_iter: int = 30,
    hmm_search_reps: int = 3,
) -> list[DetectorOutput]:
    outputs: list[DetectorOutput] = []
    if "rolling_quantile" in detectors:
        rq = RollingQuantileDetector()
        outputs.append(
            DetectorOutput(
                name="rolling_quantile",
                timestamps=timestamps,
                labels=rq.fit(close),
                diagnostics={"convergence_ok": True, "deterministic": True},
            )
        )
    if "hmm" in detectors:
        hmm = HMMDetector(
            k_regimes=hmm_regimes,
            em_iter=hmm_em_iter,
            search_reps=hmm_search_reps,
        )
        labels = hmm.fit(close)
        outputs.append(
            DetectorOutput(
                name="hmm",
                timestamps=timestamps,
                labels=labels,
                soft_probabilities=hmm.filtered_probs_,
                diagnostics={
                    "convergence_ok": bool(hmm.convergence_ok_),
                    "deterministic": True,
                    "variance_separation": tuple(
                        float(hmm.sigma2_sorted_[i + 1] / hmm.sigma2_sorted_[i])
                        for i in range(len(hmm.sigma2_sorted_) - 1)
                        if hmm.sigma2_sorted_[i] > 0.0
                    ),
                    "sigma2_sorted": hmm.sigma2_sorted_.tolist(),
                    "hmm_fallback": hmm.hmm_fallback_,
                    "hmm_regimes": int(hmm_regimes),
                    "hmm_em_iter": int(hmm_em_iter),
                    "hmm_search_reps": int(hmm_search_reps),
                    "hmm_profile": (
                        "explicit_2_state_harmonized_profile"
                        if hmm_regimes == 2
                        else "explicit_3_state_harmonized_profile"
                    ),
                },
            )
        )
    if "ms_variance" in detectors:
        labels, diagnostics = _fit_ms_variance_labels(close, k_regimes=3)
        outputs.append(
            DetectorOutput(
                name="ms_variance",
                timestamps=timestamps,
                labels=labels,
                diagnostics=diagnostics,
            )
        )
    if "vr_cascade" in detectors:
        log_close = np.log(close)
        r = np.empty(len(close), dtype=np.float64)
        r[0] = 0.0
        r[1:] = np.diff(log_close)
        vr, z = rolling_vr_m2_z_arrays(r, W=W, q=q, N=len(close), stride=1)
        labels = map_score_to_three_states(vr)
        outputs.append(
            DetectorOutput(
                name="vr_cascade",
                timestamps=timestamps,
                labels=labels,
                target_statistic=vr,
                diagnostics={
                    "convergence_ok": True,
                    "deterministic": True,
                    "target": "serial_dependence",
                    "W": W,
                    "q": q,
                    "z_median": float(np.nanmedian(z)),
                },
                common_state_eligible=False,
            )
        )
    return outputs


def run_real(
    out: Path,
    detector_names: list[str],
    W: int,
    q: int,
    symbol: str,
    start_date: str,
    end_date: str,
    hmm_regimes: int = 2,
    hmm_em_iter: int = 30,
    hmm_search_reps: int = 3,
) -> dict[str, Any]:
    close, timestamps, span = _load_btc_window(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
    )
    detectors = build_real_detectors(
        close,
        timestamps,
        detector_names,
        W=W,
        q=q,
        hmm_regimes=hmm_regimes,
        hmm_em_iter=hmm_em_iter,
        hmm_search_reps=hmm_search_reps,
    )
    forward = forward_returns_from_close(close, horizon=q)
    forward_by_ts = {int(ts): float(ret) for ts, ret in zip(timestamps, forward, strict=True)}
    artifact = build_artifact(
        detectors,
        data_span=span,
        forward_returns_by_timestamp=forward_by_ts,
        run_mode="real",
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_to_serializable(artifact), indent=2), encoding="utf-8")
    return artifact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Write a deterministic synthetic smoke artifact.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("backtest_results/harmonized_benchmark/harmonized_benchmark_smoke.json"),
        help="Output JSON artifact path.",
    )
    parser.add_argument(
        "--detectors",
        nargs="+",
        default=["rolling_quantile", "hmm", "ms_variance", "vr_cascade"],
        choices=["rolling_quantile", "hmm", "ms_variance", "vr_cascade"],
        help="Detector adapters to run for real BTCUSDT mode.",
    )
    parser.add_argument("--W", type=int, default=120, help="VR window for real-mode VR adapter.")
    parser.add_argument("--q", type=int, default=5, help="Forward/economic horizon and VR horizon.")
    parser.add_argument(
        "--hmm-regimes",
        type=int,
        default=2,
        choices=[2, 3],
        help="HMM state count for real-mode benchmark; default is explicit 2-state harmonized profile.",
    )
    parser.add_argument(
        "--hmm-em-iter",
        type=int,
        default=30,
        help="HMM EM iterations for the explicit real-mode benchmark profile.",
    )
    parser.add_argument(
        "--hmm-search-reps",
        type=int,
        default=3,
        help="HMM restart count for the explicit real-mode benchmark profile.",
    )
    parser.add_argument("--symbol", default="BTCUSDT", help="Symbol to load in real mode.")
    parser.add_argument(
        "--start",
        default="2022-10-01",
        help="UTC start date/timestamp for real mode; keeps native 1-minute bars.",
    )
    parser.add_argument(
        "--end",
        default="2022-12-31",
        help="UTC end date/timestamp for real mode; keeps native 1-minute bars.",
    )
    args = parser.parse_args(argv)

    if args.smoke:
        artifact = run_smoke(args.out)
    else:
        artifact = run_real(
            args.out,
            detector_names=args.detectors,
            W=args.W,
            q=args.q,
            symbol=args.symbol,
            start_date=args.start,
            end_date=args.end,
            hmm_regimes=args.hmm_regimes,
            hmm_em_iter=args.hmm_em_iter,
            hmm_search_reps=args.hmm_search_reps,
        )
    print(
        f"Wrote {args.out} "
        f"({artifact['disposition']['classification']}, {artifact['benchmark']['run_mode']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
