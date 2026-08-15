"""Protocol-v5 EUR/USD rolling-quantile calibration and certificate evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

import scripts._bootstrap  # noqa: F401

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "external" / "histdata" / "EURUSD" / "tick"
MANIFEST_DIR = REPO_ROOT / "evidence" / "data"
DEVELOPMENT_START = "2021-05-30T00:00:00Z"
DEVELOPMENT_END = "2025-12-31T23:59:59Z"
EVALUATION_START = "2026-01-01T00:00:00Z"
EVALUATION_END = "2026-07-31T23:59:59Z"
RV_WINDOW = 60
PCT_WINDOW = 43_200
SCALES = (1.25, 1.5, 2.0, 3.0, 4.0)


def _epoch(value: str) -> int:
    return int(np.datetime64(value.replace("Z", "")).astype("datetime64[s]").astype(np.int64))


def load_minutes(*, start: str, end: str) -> dict[str, np.ndarray]:
    """Load normalized monthly partitions covering an inclusive UTC span."""
    import pandas as pd  # noqa: PLC0415

    start_epoch, end_epoch = _epoch(start), _epoch(end)
    frames = []
    for path in sorted(DATA_DIR.glob("EURUSD_*_m1_bidask.csv")):
        frame = pd.read_csv(
            path,
            usecols=["timestamp", "bid", "ask", "mid", "n_ticks"],
            dtype={"bid": "float64", "ask": "float64", "mid": "float64", "n_ticks": "int64"},
        )
        stamp = pd.to_datetime(frame.pop("timestamp"), utc=True)
        frame.insert(0, "timestamp", stamp.astype("int64") // 1_000_000_000)
        mask = (frame["timestamp"] >= start_epoch) & (frame["timestamp"] <= end_epoch)
        if bool(mask.any()):
            frames.append(frame.loc[mask])
    if not frames:
        raise FileNotFoundError("no normalized EUR/USD partitions cover the requested span")
    joined = pd.concat(frames, ignore_index=True)
    return {column: joined[column].to_numpy() for column in joined.columns}


def consolidate_minutes(data: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Sort UTC minutes, keep the last quote, and preserve total tick counts."""
    timestamp = np.asarray(data["timestamp"], dtype=np.int64)
    order = np.argsort(timestamp, kind="stable")
    sorted_data = {name: np.asarray(values)[order] for name, values in data.items()}
    unique, first, counts = np.unique(sorted_data["timestamp"], return_index=True, return_counts=True)
    last = first + counts - 1
    clean = {
        "timestamp": unique,
        "bid": np.asarray(sorted_data["bid"], dtype=np.float64)[last],
        "ask": np.asarray(sorted_data["ask"], dtype=np.float64)[last],
        "mid": np.asarray(sorted_data["mid"], dtype=np.float64)[last],
        "n_ticks": np.add.reduceat(
            np.asarray(sorted_data["n_ticks"], dtype=np.int64),
            first,
        ),
    }
    return clean, {
        "input_rows": int(timestamp.size),
        "output_rows": int(unique.size),
        "duplicate_rows": int(np.sum(counts - 1)),
    }


def _finish_clock(data: dict[str, np.ndarray], close_indices: list[int]) -> dict[str, np.ndarray]:
    indices = np.asarray(close_indices, dtype=np.int64)
    return {
        "timestamp": data["timestamp"][indices],
        "bid": data["bid"][indices],
        "ask": data["ask"][indices],
        "mid": data["mid"][indices],
        "n_ticks": data["n_ticks"][indices],
    }


def build_tick_clock(data: dict[str, np.ndarray], threshold: int) -> dict[str, np.ndarray]:
    """Causal tick-count bars closed when accumulated source ticks cross threshold."""
    if threshold <= 0:
        raise ValueError("tick threshold must be positive")
    accumulated = 0
    close_indices: list[int] = []
    for index, ticks in enumerate(data["n_ticks"]):
        accumulated += int(ticks)
        if accumulated >= threshold:
            close_indices.append(index)
            accumulated = 0
    return _finish_clock(data, close_indices)


def build_intrinsic_clock(data: dict[str, np.ndarray], threshold: float) -> dict[str, np.ndarray]:
    """Causal intrinsic-variance bars closed on cumulative squared log return."""
    if threshold <= 0.0:
        raise ValueError("intrinsic threshold must be positive")
    returns = np.diff(np.log(data["mid"]), prepend=np.log(data["mid"][0]))
    accumulated = 0.0
    close_indices: list[int] = []
    for index, contribution in enumerate(returns * returns):
        accumulated += float(contribution)
        if accumulated >= threshold:
            close_indices.append(index)
            accumulated = 0.0
    return _finish_clock(data, close_indices)


def episode_successes(
    segments: np.ndarray,
    thresholds: np.ndarray,
    *,
    scale: float,
) -> np.ndarray:
    """Apply the frozen six-hour burst event to batched return segments."""
    returns = np.asarray(segments, dtype=np.float64).copy()
    if returns.ndim != 2 or returns.shape[1] != 419:
        raise ValueError("segments must have shape (draws, 419)")
    if thresholds.shape != (returns.shape[0], 360):
        raise ValueError("thresholds must have shape (draws, 360)")
    returns[:, 59:] *= scale
    squared = returns * returns
    cumulative = np.pad(np.cumsum(squared, axis=1), ((0, 0), (1, 0)))
    rv = np.sqrt((cumulative[:, 60:] - cumulative[:, :-60]) / 60.0)
    extreme = rv > thresholds
    occupancy = np.mean(extreme[:, 60:], axis=1)
    return occupancy >= 0.50


def paired_utility(
    forward_return: np.ndarray,
    exposure: np.ndarray,
    spread_fraction: np.ndarray,
    *,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Paired policy-minus-baseline utility under the frozen cost convention."""
    returns = np.asarray(forward_return, dtype=np.float64)
    quantity = np.asarray(exposure, dtype=np.float64)
    spread = np.asarray(spread_fraction, dtype=np.float64)
    change = np.abs(np.diff(quantity, prepend=quantity[0]))
    costs = change * (0.5 * spread + 0.0001)
    policy = quantity * returns - 0.5 * gamma * (quantity * returns) ** 2 - costs
    baseline = returns - 0.5 * gamma * returns**2
    return policy - baseline, costs


def _manifest_hashes(years: list[str]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for year in years:
        name = f"eurusd-histdata-{year}-manifest.json"
        path = MANIFEST_DIR / name
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def calibrate_development() -> dict[str, Any]:
    """Freeze alternative-clock thresholds and gamma using development data only."""
    raw = load_minutes(start=DEVELOPMENT_START, end=DEVELOPMENT_END)
    data, consolidation = consolidate_minutes(raw)
    returns = np.diff(np.log(data["mid"]), prepend=np.log(data["mid"][0]))
    tick_threshold = int(round(float(np.mean(data["n_ticks"]))))
    intrinsic_threshold = float(np.mean(returns[1:] ** 2))
    decision = np.arange(0, len(returns) - 60, 60, dtype=np.int64)
    forward = np.log(data["mid"][decision + 60] / data["mid"][decision])
    gamma = float(1.0 / np.var(forward, ddof=1))
    return {
        "schema_version": 1,
        "artifact": "eurusd_development_calibration",
        "development_span": {"start": DEVELOPMENT_START, "end": DEVELOPMENT_END},
        "n_minutes": int(data["timestamp"].size),
        "consolidation": consolidation,
        "tick_threshold": tick_threshold,
        "intrinsic_variance_threshold": intrinsic_threshold,
        "gamma": gamma,
        "detector": {
            "rv_window": RV_WINDOW,
            "pct_window": PCT_WINDOW,
            "p_elevated": 0.75,
            "p_extreme": 0.95,
            "strict_extreme": True,
        },
        "data_hashes": _manifest_hashes(["2021", "2022", "2023", "2024", "2025"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.calibrate:
        parser.error("one of --calibrate or the certificate evidence mode is required")
    artifact = calibrate_development()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
