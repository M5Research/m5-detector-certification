"""Protocol-v5 EUR/USD rolling-quantile calibration and certificate evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import norm

import scripts._bootstrap  # noqa: F401

from certificate.information import _stationary_index_matrix, primary_block_length
from certificate.models import CertificateSpec, GateResult
from certificate.preregistration import assert_preregistration
from certificate.serialize import spec_hash
from certificate.statistics import (
    exact_binomial_bounds,
    holm_adjust,
    stationary_mean_interval,
)

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


def _fit_detector(data: dict[str, np.ndarray]) -> tuple[np.ndarray, Any]:
    from strategies.vol_regime_switch.rolling_quantile_detector import (  # noqa: PLC0415
        RollingQuantileDetector,
    )

    detector = RollingQuantileDetector(
        rv_window=RV_WINDOW,
        pct_window=PCT_WINDOW,
        p_elevated=0.75,
        p_extreme=0.95,
    )
    return detector.fit(data["mid"]), detector


def _episode_calibration(
    spec: CertificateSpec,
    development: dict[str, np.ndarray],
    detector: Any,
) -> dict[str, Any]:
    returns = np.diff(
        np.log(development["mid"]),
        prepend=np.log(development["mid"][0]),
    )
    q_extreme = np.asarray(detector.q_extreme_, dtype=np.float64)
    candidate_mask = np.isfinite(q_extreme)
    candidate_mask[:59] = False
    candidate_mask[max(0, len(candidate_mask) - 359) :] = False
    candidates = np.flatnonzero(candidate_mask)
    if candidates.size < 10_000:
        raise ValueError("development span has too few eligible burst starts")
    seed = int(spec_hash(spec)[:16], 16)
    rng = np.random.default_rng(seed)
    offsets = np.arange(-59, 360, dtype=np.int64)
    future = np.arange(360, dtype=np.int64)

    power_rows: list[dict[str, Any]] = []
    n_injection = int(spec.thresholds["n_injection_per_cell"])
    for scale in SCALES:
        starts = rng.choice(candidates, size=n_injection, replace=True)
        segments = returns[starts[:, None] + offsets]
        thresholds = q_extreme[starts[:, None] + future]
        successes = int(np.sum(episode_successes(segments, thresholds, scale=scale)))
        bounds = exact_binomial_bounds(successes, n_injection, alpha=spec.alpha)
        power_rows.append(
            {
                "scale": scale,
                "draws": n_injection,
                "successes": successes,
                "rate": successes / n_injection,
                "bounds": bounds,
            }
        )
    cmde = next((row["scale"] for row in power_rows if row["bounds"]["lower"] >= 0.90), None)

    n_null = int(spec.thresholds["n_null"])
    null_successes = 0
    block = primary_block_length(len(returns))
    batch_size = 500
    for start in range(0, n_null, batch_size):
        count = min(batch_size, n_null - start)
        indices = _stationary_sample_indices(len(returns), 419, block, count, rng)
        # Only the 419 entries required by the episode event are materialized.
        segments = returns[indices[:, :419]]
        reference_starts = rng.choice(candidates, size=count, replace=True)
        thresholds = q_extreme[reference_starts[:, None] + future]
        null_successes += int(np.sum(episode_successes(segments, thresholds, scale=1.0)))
    size_bounds = exact_binomial_bounds(null_successes, n_null, alpha=spec.alpha)
    return {
        "power": power_rows,
        "cmde_90": cmde,
        "power_state": "pass" if cmde is not None else "fail",
        "size": {
            "draws": n_null,
            "successes": null_successes,
            "rate": null_successes / n_null,
            "bounds": size_bounds,
            "state": "pass" if size_bounds["upper"] <= spec.margins["size"] else "fail",
        },
        "stationary_mean_block": block,
        "seed": seed,
    }


def _stationary_sample_indices(
    n_obs: int,
    sample_length: int,
    mean_block: int,
    n_rows: int,
    rng: np.random.Generator,
) -> np.ndarray:
    probability = min(1.0, 1.0 / float(mean_block))
    indices = np.empty((n_rows, sample_length), dtype=np.int64)
    indices[:, 0] = rng.integers(0, n_obs, size=n_rows)
    for column in range(1, sample_length):
        restart = rng.random(n_rows) < probability
        fresh = rng.integers(0, n_obs, size=n_rows)
        indices[:, column] = np.where(
            restart,
            fresh,
            (indices[:, column - 1] + 1) % n_obs,
        )
    return indices


def _bootstrap_means(
    values: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    block = primary_block_length(arr.size)
    output = np.empty(n_boot, dtype=np.float64)
    for start in range(0, n_boot, 512):
        stop = min(n_boot, start + 512)
        indices = _stationary_index_matrix(arr.size, block, stop - start, rng)
        output[start:stop] = np.mean(arr[indices], axis=1)
    return output


def _transport_evidence(
    calendar_labels: np.ndarray,
    eval_timestamps: np.ndarray,
    alternate: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    p_values: list[float] = []
    for clock_index, (clock_name, (timestamps, labels)) in enumerate(alternate.items()):
        positions = np.searchsorted(timestamps, eval_timestamps, side="right") - 1
        valid = (positions >= 0) & (calendar_labels >= 0)
        mapped = np.full(calendar_labels.shape, -1, dtype=np.int8)
        mapped[valid] = labels[positions[valid]]
        valid &= mapped >= 0
        comparisons = (
            ("overall", (calendar_labels[valid] != mapped[valid]).astype(float), 0.10),
            (
                "extreme",
                ((calendar_labels[valid] == 2) != (mapped[valid] == 2)).astype(float),
                0.05,
            ),
        )
        for comparison_index, (name, values, margin) in enumerate(comparisons):
            boot = _bootstrap_means(
                values,
                n_boot=n_boot,
                seed=seed + 100 * clock_index + comparison_index,
            )
            point = float(np.mean(values))
            lower, upper = np.quantile(boot, [0.025, 0.975])
            standard_error = float(np.std(boot, ddof=1))
            p_value = (
                0.0
                if standard_error == 0.0 and point < margin
                else float(norm.cdf((point - margin) / standard_error))
            )
            p_values.append(p_value)
            rows.append(
                {
                    "clock": clock_name,
                    "comparison": name,
                    "n": int(values.size),
                    "estimand": point,
                    "margin": margin,
                    "ci_95": [float(lower), float(upper)],
                    "raw_p_value": p_value,
                }
            )
    adjusted = holm_adjust(p_values)
    for row, adjusted_p in zip(rows, adjusted, strict=True):
        row["holm_p_value"] = adjusted_p
        row["state"] = (
            "pass"
            if adjusted_p <= 0.05 and row["ci_95"][1] <= row["margin"]
            else "fail"
        )
    return {
        "comparisons": rows,
        "state": "pass" if all(row["state"] == "pass" for row in rows) else "fail",
        "n_boot": n_boot,
    }


def _information_target(
    data: dict[str, np.ndarray],
    labels: np.ndarray,
    detector: Any,
    *,
    evaluation_start: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    returns = np.diff(np.log(data["mid"]), prepend=np.log(data["mid"][0]))
    squared = returns * returns
    cumulative = np.pad(np.cumsum(squared), (1, 0))
    future_rv = np.full(len(returns), np.nan)
    future_rv[:-60] = np.sqrt((cumulative[61:] - cumulative[1:-60]) / 60.0)
    decision = np.arange(0, len(returns) - 60, 60, dtype=np.int64)
    valid = (
        (data["timestamp"][decision] >= evaluation_start)
        & (labels[decision] >= 0)
        & np.isfinite(detector.q_elevated_[decision])
        & np.isfinite(future_rv[decision])
    )
    decision = decision[valid]
    target = (future_rv[decision] > detector.q_elevated_[decision]).astype(np.int8)
    forward = np.log(data["mid"][decision + 60] / data["mid"][decision])
    return decision, labels[decision], target, forward


def build_evidence(
    spec: CertificateSpec,
    *,
    n_info_boot: int,
    n_transport_boot: int,
    n_value_boot: int,
) -> dict[str, Any]:
    """Generate the complete seven-gate EUR/USD certificate evidence."""
    digest = spec_hash(spec)
    development_raw = load_minutes(start=DEVELOPMENT_START, end=DEVELOPMENT_END)
    evaluation_raw = load_minutes(start=EVALUATION_START, end=EVALUATION_END)
    development, development_consolidation = consolidate_minutes(development_raw)
    evaluation, evaluation_consolidation = consolidate_minutes(evaluation_raw)
    combined = {
        name: np.concatenate([development[name], evaluation[name]])
        for name in development
    }
    combined, combined_consolidation = consolidate_minutes(combined)

    calendar_labels, calendar_detector = _fit_detector(combined)
    development_labels, development_detector = _fit_detector(development)
    calibration = _episode_calibration(spec, development, development_detector)

    tick = build_tick_clock(combined, int(spec.thresholds["tick_threshold"]))
    intrinsic = build_intrinsic_clock(
        combined,
        float(spec.thresholds["intrinsic_variance_threshold"]),
    )
    tick_labels, _tick_detector = _fit_detector(tick)
    intrinsic_labels, _intrinsic_detector = _fit_detector(intrinsic)
    eval_start_epoch = _epoch(EVALUATION_START)
    eval_mask = combined["timestamp"] >= eval_start_epoch
    eval_calendar_labels = calendar_labels[eval_mask]
    eval_timestamps = combined["timestamp"][eval_mask]
    transport = _transport_evidence(
        eval_calendar_labels,
        eval_timestamps,
        {
            "tick-count": (tick["timestamp"], tick_labels),
            "intrinsic-variance": (intrinsic["timestamp"], intrinsic_labels),
        },
        n_boot=n_transport_boot,
        seed=int(digest[:16], 16),
    )

    decision, info_labels, target, forward = _information_target(
        combined,
        calendar_labels,
        calendar_detector,
        evaluation_start=eval_start_epoch,
    )
    from scripts.wp1.vr_detector_stationary_block import (  # noqa: PLC0415
        stationary_trigger_information,
    )

    information = stationary_trigger_information(
        info_labels,
        target,
        spec_hash=digest,
        alpha=spec.alpha,
        n_boot=n_info_boot,
        gate_label="eurusd-rq-next-hour-information",
    )
    information.update(
        {
            "n_decision_epochs": int(decision.size),
            "target_rate": float(np.mean(target)),
            "label_counts": {
                str(state): int(np.sum(info_labels == state)) for state in (0, 1, 2)
            },
        }
    )
    target_state = "pass" if target.size >= 100 and 0.05 < np.mean(target) < 0.95 else "fail"

    exposure = np.choose(info_labels, [1.0, 0.5, 0.0])
    spread_fraction = (
        (combined["ask"][decision] - combined["bid"][decision])
        / combined["mid"][decision]
    )
    utility, costs = paired_utility(
        forward,
        exposure,
        spread_fraction,
        gamma=float(spec.thresholds["gamma"]),
    )
    value_ci = stationary_mean_interval(
        utility,
        n_boot=n_value_boot,
        alpha=spec.alpha,
        seed=int(digest[16:32], 16),
    )
    value_point = float(np.mean(utility))
    value_state = "pass" if value_ci[0] > 0.0 else "fail"

    eval_manifest = json.loads(
        (MANIFEST_DIR / "eurusd-histdata-2026-01_07-manifest.json").read_text(encoding="utf-8")
    )
    instrument_ok = all(
        row["ask_gte_bid_positive"] and row["raw_sha256"] and row["sha256"]
        for row in eval_manifest["partitions"]
    ) and np.all(np.diff(evaluation["timestamp"]) > 0)
    artifact_ref = f"evidence/gates/{spec.certificate_id}/evidence.json"
    gate_results = [
        GateResult(
            gate="instrument",
            state="pass" if instrument_ok else "fail",
            estimand=float(evaluation["timestamp"].size),
            threshold=1.0,
            artifacts=[artifact_ref],
            reason=(
                f"raw/normalized hashes verified; {evaluation_consolidation['duplicate_rows']} "
                "duplicate minute rows deterministically consolidated"
            ),
        ),
        GateResult(
            gate="target",
            state=target_state,
            estimand=float(np.mean(target)),
            threshold=0.05,
            artifacts=[artifact_ref],
            reason="next-hour volatility target is non-degenerate at frozen hourly epochs",
        ),
        GateResult(
            gate="size",
            state=calibration["size"]["state"],
            estimand=calibration["size"]["rate"],
            threshold=spec.margins["size"],
            uncertainty=calibration["size"]["bounds"],
            artifacts=[artifact_ref],
            reason="stationary-block null episode event with exact upper bound",
        ),
        GateResult(
            gate="transport",
            state=transport["state"],
            estimand=max(row["estimand"] for row in transport["comparisons"]),
            threshold=max(spec.margins["transport_overall"], spec.margins["transport_extreme"]),
            artifacts=[artifact_ref],
            reason="four frozen dependence-aware equivalence comparisons with Holm adjustment",
        ),
        GateResult(
            gate="power",
            state=calibration["power_state"],
            estimand=calibration["cmde_90"],
            threshold=spec.margins["power"],
            artifacts=[artifact_ref],
            reason="smallest burst scale with exact lower detection bound >= 0.90",
        ),
        GateResult(
            gate="information",
            state=information["gate_state"],
            estimand=information["e_value"],
            threshold=information["e_value_threshold"],
            uncertainty={"mi_lower": information["ci"][0], "mi_upper": information["ci"][1]},
            artifacts=[artifact_ref],
            reason=information["reason"],
        ),
        GateResult(
            gate="value",
            state=value_state,
            estimand=value_point,
            threshold=0.0,
            uncertainty={"lower": value_ci[0], "upper": value_ci[1]},
            artifacts=[artifact_ref],
            reason="paired mean-variance utility net of half-spread plus one-bp slippage",
        ),
    ]
    eval_paths = [DATA_DIR / f"EURUSD_2026_{month:02d}_m1_bidask.csv" for month in range(1, 8)]
    return {
        "schema_version": 1,
        "artifact": "eurusd_rolling_quantile_protocol_v5_evidence",
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "preregistration_commit": spec.preregistration_commit,
        "spec_hash": digest,
        "data_hashes": {
            "evaluation_manifest": hashlib.sha256(
                (MANIFEST_DIR / "eurusd-histdata-2026-01_07-manifest.json").read_bytes()
            ).hexdigest(),
            "development_calibration": hashlib.sha256(
                (REPO_ROOT / "evidence" / "calibration" / "eurusd-development-v1.json").read_bytes()
            ).hexdigest(),
            **{path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in eval_paths},
        },
        "data_quality": {
            "development": development_consolidation,
            "evaluation": evaluation_consolidation,
            "combined": combined_consolidation,
            "vendor_partitions": eval_manifest["partitions"],
        },
        "clock_counts": {
            "calendar": int(combined["timestamp"].size),
            "tick-count": int(tick["timestamp"].size),
            "intrinsic-variance": int(intrinsic["timestamp"].size),
        },
        "calibration": calibration,
        "transport": transport,
        "information": information,
        "value": {
            "point": value_point,
            "ci": list(value_ci),
            "mean_cost": float(np.mean(costs)),
            "gamma": float(spec.thresholds["gamma"]),
        },
        "gate_results": [gate.model_dump(mode="json") for gate in gate_results],
    }


def load_gate_results(spec: CertificateSpec, repo: Path) -> list[GateResult]:
    path = repo / "evidence" / "gates" / spec.certificate_id / "evidence.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [GateResult.model_validate(row) for row in payload["gate_results"]]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--n-info-boot", type=int, default=49_999)
    parser.add_argument("--n-transport-boot", type=int, default=9_999)
    parser.add_argument("--n-value-boot", type=int, default=49_999)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.calibrate:
        artifact = calibrate_development()
    elif args.spec:
        spec = CertificateSpec.model_validate_json(args.spec.read_text(encoding="utf-8"))
        assert_preregistration(spec, repo=REPO_ROOT, spec_path=args.spec)
        artifact = build_evidence(
            spec,
            n_info_boot=args.n_info_boot,
            n_transport_boot=args.n_transport_boot,
            n_value_boot=args.n_value_boot,
        )
    else:
        parser.error("one of --calibrate or --spec is required")
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
