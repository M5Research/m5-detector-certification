"""Protocol-v5 BTCUSDT q=2 successor calibration and evidence runner."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import norm

import scripts._bootstrap  # noqa: F401

from certificate.information import primary_block_length, stationary_block_indices
from certificate.models import CertificateSpec, GateResult
from certificate.preregistration import assert_preregistration
from certificate.serialize import spec_hash
from certificate.statistics import exact_binomial_bounds, stationary_mean_interval

W = 120
Q = 2
RECENTER_CENTER = -0.004886325945105274
RECENTER_SCALE = 0.010302944723275254
AMPLITUDES = (0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.15)
TRANSPORT_MARGIN = 0.019748
N_NULL = 10_000
N_INJECTION = 2_000
N_WINDOWS = 100
REPO_ROOT = Path(__file__).resolve().parents[2]
BTC_DIR = REPO_ROOT / "data" / "external" / "binance" / "BTCUSDT" / "1m"
BTC_MANIFEST = REPO_ROOT / "evidence" / "data" / "btcusdt-binance-2026-06_07-manifest.json"


def _median_vr2(returns: np.ndarray) -> np.ndarray:
    """Median q=2 variance-ratio departure across each draw's windows."""
    one_var = np.var(returns, axis=-1, ddof=1)
    two = returns[..., :-1] + returns[..., 1:]
    two_var = np.var(two, axis=-1, ddof=1)
    vr_departure = np.divide(
        two_var,
        2.0 * one_var,
        out=np.zeros_like(two_var),
        where=one_var > 0.0,
    ) - 1.0
    return np.median(vr_departure, axis=1)


def simulate_recentered_vr_draws(
    *,
    phi: float,
    n_draws: int,
    n_windows: int = N_WINDOWS,
    seed: int,
    batch_size: int = 128,
) -> np.ndarray:
    """Simulate the existing Gaussian AR(1) family and return recentered statistics."""
    if not 0.0 <= phi < 1.0:
        raise ValueError("phi must be in [0, 1)")
    rng = np.random.default_rng(seed)
    output = np.empty(n_draws, dtype=np.float64)
    scale = float(np.sqrt(1.0 - phi * phi))
    for start in range(0, n_draws, batch_size):
        stop = min(n_draws, start + batch_size)
        innovation = rng.normal(size=(stop - start, n_windows, W))
        series = np.empty_like(innovation)
        series[..., 0] = innovation[..., 0]
        for index in range(1, W):
            series[..., index] = phi * series[..., index - 1] + scale * innovation[..., index]
        raw = _median_vr2(series)
        output[start:stop] = (raw - RECENTER_CENTER) / RECENTER_SCALE
    return output


def calibrate_successor(
    *,
    amplitudes: list[float],
    phi_by_amplitude: dict[float, float],
    n_null: int = N_NULL,
    n_injection: int = N_INJECTION,
    n_windows: int = N_WINDOWS,
    alpha: float = 0.025,
    seed: int = 43,
) -> dict[str, Any]:
    """Run size and power calibration at the revision-specific alpha."""
    critical = float(norm.ppf(1.0 - alpha))
    null = simulate_recentered_vr_draws(
        phi=0.0,
        n_draws=n_null,
        n_windows=n_windows,
        seed=seed,
    )
    null_fires = int(np.sum(null > critical))
    size_bounds = exact_binomial_bounds(null_fires, n_null, alpha=alpha)
    size = {
        "draws": n_null,
        "fires": null_fires,
        "rate": null_fires / n_null,
        "bounds": size_bounds,
        "threshold": critical,
        "state": "pass" if size_bounds["upper"] <= alpha else "fail",
    }
    power: list[dict[str, Any]] = []
    for index, amplitude in enumerate(amplitudes):
        draws = simulate_recentered_vr_draws(
            phi=float(phi_by_amplitude[amplitude]),
            n_draws=n_injection,
            n_windows=n_windows,
            seed=seed + 10_000 * (index + 1),
        )
        fires = int(np.sum(draws > critical))
        bounds = exact_binomial_bounds(fires, n_injection, alpha=alpha)
        power.append(
            {
                "amplitude": float(amplitude),
                "phi": float(phi_by_amplitude[amplitude]),
                "draws": n_injection,
                "fires": fires,
                "rate": fires / n_injection,
                "bounds": bounds,
            }
        )
    cmde = next((row["amplitude"] for row in power if row["bounds"]["lower"] >= 0.90), None)
    return {
        "method": "Gaussian AR(1), 100 independent W=120 windows per draw, median q=2 VR departure",
        "recenter_center": RECENTER_CENTER,
        "recenter_scale": RECENTER_SCALE,
        "alpha": alpha,
        "size": size,
        "power": power,
        "cmde_90": cmde,
        "power_state": "pass" if cmde is not None else "fail",
    }


def transport_tost(
    calendar: np.ndarray,
    volume: np.ndarray,
    *,
    margin: float,
    alpha: float,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    """Dependence-aware TOST for the single frozen calendar--volume comparison."""
    left = np.asarray(calendar, dtype=np.float64)
    right = np.asarray(volume, dtype=np.float64)
    left = left[np.isfinite(left)]
    right = right[np.isfinite(right)]
    if left.size < 2 or right.size < 2:
        raise ValueError("transport samples are too small")
    rng = np.random.default_rng(seed)
    left_block = primary_block_length(left.size)
    right_block = primary_block_length(right.size)
    boot = np.empty(n_boot, dtype=np.float64)
    for index in range(n_boot):
        left_idx = stationary_block_indices(left.size, left_block, rng)
        right_idx = stationary_block_indices(right.size, right_block, rng)
        boot[index] = float(np.median(left[left_idx]) - np.median(right[right_idx]))
    point = float(np.median(left) - np.median(right))
    lower, upper = np.quantile(boot, [alpha, 1.0 - alpha])
    p_lower = float((np.sum(boot <= -margin) + 1) / (n_boot + 1))
    p_upper = float((np.sum(boot >= margin) + 1) / (n_boot + 1))
    passed = float(lower) > -margin and float(upper) < margin
    return {
        "estimand": point,
        "margin": margin,
        "ci": [float(lower), float(upper)],
        "tost_p_value": max(p_lower, p_upper),
        "n_calendar": int(left.size),
        "n_volume": int(right.size),
        "n_boot": n_boot,
        "state": "pass" if passed else "fail",
    }


def load_holdout() -> dict[str, np.ndarray]:
    """Load the two checksum-verified normalized holdout partitions."""
    rows: list[dict[str, str]] = []
    for month in ("2026-06", "2026-07"):
        path = BTC_DIR / f"BTCUSDT-1m-{month}.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    timestamp = np.asarray([int(row["timestamp"]) for row in rows], dtype=np.int64)
    if timestamp.size != 87_840 or np.any(np.diff(timestamp) != 60_000):
        raise ValueError("BTC holdout must contain 87,840 contiguous one-minute rows")
    return {
        "timestamp": timestamp,
        "open": np.asarray([float(row["open"]) for row in rows]),
        "high": np.asarray([float(row["high"]) for row in rows]),
        "low": np.asarray([float(row["low"]) for row in rows]),
        "close": np.asarray([float(row["close"]) for row in rows]),
        "volume": np.asarray([float(row["volume"]) for row in rows]),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _vr_samples(close: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from scripts.wp1.vr_significance import compute_rolling_vr_and_z_strided  # noqa: PLC0415

    vr, z = compute_rolling_vr_and_z_strided(close, W=W, q=Q, stride=W)
    return vr[np.isfinite(vr)], z[np.isfinite(z)]


def _trigger_information(
    close: np.ndarray,
    *,
    q: int,
    digest: str,
    alpha: float,
    n_boot: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    from scripts.wp1.harmonized_benchmark import (  # noqa: PLC0415
        forward_returns_from_close,
        vr_holm_trigger_labels,
    )
    from scripts.wp1.vr_detector_stationary_block import (  # noqa: PLC0415
        stationary_trigger_information,
    )

    trigger = vr_holm_trigger_labels(close, W=W, target_q=q, stride=W, alpha=alpha)
    indices = np.asarray(trigger["indices"], dtype=np.int64)
    labels = np.asarray(trigger["labels"], dtype=np.int8)
    forward = forward_returns_from_close(close, q)[indices]
    finite = np.isfinite(forward)
    labels, forward = labels[finite], forward[finite]
    target = (forward > 0.0).astype(np.int8)
    result = stationary_trigger_information(
        labels,
        target,
        spec_hash=digest,
        alpha=alpha,
        n_boot=n_boot,
        gate_label=f"btc-vr-q{q}-information",
    )
    result.update(
        {
            "q": q,
            "n_decision_epochs": int(labels.size),
            "n_triggers": int(np.sum(labels)),
            "trigger_rate": float(np.mean(labels)) if labels.size else 0.0,
        }
    )
    return result, labels, forward


def build_evidence(
    spec: CertificateSpec,
    *,
    n_info_boot: int,
    n_transport_boot: int,
) -> dict[str, Any]:
    """Generate every required BTC successor gate from frozen inputs."""
    digest = spec_hash(spec)
    holdout = load_holdout()
    manifest = json.loads(BTC_MANIFEST.read_text(encoding="utf-8"))
    checksum_ok = all(row["vendor_checksum_verified"] for row in manifest["partitions"])

    from scripts.wp1.gauge_bars import build_volume_bars  # noqa: PLC0415
    from scripts.wp1.signal_injection import resolve_phi  # noqa: PLC0415

    phi_by_amplitude = {amplitude: resolve_phi(amplitude, Q) for amplitude in AMPLITUDES}
    calibration = calibrate_successor(
        amplitudes=list(AMPLITUDES),
        phi_by_amplitude=phi_by_amplitude,
        n_null=int(spec.thresholds["n_null"]),
        n_injection=int(spec.thresholds["n_injection_per_cell"]),
        n_windows=int(spec.thresholds["n_windows_per_draw"]),
        alpha=spec.alpha,
        seed=43,
    )

    calendar_vr, calendar_z = _vr_samples(holdout["close"])
    volume_bars = build_volume_bars(holdout, spec.thresholds["volume_threshold_btc"])
    volume_vr, _volume_z = _vr_samples(volume_bars["close"])
    transport = transport_tost(
        calendar_vr,
        volume_vr,
        margin=spec.margins["transport"],
        alpha=spec.alpha,
        n_boot=n_transport_boot,
        seed=43,
    )

    observed_median_z = float(np.median(calendar_z))
    observed_recentered = (observed_median_z - RECENTER_CENTER) / RECENTER_SCALE
    critical = float(norm.ppf(1.0 - spec.alpha))
    target_state = "pass" if observed_recentered > critical else "fail"

    info_q2, labels_q2, forward_q2 = _trigger_information(
        holdout["close"],
        q=2,
        digest=digest,
        alpha=spec.alpha,
        n_boot=n_info_boot,
    )
    info_q5, _labels_q5, _forward_q5 = _trigger_information(
        holdout["close"],
        q=5,
        digest=digest,
        alpha=spec.alpha,
        n_boot=n_info_boot,
    )

    trigger = labels_q2 == 1
    if np.any(trigger) and np.any(~trigger):
        value_series = np.where(trigger, forward_q2, 0.0)
        value_point = float(np.mean(value_series))
        value_ci = stationary_mean_interval(
            value_series,
            n_boot=n_transport_boot,
            alpha=spec.alpha,
            seed=int(digest[:16], 16),
        )
        value_state = "pass" if value_ci[0] > 0.0 else "fail"
        value_reason = "cost-free scientific trigger utility; no economic cost policy in scope"
    else:
        value_point, value_ci, value_state = 0.0, (0.0, 0.0), "fail"
        value_reason = "trigger labels are degenerate; no decision value can be established"

    artifact_ref = f"evidence/gates/{spec.certificate_id}/evidence.json"
    gate_results = [
        GateResult(
            gate="instrument",
            state="pass" if checksum_ok else "fail",
            estimand=float(sum(row["row_count"] for row in manifest["partitions"])),
            threshold=87_840.0,
            artifacts=[artifact_ref],
            reason="vendor checksums verified and normalized minutes contiguous",
        ),
        GateResult(
            gate="target",
            state=target_state,
            estimand=observed_recentered,
            threshold=critical,
            artifacts=[artifact_ref],
            reason="one-sided observed median q=2 M2 statistic under frozen recentering",
        ),
        GateResult(
            gate="size",
            state=calibration["size"]["state"],
            estimand=calibration["size"]["rate"],
            threshold=spec.alpha,
            uncertainty=calibration["size"]["bounds"],
            artifacts=[artifact_ref],
            reason="exact one-sided Clopper-Pearson upper bound",
        ),
        GateResult(
            gate="transport",
            state=transport["state"],
            estimand=transport["estimand"],
            threshold=spec.margins["transport"],
            uncertainty={"lower": transport["ci"][0], "upper": transport["ci"][1]},
            artifacts=[artifact_ref],
            reason="single frozen calendar-volume dependence-aware TOST",
        ),
        GateResult(
            gate="power",
            state=calibration["power_state"],
            estimand=calibration["cmde_90"],
            threshold=spec.margins["power"],
            artifacts=[artifact_ref],
            reason="smallest declared amplitude with exact lower power bound >= 0.90",
        ),
        GateResult(
            gate="information",
            state=info_q2["gate_state"],
            estimand=info_q2["e_value"],
            threshold=info_q2["e_value_threshold"],
            uncertainty={"mi_lower": info_q2["ci"][0], "mi_upper": info_q2["ci"][1]},
            artifacts=[artifact_ref],
            reason=info_q2["reason"],
        ),
        GateResult(
            gate="value",
            state=value_state,
            estimand=value_point,
            threshold=0.0,
            uncertainty={"lower": value_ci[0], "upper": value_ci[1]},
            artifacts=[artifact_ref],
            reason=value_reason,
        ),
    ]
    normalized_paths = [BTC_DIR / f"BTCUSDT-1m-{month}.csv" for month in ("2026-06", "2026-07")]
    return {
        "schema_version": 1,
        "artifact": "btcusdt_vr_q2_protocol_v5_evidence",
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "preregistration_commit": spec.preregistration_commit,
        "spec_hash": digest,
        "data_hashes": {
            "vendor_manifest": _sha256(BTC_MANIFEST),
            **{path.name: _sha256(path) for path in normalized_paths},
        },
        "calibration": calibration,
        "observed": {
            "median_z_m2": observed_median_z,
            "recentered_z": observed_recentered,
            "critical": critical,
        },
        "transport": transport,
        "information": {"q2": info_q2, "q5": info_q5},
        "value": {"point": value_point, "ci": list(value_ci), "reason": value_reason},
        "gate_results": [gate.model_dump(mode="json") for gate in gate_results],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-info-boot", type=int, default=49_999)
    parser.add_argument("--n-transport-boot", type=int, default=9_999)
    parser.add_argument(
        "--information-dir",
        type=Path,
        help="directory for separate q=2 and q=5 stationary-block artifacts",
    )
    args = parser.parse_args(argv)
    spec = CertificateSpec.model_validate_json(args.spec.read_text(encoding="utf-8"))
    assert_preregistration(spec, repo=REPO_ROOT, spec_path=args.spec)
    artifact = build_evidence(
        spec,
        n_info_boot=args.n_info_boot,
        n_transport_boot=args.n_transport_boot,
    )
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    info_dir = args.information_dir or args.out.parent
    info_dir.mkdir(parents=True, exist_ok=True)
    for label in ("q2", "q5"):
        info_path = info_dir / f"information-{label}.json"
        if info_path.exists():
            raise FileExistsError(f"refusing to overwrite {info_path}")
        info_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "artifact": "btc_vr_stationary_block_information",
                    "created_at": artifact["created_at"],
                    "preregistration_commit": artifact["preregistration_commit"],
                    "spec_hash": artifact["spec_hash"],
                    "data_hashes": artifact["data_hashes"],
                    "result": artifact["information"][label],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
