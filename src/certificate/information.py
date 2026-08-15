"""Dependence-preserving stationary-block mutual information.

Certificate-grade information evidence. New outputs never include
``permutation_p``. Production callers use B=49,999; tests pass a reduced B.
"""
from __future__ import annotations

import math
import hashlib
from typing import Any

import numpy as np

DEFAULT_N_BOOT = 49_999


def seed_from_spec_hash(spec_hash: str, label: str) -> int:
    """Derive a deterministic, domain-separated 64-bit seed."""
    if len(spec_hash) != 64:
        raise ValueError("spec_hash must be a 64-character SHA-256 digest")
    try:
        spec_bytes = bytes.fromhex(spec_hash)
    except ValueError as exc:
        raise ValueError("spec_hash must be hexadecimal") from exc
    digest = hashlib.sha256(spec_bytes + b"\0" + label.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def primary_block_length(n: int) -> int:
    if n <= 0:
        raise ValueError("n must be positive")
    return int(math.ceil(n ** (1.0 / 3.0)))


def stationary_block_indices(
    n_obs: int,
    mean_block: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if n_obs <= 0:
        raise ValueError("n_obs must be positive")
    if mean_block <= 0:
        raise ValueError("mean_block must be positive")
    p_new_block = min(1.0, 1.0 / float(mean_block))
    idx = np.empty(n_obs, dtype=np.int64)
    idx[0] = int(rng.integers(0, n_obs))
    for i in range(1, n_obs):
        if rng.random() < p_new_block:
            idx[i] = int(rng.integers(0, n_obs))
        else:
            idx[i] = (idx[i - 1] + 1) % n_obs
    return idx


def plugin_mutual_information(x: np.ndarray, y: np.ndarray) -> float:
    x_arr = np.asarray(x, dtype=np.int64)
    y_arr = np.asarray(y, dtype=np.int64)
    if x_arr.shape != y_arr.shape:
        raise ValueError("x and y must have the same shape")
    n = int(x_arr.size)
    if n == 0:
        return 0.0
    x_codes, x_inv = np.unique(x_arr, return_inverse=True)
    y_codes, y_inv = np.unique(y_arr, return_inverse=True)
    joint = np.zeros((x_codes.size, y_codes.size), dtype=np.float64)
    np.add.at(joint, (x_inv, y_inv), 1.0)
    joint /= n
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.divide(joint, px * py, out=np.ones_like(joint), where=joint > 0)
        terms = np.where(joint > 0, joint * np.log(ratio), 0.0)
    return float(np.sum(terms))


def miller_madow_mutual_information(x: np.ndarray, y: np.ndarray) -> float:
    x_arr = np.asarray(x, dtype=np.int64)
    y_arr = np.asarray(y, dtype=np.int64)
    n = int(x_arr.size)
    if n == 0:
        return 0.0
    plugin = plugin_mutual_information(x_arr, y_arr)
    x_codes, x_inv = np.unique(x_arr, return_inverse=True)
    y_codes, y_inv = np.unique(y_arr, return_inverse=True)
    joint = np.zeros((x_codes.size, y_codes.size), dtype=np.int64)
    np.add.at(joint, (x_inv, y_inv), 1)
    b_xy = int(np.count_nonzero(joint))
    b_x = int(np.count_nonzero(joint.sum(axis=1)))
    b_y = int(np.count_nonzero(joint.sum(axis=0)))
    return float(plugin + (b_xy - b_x - b_y + 1) / (2.0 * n))


def _pvalue(observed: float, null_values: np.ndarray) -> float:
    b = int(null_values.size)
    if b == 0:
        return 1.0
    return float((np.sum(null_values >= observed) + 1.0) / (b + 1.0))


def _one_reference(
    x: np.ndarray,
    y: np.ndarray,
    *,
    mean_block: int,
    n_boot: int,
    alpha: float,
    paired_rng: np.random.Generator,
    null_rng: np.random.Generator,
) -> dict[str, Any]:
    n = int(x.size)
    plugin = plugin_mutual_information(x, y)
    miller = miller_madow_mutual_information(x, y)
    paired = np.empty(n_boot, dtype=np.float64)
    null = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        paired_idx = stationary_block_indices(n, mean_block, paired_rng)
        paired[i] = miller_madow_mutual_information(x[paired_idx], y[paired_idx])
        x_idx = stationary_block_indices(n, mean_block, null_rng)
        y_idx = stationary_block_indices(n, mean_block, null_rng)
        null[i] = miller_madow_mutual_information(x[x_idx], y[y_idx])
    ci = np.quantile(paired, [alpha / 2.0, 1.0 - alpha / 2.0]).tolist()
    p_value = _pvalue(miller, null)
    return {
        "plugin_mi": plugin,
        "miller_madow_mi": miller,
        "ci": [float(ci[0]), float(ci[1])],
        "p_value": p_value,
        "mean_block": int(mean_block),
        "n_boot": int(n_boot),
        "n": n,
        "confidence_level": float(1.0 - alpha),
    }


def stationary_block_information(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_boot: int = DEFAULT_N_BOOT,
    spec_hash: str,
    gate_label: str = "information",
    alpha: float = 0.05,
    force_unstable: bool = False,
) -> dict[str, Any]:
    x_arr = np.asarray(x, dtype=np.int64)
    y_arr = np.asarray(y, dtype=np.int64)
    if x_arr.shape != y_arr.shape:
        raise ValueError("x and y must have the same shape")
    n = int(x_arr.size)
    primary = max(1, primary_block_length(n))
    half = max(1, primary // 2)
    double = max(1, primary * 2)
    def reference(mean_block: int, label: str) -> dict[str, Any]:
        return _one_reference(
            x_arr,
            y_arr,
            mean_block=mean_block,
            n_boot=n_boot,
            alpha=alpha,
            paired_rng=np.random.default_rng(
                seed_from_spec_hash(spec_hash, f"{gate_label}:{label}:paired")
            ),
            null_rng=np.random.default_rng(
                seed_from_spec_hash(spec_hash, f"{gate_label}:{label}:null")
            ),
        )

    primary_result = reference(primary, "primary")
    half_result = reference(half, "half")
    double_result = reference(double, "double")
    primary_pass = primary_result["p_value"] <= alpha
    half_pass = half_result["p_value"] <= alpha
    double_pass = double_result["p_value"] <= alpha
    reversed_conclusion = (half_pass != primary_pass) or (double_pass != primary_pass)
    if force_unstable:
        reversed_conclusion = True
    if reversed_conclusion:
        gate_state = "invalid"
        reason = "unstable_reference"
    elif primary_pass:
        gate_state = "pass"
        reason = "stationary-block independence null rejected"
    else:
        gate_state = "fail"
        reason = "stationary-block independence null not rejected"
    result = {
        **primary_result,
        "gate_state": gate_state,
        "reason": reason,
        "alpha": float(alpha),
        "confidence_level": float(1.0 - alpha),
        "seed_derivation": "sha256(spec_hash || NUL || domain_label)",
        "sensitivity": {
            "half": half_result,
            "double": double_result,
        },
    }
    assert "permutation_p" not in result
    return result
