"""Empirical VR null reference for the Phase 15 manuscript revision.

The script computes finite-sample reference distributions for the median
Lo--MacKinlay M2 statistic at the cascade's primary cells.  It supports block,
stationary, wild, Student-t, and rank/sign nulls, then recomputes the Holm
family over the requested cell set for each null specification.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import norm

os.environ.setdefault("OMP_NUM_THREADS", "1")

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backtest.utils import PROJECT_ROOT  # noqa: E402

import scripts._bootstrap  # noqa: F401, E402
from scripts.wp1 import vr_significance  # noqa: E402

PRECOMPUTED_NPZ = PROJECT_ROOT / "data" / "injection_runs" / "precomputed.npz"
INJECTION_DIR = PROJECT_ROOT / "data" / "injection_runs"
OUT_DIR = PROJECT_ROOT / "backtest_results" / "empirical_vr_null"
DEFAULT_CELLS = ((120, 5), (120, 2))
PRIMARY_GATE_CELLS = ((60, 5), (120, 5), (240, 15))
PRIMARY_Q_GRID = (2, 5, 15, 60)
ALL_PRIMARY_CELLS = tuple(
    sorted(set(PRIMARY_GATE_CELLS).union((120, q) for q in PRIMARY_Q_GRID))
)
DEFAULT_BLOCK = 120
DEFAULT_N_BOOT = 100
DEFAULT_SEED = 43
DEFAULT_NULL_METHODS = ("circular-block",)
BLOCK_NULL_METHODS = frozenset({"circular-block", "moving-block", "stationary-block"})
NULL_ALIASES = {
    "circular": "circular-block",
    "circular-block": "circular-block",
    "moving": "moving-block",
    "moving-block": "moving-block",
    "stationary": "stationary-block",
    "stationary-block": "stationary-block",
    "wild": "wild",
    "student": "student-t",
    "student-t": "student-t",
    "rank": "rank-sign",
    "sign": "rank-sign",
    "rank-sign": "rank-sign",
}

__all__ = [
    "DEFAULT_CELLS",
    "ALL_PRIMARY_CELLS",
    "circular_block_bootstrap_indices",
    "empirical_pvalue",
    "holm_adjust_pvalues",
    "median_z_for_returns",
    "moving_block_bootstrap_indices",
    "run_empirical_vr_null",
    "stationary_block_bootstrap_indices",
]


def circular_block_bootstrap_indices(
    n_obs: int,
    block: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return n_obs circular block-bootstrap indices."""
    if n_obs <= 0:
        raise ValueError("n_obs must be positive")
    if block <= 0:
        raise ValueError("block must be positive")
    n_blocks = int(np.ceil(n_obs / block))
    starts = rng.integers(0, n_obs, size=n_blocks)
    offsets = np.arange(block)
    return ((starts[:, None] + offsets[None, :]) % n_obs).ravel()[:n_obs]


def moving_block_bootstrap_indices(
    n_obs: int,
    block: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return n_obs moving-block bootstrap indices without circular wrapping."""
    if n_obs <= 0:
        raise ValueError("n_obs must be positive")
    if block <= 0:
        raise ValueError("block must be positive")
    block = min(int(block), int(n_obs))
    n_blocks = int(np.ceil(n_obs / block))
    max_start = max(0, n_obs - block)
    starts = rng.integers(0, max_start + 1, size=n_blocks)
    offsets = np.arange(block)
    return (starts[:, None] + offsets[None, :]).ravel()[:n_obs]


def stationary_block_bootstrap_indices(
    n_obs: int,
    block: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return n_obs stationary-bootstrap indices with mean block length block."""
    if n_obs <= 0:
        raise ValueError("n_obs must be positive")
    if block <= 0:
        raise ValueError("block must be positive")
    p_new_block = min(1.0, 1.0 / float(block))
    idx = np.empty(n_obs, dtype=np.int64)
    idx[0] = int(rng.integers(0, n_obs))
    for i in range(1, n_obs):
        if rng.random() < p_new_block:
            idx[i] = int(rng.integers(0, n_obs))
        else:
            idx[i] = (idx[i - 1] + 1) % n_obs
    return idx


def _close_from_returns(returns: np.ndarray) -> np.ndarray:
    close = np.exp(np.cumsum(np.asarray(returns, dtype=np.float64)))
    close[~np.isfinite(close)] = np.nan
    return close


def median_z_for_returns(returns: np.ndarray, *, W: int, q: int) -> dict:
    """Compute non-overlapping median z_m2 for a return path."""
    close = _close_from_returns(returns)
    _vr, z = vr_significance.compute_rolling_vr_and_z_strided(
        close,
        W=int(W),
        q=int(q),
        stride=int(W),
    )
    finite_z = z[np.isfinite(z)]
    if len(finite_z) == 0:
        return {
            "median_z_m2": float("nan"),
            "n_windows": 0,
            "asymptotic_two_sided_p": float("nan"),
        }
    median_z = float(np.median(finite_z))
    return {
        "median_z_m2": median_z,
        "n_windows": int(len(finite_z)),
        "asymptotic_two_sided_p": 2.0 * float(norm.sf(abs(median_z))),
    }


def empirical_pvalue(
    null_stats: np.ndarray,
    observed: float,
    *,
    alternative: str = "two-sided",
) -> float:
    """Bootstrap p-value with plus-one correction."""
    stats = np.asarray(null_stats, dtype=np.float64)
    stats = stats[np.isfinite(stats)]
    if len(stats) == 0 or not np.isfinite(observed):
        return float("nan")
    if alternative == "two-sided":
        count = int(np.sum(np.abs(stats) >= abs(observed)))
    elif alternative == "greater":
        count = int(np.sum(stats >= observed))
    elif alternative == "less":
        count = int(np.sum(stats <= observed))
    else:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")
    return float((count + 1) / (len(stats) + 1))


def holm_adjust_pvalues(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values in original order."""
    if not pvalues:
        return []
    clean = [float(p) if np.isfinite(p) else 1.0 for p in pvalues]
    m = len(clean)
    order = sorted(range(m), key=lambda i: clean[i])
    adjusted = [1.0] * m
    prev = 0.0
    for rank, idx in enumerate(order):
        adj = min(1.0, clean[idx] * (m - rank))
        adj = max(adj, prev)
        adjusted[idx] = adj
        prev = adj
    return adjusted


def _resolve_code_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "log", "--format=%H", "-1"],
            capture_output=True,
            text=True,
            cwd=str(_REPO_ROOT),
            check=False,
        )
    except Exception:
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def _normalize_null_methods(null_methods: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    normalized = []
    for method in null_methods:
        key = str(method).strip().lower()
        if not key:
            continue
        if key not in NULL_ALIASES:
            valid = ", ".join(sorted(NULL_ALIASES))
            raise ValueError(f"unknown null method {method!r}; valid: {valid}")
        canonical = NULL_ALIASES[key]
        if canonical not in normalized:
            normalized.append(canonical)
    if not normalized:
        raise ValueError("at least one null method is required")
    return tuple(normalized)


def _cell_records(cells: tuple[tuple[int, int], ...]) -> list[dict]:
    return [{"W": int(W), "q": int(q)} for W, q in cells]


def _estimate_student_t_df(samples: np.ndarray) -> float:
    finite = np.asarray(samples, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) < 10:
        return 30.0
    centered = finite - float(np.mean(finite))
    m2 = float(np.mean(centered * centered))
    if m2 <= 0.0 or not np.isfinite(m2):
        return 30.0
    m4 = float(np.mean(centered ** 4))
    excess = m4 / (m2 * m2) - 3.0
    if not np.isfinite(excess) or excess <= 0.0:
        return 30.0
    return float(np.clip(6.0 / excess + 4.0, 4.5, 80.0))


def _student_t_null_returns(
    centered_returns: np.ndarray,
    rng: np.random.Generator,
    sigma_t: np.ndarray | None,
) -> np.ndarray:
    if sigma_t is not None:
        sigma = np.asarray(sigma_t, dtype=np.float64)
        safe_sigma = np.where(np.isfinite(sigma) & (sigma > 0.0), sigma, np.nan)
        median_sigma = float(np.nanmedian(safe_sigma))
        safe_sigma = np.where(np.isfinite(safe_sigma), safe_sigma, median_sigma)
        residuals = centered_returns / safe_sigma
        df = _estimate_student_t_df(residuals)
        innovations = rng.standard_t(df, size=len(centered_returns))
        innovations = innovations / np.sqrt(df / (df - 2.0))
        return innovations * safe_sigma

    df = _estimate_student_t_df(centered_returns)
    scale = float(np.std(centered_returns, ddof=1))
    innovations = rng.standard_t(df, size=len(centered_returns))
    innovations = innovations / np.sqrt(df / (df - 2.0))
    return innovations * scale


def _rank_sign_null_returns(
    centered_returns: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    magnitudes = np.abs(centered_returns)
    permuted = magnitudes[rng.permutation(len(magnitudes))]
    signs = rng.choice(np.array([-1.0, 1.0]), size=len(magnitudes))
    return permuted * signs


def _generate_null_returns(
    method: str,
    centered_returns: np.ndarray,
    rng: np.random.Generator,
    block: int | None,
    sigma_t: np.ndarray | None,
) -> np.ndarray:
    if method == "circular-block":
        if block is None:
            raise ValueError("block length required for circular-block")
        return centered_returns[circular_block_bootstrap_indices(len(centered_returns), block, rng)]
    if method == "moving-block":
        if block is None:
            raise ValueError("block length required for moving-block")
        return centered_returns[moving_block_bootstrap_indices(len(centered_returns), block, rng)]
    if method == "stationary-block":
        if block is None:
            raise ValueError("block length required for stationary-block")
        return centered_returns[stationary_block_bootstrap_indices(len(centered_returns), block, rng)]
    if method == "wild":
        signs = rng.choice(np.array([-1.0, 1.0]), size=len(centered_returns))
        return centered_returns * signs
    if method == "student-t":
        return _student_t_null_returns(centered_returns, rng, sigma_t)
    if method == "rank-sign":
        return _rank_sign_null_returns(centered_returns, rng)
    raise ValueError(f"unknown null method: {method}")


def _summarize_null(null_stats: np.ndarray) -> dict:
    stats = np.asarray(null_stats, dtype=np.float64)
    stats = stats[np.isfinite(stats)]
    if len(stats) == 0:
        return {"n_boot_finite": 0}
    return {
        "n_boot_finite": int(len(stats)),
        "median_z_p05": float(np.quantile(stats, 0.05)),
        "median_z_p50": float(np.quantile(stats, 0.50)),
        "median_z_p95": float(np.quantile(stats, 0.95)),
        "empirical_size_alpha_0.05_asymptotic_two_sided": float(
            np.mean(2.0 * norm.sf(np.abs(stats)) < 0.05)
        ),
    }


def _extract_cell_median_z(path: Path, q: int) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    z_vals = []
    for draw in data.get("per_draw", []):
        for item in draw.get("holm_ordered", []):
            if int(item.get("q", -1)) == int(q):
                z_vals.append(float(item["median_z_m2"]))
                break
    finite = np.asarray(z_vals, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        median_z = float("nan")
    else:
        median_z = float(np.median(finite))
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "delta_target": float(data["delta_target"]),
        "W": int(data["W"]),
        "q": int(data["q"]),
        "P_det": float(data["P_det"]),
        "n_fires": int(data["n_fires"]),
        "N_mc": int(data["N_mc"]),
        "median_draw_median_z_m2": median_z,
        "n_draws_with_z": int(len(finite)),
    }


def _addendum_summaries(null_by_cell: dict[tuple[int, int], np.ndarray]) -> list[dict]:
    summaries: list[dict] = []
    for q in (2, 5):
        key = (120, q)
        if key not in null_by_cell:
            continue
        path = INJECTION_DIR / f"inj_d0.15_q{q}_W120.json"
        if not path.exists():
            continue
        item = _extract_cell_median_z(path, q)
        null_stats = null_by_cell[key]
        observed = item["median_draw_median_z_m2"]
        item["empirical_two_sided_p_vs_null"] = empirical_pvalue(
            null_stats,
            observed,
            alternative="two-sided",
        )
        item["empirical_positive_gate_p_vs_null"] = empirical_pvalue(
            null_stats,
            observed,
            alternative="greater",
        )
        summaries.append(item)
    return summaries


def _block_sensitivity_rows(results: list[dict]) -> list[dict]:
    rows_by_key: dict[tuple[str, int, int], dict] = {}
    for result in results:
        method = result["null_method"]
        block_len = result.get("block_length")
        if block_len is None:
            continue
        for cell in result["cells"]:
            key = (method, int(cell["W"]), int(cell["q"]))
            row = rows_by_key.setdefault(
                key,
                {
                    "null_method": method,
                    "W": int(cell["W"]),
                    "q": int(cell["q"]),
                    "two_sided_p_by_block": {},
                    "holm_p_by_block": {},
                },
            )
            row["two_sided_p_by_block"][str(block_len)] = cell[
                "empirical_two_sided_p_vs_null"
            ]
            row["holm_p_by_block"][str(block_len)] = cell[
                "holm_adjusted_two_sided_p"
            ]
    return list(rows_by_key.values())


def run_empirical_vr_null(
    returns: np.ndarray,
    *,
    cells: tuple[tuple[int, int], ...] = DEFAULT_CELLS,
    n_boot: int = DEFAULT_N_BOOT,
    block: int | None = DEFAULT_BLOCK,
    blocks: tuple[int, ...] | None = None,
    seed: int = DEFAULT_SEED,
    null_methods: tuple[str, ...] = DEFAULT_NULL_METHODS,
    sigma_t: np.ndarray | None = None,
) -> dict:
    """Compute observed and empirical-null median-z references."""
    if n_boot <= 0:
        raise ValueError("n_boot must be positive")
    returns_arr = np.asarray(returns, dtype=np.float64)
    finite_mask = np.isfinite(returns_arr)
    returns_arr = returns_arr[finite_mask]
    if len(returns_arr) <= max(W for W, _q in cells):
        raise ValueError("returns must be longer than the largest W")
    for W, q in cells:
        if int(q) < 2 or int(W) < int(q) + 2:
            raise ValueError(f"invalid VR cell W={W}, q={q}; require W >= q + 2")

    if sigma_t is not None:
        sigma_arr = np.asarray(sigma_t, dtype=np.float64)
        if len(sigma_arr) != len(finite_mask):
            raise ValueError("sigma_t must have the same length as returns")
        sigma_arr = sigma_arr[finite_mask]
    else:
        sigma_arr = None

    if blocks is None:
        if block is None:
            blocks = (DEFAULT_BLOCK,)
        else:
            blocks = (int(block),)
    blocks = tuple(int(b) for b in blocks)
    if any(b <= 0 for b in blocks):
        raise ValueError("all block lengths must be positive")
    null_methods = _normalize_null_methods(null_methods)

    centered = returns_arr - float(np.mean(returns_arr))

    observed_by_cell: dict[tuple[int, int], dict] = {}
    for W, q in cells:
        observed_by_cell[(int(W), int(q))] = median_z_for_returns(
            returns_arr,
            W=int(W),
            q=int(q),
        )

    results: list[dict] = []
    first_null_by_cell: dict[tuple[int, int], np.ndarray] | None = None
    for method_idx, method in enumerate(null_methods):
        method_blocks: tuple[int | None, ...]
        if method in BLOCK_NULL_METHODS:
            method_blocks = tuple(blocks)
        else:
            method_blocks = (None,)

        for block_idx, block_len in enumerate(method_blocks):
            rng_seed = int(seed + method_idx * 1_000_003 + block_idx * 10_007 + (block_len or 0))
            rng = np.random.default_rng(rng_seed)
            null_by_cell: dict[tuple[int, int], np.ndarray] = {
                (int(W), int(q)): np.empty(n_boot, dtype=np.float64)
                for W, q in cells
            }
            for b in range(n_boot):
                boot_returns = _generate_null_returns(
                    method,
                    centered,
                    rng,
                    block_len,
                    sigma_arr,
                )
                for W, q in cells:
                    stat = median_z_for_returns(boot_returns, W=int(W), q=int(q))
                    null_by_cell[(int(W), int(q))][b] = stat["median_z_m2"]

            if first_null_by_cell is None:
                first_null_by_cell = null_by_cell

            cell_rows = []
            raw_two_sided = []
            for W, q in cells:
                key = (int(W), int(q))
                observed = observed_by_cell[key]
                null_stats = null_by_cell[key]
                p_two_sided = empirical_pvalue(
                    null_stats,
                    observed["median_z_m2"],
                    alternative="two-sided",
                )
                raw_two_sided.append(p_two_sided)
                cell_rows.append(
                    {
                        "W": key[0],
                        "q": key[1],
                        "observed_uninjected": observed,
                        "empirical_two_sided_p_vs_null": p_two_sided,
                        "empirical_positive_gate_p_vs_null": empirical_pvalue(
                            null_stats,
                            observed["median_z_m2"],
                            alternative="greater",
                        ),
                        "null_summary": _summarize_null(null_stats),
                    }
                )

            holm_adjusted = holm_adjust_pvalues(raw_two_sided)
            for row, adj in zip(cell_rows, holm_adjusted, strict=True):
                row["holm_adjusted_two_sided_p"] = adj
                row["holm_reject_two_sided_alpha_0.05"] = bool(adj < 0.05)

            results.append(
                {
                    "null_method": method,
                    "block_length": block_len,
                    "n_boot": int(n_boot),
                    "rng_seed": rng_seed,
                    "cells": cell_rows,
                    "family_summary": {
                        "family_size": len(cell_rows),
                        "min_raw_two_sided_p": float(np.nanmin(raw_two_sided)),
                        "min_holm_adjusted_two_sided_p": float(np.nanmin(holm_adjusted)),
                        "n_holm_rejections_alpha_0.05": int(
                            sum(row["holm_reject_two_sided_alpha_0.05"] for row in cell_rows)
                        ),
                    },
                }
            )

    return {
        "description": (
            "Empirical VR null program: recomputes non-overlapping median z_m2 "
            "under block, wild, Student-t, and rank/sign finite-sample nulls."
        ),
        "manifest": {
            "code_commit": _resolve_code_commit(),
            "seed": int(seed),
            "n_boot": int(n_boot),
            "cells": _cell_records(cells),
            "null_methods": list(null_methods),
            "block_lengths": list(blocks),
            "n_returns": int(len(returns_arr)),
            "student_t_sigma_source": "sigma_t" if sigma_arr is not None else "sample_std",
        },
        "n_boot": int(n_boot),
        "seed": int(seed),
        "n_returns": int(len(returns_arr)),
        "year_2026_loaded": False,
        "results": results,
        "block_sensitivity": _block_sensitivity_rows(results),
        "addendum_cells": _addendum_summaries(first_null_by_cell or {}),
    }


def _parse_cells(raw: str) -> tuple[tuple[int, int], ...]:
    if raw.strip().lower() == "all-primary":
        return ALL_PRIMARY_CELLS
    cells = []
    for item in raw.split(","):
        if not item.strip():
            continue
        W_raw, q_raw = item.split(":")
        cells.append((int(W_raw), int(q_raw)))
    return tuple(cells)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, default=PRECOMPUTED_NPZ)
    parser.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    parser.add_argument(
        "--block",
        type=int,
        default=DEFAULT_BLOCK,
        help="Legacy shorthand for a single block length.",
    )
    parser.add_argument(
        "--blocks",
        type=int,
        nargs="+",
        default=None,
        help="Block lengths for circular/moving/stationary nulls.",
    )
    parser.add_argument(
        "--nulls",
        nargs="+",
        default=list(DEFAULT_NULL_METHODS),
        help=(
            "Null methods: circular-block moving-block stationary-block "
            "wild student-t rank-sign"
        ),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--cells",
        type=_parse_cells,
        default=DEFAULT_CELLS,
        help="Comma-separated W:q cells, e.g. 120:5,120:2, or all-primary",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if not args.npz.exists():
        raise FileNotFoundError(args.npz)
    data = np.load(args.npz)
    returns = np.asarray(data["r_real"], dtype=np.float64)
    sigma_t = np.asarray(data["sigma_t"], dtype=np.float64) if "sigma_t" in data else None

    report = run_empirical_vr_null(
        returns,
        cells=args.cells,
        n_boot=args.n_boot,
        block=args.block,
        blocks=tuple(args.blocks) if args.blocks is not None else None,
        seed=args.seed,
        null_methods=tuple(args.nulls),
        sigma_t=sigma_t,
    )
    report["manifest"]["source_npz"] = str(args.npz.relative_to(PROJECT_ROOT))
    report["manifest"]["generated_utc"] = datetime.now(timezone.utc).isoformat()

    out_path = args.out
    if out_path is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_path = OUT_DIR / f"empirical_vr_null_{stamp}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Wrote {out_path}")
    for result in report["results"]:
        block_label = (
            f", block={result['block_length']}"
            if result["block_length"] is not None
            else ""
        )
        print(f"Null={result['null_method']}{block_label}:")
        for row in result["cells"]:
            obs = row["observed_uninjected"]
            null = row["null_summary"]
            print(
                f"  W={row['W']} q={row['q']}: observed median_z={obs['median_z_m2']:.4f}, "
                f"emp_p={row['empirical_two_sided_p_vs_null']:.4f}, "
                f"holm={row['holm_adjusted_two_sided_p']:.4f}, "
                f"null_p05/p50/p95={null['median_z_p05']:.4f}/"
                f"{null['median_z_p50']:.4f}/{null['median_z_p95']:.4f}"
            )
    for item in report["addendum_cells"]:
        print(
            f"Addendum q={item['q']}: P_det={item['P_det']:.3f}, "
            f"median_draw_z={item['median_draw_median_z_m2']:.4f}, "
            f"emp_pos_p={item['empirical_positive_gate_p_vs_null']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
