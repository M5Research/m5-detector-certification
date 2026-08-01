"""Recentered-reference repair: the §4.8 exploratory instrument repair.

This is the generator for
`backtest_results/reference_repair/recentered_reference_repair_20260710.json`,
which supports the manuscript's repair table, Figure 3, and claim C9 — the
protocol's first *admissible* certificate and the only positive result in the
paper. It previously had no generator in either repository, which made the one
positive result the only result a reader could not reproduce from versioned
code.

WHAT THE REPAIR DOES
--------------------
The size gate found the cascade's asymptotic N(0,1) reference mis-centered: the
null median of the decision statistic `median_z_m2` sits well away from zero
(around -0.005 at q=2 and -0.106 at q=5) rather than at it. The repair is the
standard empirical-null calibration that the size gate itself prescribes, and
it introduces no new simulation:

  1. From the calibration cell (delta = 0.0005, i.e. negligible injected
     signal) collect the per-draw `median_z_m2` values for each (W, q).
  2. Estimate the empirical null location and scale:
         mu_0(W,q)    = median of those values
         sigma_0(W,q) = standard deviation of those values
  3. Recentre the statistic:  Z_rec = (median_z_m2 - mu_0) / sigma_0
  4. Re-threshold the FROZEN per-draw values, one-sided positive:
         fires  <=>  Z_rec > z_{1-alpha},  alpha = 0.05
  5. Recompute P_det per cell by counting re-thresholded draws. No draw is
     re-simulated; the frozen Monte Carlo output is re-read under a corrected
     reference.
  6. cMDE = delta_90 = smallest executed delta with P_det >= 0.90, and
     delta_50 likewise at 0.50.
  7. Out-of-sample size: split the calibration cell's draws in half by draw
     index, estimate (mu_0, sigma_0) on the first half, and report the fire
     rate on the second. That rate is the OOS false-alarm column.

REQUIRED INPUT THAT IS NOT PUBLISHED
------------------------------------
Steps 1 and 4 need the per-draw `median_z_m2` values. The published injection
cells do not carry them: each per-draw record holds only `cascade_fired`,
`holm_ordered`, `mc_idx` and `seed`. `precomputed.npz` holds the injection
*inputs* (`r_real`, `sigma_t`, `timestamps`), not the per-draw statistic.

So this script cannot regenerate the artifact from what this repository
publishes. That is the same root cause as the Appendix B schema deviation
recorded in `prereg/DEVIATIONS.md` D6: the artifact schema drops per-draw
fields that the frozen protocol required. Emitting `median_z_m2` per draw
(alongside the required `status` and `n_failed`) when the grid is next
regenerated makes both D6 and this script's input gap go away together.

Regenerating the per-draw statistic from scratch means re-running the
injection cascade over the 104 cells, roughly 5h28m of CPU according to
`data/injection_runs/low_impact_run.log`, and it needs the BTC parquet data
that is not redistributed here.

WHAT THIS SCRIPT CAN DO TODAY
-----------------------------
`--verify` audits every relation in the frozen artifact that is checkable
without the missing input: that each cell's reported cMDE and delta_50 are
exactly the thresholds implied by its own recentered P_det table, that the
recentered and original P_det tables cover the same amplitudes, that the
recentring moved the null toward zero, and that reported sizes and
probabilities are in range. This does not prove the artifact was produced by
the procedure above, and the script says so rather than implying otherwise.

`--run` performs the full repair once given a per-draw statistic file, and is
the path to a genuine end-to-end reproduction.

    python scripts/wp1/reference_repair.py --verify
    python scripts/wp1/reference_repair.py --run --per-draw <path.json>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_REPO_SRC = _REPO_ROOT / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from backtest.utils import PROJECT_ROOT  # noqa: E402

import scripts._bootstrap  # noqa: F401, E402

REPAIR_DIR = PROJECT_ROOT / "backtest_results" / "reference_repair"
FROZEN_ARTIFACT = REPAIR_DIR / "recentered_reference_repair_20260710.json"
CALIBRATION_DELTA = 0.0005
ALPHA = 0.05
SEED = 43


# ---------------------------------------------------------------------------
# The repair itself
# ---------------------------------------------------------------------------


def estimate_empirical_null(values: np.ndarray) -> tuple[float, float]:
    """Location and scale of the empirical null, from the calibration cell."""
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        raise ValueError("calibration cell carries no per-draw values")
    return float(np.median(v)), float(np.std(v, ddof=1))


def recenter(values: np.ndarray, mu_0: float, sigma_0: float) -> np.ndarray:
    if not np.isfinite(sigma_0) or sigma_0 <= 0.0:
        raise ValueError(f"degenerate empirical-null scale: {sigma_0!r}")
    return (np.asarray(values, dtype=np.float64) - mu_0) / sigma_0


def fires(z_rec: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    """One-sided positive decision rule."""
    return z_rec > norm.ppf(1.0 - alpha)


def p_det(values: np.ndarray, mu_0: float, sigma_0: float, alpha: float = ALPHA) -> float:
    return float(np.mean(fires(recenter(values, mu_0, sigma_0), alpha)))


def threshold_at(pdet_by_delta: dict[float, float], level: float) -> float | None:
    """Smallest executed delta whose P_det reaches `level`. No interpolation."""
    for delta in sorted(pdet_by_delta):
        if pdet_by_delta[delta] >= level:
            return delta
    return None


def oos_size(calibration_values: np.ndarray, alpha: float = ALPHA) -> float:
    """Fire rate on the second half of the calibration draws.

    The null is estimated on the first half only, so the reported rate is a
    genuine out-of-sample false-alarm rate rather than the in-sample rate the
    recentring trivially controls.
    """
    v = np.asarray(calibration_values, dtype=np.float64)
    half = v.size // 2
    if half < 2:
        raise ValueError("calibration cell too small to split")
    mu_0, sigma_0 = estimate_empirical_null(v[:half])
    return float(np.mean(fires(recenter(v[half:], mu_0, sigma_0), alpha)))


def run_repair(per_draw: dict[tuple[int, int, float], np.ndarray]) -> dict:
    """Full repair over a {(W, q, delta): per-draw median_z_m2} mapping."""
    cells: dict[str, dict] = {}
    wq = sorted({(w, q) for (w, q, _) in per_draw})

    for w, q in wq:
        calib_key = (w, q, CALIBRATION_DELTA)
        if calib_key not in per_draw:
            raise KeyError(f"no calibration cell (delta={CALIBRATION_DELTA}) for W={w}, q={q}")
        calib = np.asarray(per_draw[calib_key], dtype=np.float64)
        mu_0, sigma_0 = estimate_empirical_null(calib)

        recentered = {
            delta: p_det(vals, mu_0, sigma_0)
            for (ww, qq, delta), vals in per_draw.items()
            if ww == w and qq == q
        }
        cells[f"W{w}_q{q}"] = {
            "null_center": mu_0,
            "null_sd": sigma_0,
            "oos_size": oos_size(calib),
            "cmde_recentered_d90": threshold_at(recentered, 0.90),
            "cmde_recentered_d50": threshold_at(recentered, 0.50),
            "recentered_pdet": {str(d): p for d, p in sorted(recentered.items())},
        }

    return {
        "artifact": "recentered_reference_repair",
        "provenance": {
            "type": "exploratory_post_freeze_repair",
            "method": "offline re-thresholding of frozen per-draw median_z_m2 "
                      "against empirical null",
            "seed": SEED,
            "alpha": ALPHA,
            "calibration_delta": CALIBRATION_DELTA,
            "split_rule": "first half of calibration draws estimates the null; "
                          "second half measures out-of-sample size",
            "source_injection_runs": "data/injection_runs/",
            "note": "NOT preregistered; confirmatory rerun required for "
                    "deployment certification.",
        },
        "cells": cells,
    }


# ---------------------------------------------------------------------------
# Verification of the frozen artifact
# ---------------------------------------------------------------------------


def verify_frozen(artifact: dict) -> list[str]:
    """Return a list of problems; empty means every checkable relation holds."""
    problems: list[str] = []

    for name, cell in artifact["cells"].items():
        recentered = {float(k): v for k, v in cell["recentered_pdet"].items()}

        for level, field in ((0.90, "cmde_recentered_d90"), (0.50, "cmde_recentered_d50")):
            expected = threshold_at(recentered, level)
            if cell[field] != expected:
                problems.append(
                    f"{name}: {field} is {cell[field]!r} but its own P_det table "
                    f"implies {expected!r}"
                )

        if "orig_pdet" in cell:
            orig = {float(k) for k in cell["orig_pdet"]}
            if orig != set(recentered):
                problems.append(
                    f"{name}: recentered and original P_det tables cover "
                    "different amplitude grids"
                )

        for delta, value in recentered.items():
            if not 0.0 <= value <= 1.0:
                problems.append(f"{name}: P_det {value!r} at delta={delta} is out of [0,1]")

        if not 0.0 <= cell["oos_size"] <= 1.0:
            problems.append(f"{name}: oos_size {cell['oos_size']!r} is out of [0,1]")

        if not cell["null_sd"] > 0.0:
            problems.append(f"{name}: null_sd {cell['null_sd']!r} is not positive")

        # The recentring exists because the asymptotic reference is mis-centred;
        # a null_center of essentially zero would mean there was nothing to fix.
        if abs(cell["null_center"]) < 1e-6:
            problems.append(
                f"{name}: null_center {cell['null_center']!r} is ~0, so the "
                "recentring would be a no-op"
            )

        d90, d90_orig = cell["cmde_recentered_d90"], cell.get("cmde_orig_d90")
        if d90 is not None and d90_orig is not None and d90 > d90_orig:
            problems.append(
                f"{name}: repair made power worse (cMDE {d90_orig} -> {d90})"
            )

    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true",
                        help="audit the frozen artifact's internal relations")
    parser.add_argument("--run", action="store_true",
                        help="perform the repair (requires --per-draw)")
    parser.add_argument("--per-draw", type=Path,
                        help="JSON mapping 'W,q,delta' -> [median_z_m2, ...]")
    parser.add_argument("--source", type=Path, default=FROZEN_ARTIFACT)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    if args.run:
        if not args.per_draw:
            raise SystemExit(
                "--run needs --per-draw. The published injection cells do not\n"
                "carry per-draw median_z_m2 (see this module's docstring and\n"
                "prereg/DEVIATIONS.md D6), so the repair cannot be regenerated\n"
                "from this repository alone."
            )
        raw = json.loads(args.per_draw.read_text(encoding="utf-8"))
        per_draw = {}
        for key, values in raw.items():
            w, q, delta = key.split(",")
            per_draw[(int(w), int(q), float(delta))] = np.asarray(values, dtype=np.float64)
        result = run_repair(per_draw)
        out = args.out or (REPAIR_DIR / "recentered_reference_repair_regenerated.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(f"wrote {out.relative_to(PROJECT_ROOT)}")
        return

    artifact = json.loads(args.source.read_text(encoding="utf-8"))
    problems = verify_frozen(artifact)
    print(f"cells checked: {len(artifact['cells'])}")
    for name, cell in artifact["cells"].items():
        print(
            f"  {name:10} mu0={cell['null_center']:+.6f} sd={cell['null_sd']:.6f} "
            f"oos={cell['oos_size']:.2f} cMDE {cell.get('cmde_orig_d90')} -> "
            f"{cell['cmde_recentered_d90']}"
        )
    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print(f"  - {p}")
        raise SystemExit(1)
    print("\nAll internally checkable relations hold.")
    print(
        "NOTE: this does not prove the artifact was produced by the documented\n"
        "procedure. That requires the per-draw median_z_m2 values, which this\n"
        "repository does not publish (see DEVIATIONS.md D6)."
    )


if __name__ == "__main__":
    main()
