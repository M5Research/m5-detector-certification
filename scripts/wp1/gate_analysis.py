"""Gate analysis: non-degeneracy + cross-regime Kruskal-Wallis + provisional PASS/FAIL.

Composes:
    py_engine.load_and_clean -> regime_detector.detect_regime ->
    predictability.compute_rolling_predictability per (W,q) ->
    check_degeneracy + non_overlapping_samples + KW/epsilon-sq ->
    provisional PASS/FAIL against the RATIFIED pre-registered thresholds ->
    backtest_results/gate/gate_report_<date>.json

Integrity controls enforced here:
    - D-07 holdout: data window end must be < 2026-01-01 (assert in main).
    - Pitfall 4 (look-ahead): only regime_detector.detect_regime is used for R_t.
      The look-ahead-biased script documented in RESEARCH.md Pitfall 4 is NEVER imported.
    - Pitfall 5 (warmup): regime >= 0 filter applied BEFORE KW grouping.
    - Pitfall 3 (overlap): KW uses non-overlapping samples at stride W.
    - D-02 headline: epsilon-squared effect size, NOT the p-value.
    - Determinism: np.random.default_rng(seed); never np.random.seed().
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

# Path bootstrap: when run as a script (python scripts/wp1/gate_analysis.py),
# Python adds the script directory to sys.path[0] rather than the repo root.
# Insert the repo root so that `backtest`, `scripts`, and `strategies` packages
# are all importable regardless of invocation style.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_REPO_SRC = _REPO_ROOT / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from backtest.utils import PROJECT_ROOT  # noqa: E402

import scripts._bootstrap  # noqa: F401, E402  (prepends repo src/ to sys.path)
from scripts.wp1.predictability import compute_rolling_predictability  # noqa: E402
from strategies.vol_regime_switch.regime_detector import detect_regime  # noqa: E402
from strategies.vol_regime_switch.regime_population import (  # noqa: E402
    non_overlapping_samples as _nl_sampler,
)

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------

GATE_DIR = PROJECT_ROOT / "backtest_results" / "gate"

# ---------------------------------------------------------------------------
# Pre-registered thresholds (FROZEN — from 00-PREREGISTRATION.md, Section 3)
# DO NOT CHANGE after ratification.
# ---------------------------------------------------------------------------

# Non-degeneracy thresholds (D-04)
DEGEN_CV_FLOOR: float = 0.05        # CV = std/mean; series near-constant if CV < this
DEGEN_IQR_FLOOR: float = 1e-5       # IQR = Q75-Q25; collapsed middle if IQR < this
DEGEN_RAIL_LEVEL: float = 0.95      # |VR(q)-1| >= 0.95 counts as "pinned at rail"
DEGEN_RAIL_FRAC_CUTOFF: float = 0.05  # > 5% of values at rail => degenerate

# Non-trivial cross-regime effect threshold (D-02, Section 5)
EPSILON_SQ_FLOOR: float = 0.01      # epsilon-squared > 0.01 => non-trivial (small effect)

# Pinned (W, q) triple (D-05, Section 2)
WQ_GRID: list[tuple[int, int]] = [(60, 5), (120, 5), (240, 15)]

# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def check_degeneracy(pred_t: np.ndarray) -> dict:
    """Non-degeneracy check on a predictability series.

    Uses the RATIFIED pre-registered thresholds (FROZEN constants above).
    The series must already be filtered to finite, non-warmup values by the
    caller, OR this function will filter internally.

    Parameters
    ----------
    pred_t : np.ndarray
        1-D array of predictability_t values (|VR(q)-1|).  May contain NaN.

    Returns
    -------
    dict with keys:
        std, cv, iqr, rail_frac, zero_frac (floats)
        is_degenerate (bool)  -- True if ANY degeneracy condition holds.

    Degeneracy conditions (ANY one => degenerate):
        std == 0           (constant series)
        cv < DEGEN_CV_FLOOR    (near-constant)
        iqr < DEGEN_IQR_FLOOR  (no spread in middle 50%)
        rail_frac > DEGEN_RAIL_FRAC_CUTOFF  (saturation)
    """
    finite = pred_t[np.isfinite(pred_t)]
    if len(finite) == 0:
        return {
            "std": float("nan"),
            "cv": float("nan"),
            "iqr": float("nan"),
            "rail_frac": float("nan"),
            "zero_frac": float("nan"),
            "is_degenerate": True,
        }

    std_ = float(np.std(finite, ddof=1))
    mean_ = float(np.mean(finite))
    cv = std_ / mean_ if mean_ > 0 else float("inf")
    q25, q75 = np.percentile(finite, [25, 75])
    iqr = float(q75 - q25)
    rail_frac = float(np.mean(finite >= DEGEN_RAIL_LEVEL))
    zero_frac = float(np.mean(finite <= 0.001))

    is_degenerate = (
        (std_ == 0.0)
        or (cv < DEGEN_CV_FLOOR)
        or (iqr < DEGEN_IQR_FLOOR)
        or (rail_frac > DEGEN_RAIL_FRAC_CUTOFF)
    )

    return {
        "std": std_,
        "cv": float(cv),
        "iqr": iqr,
        "rail_frac": rail_frac,
        "zero_frac": zero_frac,
        "is_degenerate": bool(is_degenerate),
    }


def non_overlapping_samples(series: np.ndarray, stride: int) -> np.ndarray:
    """Extract non-overlapping samples from a series at stride W by array index.

    Returns every stride-th element starting from index stride-1 (the last
    element of the first non-overlapping window).

    Parameters
    ----------
    series : np.ndarray
        1-D array of values.
    stride : int
        Window width W; samples are separated by exactly W array positions.

    Returns
    -------
    np.ndarray : series[stride-1::stride]

    Notes
    -----
    KNOWN LIMITATION (Phase-0 ratified; Phase-1 prerequisite to correct):
    This function samples at stride W on the *array index* of whatever series
    is passed in.  In gate analysis the input is the post-filter (compacted)
    array after warmup/NaN removal, so consecutive samples are W array slots
    apart but are NOT guaranteed to be W bars apart in TIME.  Each interior NaN
    that was removed shifts every subsequent sample one slot earlier in time,
    potentially allowing overlap between the underlying W-bar price windows that
    two adjacent KW samples were computed from.

    For the Phase-0 gate this is inconsequential: the effect size was ~0.001,
    far below the 0.01 floor, and compaction was ~0.25%.  The ratified verdict
    is unaffected.  Correcting to time-axis sampling (sample at stride W on the
    original index, then intersect with validity) is a Phase-1 prerequisite
    before this sampler is reused in later analyses.

    Examples
    --------
    non_overlapping_samples(np.arange(600.0), stride=120)
    -> array([119., 239., 359., 479., 599.])  length 5, first element 119
    """
    return series[stride - 1 :: stride]


def run_gate(
    pred_series: dict,
    regime: np.ndarray,
    wq_grid: list,
    strides: dict | None = None,
) -> dict:
    """Run the gate analysis: degeneracy + KW/epsilon-sq per (W,q).

    Parameters
    ----------
    pred_series : dict
        Mapping from wq key -> 1-D np.ndarray of predictability_t values
        (same length as regime array; NaN for warmup bars).
    regime : np.ndarray
        1-D int8 array of regime labels (-1=warmup, 0/1/2=active).
        Same length as each series in pred_series.
    wq_grid : list
        List of keys to process; each key must exist in pred_series.
    strides : dict, optional
        Mapping from wq key -> integer stride for non-overlapping sampling.
        If a key is absent or strides is None, stride=1 is used (all elements
        treated as non-overlapping -- appropriate for unit tests with small N).

    Returns
    -------
    dict with keys:
        'verdict' : str -- 'PASS' or 'FAIL'
        'per_wq'  : list of dicts, one per entry in wq_grid, each containing:
                    W, q, n_samples, degeneracy metrics (std,cv,iqr,rail_frac,
                    zero_frac,is_degenerate), per_regime_stats, kw_h, kw_pval,
                    epsilon_sq, setting_verdict, notes
        'reasons' : list of str

    Verdict logic (verbatim per 00-PREREGISTRATION.md Section 6):
        PASS iff EXISTS wq such that NOT degenerate(wq) AND epsilon_sq(wq) > 0.01
        FAIL iff FOR ALL wq: degenerate(wq) OR epsilon_sq(wq) <= 0.01 OR INCONCLUSIVE(wq)
    """
    if strides is None:
        strides = {}

    per_wq: list[dict] = []
    reasons: list[str] = []
    any_pass = False

    for wq_key in wq_grid:
        stride = strides.get(wq_key, 1)

        # Extract the W and q values from the key (any type -- int or str)
        W_val = wq_key[0]
        q_val = wq_key[1]

        series = pred_series[wq_key]
        series = np.asarray(series, dtype=np.float64)

        # ----------------------------------------------------------------
        # Pitfall 5: filter to regime >= 0 BEFORE any grouping or sampling
        # ----------------------------------------------------------------
        valid_mask = (regime >= 0) & np.isfinite(series)
        pred_valid = series[valid_mask]

        # Degeneracy check on the full (non-warmup, finite) series
        degen = check_degeneracy(pred_valid)

        # ----------------------------------------------------------------
        # Non-overlapping samples for KW (Pitfall 3)
        # D-06: use the shared WR-01-corrected sampler from regime_population.
        # Pass the FULL-LENGTH arrays; the sampler applies stride on the original
        # time-axis index and intersects (regime >= 0) & isfinite(series) internally.
        # ----------------------------------------------------------------
        pred_nl, regime_nl, _retained_idx = _nl_sampler(series, regime, stride)
        n_total = len(pred_nl)

        # Per-regime distributions
        per_regime_stats: dict[str, dict] = {}
        for r_label in (0, 1, 2):
            mask_r = regime_nl == r_label
            vals = pred_nl[mask_r]
            if len(vals) == 0:
                per_regime_stats[str(r_label)] = {
                    "n": 0, "mean": float("nan"), "median": float("nan"),
                    "std": float("nan"), "q25": float("nan"), "q75": float("nan"),
                }
            else:
                q25r, q75r = np.percentile(vals, [25, 75])
                per_regime_stats[str(r_label)] = {
                    "n": int(len(vals)),
                    "mean": float(np.mean(vals)),
                    "median": float(np.median(vals)),
                    "std": float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0),
                    "q25": float(q25r),
                    "q75": float(q75r),
                }

        # ----------------------------------------------------------------
        # Kruskal-Wallis + epsilon-squared
        # ----------------------------------------------------------------
        groups = [pred_nl[regime_nl == r] for r in (0, 1, 2)]
        groups_valid = [g for g in groups if len(g) >= 2]

        if len(groups_valid) < 2:
            # Insufficient data for KW
            kw_h = float("nan")
            kw_pval = float("nan")
            epsilon_sq = float("nan")
            setting_verdict = "INCONCLUSIVE"
            notes = "insufficient groups (need >= 2 regime groups with >= 2 samples each)"
            reasons.append(f"({W_val},{q_val}): {notes}")
        else:
            h_stat, p_val = scipy_stats.kruskal(*groups_valid)
            kw_h = float(h_stat)
            kw_pval = float(p_val)

            if not np.isfinite(kw_h):
                # All sampled values are identical (complete tie) -- KW rank
                # denominator goes to zero and SciPy returns NaN.  This is a
                # degeneracy condition, not a measured trivial effect; classify
                # as INCONCLUSIVE rather than FAIL to avoid misleading notes.
                setting_verdict = "INCONCLUSIVE"
                notes = "Kruskal-Wallis undefined (all-tied ranks across sampled groups)"
                epsilon_sq = float("nan")
                reasons.append(f"({W_val},{q_val}): KW undefined (tied ranks)")
            else:
                # epsilon-squared = H / ((n^2 - 1) / (n + 1))  [Tomczak & Tomczak 2014]
                n_kw = sum(len(g) for g in groups_valid)
                epsilon_sq = float(kw_h / ((n_kw ** 2 - 1) / (n_kw + 1))) if n_kw > 1 else float("nan")

                if degen["is_degenerate"]:
                    setting_verdict = "FAIL"
                    notes = "degenerate series"
                    reasons.append(f"({W_val},{q_val}): degenerate")
                elif epsilon_sq > EPSILON_SQ_FLOOR:
                    setting_verdict = "PASS"
                    notes = f"non-degenerate AND epsilon_sq={epsilon_sq:.4f} > {EPSILON_SQ_FLOOR}"
                    any_pass = True
                else:
                    setting_verdict = "FAIL"
                    notes = f"epsilon_sq={epsilon_sq:.4f} <= {EPSILON_SQ_FLOOR} (trivial cross-regime effect)"
                    reasons.append(f"({W_val},{q_val}): epsilon_sq <= {EPSILON_SQ_FLOOR}")

        per_wq.append({
            "W": W_val,
            "q": q_val,
            "n_samples": n_total,
            "std": degen["std"],
            "cv": degen["cv"],
            "iqr": degen["iqr"],
            "rail_frac": degen["rail_frac"],
            "zero_frac": degen["zero_frac"],
            "is_degenerate": degen["is_degenerate"],
            "per_regime_stats": per_regime_stats,
            "kw_h": kw_h,
            "kw_pval": kw_pval,
            "epsilon_sq": epsilon_sq,
            "setting_verdict": setting_verdict,
            "notes": notes,
        })

    verdict = "PASS" if any_pass else "FAIL"

    return {
        "verdict": verdict,
        "per_wq": per_wq,
        "reasons": reasons,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> int:
    """Load 2021-2025, detect regime, compute predictability per (W,q), run gate."""
    import datetime as _dt

    from scripts.wp1.py_engine import load_and_clean

    # -----------------------------------------------------------------------
    # D-07 holdout: define the data window and assert it ends before 2026-01-01
    # -----------------------------------------------------------------------
    # 2021-01-01 00:00:00 UTC in epoch milliseconds
    start_ms = int(
        _dt.datetime(2021, 1, 1, tzinfo=_dt.UTC).timestamp() * 1000
    )
    # 2025-12-31 23:59:59 UTC in epoch milliseconds
    end_ms = int(
        _dt.datetime(2025, 12, 31, 23, 59, 59, tzinfo=_dt.UTC).timestamp() * 1000
    )

    # D-07 guard: verify the window end is strictly before 2026-01-01
    holdout_boundary_ms = int(
        _dt.datetime(2026, 1, 1, tzinfo=_dt.UTC).timestamp() * 1000
    )
    assert end_ms < holdout_boundary_ms, (
        f"D-07 VIOLATION: data window end ({end_ms}) must be < 2026-01-01 "
        f"({holdout_boundary_ms}). The 2026 holdout must NOT be loaded."
    )

    data_path = str(PROJECT_ROOT / "data" / "binance_futures")
    symbol = "BTCUSDT"

    print(f"Loading {symbol} 2021-2025 (D-07 holdout: 2026 NOT loaded)...")
    data = load_and_clean(
        data_path=data_path,
        symbol=symbol,
        start_ms=start_ms,
        end_ms=end_ms,
        max_gap_allowed_mins=60,
    )

    close = data["close"]
    timestamps = data["timestamp"]
    N = len(close)

    # Confirm data span
    ts_start = _dt.datetime.fromtimestamp(timestamps[0] / 1000, tz=_dt.UTC)
    ts_end = _dt.datetime.fromtimestamp(timestamps[-1] / 1000, tz=_dt.UTC)
    print(f"Data span: {ts_start.date()} -> {ts_end.date()} ({N:,} bars)")

    # D-07 double-check: last timestamp must be before 2026-01-01
    assert ts_end.year < 2026, (
        f"D-07 VIOLATION: last loaded bar is in {ts_end.year} (year=2026 partition was read). "
        "The 2026 holdout must remain UNTOUCHED."
    )

    # -----------------------------------------------------------------------
    # Detect regime (causal R_t from regime_detector -- Pitfall 4 enforced)
    # -----------------------------------------------------------------------
    # REGIME CLASSIFICATION PATH DISCLOSURE (WR-04):
    # Passing no `config` argument selects the STATIC threshold path inside
    # detect_regime (low_vol_threshold=1.2, extreme_vol_threshold=2.5 on the
    # fast/slow RMS-vol ratio).  This is the DEFAULT_REGIME_CONFIG path, which
    # contains no p_low/p_high keys, so the dynamic rolling-quantile branch is
    # NOT used.  The strategy's DEFAULT_INTEGRATION_CONFIG also lacks p_low/p_high
    # and therefore uses the same static threshold path -- so the gate's regime
    # definition matches the strategy's default configuration.  This is the
    # ratified regime definition for Phase-0; if a quantile-based definition is
    # desired in future phases it must be explicitly configured and re-registered.
    print("Detecting regime labels...")
    regime = detect_regime({"close": close})  # int8 array; -1=warmup

    # -----------------------------------------------------------------------
    # Compute rolling predictability per (W, q)
    # -----------------------------------------------------------------------
    print(f"Computing predictability_t for {len(WQ_GRID)} (W,q) settings...")
    pred_series: dict[tuple[int, int], np.ndarray] = {}
    for W, q in WQ_GRID:
        print(f"  (W={W}, q={q})...")
        pred_series[(W, q)] = compute_rolling_predictability(close, W=W, q=q)

    # -----------------------------------------------------------------------
    # Strides for non-overlapping KW sampling (stride = W for each setting)
    # -----------------------------------------------------------------------
    strides = {(W, q): W for W, q in WQ_GRID}

    # -----------------------------------------------------------------------
    # Run gate analysis
    # -----------------------------------------------------------------------
    print("Running gate analysis (degeneracy + KW + epsilon-sq)...")
    gate_result = run_gate(
        pred_series=pred_series,
        regime=regime,
        wq_grid=WQ_GRID,
        strides=strides,
    )

    # -----------------------------------------------------------------------
    # Assemble gate report
    # -----------------------------------------------------------------------
    run_date = datetime.now(tz=_dt.UTC).strftime("%Y%m%d_%H%M%S")
    report = {
        "run_date": run_date,
        "data_span": {
            "start": str(ts_start.date()),
            "end": str(ts_end.date()),
            "n_bars": N,
            "year_2026_loaded": False,
        },
        "preregistration": {
            "source": "00-PREREGISTRATION.md (status: ratified 2026-06-01)",
            "epsilon_sq_floor": EPSILON_SQ_FLOOR,
            "cv_floor": DEGEN_CV_FLOOR,
            "iqr_floor": DEGEN_IQR_FLOOR,
            "rail_level": DEGEN_RAIL_LEVEL,
            "rail_frac_cutoff": DEGEN_RAIL_FRAC_CUTOFF,
        },
        "verdict": gate_result["verdict"],
        "verdict_note": (
            "PROVISIONAL — computed against ratified pre-registered thresholds. "
            "FINAL verdict requires human ratification (D-01)."
        ),
        "per_wq": gate_result["per_wq"],
        "reasons": gate_result["reasons"],
    }

    # Write artifact
    GATE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = GATE_DIR / f"gate_report_{run_date}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Print summary
    print("\n" + "=" * 60)
    print(f"GATE REPORT: {out_path}")
    print(f"Data span: {ts_start.date()} -> {ts_end.date()} ({N:,} bars)")
    print("2026 holdout: NOT loaded (D-07 guard held)")
    print()
    print("Per-(W,q) results:")
    for row in gate_result["per_wq"]:
        deg_str = "DEGENERATE" if row["is_degenerate"] else "non-degenerate"
        eps = row["epsilon_sq"]
        eps_str = "n/a" if np.isnan(eps) else f"{eps:.4f}"
        print(
            f"  (W={row['W']}, q={row['q']}): {deg_str}, "
            f"n_samples={row['n_samples']}, "
            f"epsilon_sq={eps_str}, "
            f"verdict={row['setting_verdict']}"
        )
    print()
    print(f"PROVISIONAL VERDICT: {gate_result['verdict']}")
    print("(FINAL verdict requires human ratification per D-01 — not autonomous)")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
