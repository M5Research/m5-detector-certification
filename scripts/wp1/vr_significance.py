"""Pure-function VR-significance math module.

Activates the z_m2 M2 statistic from _vr_m2_kernel, computes per-window
|VR(q)-1| population + median CI, Holm over q-family (family_size=4),
VR horizon profile, and LOW-regime noise floor per the frozen protocol
in 07-PREREGISTRATION.md (freeze commit 720c1d4).

This module is import-clean: module-level imports are ONLY numpy.
scipy.stats is DEFERRED inside functions that need it, so the module
can be imported at pytest collection time without touching any real-data
pipeline.

Frozen constants (consumed, never recomputed):
  - Closure threshold   : |VR(q)-1| < 0.001  (§7)
  - Holm family size    : 4  (§8, Pre-Check B)
  - Bootstrap seed      : 43  (D-B8)
  - q-grid              : {2, 5, 15, 60}  (§6; q=1 EXCLUDED)
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Imports from sibling modules (not at module scope to maintain import-clean
# discipline — these are resolved at function call time)
# ---------------------------------------------------------------------------
# _vr_m2_kernel is imported inside compute_rolling_vr_and_z
# apply_holm is imported inside apply_holm_b


# ---------------------------------------------------------------------------
# Rolling VR and z_m2 (z_m2 ACTIVATION — D-B8)
# ---------------------------------------------------------------------------


def compute_rolling_vr_and_z(
    close: np.ndarray,
    W: int,
    q: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Causal rolling VR and z_m2 arrays, both NaN at warmup 0..W-2.

    Activates the z_m2 value that compute_rolling_predictability discards at
    predictability.py:L311. Both return values from _vr_m2_kernel are captured:
    ``vr, z = _vr_m2_kernel(r_window, q)`` — never ``vr, _``.

    Parameters
    ----------
    close : np.ndarray
        1-D array of close prices (positive, no NaN/Inf).
    W : int
        Rolling window size in bars (>= q + 2).
    q : int
        Aggregation horizon (>= 2). Raises ValueError for q < 2.

    Returns
    -------
    vr_arr : np.ndarray of float64, length N
        Per-bar |VR(q)-1| values. NaN at warmup bars and degenerate windows.
    z_arr : np.ndarray of float64, length N
        Per-bar M2 z_m2 values. NaN at warmup bars and degenerate windows.

    Raises
    ------
    ValueError
        If q < 2 (VR(1) is identically 1; q=1 excluded per §6).
        If W < q + 2 (window too short to compute VR(q)).
    """
    if q < 2:
        raise ValueError(
            f"q must be >= 2; VR(1) is identically 1 (got q={q})."
        )
    if W < q + 2:
        raise ValueError(
            f"W must be >= q + 2 = {q + 2} to compute VR(q) (got W={W})."
        )

    from scripts.wp1.predictability import rolling_vr_m2_z_arrays  # noqa: PLC0415

    close = np.asarray(close, dtype=np.float64)
    N = len(close)

    if N < W:
        return (
            np.full(N, np.nan, dtype=np.float64),
            np.full(N, np.nan, dtype=np.float64),
        )

    log_close = np.log(close)
    r_full = np.empty(N, dtype=np.float64)
    r_full[0] = 0.0
    r_full[1:] = np.diff(log_close)

    return rolling_vr_m2_z_arrays(r_full, W, q, N, stride=1)


def compute_rolling_vr_and_z_strided(
    close: np.ndarray,
    W: int,
    q: int,
    stride: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Strided rolling VR for cascade paths (stride=W by default)."""
    if q < 2:
        raise ValueError(
            f"q must be >= 2; VR(1) is identically 1 (got q={q})."
        )
    if W < q + 2:
        raise ValueError(
            f"W must be >= q + 2 = {q + 2} to compute VR(q) (got W={W})."
        )
    from scripts.wp1.predictability import rolling_vr_m2_z_arrays  # noqa: PLC0415

    close = np.asarray(close, dtype=np.float64)
    N = len(close)
    if N < W:
        return (
            np.full(N, np.nan, dtype=np.float64),
            np.full(N, np.nan, dtype=np.float64),
        )
    step = W if stride is None else stride
    log_close = np.log(close)
    r_full = np.empty(N, dtype=np.float64)
    r_full[0] = 0.0
    r_full[1:] = np.diff(log_close)
    return rolling_vr_m2_z_arrays(r_full, W, q, N, stride=step)


# ---------------------------------------------------------------------------
# Circular block-bootstrap CI on median |VR(q)-1|
# ---------------------------------------------------------------------------


def median_vr_dep_boot_ci(
    pred_nl: np.ndarray,
    block: int,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 43,
) -> tuple[float, float, float]:
    """Circular block-bootstrap 95% CI on the MEDIAN of pred_nl.

    Reuses the circular block-bootstrap MECHANISM of epsilon_sq_boot_ci
    (regime_population.py L81-145), changing ONLY the statistic to
    np.median (not _epsilon_sq_kw). No regime_nl argument needed (pooled,
    not grouped). block=W (caller supplies window width; NOT block=10 from
    regate — Pitfall 4).

    Parameters
    ----------
    pred_nl : np.ndarray
        Non-overlapping |VR(q)-1| values (output of non_overlapping_samples).
    block : int
        Block size for circular bootstrap. Pass block=W (window width).
    n_boot : int
        Number of bootstrap resamples (default 2000).
    alpha : float
        Coverage level: CI covers 1-alpha (default 0.05 for 95% CI).
    seed : int
        RNG seed (BOOT_SEED=43; never np.random.seed). Frozen D-B8.

    Returns
    -------
    (point, lo, hi) : 95% percentile CI on median |VR-1|.
        Returns (point, nan, nan) if n < 2*block or fewer than 10 finite boots.

    Notes
    -----
    Uses np.random.default_rng(seed) exclusively (determinism guarantee).
    The legacy seeding API (np.random.seed) is not used in this module.
    The CI is clamped to contain the point estimate (lo <= point <= hi).
    """
    pred_nl = np.asarray(pred_nl, dtype=np.float64)
    n = len(pred_nl)
    point = float(np.median(pred_nl))

    if n < block * 2:
        return point, float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    boots = np.empty(n_boot, dtype=np.float64)

    for b in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]) % n
        flat_idx = idx.ravel()[:n]
        pred_b = pred_nl[flat_idx]
        boots[b] = np.median(pred_b)

    boots = boots[np.isfinite(boots)]
    if len(boots) < 10:
        return point, float("nan"), float("nan")

    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    # Clamp CI to contain the point estimate
    lo = min(lo, point)
    hi = max(hi, point)
    return point, lo, hi


# ---------------------------------------------------------------------------
# VR significance summary for a single q
# ---------------------------------------------------------------------------


def compute_vr_significance(
    pred_nl: np.ndarray,
    z_nl: np.ndarray,
) -> dict:
    """Effect-size-first significance summary for a single q.

    The MEDIAN of |VR(q)-1| is the central closure statistic (pinned by the
    regate distributional precedent PRE-RUN, RF-1). Mean is reported
    alongside but is NOT the closure statistic.

    The aggregate two-tailed p-value uses the MEDIAN z_m2 across
    non-overlapping windows (representative z for the cell), following
    the approach: p = 2 * norm.sf(|median_z|). This is consistent with
    reporting the median as the primary central statistic (§6).

    Parameters
    ----------
    pred_nl : np.ndarray
        Non-overlapping |VR(q)-1| values (output of non_overlapping_samples).
    z_nl : np.ndarray
        Non-overlapping z_m2 values corresponding to pred_nl windows.

    Returns
    -------
    dict with keys:
        'median_vr_dep'  : float  — MEDIAN |VR(q)-1| (closure statistic)
        'mean_vr_dep'    : float  — mean |VR(q)-1| (reported alongside, NOT closure)
        'n_nl'           : int    — number of non-overlapping windows
        'closed'         : bool   — True iff median_vr_dep < 0.001 (§7 frozen floor)
        'p_twotailed'    : float  — two-tailed p from median z_m2 (norm.sf)
        'median_z_m2'    : float  — median z_m2 statistic
    """
    from scipy.stats import norm  # noqa: PLC0415 — deferred import

    pred_nl = np.asarray(pred_nl, dtype=np.float64)
    z_nl = np.asarray(z_nl, dtype=np.float64)

    finite_pred = pred_nl[np.isfinite(pred_nl)]
    finite_z = z_nl[np.isfinite(z_nl)]

    if len(finite_pred) == 0:
        return {
            "median_vr_dep": float("nan"),
            "mean_vr_dep": float("nan"),
            "n_nl": 0,
            "closed": False,
            "p_twotailed": float("nan"),
            "median_z_m2": float("nan"),
        }

    median_vr_dep = float(np.median(finite_pred))
    mean_vr_dep = float(np.mean(finite_pred))
    # Closure threshold: 0.001 consumed verbatim from §7 (never recomputed)
    closed = bool(median_vr_dep < 0.001)

    if len(finite_z) == 0:
        p_twotailed = float("nan")
        median_z_m2 = float("nan")
    else:
        median_z_m2 = float(np.median(finite_z))
        p_twotailed = 2.0 * float(norm.sf(abs(median_z_m2)))

    return {
        "median_vr_dep": median_vr_dep,
        "mean_vr_dep": mean_vr_dep,
        "n_nl": int(len(finite_pred)),
        "closed": closed,
        "p_twotailed": p_twotailed,
        "median_z_m2": median_z_m2,
    }


# ---------------------------------------------------------------------------
# Holm correction for Pre-Check B (family_size=4 FROZEN)
# ---------------------------------------------------------------------------


def apply_holm_b(pvalues: list[float]) -> list[float]:
    """Holm correction for Pre-Check B family (4 tests, one per q). §8.

    Thin wrapper calling nested_test.apply_holm with family_size=4.
    family_size=4 is FROZEN by §8 (Pre-Check B denominator); never 3.

    Parameters
    ----------
    pvalues : list[float]
        Raw p-values (one per q in the frozen q-grid {2,5,15,60}).

    Returns
    -------
    list[float]
        Holm-adjusted p-values in the same input order.
    """
    from scripts.wp1.nested_test import apply_holm as _apply_holm_base  # noqa: PLC0415

    return _apply_holm_base(pvalues, family_size=4)  # family_size=4 FROZEN; never 3


# ---------------------------------------------------------------------------
# VR horizon profile (microstructure artifact 1, D-B4/RF-6)
# ---------------------------------------------------------------------------


def compute_vr_horizon_profile(
    close: np.ndarray,
    W: int,
    q_grid: tuple[int, ...],
) -> dict:
    """Microstructure artifact 1: median |VR(q)-1| per q across the full series.

    For each q in q_grid, computes the rolling VR departure and returns the
    median over valid (non-NaN) bars. The |VR(2)-1| vs |VR(60)-1| shape is
    the bid-ask-bounce signature: bounce inflates VR most strongly at short
    horizons (§6). This is standalone diagnostic data; the 'contextualizes-A'
    linkage is deferred to Phase 10 per D-B10.

    Parameters
    ----------
    close : np.ndarray
        1-D array of close prices.
    W : int
        Rolling window width.
    q_grid : tuple[int, ...]
        Aggregation horizons to profile (frozen §6: (2, 5, 15, 60)).

    Returns
    -------
    dict
        {q: median_vr_dep} for each q in q_grid.
        NaN entries for degenerate grids.
    """
    result = {}
    for q in q_grid:
        vr_arr, _ = compute_rolling_vr_and_z(close, W, q)
        valid = vr_arr[np.isfinite(vr_arr)]
        result[q] = float(np.median(valid)) if len(valid) > 0 else float("nan")
    return result


# ---------------------------------------------------------------------------
# LOW-regime noise floor (microstructure artifact 2, D-B4/RF-6)
# ---------------------------------------------------------------------------


def compute_low_noise_floor(
    close: np.ndarray,
    regime: np.ndarray,
    W: int,
    q_grid: tuple[int, ...],
) -> dict:
    """Microstructure artifact 2: median |VR(q)-1| within LOW persistence windows.

    A LOW persistence window is a contiguous run of regime == 0 bars of length
    >= W. Within these runs only, non-overlapping |VR(q)-1| samples are taken
    and the median is returned per q. This is the 'steady-state LOW floor'
    attributable to residual microstructure noise (bid-ask-bounce, discretization)
    rather than regime transitions.

    The regime array is passed in by the caller (regenerated inline via
    RollingQuantileDetector().fit(close) per D-B7 — this module does NOT load
    labels itself).

    Parameters
    ----------
    close : np.ndarray
        1-D array of close prices.
    regime : np.ndarray
        1-D int8 regime labels (-1=warmup, 0=LOW, 1=ELEVATED, 2=EXTREME).
        Same length as close.
    W : int
        Rolling window width; also the minimum run length for persistence.
    q_grid : tuple[int, ...]
        Aggregation horizons (frozen §6: (2, 5, 15, 60)).

    Returns
    -------
    dict
        {q: median_vr_dep} computed only over LOW persistence windows.
        NaN if no qualifying persistence window exists.
    """
    regime = np.asarray(regime)
    N = len(regime)

    # Identify contiguous LOW runs of length >= W using run-length encoding
    low_mask = (regime == 0).astype(np.int8)

    # Build a boolean mask of indices belonging to LOW persistence windows (runs >= W)
    persistence_mask = np.zeros(N, dtype=bool)
    i = 0
    while i < N:
        if low_mask[i]:
            # Find end of this LOW run
            j = i
            while j < N and low_mask[j]:
                j += 1
            run_len = j - i
            if run_len >= W:
                persistence_mask[i:j] = True
            i = j
        else:
            i += 1

    result = {}
    for q in q_grid:
        vr_arr, _ = compute_rolling_vr_and_z(close, W, q)

        # Within persistence windows only, take non-overlapping samples
        # Using the same stride mask as non_overlapping_samples, restricted to persistence bars
        idx = np.arange(N)
        stride_mask = ((idx - (W - 1)) % W == 0) & (idx >= W - 1)
        valid_mask = stride_mask & persistence_mask & np.isfinite(vr_arr)
        samples = vr_arr[valid_mask]

        result[q] = float(np.median(samples)) if len(samples) > 0 else float("nan")

    return result


# ---------------------------------------------------------------------------
# Per-year breakdown
# ---------------------------------------------------------------------------


def compute_per_year_breakdown(
    pred_nl: np.ndarray,
    timestamps_nl: np.ndarray,
    z_nl: np.ndarray,
) -> dict:
    """Group non-overlapping windows by calendar year of each window's end bar.

    Each window is assigned to the calendar year of its END bar timestamp
    (epoch-ms → UTC year), NOT by bar index (Pitfall 7). For each year
    returns the median |VR(q)-1|, a (degenerate) per-year CI note, and n_nl.

    Parameters
    ----------
    pred_nl : np.ndarray
        Non-overlapping |VR(q)-1| values.
    timestamps_nl : np.ndarray
        Epoch-ms timestamps of each window's end bar (aligned to pred_nl).
        These are timestamps[retained_idx] passed by the orchestrator.
    z_nl : np.ndarray
        Non-overlapping z_m2 values (aligned to pred_nl).

    Returns
    -------
    dict
        {year: {'median_vr_dep': float, 'mean_vr_dep': float, 'n_nl': int}}
        Only years with at least one sample are included.
    """
    pred_nl = np.asarray(pred_nl, dtype=np.float64)
    timestamps_nl = np.asarray(timestamps_nl, dtype=np.float64)

    # Convert epoch-ms to UTC year
    import datetime as _dt  # noqa: PLC0415 — standard library

    years = np.array([
        _dt.datetime.utcfromtimestamp(ts / 1000.0).year
        for ts in timestamps_nl
    ], dtype=np.int32)

    result = {}
    for year in sorted(np.unique(years)):
        mask = (years == year) & np.isfinite(pred_nl)
        group = pred_nl[mask]
        if len(group) == 0:
            continue
        result[int(year)] = {
            "median_vr_dep": float(np.median(group)),
            "mean_vr_dep": float(np.mean(group)),
            "n_nl": int(len(group)),
        }

    return result


# ---------------------------------------------------------------------------
# MDE (minimum detectable VR departure, Phase 10 SC2 feed)
# ---------------------------------------------------------------------------


def compute_mde_vr(
    n_nl: int,
    alpha: float = 0.05,
    power: float = 0.80,
) -> dict:
    """Analytic 80%-power MDE for two-tailed z-test on |VR(q)-1|.

    Feeds Phase 10 SC2. The analytic formula (RF-5):
        z_alpha_half = norm.ppf(1 - alpha/2)   # 1.96 at alpha=0.05
        z_beta       = norm.ppf(power)           # 0.842 at power=0.80
        se_vr_dep    = 1 / sqrt(n_nl)           # approx SE of median |VR-1| under H0
        MDE          = (z_alpha_half + z_beta) * se_vr_dep

    Parameters
    ----------
    n_nl : int
        Number of non-overlapping observations (effective N).
    alpha : float
        Significance level (default 0.05).
    power : float
        Target power (default 0.80).

    Returns
    -------
    dict with keys:
        'n_nl'               : int
        'alpha'              : float
        'power_target'       : float
        'mde_vr_departure'   : float
        'z_alpha_half'       : float
        'z_beta'             : float
    """
    from scipy.stats import norm  # noqa: PLC0415 — deferred import

    z_alpha_half = float(norm.ppf(1 - alpha / 2))   # 1.96 at alpha=0.05
    z_beta = float(norm.ppf(power))                   # 0.842 at power=0.80
    se_vr_dep = 1.0 / float(np.sqrt(n_nl))
    mde = (z_alpha_half + z_beta) * se_vr_dep

    return {
        "n_nl": int(n_nl),
        "alpha": float(alpha),
        "power_target": float(power),
        "mde_vr_departure": float(mde),
        "z_alpha_half": z_alpha_half,
        "z_beta": z_beta,
    }
