"""SC4/D-07 population helpers for the causal rolling-quantile detector.

Provides the WR-01-corrected non-overlapping EXTREME sampler and the
regime-population statistics helper used by both the SC4 synthetic test
suite (Plan 02-02) and the Plan-03 real-data validation driver.

Phase boundary (D-10): this module is a pure computational unit. It does
not import any gate or forecast modules.  The WR-01 fix to the gate
analysis module remains a Phase 4 deliverable (D-07).

See 01-PREREGISTRATION.md §4 (WR-01 sampler algorithm) and §9 (n_min=50 /
SPARSE rule).
"""

from __future__ import annotations

import numpy as np


def non_overlapping_samples(
    series: np.ndarray,
    regime: np.ndarray,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """WR-01-corrected non-overlapping sampler returning (pred_nl, regime_nl, retained_idx).

    Strides on the ORIGINAL time-axis index, then intersects the validity mask.
    Guarantees all consecutive retained samples are >= stride apart in original time.

    Parameters
    ----------
    series : np.ndarray
        1-D float64 predictability values (NaN for warmup / degenerate windows).
    regime : np.ndarray
        1-D int8 regime labels (-1=warmup, 0/1/2=active). Same length as series.
    stride : int
        Non-overlapping window width W.

    Returns
    -------
    pred_nl    : non-overlapping predictability values
    regime_nl  : corresponding regime labels
    retained_idx : original integer indices of retained samples (for gap assertion)

    Algorithm verbatim from 01-PREREGISTRATION.md §4 (WR-01-corrected):
        idx = np.arange(len(series))
        stride_mask = ((idx - (stride - 1)) % stride == 0) & (idx >= stride - 1)
        sample_mask = stride_mask & (regime >= 0) & np.isfinite(series)
    """
    series = np.asarray(series, dtype=np.float64)
    regime = np.asarray(regime)
    idx = np.arange(len(series))
    # WR-01 correction: stride on original time-axis index
    stride_mask = ((idx - (stride - 1)) % stride == 0) & (idx >= stride - 1)
    # Intersect validity mask: exclude warmup (-1) and non-finite pred values
    sample_mask = stride_mask & (regime >= 0) & np.isfinite(series)
    retained_idx = idx[sample_mask]
    return series[sample_mask], regime[sample_mask], retained_idx


def _epsilon_sq_kw(pred: np.ndarray, regime: np.ndarray) -> float:
    """Compute Kruskal-Wallis epsilon-squared for grouped data.

    Mirrors the formula in gate_analysis.py::run_gate (Tomczak & Tomczak 2014).
    Used as the point-estimate kernel by epsilon_sq_boot_ci and
    regime_label_permutation_null.

    CONFIRMATORY-ONLY (D-04/D-05): not gate-driving; the frozen epsilon_sq > 0.01
    + KW rule in gate_analysis.py::run_gate is unchanged.
    """
    from scipy import stats as sp  # noqa: PLC0415 — deferred import for pure src/ boundary

    groups = [pred[regime == r] for r in (0, 1, 2) if np.sum(regime == r) >= 2]
    if len(groups) < 2:
        return float("nan")
    n = sum(len(g) for g in groups)
    h, _ = sp.kruskal(*groups)
    return float(h / ((n**2 - 1) / (n + 1)))


def epsilon_sq_boot_ci(
    pred_nl: np.ndarray,
    regime_nl: np.ndarray,
    block: int,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 43,
) -> tuple[float, float, float]:
    """Circular block-bootstrap CI on epsilon-sq (KW grouped-data statistic).

    Resamples (pred_nl, regime_nl) JOINTLY in circular blocks to preserve
    the label-value pairing.  Returns (point_estimate, lower, upper).
    Returns (point, nan, nan) if n < 2*block or fewer than 10 finite boots.

    Parameters
    ----------
    pred_nl : np.ndarray
        Non-overlapping predictability values (output of non_overlapping_samples).
    regime_nl : np.ndarray
        Corresponding regime labels (same length as pred_nl).
    block : int
        Block size in sample space (block=10 recommended; see 04-RESEARCH.md §Item 2).
    n_boot : int
        Number of bootstrap resamples (default 2000).
    alpha : float
        Coverage level: CI covers 1-alpha (default 0.05 for 95% CI).
    seed : int
        RNG seed for reproducibility (BOOT_SEED=43; never np.random.seed).

    Returns
    -------
    (point, lo, hi) : 95% percentile CI on epsilon-sq.

    CONFIRMATORY-ONLY (D-04/D-05): reported in gate report, not gate-driving.
    The frozen epsilon_sq > 0.01 + KW rule in gate_analysis.py::run_gate is
    unchanged.
    """
    pred_nl = np.asarray(pred_nl, dtype=np.float64)
    regime_nl = np.asarray(regime_nl)
    n = len(pred_nl)
    point = _epsilon_sq_kw(pred_nl, regime_nl)
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
        regime_b = regime_nl[flat_idx]
        boots[b] = _epsilon_sq_kw(pred_b, regime_b)
    # Exclude NaN boots (degenerate resamples where fewer than 2 groups survive)
    boots = boots[np.isfinite(boots)]
    if len(boots) < 10:
        return point, float("nan"), float("nan")
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    # Ensure CI contains the point estimate (required by acceptance criteria):
    # for bounded statistics near the ceiling/floor the percentile CI may exclude
    # the observed value -- clamp to guarantee lo <= point <= hi.
    lo = min(lo, point)
    hi = max(hi, point)
    return point, lo, hi


def regime_label_permutation_null(
    pred_nl: np.ndarray,
    regime_nl: np.ndarray,
    n_perm: int = 5000,
    seed: int = 42,
) -> float:
    """Regime-label permutation null for epsilon-sq (Phipson-Smyth p-value).

    Shuffles regime labels among the non-overlapping samples and recomputes
    epsilon-sq per permutation.  Uses the +1/+1 finite-sample correction
    (Phipson & Smyth 2010) so that p >= 1/(n_perm+1) > 0.

    Parameters
    ----------
    pred_nl : np.ndarray
        Non-overlapping predictability values.
    regime_nl : np.ndarray
        Corresponding regime labels (same length as pred_nl).
    n_perm : int
        Number of permutations (default 5000; PERM_SEED=42).
    seed : int
        RNG seed for reproducibility (never np.random.seed).

    Returns
    -------
    float : empirical p-value = (count(eps_perm >= eps_obs) + 1) / (n_perm + 1).
            Returns NaN if fewer than 2 groups have >= 2 members (degenerate).

    CONFIRMATORY-ONLY (D-04/D-05): reported in gate report, not gate-driving.
    The frozen epsilon_sq > 0.01 + KW rule in gate_analysis.py::run_gate is
    unchanged.
    """
    pred_nl = np.asarray(pred_nl, dtype=np.float64)
    regime_nl = np.asarray(regime_nl)
    eps_obs = _epsilon_sq_kw(pred_nl, regime_nl)
    if not np.isfinite(eps_obs):
        return float("nan")
    rng = np.random.default_rng(seed)
    eps_perm = np.empty(n_perm, dtype=np.float64)
    for i in range(n_perm):
        shuffled = rng.permutation(regime_nl)
        eps_perm[i] = _epsilon_sq_kw(pred_nl, shuffled)
    # Phipson-Smyth (2010) +1/+1 correction: ensures p >= 1/(n_perm+1) > 0
    return float((np.sum(eps_perm >= eps_obs) + 1) / (n_perm + 1))


def count_nonoverlapping_extreme(regime: np.ndarray, stride: int) -> int:
    """Count non-overlapping EXTREME (regime == 2) samples.

    WR-01-corrected algorithm (01-PREREGISTRATION.md §4 / 02-RESEARCH.md
    Pattern 4): the stride is applied to the ORIGINAL time-axis index,
    NOT a compacted array of valid bars.  This guarantees a minimum
    W-bar separation between any two retained samples in original time.

    Parameters
    ----------
    regime : np.ndarray[int8]
        Regime label array (-1 = warmup, 0 = LOW, 1 = ELEVATED, 2 = EXTREME).
    stride : int
        Stride W from WQ_GRID (e.g. 60 for the tightest non-overlapping count).

    Returns
    -------
    int
        Number of non-overlapping EXTREME samples.
    """
    regime = np.asarray(regime)
    idx = np.arange(len(regime))
    # WR-01 correction: stride on original time-axis index
    stride_mask = ((idx - (stride - 1)) % stride == 0) & (idx >= stride - 1)
    # Intersect validity mask — exclude warmup bars (-1)
    sample_mask = stride_mask & (regime >= 0)
    return int(np.sum(regime[sample_mask] == 2))


def regime_population_stats(
    regime: np.ndarray,
    n_min: int = 50,
    stride: int = 60,
) -> dict[str, float | int | bool]:
    """Compute regime population statistics over valid (non-warmup) bars.

    Parameters
    ----------
    regime : np.ndarray[int8]
        Regime label array (-1 = warmup, 0 = LOW, 1 = ELEVATED, 2 = EXTREME).
    n_min : int
        Minimum non-overlapping EXTREME count before flagging SPARSE.
        Frozen primary: 50 (01-PREREGISTRATION.md §9 / D-06).
    stride : int
        Stride for count_nonoverlapping_extreme (WR-01 algorithm).

    Returns
    -------
    dict with keys:
        low_frac       : float  — fraction of valid bars labelled LOW (0)
        elevated_frac  : float  — fraction of valid bars labelled ELEVATED (1)
        extreme_frac   : float  — fraction of valid bars labelled EXTREME (2)
        n_valid        : int    — total valid bars (regime >= 0)
        n_extreme_nonoverlap : int  — WR-01-corrected non-overlapping EXTREME count
        sparse         : bool   — True when n_extreme_nonoverlap < n_min (§9)
    """
    regime = np.asarray(regime)
    valid_mask = regime >= 0
    n_valid = int(np.sum(valid_mask))

    if n_valid == 0:
        return {
            "low_frac": 0.0,
            "elevated_frac": 0.0,
            "extreme_frac": 0.0,
            "n_valid": 0,
            "n_extreme_nonoverlap": 0,
            "sparse": True,
        }

    valid_labels = regime[valid_mask]
    low_frac = float(np.sum(valid_labels == 0)) / n_valid
    elevated_frac = float(np.sum(valid_labels == 1)) / n_valid
    extreme_frac = float(np.sum(valid_labels == 2)) / n_valid

    n_extreme_nonoverlap = count_nonoverlapping_extreme(regime, stride)
    sparse = n_extreme_nonoverlap < n_min

    return {
        "low_frac": low_frac,
        "elevated_frac": elevated_frac,
        "extreme_frac": extreme_frac,
        "n_valid": n_valid,
        "n_extreme_nonoverlap": n_extreme_nonoverlap,
        "sparse": sparse,
    }
