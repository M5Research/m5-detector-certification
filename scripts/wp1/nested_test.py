"""Pure-function nested-transition-vs-levels test math module.

Implements the HAC-robust Wald chi2(1) test on the transition indicator T_t
coefficient in the nested TRANSITION vs LEVEL OLS models, per the frozen
protocol in 07-PREREGISTRATION.md (freeze commit 720c1d4).

Estimand: whether the regime-transition event T_t = 1[R_t != R_{t-1}] adds
predictive content for |VR(q)-1| over and above current and lagged regime
levels (R_t, R_{t-1}) dummies.

This module is import-clean: module-level imports are ONLY numpy.
statsmodels.api, scipy.stats, and scipy.optimize are DEFERRED imports inside
the functions that need them, so the module can be imported at pytest collection
time without touching any parquet or real-data pipeline.

All frozen analytical choices (thresholds, denominators, seeds) are consumed
verbatim from 07-PREREGISTRATION.md; none is re-derived or re-tuned here.

Frozen constants (consumed, never recomputed):
  - ΔR² threshold        : 0.005      (§7)
  - |β_T| threshold      : 0.01       (§7)
  - Degeneracy cond cutoff: 1e10       (§4)
  - Holm family size      : 3          (§8)
  - Under-powered cutoff  : < 50       (§5)
  - Bootstrap seed        : 43         (D-06)
  - Min-separation        : W bars     (§5)

References
----------
- 07-PREREGISTRATION.md (freeze commit 720c1d4), §3-§8
- 08-RESEARCH.md Q1-Q7 (statsmodels 0.14.6 HAC-Wald API verified by live execution)
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Design matrix construction
# ---------------------------------------------------------------------------


def build_design_matrix(
    regime_nl: np.ndarray,
    retained_idx: np.ndarray,
    regime_full: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Build the TRANSITION design matrix X and metadata dict.

    Columns: [const, R_t==ELEVATED, R_t==EXTREME, R_{t-1}==ELEVATED,
              R_{t-1}==EXTREME, T_t]

    LOW (regime 0) is the dummy reference category.
    R_{t-1} = regime_full[retained_idx[i] - 1] (immediately preceding calendar
    bar, per RESEARCH A1), with a boundary guard for retained_idx[i] == 0.
    T_t = 1[R_t != R_{t-1}] computed from calendar-aligned labels.

    Parameters
    ----------
    regime_nl : np.ndarray
        Regime labels at each retained (non-overlapping) sample index, shape (n,).
        Values: 0=LOW, 1=ELEVATED, 2=EXTREME.
    retained_idx : np.ndarray
        Original calendar-time indices of the retained samples, shape (n,).
    regime_full : np.ndarray
        Full-length regime label array (all calendar bars), used to look up
        R_{t-1} = regime_full[retained_idx[i] - 1].

    Returns
    -------
    X : np.ndarray, shape (n, 6)
        TRANSITION design matrix (includes T_t column).
    meta : dict
        {'ncols': 6, 'col_T': 5,
         'column_names': ['const','R_elev','R_extr','R1_elev','R1_extr','T_t'],
         'X_level': np.ndarray shape (n, 5)}  — LEVEL slice (no T_t) for ΔR².
    """
    n = len(regime_nl)
    regime_nl = np.asarray(regime_nl, dtype=np.int8)
    retained_idx = np.asarray(retained_idx, dtype=np.int64)
    regime_full = np.asarray(regime_full, dtype=np.int8)

    # Current-regime dummies (LOW is reference)
    R_elev = (regime_nl == 1).astype(np.float64)
    R_extr = (regime_nl == 2).astype(np.float64)

    # Lagged-regime dummies using calendar-time lag
    # R_{t-1} = regime at the bar immediately before each retained window end
    R1_raw = np.where(
        retained_idx > 0,
        regime_full[np.maximum(retained_idx - 1, 0)],
        np.int8(-1),  # boundary: first bar has no predecessor
    )
    R1_elev = (R1_raw == 1).astype(np.float64)
    R1_extr = (R1_raw == 2).astype(np.float64)

    # Transition indicator T_t = 1[R_t != R_{t-1}]
    # Boundary: if retained_idx[i] == 0, no valid R_{t-1} -> T_t = 0
    valid_lag = retained_idx > 0
    T_t = np.where(
        valid_lag,
        (regime_nl != R1_raw).astype(np.float64),
        0.0,
    )

    const = np.ones(n, dtype=np.float64)
    X_level = np.column_stack([const, R_elev, R_extr, R1_elev, R1_extr])
    X_trans = np.column_stack([const, R_elev, R_extr, R1_elev, R1_extr, T_t])

    col_T = 5
    ncols = 6
    column_names = ["const", "R_elev", "R_extr", "R1_elev", "R1_extr", "T_t"]

    meta = {
        "ncols": ncols,
        "col_T": col_T,
        "column_names": column_names,
        "X_level": X_level,
    }
    return X_trans, meta


# ---------------------------------------------------------------------------
# Degeneracy guard
# ---------------------------------------------------------------------------


def check_degeneracy(X: np.ndarray) -> dict:
    """Check design matrix for rank deficiency or severe ill-conditioning.

    Declares X degenerate if:
      - np.linalg.matrix_rank(X) < X.shape[1]  (rank deficiency), OR
      - np.linalg.cond(X) > 1e10               (frozen §4 ill-conditioning cutoff)

    Parameters
    ----------
    X : np.ndarray, shape (n, ncols)

    Returns
    -------
    dict with keys:
      'rank'       : int   — matrix rank
      'cond'       : float — condition number
      'degenerate' : bool  — True if rank < ncols OR cond > 1e10
    """
    rank = int(np.linalg.matrix_rank(X))
    cond = float(np.linalg.cond(X))
    # 1e10 is frozen (§4) — do not change
    degenerate = bool(rank < X.shape[1] or cond > 1e10)
    return {"rank": rank, "cond": cond, "degenerate": degenerate}


# ---------------------------------------------------------------------------
# HAC-robust Wald tests
# ---------------------------------------------------------------------------


def run_wald_hac(
    y: np.ndarray,
    X: np.ndarray,
    col_T: int,
    L: int,
) -> dict:
    """HAC-robust Wald chi2(1) on the T_t coefficient.

    Fits sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': L,
    'use_correction': True}), builds restriction matrix R with R[0, col_T]=1,
    runs result.wald_test(R, scalar=True).

    Parameters
    ----------
    y : np.ndarray, shape (n,)
        Non-overlapping DV values (|VR(q)-1|).
    X : np.ndarray, shape (n, ncols)
        TRANSITION design matrix (including T_t at col_T).
    col_T : int
        Column index of T_t in X.
    L : int
        Newey-West maxlags.

    Returns
    -------
    dict with keys: statistic, pvalue, df, distribution, beta_T, L

    Notes
    -----
    CRITICAL: reads df from wt.df_denom — NOT wt.df, which does not exist
    in statsmodels 0.14.6 (raises AttributeError). [RESEARCH Pitfall 1]
    """
    import statsmodels.api as sm  # noqa: PLC0415 — deferred import

    result_hac = sm.OLS(y, X).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": L, "use_correction": True},
    )
    ncols = X.shape[1]
    R = np.zeros((1, ncols))
    R[0, col_T] = 1.0
    wt = result_hac.wald_test(R, scalar=True)
    return {
        "statistic": float(wt.statistic),
        "pvalue": float(wt.pvalue),
        "df": int(wt.df_denom),  # .df does not exist; use .df_denom
        "distribution": wt.distribution,  # 'chi2' when scalar=True
        "beta_T": float(result_hac.params[col_T]),
        "L": L,
    }


def run_wald_hac_joint(
    y: np.ndarray,
    X: np.ndarray,
    col_E: int,
    col_D: int,
    L: int,
) -> dict:
    """§4 fallback: joint HAC-robust Wald chi2(2) on E_t and D_t coefficients.

    Used when the degeneracy guard trips and n_transitions >= 100 per direction.
    Tests the joint null H0: beta_E = 0 AND beta_D = 0 via a (2, ncols)
    restriction matrix.

    Parameters
    ----------
    y : np.ndarray, shape (n,)
    X : np.ndarray, shape (n, ncols)
        FALLBACK design matrix with E_t at col_E and D_t at col_D.
    col_E : int
        Column index of E_t (escalation indicator).
    col_D : int
        Column index of D_t (de-escalation indicator).
    L : int
        Newey-West maxlags.

    Returns
    -------
    dict with keys: statistic, pvalue, df (=2), distribution, beta_T (nan), L
    """
    import statsmodels.api as sm  # noqa: PLC0415 — deferred import

    result_hac = sm.OLS(y, X).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": L, "use_correction": True},
    )
    ncols = X.shape[1]
    R_joint = np.zeros((2, ncols))
    R_joint[0, col_E] = 1.0  # E_t coefficient = 0
    R_joint[1, col_D] = 1.0  # D_t coefficient = 0
    wt = result_hac.wald_test(R_joint, scalar=True)
    return {
        "statistic": float(wt.statistic),
        "pvalue": float(wt.pvalue),
        "df": int(wt.df_denom),  # should be 2 for chi2(2)
        "distribution": wt.distribution,
        "beta_T": float("nan"),  # joint test; individual betas not reported here
        "L": L,
    }


# ---------------------------------------------------------------------------
# Delta R²
# ---------------------------------------------------------------------------


def compute_delta_r2(
    y: np.ndarray,
    X_level: np.ndarray,
    X_trans: np.ndarray,
) -> float:
    """R²(TRANSITION) - R²(LEVEL) from two plain OLS fits.

    Uses plain .fit() (no cov_type) for R² extraction. The HAC fit is for
    the Wald test only; R² is identical between HAC and plain OLS fits
    (HAC only changes the covariance matrix, not point estimates). [RESEARCH Q2]
    Raw (not adjusted) R² — the frozen threshold ΔR² >= 0.005 is on raw R².

    Parameters
    ----------
    y : np.ndarray, shape (n,)
    X_level : np.ndarray, shape (n, ncols_level)   — LEVEL model (no T_t)
    X_trans : np.ndarray, shape (n, ncols_trans)   — TRANSITION model (with T_t)

    Returns
    -------
    float — delta R² (always >= 0 since TRANSITION nests LEVEL)
    """
    import statsmodels.api as sm  # noqa: PLC0415 — deferred import

    m_level = sm.OLS(y, X_level).fit()
    m_trans = sm.OLS(y, X_trans).fit()
    return float(m_trans.rsquared - m_level.rsquared)


# ---------------------------------------------------------------------------
# Block-bootstrap CI on beta_T
# ---------------------------------------------------------------------------


def beta_t_boot_ci(
    y_nl: np.ndarray,
    X_nl: np.ndarray,
    col_T: int,
    block: int,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 43,
) -> tuple[float, float, float]:
    """Circular block-bootstrap 95% CI on beta_T.

    Resamples (y_nl, X_nl) rows JOINTLY in circular blocks (same algorithm
    as epsilon_sq_boot_ci from regime_population.py) and refits sm.OLS each
    time to get beta_T from each resample.

    Parameters
    ----------
    y_nl : np.ndarray, shape (n,)
        Non-overlapping DV values.
    X_nl : np.ndarray, shape (n, ncols)
        TRANSITION design matrix rows corresponding to y_nl.
    col_T : int
        Column index of T_t in X_nl.
    block : int
        Block size in sample space (= W per D-06; NOT 10).
    n_boot : int
        Number of bootstrap resamples (default 2000).
    alpha : float
        Coverage level: CI covers 1-alpha (default 0.05 for 95% CI).
    seed : int
        RNG seed (BOOT_SEED=43 frozen D-06; never np.random.seed).

    Returns
    -------
    (point_estimate, lo, hi) : tuple[float, float, float]
        Returns (point, nan, nan) when n < 2*block or fewer than 10 finite boots.

    Notes
    -----
    Adapted verbatim from epsilon_sq_boot_ci (D-06); inner kernel replaced
    with sm.OLS(...).fit().params[col_T]. [RESEARCH Q3, PATTERNS.md]
    """
    import statsmodels.api as sm  # noqa: PLC0415 — deferred import

    y_nl = np.asarray(y_nl, dtype=np.float64)
    X_nl = np.asarray(X_nl, dtype=np.float64)
    n = len(y_nl)

    # Point estimate
    point = float(sm.OLS(y_nl, X_nl).fit().params[col_T])

    if n < block * 2:
        return point, float("nan"), float("nan")

    rng = np.random.default_rng(seed)  # mandatory: default_rng, never np.random.seed
    n_blocks = int(np.ceil(n / block))
    boots = np.empty(n_boot, dtype=np.float64)

    for b in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]) % n
        flat_idx = idx.ravel()[:n]
        y_b = y_nl[flat_idx]
        X_b = X_nl[flat_idx]
        try:
            coef = sm.OLS(y_b, X_b).fit().params[col_T]
            boots[b] = float(coef)
        except Exception:  # noqa: BLE001
            boots[b] = float("nan")

    boots = boots[np.isfinite(boots)]
    if len(boots) < 10:
        return point, float("nan"), float("nan")

    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    # Ensure CI contains the point estimate (clamp to lo <= point <= hi)
    lo = min(lo, point)
    hi = max(hi, point)
    return point, lo, hi


# ---------------------------------------------------------------------------
# Minimum-separation filter
# ---------------------------------------------------------------------------


def apply_min_sep_filter(
    regime: np.ndarray,
    retained_idx: np.ndarray,
    W: int,
) -> dict:
    """Apply minimum-separation filter to transition events.

    Two transition events are too close if separated by fewer than W bars
    (the DV window width). Events closer than W bars share overlapping
    |VR(q)-1| windows and contaminate one another (frozen §5).

    Retains, for each transition direction (UP: LOW→ELEVATED/EXTREME or
    ELEVATED→EXTREME; DOWN: vice versa), only those events whose post-
    transition window of length W does not overlap the next transition.

    Parameters
    ----------
    regime : np.ndarray
        Full-length regime label array (calendar bars). -1 = warmup.
    retained_idx : np.ndarray
        Non-overlapping sample indices (from non_overlapping_samples).
    W : int
        DV window width (also the minimum separation length, frozen §5).

    Returns
    -------
    dict with keys:
      'n_transitions_before'  : int   — total transition events before filter
      'n_UP_before'           : int   — UP events before filter
      'n_DOWN_before'         : int   — DOWN events before filter
      'n_UP_after'            : int   — UP events surviving filter
      'n_DOWN_after'          : int   — DOWN events surviving filter
      'n_transitions_after'   : int   — total events surviving filter
      'under_powered'         : bool  — True if < 50 survive any stratum (frozen §5)
      'n_transitions_UP'      : int   — alias for n_UP_after (for fallback gate)
      'n_transitions_DOWN'    : int   — alias for n_DOWN_after (for fallback gate)
    """
    regime = np.asarray(regime, dtype=np.int8)
    retained_idx = np.asarray(retained_idx, dtype=np.int64)

    # Identify transition events at retained indices
    # A transition at retained_idx[i] means regime[retained_idx[i]] != R_{t-1}
    # where R_{t-1} = regime[retained_idx[i] - 1] (boundary guard applied)
    up_events = []    # calendar indices of UP transitions
    down_events = []  # calendar indices of DOWN transitions

    for idx in retained_idx:
        if idx <= 0:
            continue
        r_curr = int(regime[idx])
        r_prev = int(regime[idx - 1])
        if r_curr < 0 or r_prev < 0:
            continue
        if r_curr > r_prev:
            up_events.append(int(idx))
        elif r_curr < r_prev:
            down_events.append(int(idx))

    n_UP_before = len(up_events)
    n_DOWN_before = len(down_events)
    n_before = n_UP_before + n_DOWN_before

    def _apply_min_sep(events: list[int], min_sep: int) -> list[int]:
        """Keep events separated by at least min_sep bars (greedy forward pass)."""
        if not events:
            return []
        events_sorted = sorted(events)
        kept = [events_sorted[0]]
        for ev in events_sorted[1:]:
            if ev - kept[-1] >= min_sep:
                kept.append(ev)
        return kept

    up_kept = _apply_min_sep(up_events, W)
    down_kept = _apply_min_sep(down_events, W)

    n_UP_after = len(up_kept)
    n_DOWN_after = len(down_kept)
    n_after = n_UP_after + n_DOWN_after

    # Under-powered: < 50 survivors in ANY stratum (frozen §5)
    under_powered = bool(n_UP_after < 50 or n_DOWN_after < 50)

    return {
        "n_transitions_before": n_before,
        "n_UP_before": n_UP_before,
        "n_DOWN_before": n_DOWN_before,
        "n_UP_after": n_UP_after,
        "n_DOWN_after": n_DOWN_after,
        "n_transitions_after": n_after,
        "under_powered": under_powered,
        "n_transitions_UP": n_UP_after,    # alias for fallback gate
        "n_transitions_DOWN": n_DOWN_after,  # alias for fallback gate
    }


# ---------------------------------------------------------------------------
# Holm-Bonferroni correction
# ---------------------------------------------------------------------------


def apply_holm(
    pvalues: list[float],
    family_size: int = 3,
) -> list[float]:
    """Holm-Bonferroni correction with frozen family_size=3 (§8).

    Returns adjusted p-values in INPUT order (not sorted order).

    Parameters
    ----------
    pvalues : list[float]
        Raw p-values (up to family_size members).
    family_size : int
        Holm denominator — hardcoded default 3 (frozen §8 for Pre-Check A).

    Returns
    -------
    list[float]
        Holm-adjusted p-values in the same order as input pvalues.

    Algorithm
    ---------
    1. Sort indices by raw p-value (ascending).
    2. For each rank r (0-based), adjusted = min(1.0, p * (family_size - r)).
    3. Enforce step-down monotonicity: adjusted[i] >= adjusted[i-1] in sorted order.
    4. Return in original input order.
    """
    k = family_size
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted = [0.0] * len(pvalues)
    for rank, idx in enumerate(order):
        adjusted[idx] = min(1.0, pvalues[idx] * (k - rank))
    # Enforce monotonicity: adjusted[order[i]] >= adjusted[order[i-1]] in sorted order
    for i in range(1, len(order)):
        adjusted[order[i]] = max(adjusted[order[i]], adjusted[order[i - 1]])
    return adjusted


# ---------------------------------------------------------------------------
# MDE (minimum detectable effect)
# ---------------------------------------------------------------------------


def compute_mde(
    n_nl: int,
    alpha: float = 0.05,
    power: float = 0.80,
) -> dict:
    """Analytic 80%-power MDE for chi2(1) Wald test (D-03).

    Computes the minimum detectable partial-R² at the given power and alpha
    for a chi2(1) Wald test with n_nl non-overlapping observations.

    Parameters
    ----------
    n_nl : int
        Number of non-overlapping observations.
    alpha : float
        Significance level (default 0.05).
    power : float
        Target power (default 0.80).

    Returns
    -------
    dict with keys:
      'n_nl'            : int
      'alpha'           : float
      'power_target'    : float
      'ncp_needed'      : float — non-centrality parameter for target power
      'mde_partial_r2'  : float — minimum detectable partial R² (NCP / (n + NCP))
      'crit_chi2_1'     : float — critical value chi2(1, 1-alpha)

    Notes
    -----
    Verified by live computation: at n_nl=20126, NCP=7.849, partial_R2=0.000390.
    The MDE (0.000390) is 13x below the frozen revival threshold (0.005). [RESEARCH Q4]
    """
    from scipy.stats import chi2, ncx2  # noqa: PLC0415 — deferred import
    from scipy.optimize import brentq    # noqa: PLC0415 — deferred import

    crit = float(chi2.ppf(1 - alpha, df=1))

    def _power_minus_target(nc: float) -> float:
        return float(1 - ncx2.cdf(crit, df=1, nc=nc)) - power

    ncp_needed = float(brentq(_power_minus_target, 0.001, 1000.0))
    mde_partial_r2 = float(ncp_needed / (n_nl + ncp_needed))

    return {
        "n_nl": int(n_nl),
        "alpha": float(alpha),
        "power_target": float(power),
        "ncp_needed": ncp_needed,
        "mde_partial_r2": mde_partial_r2,
        "crit_chi2_1": crit,
    }


# ---------------------------------------------------------------------------
# D-07 Disposition rule
# ---------------------------------------------------------------------------


def apply_disposition_rule(per_wq_results: list[dict]) -> dict:
    """D-07 disposition rule: A REVIVED iff primary (120,5) passes AND >= 2/3 cells pass.

    A cell passes the FULL CONJUNCTION iff:
      - Wald rejects (pvalue < 0.05, where pvalue is the Holm-adjusted value
        or raw pvalue — caller decides which to use; this function reads
        'cell_passes_full_conjunction' from each result dict), AND
      - ΔR² >= 0.005 (frozen §7), AND
      - |β_T| >= 0.01 (frozen §7)

    D-07 EXACTLY (frozen):
      A REVIVED iff:
        (a) primary (120,5) cell passes the full conjunction AND
        (b) >= 2 of the 3 grid cells each independently pass that same full conjunction.
      A KILLED otherwise.

    Parameters
    ----------
    per_wq_results : list[dict]
        One entry per (W,q) cell. Each dict must contain:
          - 'W'   : int
          - 'q'   : int
          - 'cell_passes_full_conjunction' : bool
            (True iff Wald rejects AND delta_r2 >= 0.005 AND abs(beta_T) >= 0.01)

    Returns
    -------
    dict with keys:
      'disposition'       : str  — 'A REVIVED' or 'A KILLED'
      'primary_passes'    : bool — whether primary (120,5) passes full conjunction
      'n_cells_passing'   : int  — how many of the 3 cells pass full conjunction
      'rule'              : str  — literal D-07 rule description

    Notes
    -----
    Frozen thresholds (0.005, 0.01) are consumed from the 'cell_passes_full_conjunction'
    field of each result, not re-evaluated here. This function only defines how
    the 3 cells combine. The caller is responsible for computing
    'cell_passes_full_conjunction' using those frozen thresholds.
    """
    # Identify the primary cell (120,5)
    primary_passes = False
    for cell in per_wq_results:
        if int(cell["W"]) == 120 and int(cell["q"]) == 5:
            primary_passes = bool(cell["cell_passes_full_conjunction"])
            break

    # Count cells passing the full conjunction
    n_cells_passing = sum(
        1 for cell in per_wq_results
        if bool(cell["cell_passes_full_conjunction"])
    )

    # D-07: both conditions must hold
    revived = primary_passes and (n_cells_passing >= 2)
    disposition = "A REVIVED" if revived else "A KILLED"

    return {
        "disposition": disposition,
        "primary_passes": primary_passes,
        "n_cells_passing": n_cells_passing,
        "rule": "D-07: primary (120,5) full conjunction AND >= 2/3 cells",
    }
