"""Shared realized-volatility helper for the VolRegime-Engine detectors.

Provides the canonical causal RV recipe (rolling RMS of log-returns) used by
BOTH the rolling-quantile detector (primary, §7.1) and the HMM detector
(robustness, §7.2) so that their endog series are byte-identical — exactly
what the Phase-4 cross-detector robustness comparison must isolate as the
ONLY difference between the two labeling methods.

Pre-registration reference:
  01-PREREGISTRATION.md §7.2 — "same RV construction as the rolling-quantile
  detector" (D-05 mandate). Freeze commit: 169fc20.

Key implementation notes:
  - r[0] = 0 by np.diff(prepend=log_c[0]) convention; the first log-return is
    defined as zero because there is no prior bar to difference against.
  - Simple-RMS path (ewma=False, frozen primary): first rv_window-1 bars are NaN
    (59 warmup bars for the default rv_window=60). This matches the warmup
    behavior of RollingQuantileDetector._compute_rv verbatim.
  - EWMA path (ewma=True, secondary variant §8): polars ewm_mean(alpha=
    2/(rv_window+1), adjust=False, min_samples=1) — no warmup NaN.

Phase boundary (D-10/scope fence):
  This module is a pure numerical helper. It does NOT import and MUST NOT import
  any gate, forecast, or predictability module (gate_analysis, predictability,
  regate_analysis, etc.). Phase 4 owns gate execution.
"""

from __future__ import annotations

import numpy as np
import polars as pl


def compute_rv(
    close: np.ndarray,
    rv_window: int = 60,
    ewma: bool = False,
) -> np.ndarray:
    """Compute causal realized volatility (rolling RMS of log-returns).

    Shared by RollingQuantileDetector and HMMDetector to guarantee byte-identical
    RV (D-05). Frozen primary: rv_window=60, ewma=False (simple RMS).

    Parameters
    ----------
    close : np.ndarray
        Price series (1-D, positive, finite). Caller is responsible for
        D-13 NaN/Inf/≤0 validation before calling this function.
    rv_window : int
        Rolling-RMS window in bars (frozen primary: 60).
    ewma : bool
        If False (frozen primary), use cumsum rolling-RMS recipe.
        If True (secondary variant §8), use polars EWMA with
        alpha = 2/(rv_window+1).

    Returns
    -------
    rv : np.ndarray
        Float64 RV array of length len(close).
        ewma=False: first rv_window-1 bars are NaN (59 warmup bars for
            the default rv_window=60). r[0] = 0 by prepend convention.
        ewma=True: no warmup NaN (all bars valid from bar 0).
    """
    log_c = np.log(close)
    r = np.diff(log_c, prepend=log_c[0])  # r[0] = 0 by convention
    r2 = r ** 2

    if ewma:
        # Secondary variant (§8): polars ewm_mean for causal EWMA vol
        alpha = 2.0 / (rv_window + 1)
        rv_sq = (
            pl.Series(r2)
            .ewm_mean(alpha=alpha, adjust=False, min_samples=1)
            .to_numpy()
        )
        return np.sqrt(rv_sq)
    else:
        # Primary path: cumsum rolling-RMS (Pattern 1 / STACK.md)
        cumr2 = np.cumsum(np.concatenate([[0.0], r2]))
        rv = np.full(len(close), np.nan)
        rv[rv_window - 1:] = np.sqrt(
            (cumr2[rv_window:] - cumr2[:-rv_window]) / rv_window
        )
        return rv
