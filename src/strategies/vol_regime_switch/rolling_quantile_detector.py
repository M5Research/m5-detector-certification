"""Causal rolling-quantile volatility-regime detector.

Implements the PRIMARY detector specified in 01-PREREGISTRATION.md §7.1
(frozen at commit 169fc20) with the §14.5 amendment resolving the
warmup / min_samples contradiction (D-01).

Frozen primary parameters:
  rv_window=60, pct_window=43200, p_elevated=0.75, p_extreme=0.95,
  ewma=False, pct_mode="rolling"

Key implementation notes (D-01):
  - min_samples=None on rolling_quantile (= full window_size warmup)
  - interpolation="linear" (frozen spec)
  - First valid regime bar: rv_window + pct_window - 2
  - Polars treats NaN as valid float, so 59 leading NaN rv values propagate
    NaN through the quantile for 59 extra bars beyond pct_window - 1.
  - The validity mask `valid = ~isnan(rv) & ~isnan(q_el) & ~isnan(q_ex)`
    correctly implements D-01 full-window warmup.

Phase boundary (D-10): this module is a pure computational unit. It does
not import any gate or forecast modules. No VR / epsilon-squared is
computed here. Phase 4 owns gate execution.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


class RollingQuantileDetector:
    """Causal rolling-quantile volatility-regime detector.

    Implements the frozen §7.1 primary detector spec (01-PREREGISTRATION.md,
    commit 169fc20) with the §14.5 warmup amendment (D-01, 2026-06-02).

    Labels: -1 = warmup, 0 = LOW, 1 = ELEVATED, 2 = EXTREME
    Boundary: EXTREME = rv > q_extreme (strict >, per §7.1)

    Parameters
    ----------
    rv_window : int
        Rolling-RMS window in bars (frozen primary: 60).
    pct_window : int
        Rolling-percentile window in bars (frozen primary: 43200 = 30 days).
    p_elevated : float
        Percentile for the LOW|ELEVATED boundary (frozen primary: 0.75).
    p_extreme : float
        Percentile for the ELEVATED|EXTREME boundary (frozen primary: 0.95).
    ewma : bool
        If True, use EWMA vol estimator (alpha = 2/(rv_window+1)) instead
        of simple-RMS.  False = frozen primary; True = secondary variant (§8).
    pct_mode : str
        "rolling" (default, frozen primary) or "expanding" (secondary variant).
    """

    def __init__(
        self,
        rv_window: int = 60,
        pct_window: int = 43200,
        p_elevated: float = 0.75,
        p_extreme: float = 0.95,
        ewma: bool = False,
        pct_mode: str = "rolling",
    ) -> None:
        self.rv_window = rv_window
        self.pct_window = pct_window
        self.p_elevated = p_elevated
        self.p_extreme = p_extreme
        self.ewma = ewma
        self.pct_mode = pct_mode

        # Attributes set after fit()
        self.rv_: np.ndarray = np.array([], dtype=np.float64)
        self.q_elevated_: np.ndarray = np.array([], dtype=np.float64)
        self.q_extreme_: np.ndarray = np.array([], dtype=np.float64)

    def fit(self, close: np.ndarray) -> np.ndarray:
        """Compute causal int8 regime labels for the full close series.

        Parameters
        ----------
        close : array-like
            Price series (must be positive, finite, 1-D).

        Returns
        -------
        regime : np.ndarray[int8]
            Labels: -1 = warmup, 0 = LOW, 1 = ELEVATED, 2 = EXTREME.
            Length equals len(close).

        Raises
        ------
        ValueError
            On empty (length 0 returns [] not an error), NaN/Inf values,
            non-positive prices, or non-1-D input.
        """
        close = np.asarray(close, dtype=np.float64)

        if close.size == 0:
            return np.array([], dtype=np.int8)

        if close.ndim != 1:
            raise ValueError("Input 'close' must be a 1-D array.")

        # D-13: fail-loud on NaN / Inf (NO clamping, unlike v1.0 regime_detector.py)
        if np.any(np.isnan(close)) or np.any(np.isinf(close)):
            raise ValueError("Input 'close' array contains NaN or Inf values.")

        # D-13: fail-loud on non-positive prices
        if np.any(close <= 0.0):
            raise ValueError(
                "Input 'close' array contains non-positive prices (close <= 0)."
            )

        rv = self._compute_rv(close)
        q_el, q_ex = self._compute_bounds(rv)

        # D-14: degeneracy flag — q_elevated == 0 while rv > 0
        # (no epsilon-floor; frozen recipe kept exactly; flag only)
        degenerate = (
            (~np.isnan(q_el))
            & (q_el == 0.0)
            & (~np.isnan(rv))
            & (rv > 0.0)
        )
        if np.any(degenerate):
            n_degen = int(np.sum(degenerate))
            warnings.warn(
                f"D-14: q_elevated == 0 but rv > 0 at {n_degen} bar(s). "
                "The reference window may be pathologically flat. "
                "No epsilon-floor applied (frozen recipe).",
                stacklevel=2,
            )

        regime = self._label(rv, q_el, q_ex)

        # D-04: expose intermediates for validation
        self.rv_ = rv
        self.q_elevated_ = q_el
        self.q_extreme_ = q_ex

        return regime

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_rv(self, close: np.ndarray) -> np.ndarray:
        """Thin wrapper — delegates to shared compute_rv() helper (D-05)."""
        from strategies.vol_regime_switch.realized_vol import compute_rv  # noqa: PLC0415
        return compute_rv(close, rv_window=self.rv_window, ewma=self.ewma)

    def _compute_bounds(
        self, rv: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute rolling (or expanding) percentile bounds on the rv series.

        Rolling path (frozen primary, D-01):
          pl.Series(rv).fill_nan(None).rolling_quantile(p, interpolation="linear",
              window_size=pct_window, min_samples=None, center=False)

          fill_nan(None) converts NaN rv values (the rv warmup bars) to polars
          null.  This is required for D-01: polars rolling_quantile in 1.41.2
          treats NaN as a valid float and excludes it from the quantile but
          COUNTS it against window_size, meaning the actual warmup boundary
          depends on null propagation, not NaN propagation.  With NaN→null
          conversion, a window containing any null does not satisfy
          min_samples=None (= window_size), so quantile is null until the first
          window that contains ONLY non-null (i.e. non-NaN-rv) values.  The
          first such window ends at rv_window + pct_window - 2 (= the first bar
          where all pct_window values in the window have valid rv).

          After to_numpy(), polars null AND polars NaN both become NumPy NaN, so
          the validity mask ~np.isnan(q_el) correctly identifies all warmup bars.

          interpolation="linear" → frozen spec (NOT the polars default 'nearest').
          min_samples=None → equals window_size → full-window warmup (D-01).

        Expanding path (secondary variant, D-05):
          pd.Series(rv).expanding(min_periods=pct_window).quantile(p)
        """
        if self.pct_mode == "rolling":
            # NaN → null conversion ensures D-01 boundary at rv_window + pct_window - 2
            s = pl.Series(rv).fill_nan(None)
            # Pattern 2 (02-RESEARCH.md): frozen-spec call.
            # interpolation is the SECOND positional parameter in polars 1.41.2.
            # min_samples=None → window_size (D-01 full-window warmup).
            q_el = s.rolling_quantile(
                self.p_elevated,
                interpolation="linear",
                window_size=self.pct_window,
                min_samples=None,
                center=False,
            ).to_numpy()
            q_ex = s.rolling_quantile(
                self.p_extreme,
                interpolation="linear",
                window_size=self.pct_window,
                min_samples=None,
                center=False,
            ).to_numpy()
        else:
            # Expanding-window variant (secondary, D-05/PQ-5)
            import pandas as pd  # noqa: PLC0415

            rv_pd = pd.Series(rv)
            q_el = (
                rv_pd.expanding(min_periods=self.pct_window)
                .quantile(self.p_elevated)
                .values
            )
            q_ex = (
                rv_pd.expanding(min_periods=self.pct_window)
                .quantile(self.p_extreme)
                .values
            )

        return q_el, q_ex

    @staticmethod
    def _label(
        rv: np.ndarray, q_el: np.ndarray, q_ex: np.ndarray
    ) -> np.ndarray:
        """Assign int8 regime labels.

        valid mask implements D-01: a bar is valid only when all three
        intermediates are non-NaN.  This correctly handles the extra 59
        warmup bars caused by NaN rv values propagating through the quantile.

        Boundary (§7.1): EXTREME = rv > q_extreme (strict >).
        """
        valid = ~np.isnan(rv) & ~np.isnan(q_el) & ~np.isnan(q_ex)
        # Pattern 3 (02-RESEARCH.md / regime_detector.py line 164 idiom)
        regime = np.full(len(rv), -1, dtype=np.int8)
        regime[valid] = np.where(
            rv[valid] > q_ex[valid],
            2,
            np.where(rv[valid] > q_el[valid], 1, 0),
        ).astype(np.int8)
        return regime
