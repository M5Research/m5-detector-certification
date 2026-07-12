"""Causal HMM Markov-switching volatility-regime detector.

Implements the ROBUSTNESS detector specified in 01-PREREGISTRATION.md §7.2
(frozen at commit 169fc20) with the §14.5 HMM amendment bundle (D-02/D-03/D-06,
2026-06-02).

Frozen primary parameters:
  rv_window=60, k_regimes=3, em_iter=100, search_reps=10, ewma=False,
  switching_variance=True, trend="c", switching_trend=True

Key implementation notes:
  - Uses filtered_marginal_probabilities ONLY (Hamilton forward filter, §7.2).
    The smoother's counterpart is FORBIDDEN (look-ahead / Kim smoother violation).
  - Determinism: np.random.seed(PINNED_SEED + i) before each restart fit();
    random_seed= kwarg is non-functional in statsmodels 0.14.6 (RESEARCH.md PQ-1).
  - Warmup: leading NaN-RV bars are sliced, fit on finite tail, warmup bars
    reinserted as regime=-1 (D-04). No interior bar dropping.
  - log-domain floor: RV=0 bars floored to smallest strictly-positive in-sample
    RV before log transform (D-03); floor value recorded in §14.5 amendment.
  - 10 restarts via separate fit() calls (search_reps=0 each) to collect
    per-restart LLs for the §8b convergence trigger (RESEARCH.md PQ-5).
  - §8b convergence trigger: top-5 LL spread ≥ 1% of |mean LL| → 2-state fallback.

Phase boundary: pure computational unit. No gate, no predictability, no ε².
This module does NOT import gate_analysis, predictability, or regate_analysis.
"""

from __future__ import annotations

import gc
import logging
import warnings
from typing import Any

import numpy as np
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

from strategies.vol_regime_switch.realized_vol import compute_rv

logger = logging.getLogger(__name__)

PINNED_SEED: int = 42  # recorded in §14.5 amendment + 03-VALIDATION.md


class HMMDetector:
    """Causal HMM Markov-switching volatility-regime detector.

    Implements the frozen §7.2 robustness detector spec (01-PREREGISTRATION.md,
    commit 169fc20) with the §14.5 HMM amendment bundle (D-02/D-03/D-06,
    2026-06-02).

    Labels: -1 = warmup, 0 = LOW, 1 = ELEVATED, 2 = EXTREME
    Causality: Hamilton forward filter (filtered_marginal_probabilities ONLY).

    Parameters
    ----------
    rv_window : int
        Rolling-RMS window in bars (frozen primary: 60).
    k_regimes : int
        Number of regimes (frozen primary: 3; use 2 for fallback/variant).
    em_iter : int
        EM iterations per restart (frozen primary: 100; 200 on ConvergenceWarning).
    search_reps : int
        Number of independent restarts (frozen primary: 10). Each restart uses
        np.random.seed(PINNED_SEED + i) for determinism (D-07).
    ewma : bool
        If False (frozen primary), use simple-RMS RV via shared compute_rv().
        If True, use EWMA variant (secondary, §8).
    switching_variance : bool
        Allow each regime to have a distinct variance (frozen: True, §7.2).
    trend : str
        Trend specification per regime (frozen: "c" = constant, §7.2).
    switching_trend : bool
        Allow trend to switch across regimes (frozen: True, §7.2).
    """

    def __init__(
        self,
        rv_window: int = 60,
        k_regimes: int = 3,
        em_iter: int = 100,
        search_reps: int = 10,
        ewma: bool = False,
        switching_variance: bool = True,
        trend: str = "c",
        switching_trend: bool = True,
    ) -> None:
        self.rv_window = rv_window
        self.k_regimes = k_regimes
        self.em_iter = em_iter
        self.search_reps = search_reps
        self.ewma = ewma
        self.switching_variance = switching_variance
        self.trend = trend
        self.switching_trend = switching_trend

        # Attributes set after fit()
        self.rv_: np.ndarray = np.array([], dtype=np.float64)
        self.log_rv_: np.ndarray = np.array([], dtype=np.float64)
        self.filtered_probs_: np.ndarray = np.array([], dtype=np.float64)
        self.fitted_params_: np.ndarray = np.array([], dtype=np.float64)
        self.llf_: float = float("nan")
        self.convergence_ok_: bool = False
        self.hmm_fallback_: str = "none"
        self.floored_bar_count_: int = 0
        # Variance-separation diagnostics (D-10)
        self.sigma2_sorted_: np.ndarray = np.array([], dtype=np.float64)
        self.ratio_elev_low_: float = float("nan")
        self.ratio_ext_elev_: float = float("nan")
        # §8b convergence diagnostics — top-5 LL spread (D-07 / PATTERNS D-13)
        self.top5_spread_: float = float("nan")
        self.spread_fraction_: float = float("nan")

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

        # D-13: fail-loud on NaN / Inf
        if np.any(np.isnan(close)) or np.any(np.isinf(close)):
            raise ValueError("Input 'close' array contains NaN or Inf values.")

        # D-13: fail-loud on non-positive prices
        if np.any(close <= 0.0):
            raise ValueError(
                "Input 'close' array contains non-positive prices (close <= 0)."
            )

        # Step 1: RV via shared helper (D-05 — byte-identical to rolling_quantile_detector)
        rv = compute_rv(close, rv_window=self.rv_window, ewma=self.ewma)

        # Step 2: log-RV endog with floor (D-03/D-04)
        # Slice leading NaN-RV warmup bars; no interior dropping
        warmup_len = int(np.sum(np.isnan(rv)))
        rv_finite = rv[warmup_len:]

        # Minimal log-domain floor (D-03): smallest strictly-positive in-sample RV
        positive_rv = rv_finite[rv_finite > 0]
        floor_value = float(positive_rv.min()) if len(positive_rv) > 0 else 1e-10
        floored_count = int(np.sum(rv_finite == 0))
        rv_floored = np.where(rv_finite > 0, rv_finite, floor_value)
        log_rv = np.log(rv_floored)

        # Step 3: 10 restarts (search_reps=0 each) to collect per-restart LLs (§8b / PQ-5)
        # GC collect before the restart loop to free any fragmented memory (important for
        # large series where each restart allocates ~k*k*T float64 arrays).
        gc.collect()
        ll_values: list[float] = []
        conv_flags: list[bool] = []
        results_list: list[Any | None] = []

        for i in range(self.search_reps):
            np.random.seed(PINNED_SEED + i)  # D-07: NOT random_seed= kwarg (non-functional)
            mod_i = MarkovRegression(
                log_rv,
                k_regimes=self.k_regimes,
                trend=self.trend,
                switching_variance=self.switching_variance,
                switching_trend=self.switching_trend,
            )
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    res_i = mod_i.fit(em_iter=self.em_iter, search_reps=0, disp=0)
                    conv_i = any(
                        issubclass(w.category, ConvergenceWarning) for w in caught
                    )
            except Exception as exc:
                # Numerically degenerate restart (e.g. NaN params, SVD failure, MemoryError) —
                # skip and mark as unconverged (contributes to convergence_ok=False).
                logger.debug("Restart %d skipped due to numerical error: %s", i, exc)
                conv_i = True  # treat as convergence failure for §8b spread
                # Skip this restart — do not append to results_list
                # Mark with a sentinel LL of -inf so it ranks last in argmax
                ll_values.append(-float("inf"))
                conv_flags.append(conv_i)
                results_list.append(None)
                # GC collect to reclaim any memory fragmented by a failed restart
                # (important when MemoryError occurs — gives allocator a chance to coalesce)
                gc.collect()
                continue

            # §8b retry rule: ConvergenceWarning → retry at em_iter=200
            if conv_i and self.em_iter < 200:
                np.random.seed(PINNED_SEED + i)
                mod_r = MarkovRegression(
                    log_rv,
                    k_regimes=self.k_regimes,
                    trend=self.trend,
                    switching_variance=self.switching_variance,
                    switching_trend=self.switching_trend,
                )
                try:
                    with warnings.catch_warnings(record=True) as caught2:
                        warnings.simplefilter("always")
                        res_i = mod_r.fit(em_iter=200, search_reps=0, disp=0)
                        conv_i = any(
                            issubclass(w.category, ConvergenceWarning) for w in caught2
                        )
                except Exception:
                    conv_i = True  # retry also failed — mark as failed

            ll_values.append(res_i.llf)
            conv_flags.append(conv_i)
            results_list.append(res_i)

        # Step 4: §8b convergence trigger — top-5 LL spread
        ll_arr = np.array(ll_values)
        sorted_ll = np.sort(ll_arr)[::-1]  # descending (may contain -inf for failed restarts)
        # Use top-5 if ≥5 restarts; otherwise use the full set (top-n spread)
        n_for_spread = min(5, len(sorted_ll))
        if n_for_spread < 2:
            top5_spread = 0.0
        else:
            top5_spread = float(sorted_ll[0] - sorted_ll[n_for_spread - 1])
        mean_ll_abs = float(abs(np.mean(ll_arr[np.isfinite(ll_arr)]))) if np.any(np.isfinite(ll_arr)) else float("inf")
        spread_frac = top5_spread / mean_ll_abs if mean_ll_abs > 0 else float("inf")
        any_persistent_conv = any(conv_flags)
        convergence_ok = (spread_frac < 0.01) and not any_persistent_conv

        # Best restart = highest LL among successful (non-None) restarts
        best_idx = int(np.argmax(ll_arr))
        best_res = results_list[best_idx]
        if best_res is None:
            # All restarts failed — fall back to any successful one
            for _j, r in enumerate(results_list):
                if r is not None:
                    best_res = r
                    break
            if best_res is None:
                raise RuntimeError(
                    "All HMM EM restarts failed due to numerical errors. "
                    "The input series may be too short, flat, or degenerate."
                )

        # Step 5: runtime hasattr assertion (§7.2 — required per statsmodels stability note)
        assert hasattr(best_res, "filtered_marginal_probabilities"), (
            "statsmodels API: filtered_marginal_probabilities not found on result object. "
            "statsmodels has flagged this module as 'not guaranteed stable' (§7.2)."
        )

        # Step 6: variance-ascending relabeling (RESEARCH.md PQ-6 / D-08)
        param_names = best_res.model.param_names
        sigma2_idx = [j for j, n in enumerate(param_names) if n.startswith("sigma2")]
        sigmas = best_res.params[sigma2_idx]
        order = np.argsort(sigmas)  # ascending: col 0 = lowest var = LOW

        # GC collect before large post-fit array operations to reduce fragmentation.
        # After 10 restart fits, memory may be fragmented; argmax on 2.4M-bar array
        # needs a contiguous allocation that may fail without GC.
        gc.collect()

        # Use filtered (NEVER smoothed — FORBIDDEN per §7.2)
        fmp = best_res.filtered_marginal_probabilities  # shape (N-warmup, k_regimes)
        fmp_sorted = fmp[:, order]  # reorder: col 0=LOW, col 1=ELEVATED, col 2=EXTREME
        labels_finite = np.argmax(fmp_sorted, axis=1).astype(np.int8)

        # Variance-separation diagnostics (D-10)
        sorted_sigmas = sigmas[order]
        if self.k_regimes >= 2 and sorted_sigmas[0] > 0:
            ratio_elev_low = float(sorted_sigmas[1] / sorted_sigmas[0])
        else:
            ratio_elev_low = float("nan")
        if self.k_regimes >= 3 and sorted_sigmas[1] > 0:
            ratio_ext_elev = float(sorted_sigmas[2] / sorted_sigmas[1])
        else:
            ratio_ext_elev = float("nan")

        # Step 7: reinsert warmup as -1 (D-04 — no interior dropping)
        regime = np.full(len(close), -1, dtype=np.int8)
        regime[warmup_len:] = labels_finite

        # Expose intermediates for validation (mirrors rolling_quantile_detector lines 143-147)
        self.rv_ = rv
        self.log_rv_ = np.full(len(close), np.nan)
        self.log_rv_[warmup_len:] = log_rv
        self.filtered_probs_ = np.full((len(close), self.k_regimes), np.nan)
        self.filtered_probs_[warmup_len:] = fmp_sorted
        self.fitted_params_ = best_res.params.copy()
        self.llf_ = float(best_res.llf)
        self.convergence_ok_ = convergence_ok
        self.hmm_fallback_ = "none"
        self.floored_bar_count_ = floored_count
        self.sigma2_sorted_ = sorted_sigmas.copy()
        self.ratio_elev_low_ = ratio_elev_low
        self.ratio_ext_elev_ = ratio_ext_elev
        self.top5_spread_ = top5_spread
        self.spread_fraction_ = spread_frac

        return regime
