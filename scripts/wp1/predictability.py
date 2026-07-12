"""Causal rolling VR-M2 predictability estimator.

Implements the Lo-MacKinlay (1988) Variance-Ratio test with the HAC-robust M2
asymptotic variance (1990-erratum-corrected delta(j)), matching the arch
library's VarianceRatio(robust=True, debiased=True) formula exactly.

The module is causal: each predictability_t value at index t uses only close
prices at indices <= t.

Headline DV:
    predictability_t = |VR(q) - 1|

Formula source (arch._compute_statistic convention):
- y: price series of length T (= W bars in the window)
- delta_y = diff(y), length nq = T - 1
- mu = (y[-1] - y[0]) / (T - 1)   (drift, 'c' trend)
- sigma2_1 = sum((delta_y - mu)^2) / nq * nq/(nq-1)   (debiased, ddof=1)
- sigma2_q = sum((y[q:] - y[:-q] - q*mu)^2) / (nq*q) * (nq*q)/m   (debiased)
  where m = q*(nq-q+1)*(1-q/nq)
- VR(q) = sigma2_q / sigma2_1
- M2 phi = sum_{k=1}^{q-1} 4*(1-k/q)^2 * nq * z2[k:] @ z2[:-k] / sum(z2)^2
  where z2 = (delta_y - mu)^2
  (1990-erratum T-multiplier: the 'nq *' in the delta term is the correction)

References
----------
- Lo, A.W. and MacKinlay, A.C. (1988). Review of Financial Studies.
- Lo, A.W. and MacKinlay, A.C. (1990). Journal of Econometrics.
- arch library source: arch.unitroot.unitroot.VarianceRatio._compute_statistic
"""
from __future__ import annotations

import numpy as np

try:
    import numba
except ImportError:  # pragma: no cover
    numba = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Core single-window kernel
# ---------------------------------------------------------------------------


def _vr_m2_kernel(
    r: np.ndarray, q: int
) -> tuple[float, float]:
    """Compute VR(q) and M2 z-statistic matching arch.VarianceRatio(robust=True).

    Parameters
    ----------
    r : np.ndarray
        1-D float64 array of log-returns, length W (the full rolling window).
        Internally reconstructs prices as exp(cumsum(r)) of length W, then
        applies arch's formula using price first-differences (delta_y = diff(prices)).
    q : int
        Aggregation horizon (integer >= 2).

    Returns
    -------
    (vr, z_m2) : tuple[float, float]
        Both NaN when the window is degenerate (too short, zero variance, phi<=0).

    Notes
    -----
    The reconstructed price series has length W, giving nq = W - 1 price
    increments.  This matches arch.VarianceRatio(prices_of_length_W, ...).
    """
    # Reconstruct price levels from log-returns
    # prices[i] = exp(r[0] + r[1] + ... + r[i])
    prices = np.exp(np.cumsum(r))
    nobs = len(prices)
    nq = nobs - 1  # number of price increments

    if nq < q + 1:
        return float("nan"), float("nan")

    # Drift estimate (arch 'c' trend default)
    mu = (prices[-1] - prices[0]) / (nobs - 1)

    # 1-period price increments
    delta_y = np.diff(prices)

    # 1-period variance (debiased, ddof=1)
    demeaned = delta_y - mu
    sigma2_1 = np.dot(demeaned, demeaned) / (nq - 1)
    if sigma2_1 == 0.0:
        return float("nan"), float("nan")

    # q-period overlapping price increments
    delta_y_q = prices[q:] - prices[:-q]

    # bias-correction factor m = q*(nq-q+1)*(1-q/nq)
    m = q * (nq - q + 1) * (1.0 - q / nq)
    if m == 0.0:
        return float("nan"), float("nan")

    # q-period variance (debiased)
    dev_q = delta_y_q - q * mu
    sigma2_q = np.dot(dev_q, dev_q) / m

    vr = sigma2_q / sigma2_1

    # M2 heteroskedasticity-robust asymptotic variance (1990-erratum T-correction)
    # phi = sum_{k=1}^{q-1} 4*(1-k/q)^2 * nq * z2[k:] @ z2[:-k] / sum(z2)^2
    z2 = demeaned ** 2
    scale = np.sum(z2) ** 2  # (sum of z2)^2  -- arch M2 normalization
    if scale == 0.0:
        return float("nan"), float("nan")

    phi = 0.0
    for k in range(1, q):
        delta_k = nq * (z2[k:] @ z2[:-k]) / scale
        phi += 4.0 * (1.0 - k / q) ** 2 * delta_k

    if phi <= 0.0:
        return float("nan"), float("nan")

    z_m2 = float(np.sqrt(nq)) * (vr - 1.0) / float(np.sqrt(phi))
    return float(vr), float(z_m2)


if numba is not None:

    @numba.njit(cache=True)
    def _vr_m2_kernel_njit(r: np.ndarray, q: int) -> tuple[float, float]:
        nobs = len(r)
        prices = np.empty(nobs, dtype=np.float64)
        cum = 0.0
        for i in range(nobs):
            cum += r[i]
            prices[i] = np.exp(cum)

        nq = nobs - 1
        if nq < q + 1:
            return np.nan, np.nan

        mu = (prices[-1] - prices[0]) / (nobs - 1)

        demeaned = np.empty(nq, dtype=np.float64)
        for i in range(nq):
            demeaned[i] = prices[i + 1] - prices[i] - mu

        sigma2_1 = 0.0
        for i in range(nq):
            sigma2_1 += demeaned[i] * demeaned[i]
        sigma2_1 /= nq - 1
        if sigma2_1 == 0.0:
            return np.nan, np.nan

        m = q * (nq - q + 1) * (1.0 - q / nq)
        if m == 0.0:
            return np.nan, np.nan

        dev_q = np.empty(nq - q + 1, dtype=np.float64)
        for i in range(len(dev_q)):
            dev_q[i] = prices[i + q] - prices[i] - q * mu

        sigma2_q = 0.0
        for i in range(len(dev_q)):
            sigma2_q += dev_q[i] * dev_q[i]
        sigma2_q /= m

        vr = sigma2_q / sigma2_1

        z2 = np.empty(nq, dtype=np.float64)
        scale = 0.0
        for i in range(nq):
            z2[i] = demeaned[i] * demeaned[i]
            scale += z2[i]
        scale = scale * scale
        if scale == 0.0:
            return np.nan, np.nan

        phi = 0.0
        for k in range(1, q):
            acc = 0.0
            for i in range(k, nq):
                acc += z2[i] * z2[i - k]
            delta_k = nq * acc / scale
            phi += 4.0 * (1.0 - k / q) ** 2 * delta_k

        if phi <= 0.0:
            return np.nan, np.nan

        z_m2 = np.sqrt(nq) * (vr - 1.0) / np.sqrt(phi)
        return vr, z_m2

    @numba.njit(cache=True)
    def _rolling_vr_m2_z_njit(
        r_full: np.ndarray,
        W: int,
        q: int,
        vr_arr: np.ndarray,
        z_arr: np.ndarray,
        stride: int,
    ) -> None:
        n = len(r_full)
        t = W - 1
        while t < n:
            r_window = r_full[t - W + 1 : t + 1]
            vr, z = _vr_m2_kernel_njit(r_window, q)
            if not np.isnan(vr):
                vr_arr[t] = abs(vr - 1.0)
                z_arr[t] = z
            t += stride


def rolling_vr_m2_z_arrays(
    r_full: np.ndarray, W: int, q: int, N: int, stride: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Causal rolling |VR-1| and z_m2 arrays (numba-accelerated when available).

    When stride > 1, only end-of-window bars at ``W-1, W-1+stride, ...`` are
    computed; intermediate bars remain NaN. This matches non_overlapping_samples
    (stride=W) without changing retained values.
    """
    vr_arr = np.full(N, np.nan, dtype=np.float64)
    z_arr = np.full(N, np.nan, dtype=np.float64)
    if N < W:
        return vr_arr, z_arr
    step = max(int(stride), 1)
    if numba is not None:
        _rolling_vr_m2_z_njit(r_full, W, q, vr_arr, z_arr, step)
    else:
        for t in range(W - 1, N, step):
            r_window = r_full[t - W + 1 : t + 1]
            vr, z = _vr_m2_kernel(r_window, q)
            if not np.isnan(vr):
                vr_arr[t] = abs(vr - 1.0)
                z_arr[t] = z
    return vr_arr, z_arr


# ---------------------------------------------------------------------------
# Confirmatory helpers
# ---------------------------------------------------------------------------


def ljungbox_in_window(r: np.ndarray, lags: int = 5) -> tuple[float, float]:
    """Ljung-Box Q statistic and p-value for log-returns in a window.

    Delegates to statsmodels.stats.diagnostic.acorr_ljungbox so results
    match the reference implementation within floating-point precision.

    Parameters
    ----------
    r : np.ndarray
        1-D array of log-returns.
    lags : int
        Number of lags (the test uses only the final lag).

    Returns
    -------
    (lb_stat, lb_pvalue) : tuple[float, float]
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox  # local import: statsmodels is optional
    result = acorr_ljungbox(r, lags=[lags], return_df=True)
    return float(result["lb_stat"].iloc[-1]), float(result["lb_pvalue"].iloc[-1])


def abs_autocorr(r: np.ndarray, max_lag: int = 5) -> float:
    """Mean absolute autocorrelation for lags 1..max_lag.

    Confirmatory helper alongside the VR-M2 headline DV.

    Parameters
    ----------
    r : np.ndarray
        1-D array of log-returns.
    max_lag : int
        Maximum lag to include.

    Returns
    -------
    float : mean |autocorrelation| across lags 1..max_lag, or NaN if degenerate.
    """
    n = len(r)
    if n < max_lag + 1:
        return float("nan")
    mu = r.mean()
    r_c = r - mu
    var = np.dot(r_c, r_c) / n
    if var == 0.0:
        return float("nan")
    ac_sum = 0.0
    for lag in range(1, max_lag + 1):
        ac = np.dot(r_c[lag:], r_c[:-lag]) / (n * var)
        ac_sum += abs(ac)
    return ac_sum / max_lag


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class VREstimator:
    """Causal rolling VR-M2 predictability estimator.

    Parameters
    ----------
    W : int
        Rolling window size in bars (must be >= q + 2).
    q : int
        Aggregation horizon in bars (must be >= 2).

    Public Methods
    --------------
    fit(close) -> np.ndarray
        Compute the rolling predictability series |VR(q) - 1|.
    _vr_m2(r) -> tuple[float, float]
        Single-window (VR, z_m2) for cross-validation tests.
        Accepts log-returns of length W; reconstructs price levels internally.
    """

    def __init__(self, W: int, q: int) -> None:
        if q < 2:
            raise ValueError(
                f"q must be >= 2; VR(1) is identically 1 (got q={q})."
            )
        if W < q + 2:
            raise ValueError(
                f"W must be >= q + 2 = {q + 2} to compute VR(q) (got W={W})."
            )
        self.W = W
        self.q = q

    def _vr_m2(self, r: np.ndarray) -> tuple[float, float]:
        """Single-window VR and M2 z-statistic on a length-W return array.

        Parameters
        ----------
        r : np.ndarray
            Log-returns of length W (the causal window ending at bar t).
            Prices are reconstructed as exp(cumsum(r)) of length W.

        Returns
        -------
        (vr, z_m2) : tuple[float, float]
        """
        return _vr_m2_kernel(r, self.q)

    def fit(self, close: np.ndarray) -> np.ndarray:
        """Compute rolling predictability_t = |VR(q) - 1|.

        Parameters
        ----------
        close : np.ndarray
            1-D array of close prices (must be positive).

        Returns
        -------
        pred : np.ndarray of float64, same length as close.
            Indices 0..W-2 are NaN (warmup); index W-1 onward is finite
            (or NaN if the window is degenerate).
        """
        return compute_rolling_predictability(close, self.W, self.q)


def compute_rolling_predictability(
    close: np.ndarray,
    W: int,
    q: int,
) -> np.ndarray:
    """Compute causal rolling predictability_t = |VR(q) - 1|.

    Each value at index t uses only close prices at indices [t-W+1 .. t]
    (a W-bar window ending at t).  Indices 0..W-2 are set to NaN (warmup).
    The formula matches arch.unitroot.VarianceRatio(robust=True, debiased=True)
    applied to the W-bar price window.

    Parameters
    ----------
    close : np.ndarray
        1-D array of close prices (positive, no NaN/Inf).
    W : int
        Rolling window size in bars (>= q + 2).
    q : int
        Aggregation horizon (>= 2).

    Returns
    -------
    pred : np.ndarray of float64, length N.
        NaN for warmup bars (indices 0..W-2) and degenerate windows.

    Raises
    ------
    ValueError
        If q < 2 or W < q + 2.
    """
    if q < 2:
        raise ValueError(
            f"q must be >= 2; VR(1) is identically 1 (got q={q})."
        )
    if W < q + 2:
        raise ValueError(
            f"W must be >= q + 2 = {q + 2} to compute VR(q) (got W={W})."
        )

    close = np.asarray(close, dtype=np.float64)
    N = len(close)

    if N < W:
        return np.full(N, np.nan, dtype=np.float64)

    log_close = np.log(close)
    r_full = np.empty(N, dtype=np.float64)
    r_full[0] = 0.0
    r_full[1:] = np.diff(log_close)

    pred, _ = rolling_vr_m2_z_arrays(r_full, W, q, N)
    return pred
