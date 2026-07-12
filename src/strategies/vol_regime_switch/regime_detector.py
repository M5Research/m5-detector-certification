"""Volatility Regime Switch - Layer 1 Regime Detection Module."""

import logging

import numba
import numpy as np
import polars as pl

from strategies.vol_regime_switch.defaults import DEFAULT_REGIME_CONFIG

logger = logging.getLogger(__name__)


def rolling_sum(x: np.ndarray, window: int) -> np.ndarray:
    """
    Compute causal rolling sum of a 1D array.
    
    If the array length is less than the window size, returns all NaNs.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if window <= 1:
        return x.copy()
    if n < window:
        return np.full(n, np.nan, dtype=np.float64)
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = c[window:] - c[:-window]
    pad = np.full(window - 1, np.nan, dtype=np.float64)
    return np.concatenate([pad, out])


def rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    """Compute causal rolling mean of a 1D array."""
    return rolling_sum(x, window) / float(window)


def rolling_std_from_returns(r: np.ndarray, window: int) -> np.ndarray:
    """
    Compute causal Root-Mean-Square (RMS) of returns.
    
    For high-frequency returns, mean subtraction is omitted to stabilize calculations.
    """
    r = np.asarray(r, dtype=np.float64)
    r2 = r * r
    mean_r2 = rolling_mean(r2, window)
    return np.sqrt(np.maximum(mean_r2, 0.0))




@numba.njit
def ema_jit(x: np.ndarray, window: int) -> np.ndarray:
    """Compute Exponential Moving Average (EMA) causally, ignoring NaNs in Numba."""
    n = len(x)
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return out
        
    first_idx = -1
    for i in range(n):
        if not np.isnan(x[i]):
            first_idx = i
            break
            
    if first_idx == -1:
        return out
        
    idx = first_idx
    out[idx] = x[idx]
    alpha = 2.0 / (window + 1)
    
    for i in range(idx + 1, n):
        if np.isnan(x[i]):
            out[i] = out[i - 1]
        else:
            out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1]
            
    return out


def detect_regime(data: dict, config: dict | None = None) -> np.ndarray:
    """
    Compute volatility regime labels (0=LOW, 1=HIGH, 2=EXTREME).
    
    Parameters
    ----------
    data : dict
        Must contain key 'close' (1D float64 numpy array or list).
    config : dict, optional
        Keys for static threshold classification:
        - 'fast_window': int, default 20
        - 'slow_window': int, default 100
        - 'low_vol_threshold': float, default 1.2
        - 'extreme_vol_threshold': float, default 2.5
        
        Keys for dynamic quantile classification:
        - 'p_low': float, e.g. 0.30
        - 'p_high': float, e.g. 0.90
        - 'quantile_window': int, default 43200
        - 'vr_smooth_window': int, default 20
        
    Returns
    -------
    regime : np.ndarray of int8
        Same length as close. Warmup is marked as -1.
    """
    if "close" not in data:
        raise ValueError("Input dictionary 'data' must contain key 'close'.")
    
    close_raw = data["close"]
    if close_raw is None:
        raise ValueError("'close' data cannot be None.")
        
    close = np.asarray(close_raw)
    
    # Handle empty input
    if close.size == 0:
        return np.array([], dtype=np.int8)
    
    # Validate close shape is 1D
    if close.ndim != 1:
        raise ValueError("'close' must be a 1D array.")
        
    # Check for NaN or Inf in close
    if np.any(np.isnan(close)) or np.any(np.isinf(close)):
        raise ValueError("Input 'close' array contains NaN or Inf values.")
        
    # Warn and clamp if close <= 0
    if np.any(close <= 0.0):
        logger.warning(
            "Non-positive prices detected in 'close'. Clamping to 1e-12."
        )
        close = np.clip(close, 1e-12, None)
    else:
        close = close.astype(np.float64)
        
    # Load configuration
    if config is None:
        config = DEFAULT_REGIME_CONFIG
        
    fast_window = int(config.get("fast_window", 20))
    slow_window = int(config.get("slow_window", 100))
    
    if fast_window < 1 or slow_window < 1:
        raise ValueError("Window sizes must be at least 1.")
    if fast_window >= slow_window:
        raise ValueError("fast_window must be less than slow_window.")
        
    # Compute log returns: r_t = log(close_t / close_{t-1}), with r_0 = 0.
    log_close = np.log(close)
    r = np.diff(log_close, prepend=log_close[0])
    
    # Realized volatilities (RMS)
    fast_vol = rolling_std_from_returns(r, fast_window)
    slow_vol = rolling_std_from_returns(r, slow_window)
    
    # Calculate VR = fast_vol / slow_vol
    # If slow_vol is zero or NaN, treat VR as NaN.
    vr = np.full_like(fast_vol, np.nan)
    valid_slow = (~np.isnan(slow_vol)) & (slow_vol > 0.0)
    vr[valid_slow] = fast_vol[valid_slow] / slow_vol[valid_slow]
    
    # Initialize regime array (default to Warmup = -1 for NaN VR)
    regime = np.full(len(close), -1, dtype=np.int8)
    
    p_low = config.get("p_low", None)
    p_high = config.get("p_high", None)
    
    if p_low is not None and p_high is not None:
        # Dynamic rolling quantile classification
        vr_smooth_window = int(config.get("vr_smooth_window", 20))
        vr_smoothed = ema_jit(vr, vr_smooth_window)
        
        quantile_window = int(config.get("quantile_window", 43200))
        
        # Calculate rolling quantiles using Polars Series
        s = pl.Series(vr_smoothed)
        q_low = s.rolling_quantile(p_low, window_size=quantile_window, min_samples=1).to_numpy()
        q_high = s.rolling_quantile(p_high, window_size=quantile_window, min_samples=1).to_numpy()
        
        valid_vr = ~np.isnan(vr_smoothed)
        regime[valid_vr & (vr_smoothed <= q_low)] = 0
        regime[valid_vr & (vr_smoothed > q_low) & (vr_smoothed <= q_high)] = 1
        regime[valid_vr & (vr_smoothed > q_high)] = 2
    else:
        # Fallback to static threshold classification
        low_vol_threshold = config.get("low_vol_threshold", 1.2)
        extreme_vol_threshold = config.get("extreme_vol_threshold", 2.5)
        
        valid_vr = ~np.isnan(vr)
        regime[valid_vr & (vr <= low_vol_threshold)] = 0
        regime[valid_vr & (vr > low_vol_threshold) & (vr <= extreme_vol_threshold)] = 1
        regime[valid_vr & (vr > extreme_vol_threshold)] = 2
        
    return regime

