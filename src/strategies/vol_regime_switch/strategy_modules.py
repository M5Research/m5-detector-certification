"""Volatility Regime Switch - Layer 2 Strategy Modules and State Machine."""

import logging

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from strategies.vol_regime_switch.defaults import DEFAULT_STRATEGY_CONFIG
from strategies.vol_regime_switch.regime_detector import detect_regime, rolling_sum

logger = logging.getLogger(__name__)


def rolling_vwap(close: np.ndarray, volume: np.ndarray, window: int = 1440) -> np.ndarray:
    """
    Compute causal rolling Volume Weighted Average Price (VWAP).
    
    If data length is less than window size, returns all NaNs.
    If cumulative volume over the window is zero, returns NaN to prevent division instability.
    """
    close = np.asarray(close, dtype=np.float64)
    volume = np.asarray(volume, dtype=np.float64)
    
    n = len(close)
    if n < window:
        return np.full(n, np.nan, dtype=np.float64)
        
    pv = close * volume
    sum_pv = rolling_sum(pv, window)
    sum_v = rolling_sum(volume, window)
    
    vwap = np.full(n, np.nan, dtype=np.float64)
    valid_volume = (~np.isnan(sum_v)) & (sum_v > 0.0)
    vwap[valid_volume] = sum_pv[valid_volume] / sum_v[valid_volume]
    
    return vwap


def donchian_channels(high: np.ndarray, low: np.ndarray, window: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute causal Donchian Channels (highest high and lowest low of last W bars).
    
    If data length is less than window size, returns all NaNs.
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    
    n = len(high)
    if n < window:
        upper = np.full(n, np.nan, dtype=np.float64)
        lower = np.full(n, np.nan, dtype=np.float64)
        return upper, lower
        
    high_view = sliding_window_view(high, window_shape=window)
    low_view = sliding_window_view(low, window_shape=window)
    
    upper = np.concatenate([np.full(window - 1, np.nan), np.max(high_view, axis=-1)])
    lower = np.concatenate([np.full(window - 1, np.nan), np.min(low_view, axis=-1)])
    
    return upper, lower


def generate_raw_signals(data: dict, config: dict | None = None) -> np.ndarray:
    """
    Generate raw trading signals (-1.0=short, 0.0=flat, 1.0=long) using regime switching.
    
    Parameters
    ----------
    data : dict
        Must contain keys 'close', 'high', 'low', 'volume' (1D numpy arrays).
    config : dict, optional
        Keys:
        - 'fast_window': int, default 20
        - 'slow_window': int, default 100
        - 'low_vol_threshold': float, default 1.2
        - 'extreme_vol_threshold': float, default 2.5
        - 'vwap_window': int, default 1440
        - 'vwap_entry': float, default 0.02
        - 'vwap_exit': float, default 0.002
        - 'donchian_window': int, default 20
    Returns
    -------
    signals : np.ndarray of float64
        Same length as close.
    """
    # 1. Validation
    required_keys = ["close", "high", "low", "volume"]
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Input dictionary 'data' must contain key '{key}'.")
        if data[key] is None:
            raise ValueError(f"'{key}' data cannot be None.")
            
    close = np.asarray(data["close"])
    high = np.asarray(data["high"])
    low = np.asarray(data["low"])
    volume = np.asarray(data["volume"])
    
    n = len(close)
    if n == 0:
        return np.array([], dtype=np.float64)
        
    if close.ndim != 1 or high.ndim != 1 or low.ndim != 1 or volume.ndim != 1:
        raise ValueError("All inputs must be 1D arrays.")
        
    if not (len(high) == n and len(low) == n and len(volume) == n):
        raise ValueError("All input arrays must have the same length.")
        
    # Check for NaN / Inf in any of the input arrays
    for key, arr in zip(required_keys, [close, high, low, volume], strict=True):
        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
            raise ValueError(f"Input '{key}' array contains NaN or Inf values.")
            
    # Clamping negative / non-positive prices
    if np.any(close <= 0.0) or np.any(high <= 0.0) or np.any(low <= 0.0):
        logger.warning("Non-positive prices detected in inputs. Clamping to 1e-12.")
        close = np.clip(close, 1e-12, None)
        high = np.clip(high, 1e-12, None)
        low = np.clip(low, 1e-12, None)
    else:
        close = close.astype(np.float64)
        high = high.astype(np.float64)
        low = low.astype(np.float64)
        
    if np.any(volume < 0.0):
        logger.warning("Negative volume detected. Clamping to 0.0.")
        volume = np.clip(volume, 0.0, None)
    else:
        volume = volume.astype(np.float64)
        
    # 2. Config parsing
    if config is None:
        config = DEFAULT_STRATEGY_CONFIG
        
    slow_window = int(config.get("slow_window", 100))
    vwap_window = int(config.get("vwap_window", 1440))
    vwap_exit = config.get("vwap_exit", 0.002)
    donchian_window = int(config.get("donchian_window", 20))
    stop_loss_pct = config.get("stop_loss_pct", 0.015)
    take_profit_pct = config.get("take_profit_pct", 0.03)
    
    # 3. Feature Calculations
    regime = detect_regime({"close": close}, config)
    vwap = rolling_vwap(close, volume, vwap_window)
    
    # Compute deviation from VWAP safely
    deviation = np.full_like(close, np.nan)
    valid_vwap = (~np.isnan(vwap)) & (vwap > 0.0)
    deviation[valid_vwap] = close[valid_vwap] / vwap[valid_vwap] - 1.0
    
    donchian_high, donchian_low = donchian_channels(high, low, donchian_window)
    
    # Calculate standard deviation of deviation from VWAP over vwap_window bars
    from strategies.vol_regime_switch.regime_detector import rolling_std_from_returns
    clean_deviation = np.nan_to_num(deviation, nan=0.0)
    dev_std = rolling_std_from_returns(clean_deviation, vwap_window)
    
    # Calculate dynamic Z-Score
    z_score = np.full_like(deviation, np.nan)
    valid_dev_std = (~np.isnan(dev_std)) & (~np.isnan(deviation))
    z_score[valid_dev_std] = deviation[valid_dev_std] / np.maximum(dev_std[valid_dev_std], 1e-12)
    
    # 4. State Machine Loop
    signals = np.zeros(n, dtype=np.float64)
    pos = 0.0
    entry_price = 0.0
    
    for t in range(n):
        current_regime = regime[t]
        
        # (Transition handler removed to prevent regime wiggles from cutting active trades)
            
        desired_raw = 0.0
        
        if current_regime == -1 or current_regime == 2:
            # Warmup (-1) or EXTREME_VOL (2) -> Always Flat
            desired_raw = 0.0
            
        elif current_regime == 0:
            # LOW_VOL -> Module A: Conditional VWAP Mean Reversion
            # Warmup check: requires vwap_window data and valid Z-score
            if t < vwap_window or np.isnan(deviation[t]) or np.isnan(z_score[t]):
                desired_raw = 0.0
            else:
                dev = deviation[t]
                zs = z_score[t]
                if pos == 0.0:
                    if zs < -2.0:
                        desired_raw = 1.0
                    elif zs > 2.0:
                        desired_raw = -1.0
                    else:
                        desired_raw = 0.0
                elif pos > 0.0:
                    # Check exit criteria
                    stop_triggered = (close[t] / entry_price - 1.0) < -stop_loss_pct
                    tp_triggered = (close[t] / entry_price - 1.0) > take_profit_pct
                    if stop_triggered or tp_triggered or dev > -vwap_exit:
                        # Exit long. Can we flip immediately to short?
                        if zs > 2.0:
                            desired_raw = -1.0
                        else:
                            desired_raw = 0.0
                    else:
                        desired_raw = 1.0
                else:  # pos < 0.0
                    # Check exit criteria
                    stop_triggered = (close[t] / entry_price - 1.0) > stop_loss_pct
                    tp_triggered = (close[t] / entry_price - 1.0) < -take_profit_pct
                    if stop_triggered or tp_triggered or dev < vwap_exit:
                        # Exit short. Can we flip immediately to long?
                        if zs < -2.0:
                            desired_raw = 1.0
                        else:
                            desired_raw = 0.0
                    else:
                        desired_raw = -1.0
                        
        else:
            # HIGH_VOL -> Module B: Donchian Channel Breakout
            # Warmup check: requires slow_window data to evaluate high_vol regime
            if t < slow_window:
                desired_raw = 0.0
            else:
                upper_prev = donchian_high[t - 1]
                lower_prev = donchian_low[t - 1]
                
                if np.isnan(upper_prev) or np.isnan(lower_prev):
                    desired_raw = 0.0
                else:
                    long_break = close[t] > upper_prev
                    short_break = close[t] < lower_prev
                    
                    if pos == 0.0:
                        if long_break:
                            desired_raw = 1.0
                        elif short_break:
                            desired_raw = -1.0
                        else:
                            desired_raw = 0.0
                    elif pos > 0.0:
                        # Reverse to short on opposite breakout, else hold
                        desired_raw = -1.0 if short_break else 1.0
                    else:  # pos < 0.0
                        # Reverse to long on opposite breakout, else hold
                        desired_raw = 1.0 if long_break else -1.0
                        
        signals[t] = desired_raw
        
        # Track position parameters
        if desired_raw != 0.0:
            if pos == 0.0 or (pos != desired_raw):  # Entry or flip
                entry_price = close[t]
        else:
            entry_price = 0.0
            
        pos = desired_raw
        
    return signals
