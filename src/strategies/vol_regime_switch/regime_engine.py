"""Volatility Regime Switch - Integrated Strategy Module for C++ Backtester."""

import logging

import numba
import numpy as np

from strategies.vol_regime_switch.defaults import DEFAULT_INTEGRATION_CONFIG
from strategies.vol_regime_switch.regime_detector import rolling_std_from_returns
from strategies.vol_regime_switch.strategy_modules import donchian_channels, rolling_vwap

logger = logging.getLogger(__name__)


@numba.njit
def ema(x: np.ndarray, window: int) -> np.ndarray:
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


@numba.njit
def apply_drawdown_lock_jit(
    proxy_equity: float,
    peak_equity: float,
    dd_locked: bool,
    dd_cooldown: int,
    dd_threshold: float,
    cooldown_bars: int,
) -> tuple[bool, int, float]:
    drawdown = 1.0 - (proxy_equity / max(peak_equity, 1e-12))
    
    if not dd_locked and drawdown > dd_threshold:
        dd_locked = True
        dd_cooldown = cooldown_bars
        
    if dd_locked:
        dd_cooldown = max(dd_cooldown - 1, 0)
        if dd_cooldown == 0:
            dd_locked = False
            peak_equity = proxy_equity
            
    if proxy_equity > peak_equity:
        peak_equity = proxy_equity
        
    return dd_locked, dd_cooldown, peak_equity


def apply_drawdown_lock(
    proxy_equity: float,
    peak_equity: float,
    dd_locked: bool,
    dd_cooldown: int,
    dd_threshold: float = 0.10,
    cooldown_bars: int = 500,
) -> tuple[bool, int, float]:
    """Apply strategy-level drawdown lock causally."""
    return apply_drawdown_lock_jit(
        proxy_equity=proxy_equity,
        peak_equity=peak_equity,
        dd_locked=dd_locked,
        dd_cooldown=dd_cooldown,
        dd_threshold=dd_threshold,
        cooldown_bars=cooldown_bars,
    )


def utc_hour_and_dow(timestamp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract UTC hour and Day-of-Week from C++ millisecond epoch timestamps safely.
    
    Converts directly via datetime64[ms] to prevent nanosecond scale overflows.
    """
    ts = np.asarray(timestamp, dtype=np.int64)
    ts_ms = ts.astype("datetime64[ms]")
    
    hour = (ts_ms.astype("datetime64[h]").astype(np.int64) % 24).astype(np.int8)
    day = ts_ms.astype("datetime64[D]").astype(np.int64)
    dow = ((day + 3) % 7).astype(np.int8)  # Monday=0, Sunday=6
    
    return hour, dow


@numba.njit
def trade_pnl_jit(entry_price: float, exit_price: float, side: float) -> float:
    if side == 0.0 or entry_price <= 0.0 or exit_price <= 0.0:
        return 0.0
    return side * (exit_price / entry_price - 1.0)


def trade_pnl(entry_price: float, exit_price: float, side: float) -> float:
    """Compute simple trade return percentage."""
    return trade_pnl_jit(entry_price, exit_price, side)


@numba.njit
def state_machine_loop(
    close: np.ndarray,
    open_: np.ndarray,
    deviation: np.ndarray,
    z_score: np.ndarray,
    donchian_high: np.ndarray,
    donchian_low: np.ndarray,
    meta_trade_allowed: np.ndarray,
    regime: np.ndarray,
    mr_regime: int,
    breakout_regime: int,
    vwap_window: int,
    slow_window: int,
    vwap_exit: float,
    stop_loss_pct: float,
    take_profit_pct: float,
    dd_threshold: float,
    cooldown_bars: int
) -> tuple[np.ndarray, np.ndarray]:
    n = len(close)
    signals = np.zeros(n, dtype=np.float64)
    limit_flags = np.zeros(n, dtype=np.bool_)
    
    pos = 0.0
    entry_price = 0.0
    entry_side = 0.0
    
    proxy_equity = 1.0
    peak_equity = 1.0
    dd_locked = False
    dd_cooldown = 0
    
    loss_streak = 0
    loss_cooldown = 0
    
    for t in range(n):
        # Update execution-aware proxy equity (Fixes Proxy Equity Drift)
        if t > 0:
            prev_held = signals[t - 2] if t > 1 else 0.0
            curr_held = signals[t - 1]
            
            if curr_held == prev_held:
                if curr_held != 0.0:
                    bar_ret = curr_held * (close[t] / max(close[t - 1], 1e-12) - 1.0)
                else:
                    bar_ret = 0.0
            else:
                # Position changed at open[t]
                gap_ret = prev_held * (open_[t] / max(close[t - 1], 1e-12) - 1.0)
                intraday_ret = curr_held * (close[t] / max(open_[t], 1e-12) - 1.0)
                bar_ret = gap_ret + intraday_ret
                
            proxy_equity *= (1.0 + bar_ret)
            
        if proxy_equity > peak_equity:
            peak_equity = proxy_equity
            
        # Drawdown lock calculation
        drawdown = 1.0 - (proxy_equity / max(peak_equity, 1e-12))
        
        current_regime = regime[t]
        d_limit = dd_threshold if current_regime == 2 else dd_threshold + 0.05
        
        if not dd_locked and drawdown >= d_limit:
            dd_locked = True
            dd_cooldown = cooldown_bars
            
        if dd_locked:
            dd_cooldown = max(dd_cooldown - 1, 0)
            if dd_cooldown == 0:
                dd_locked = False
                peak_equity = proxy_equity
                drawdown = 0.0
                
        if proxy_equity > peak_equity:
            peak_equity = proxy_equity
            
        if loss_cooldown > 0:
            loss_cooldown -= 1
            
        # Check session / weekend entry permission
        entry_allowed = meta_trade_allowed[t]
        is_risk_locked = dd_locked or (loss_cooldown > 0)
        
        # Under risk locks, we are forced flat
        if is_risk_locked:
            desired_raw = 0.0
        else:
            if current_regime == -1 or current_regime == 1:
                desired_raw = 0.0
                
            elif current_regime == mr_regime:
                if t < vwap_window or np.isnan(deviation[t]) or np.isnan(z_score[t]):
                    desired_raw = 0.0
                else:
                    dev = deviation[t]
                    zs = z_score[t]
                    if pos == 0.0:
                        if entry_allowed:
                            if zs < -2.0:
                                desired_raw = 1.0
                            elif zs > 2.0:
                                desired_raw = -1.0
                            else:
                                desired_raw = 0.0
                        else:
                            desired_raw = 0.0
                    elif pos > 0.0:
                        stop_triggered = (close[t] / entry_price - 1.0) < -stop_loss_pct
                        tp_triggered = (close[t] / entry_price - 1.0) > take_profit_pct
                        if stop_triggered or tp_triggered or dev > -vwap_exit:
                            if entry_allowed and zs > 2.0:
                                desired_raw = -1.0
                            else:
                                desired_raw = 0.0
                        else:
                            desired_raw = 1.0
                    else:  # pos < 0.0
                        stop_triggered = (close[t] / entry_price - 1.0) > stop_loss_pct
                        tp_triggered = (close[t] / entry_price - 1.0) < -take_profit_pct
                        if stop_triggered or tp_triggered or dev < vwap_exit:
                            if entry_allowed and zs < -2.0:
                                desired_raw = 1.0
                            else:
                                desired_raw = 0.0
                        else:
                            desired_raw = -1.0
            elif current_regime == breakout_regime:
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
                            if entry_allowed:
                                if long_break:
                                    desired_raw = 1.0
                                elif short_break:
                                    desired_raw = -1.0
                                else:
                                    desired_raw = 0.0
                            else:
                                desired_raw = 0.0
                        elif pos > 0.0:
                            if short_break:
                                if entry_allowed:
                                    desired_raw = -1.0
                                else:
                                    desired_raw = 0.0
                            else:
                                desired_raw = 1.0
                        else:  # pos < 0.0
                            if long_break:
                                if entry_allowed:
                                    desired_raw = 1.0
                                else:
                                    desired_raw = 0.0
                            else:
                                desired_raw = -1.0
            else:
                desired_raw = 0.0

        forced_exit = False
        if pos != 0.0 and desired_raw == 0.0:
            if is_risk_locked:
                forced_exit = True
                
        # Calculate dynamic position sizing modifiers
        limit_flags[t] = (current_regime == 0)
        
        m_t = 1.0
        if current_regime == 0:
            m_t = 1.11
        elif current_regime == 1:
            m_t = 1.05
        elif current_regime == 2:
            m_t = 0.71
            
        theta_t = 1.0
        if drawdown <= 0.08:
            theta_t = 1.0
        elif drawdown >= d_limit:
            theta_t = 0.0
        else:
            theta_t = np.exp(-10.0 * (drawdown - 0.08))
            
        signals[t] = desired_raw * m_t * theta_t
        
        # Trade accounting for loss-streak
        prev_pos = pos
        pos = desired_raw
        
        if desired_raw != 0.0:
            if prev_pos == 0.0 or (np.sign(desired_raw) != np.sign(prev_pos)):
                entry_price = close[t]
                entry_side = np.sign(desired_raw)
        else:
            if prev_pos != 0.0:
                pnl = trade_pnl_jit(entry_price, close[t], entry_side)
                if not forced_exit:
                    if pnl < 0.0:
                        loss_streak += 1
                    else:
                        loss_streak = 0
                        
                if loss_streak >= 5:
                    loss_cooldown = 100
                    loss_streak = 0
                    
            entry_price = 0.0
            entry_side = 0.0
            
    return signals, limit_flags


def generate_signals(data: dict, config: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Integrated Volatility Regime Switch Strategy Function.

    Exposes contract: generate_signals(data: dict) -> tuple[np.ndarray, np.ndarray]
    """
    # 1. Validation & Extraction
    required_keys = ["open", "high", "low", "close", "volume", "timestamp"]
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Input dictionary 'data' must contain key '{key}'.")
        if data[key] is None:
            raise ValueError(f"'{key}' data cannot be None.")
            
    open_ = np.asarray(data["open"])
    high = np.asarray(data["high"])
    low = np.asarray(data["low"])
    close = np.asarray(data["close"])
    volume = np.asarray(data["volume"])
    timestamp = np.asarray(data["timestamp"])
    
    n = len(close)
    if n == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=bool)
        
    if not (len(open_) == n and len(high) == n and len(low) == n and len(volume) == n and len(timestamp) == n):
        raise ValueError("All input arrays must have the same length.")
        
    # Clamping negative / non-positive prices
    if np.any(close <= 0.0) or np.any(high <= 0.0) or np.any(low <= 0.0) or np.any(open_ <= 0.0):
        logger.warning("Non-positive prices detected in inputs. Clamping to 1e-12.")
        close = np.clip(close, 1e-12, None)
        high = np.clip(high, 1e-12, None)
        low = np.clip(low, 1e-12, None)
        open_ = np.clip(open_, 1e-12, None)
    else:
        close = close.astype(np.float64)
        high = high.astype(np.float64)
        low = low.astype(np.float64)
        open_ = open_.astype(np.float64)
        
    if np.any(volume < 0.0):
        logger.warning("Negative volume detected. Clamping to 0.0.")
        volume = np.clip(volume, 0.0, None)
    else:
        volume = volume.astype(np.float64)
        
    # 2. Config parsing
    config = DEFAULT_INTEGRATION_CONFIG if config is None else config
    regime_module_map = config.get("regime_module_map", "shipped")
    if regime_module_map not in ("shipped", "blueprint"):
        raise ValueError(f"regime_module_map must be 'shipped' or 'blueprint', got {regime_module_map!r}")
        
    mr_regime = 2 if regime_module_map == "shipped" else 1
    breakout_regime = 0 if regime_module_map == "shipped" else 2

    fast_window = int(config.get("fast_window", 20))
    slow_window = int(config.get("slow_window", 100))
    vwap_window = int(config.get("vwap_window", 1440))
    vwap_exit = config.get("vwap_exit", 0.002)
    donchian_window = int(config.get("donchian_window", 20))
    stop_loss_pct = config.get("stop_loss_pct", 0.015)
    take_profit_pct = config.get("take_profit_pct", 0.03)
    dd_threshold = config.get("dd_threshold", 0.10)
    cooldown_bars = int(config.get("cooldown_bars", 500))
    
    # 3. Feature Calculations
    # Volatility Regime Detection with EMA Smoothing
    # Uses local rolling_std_from_returns and ema to allow test mocking to work correctly
    log_close = np.log(close)
    r = np.diff(log_close, prepend=log_close[0])
    
    fast_vol = rolling_std_from_returns(r, fast_window)
    slow_vol = rolling_std_from_returns(r, slow_window)
    
    vr = np.full_like(fast_vol, np.nan)
    valid_slow = (~np.isnan(slow_vol)) & (slow_vol > 0.0)
    vr[valid_slow] = fast_vol[valid_slow] / slow_vol[valid_slow]
    
    regime = np.full(n, -1, dtype=np.int8)
    p_low = config.get("p_low", None)
    p_high = config.get("p_high", None)
    
    if p_low is not None and p_high is not None:
        import polars as pl
        vr_smooth_window = int(config.get("vr_smooth_window", 20))
        vr_smoothed = ema(vr, vr_smooth_window)
        
        quantile_window = int(config.get("quantile_window", 43200))
        s = pl.Series(vr_smoothed)
        q_low = s.rolling_quantile(p_low, window_size=quantile_window, min_samples=1).to_numpy()
        q_high = s.rolling_quantile(p_high, window_size=quantile_window, min_samples=1).to_numpy()
        
        valid_vr = ~np.isnan(vr_smoothed)
        regime[valid_vr & (vr_smoothed <= q_low)] = 0
        regime[valid_vr & (vr_smoothed > q_low) & (vr_smoothed <= q_high)] = 1
        regime[valid_vr & (vr_smoothed > q_high)] = 2
    else:
        # Fallback to static threshold
        low_vol_threshold = config.get("low_vol_threshold", 1.2)
        extreme_vol_threshold = config.get("extreme_vol_threshold", 2.5)
        
        vr_smooth_window = int(config.get("vr_smooth_window", 20))
        vr_smoothed = ema(vr, vr_smooth_window)
        
        valid_vr = ~np.isnan(vr_smoothed)
        regime[valid_vr & (vr_smoothed <= low_vol_threshold)] = 0
        regime[valid_vr & (vr_smoothed > low_vol_threshold) & (vr_smoothed <= extreme_vol_threshold)] = 1
        regime[valid_vr & (vr_smoothed > extreme_vol_threshold)] = 2
    
    # Strategy features
    vwap = rolling_vwap(close, volume, vwap_window)
    deviation = np.full_like(close, np.nan)
    valid_vwap = (~np.isnan(vwap)) & (vwap > 0.0)
    deviation[valid_vwap] = close[valid_vwap] / vwap[valid_vwap] - 1.0
    
    donchian_high, donchian_low = donchian_channels(high, low, donchian_window)
    
    # Calculate standard deviation of deviation from VWAP over vwap_window bars
    clean_deviation = np.nan_to_num(deviation, nan=0.0)
    dev_std = rolling_std_from_returns(clean_deviation, vwap_window)
    
    # Calculate dynamic Z-Score
    z_score = np.full_like(deviation, np.nan)
    valid_dev_std = (~np.isnan(dev_std)) & (~np.isnan(deviation))
    z_score[valid_dev_std] = deviation[valid_dev_std] / np.maximum(dev_std[valid_dev_std], 1e-12)
    
    # 3.3 Session & Weekend Filters
    hour, dow = utc_hour_and_dow(timestamp)
    in_session = (hour >= 12) & (hour < 20)
    is_weekend = dow >= 5
    meta_trade_allowed = in_session & (~is_weekend)
    
    # 4. State Machine Loop via Numba
    signals, limit_flags = state_machine_loop(
        close=close,
        open_=open_,
        deviation=deviation,
        z_score=z_score,
        donchian_high=donchian_high,
        donchian_low=donchian_low,
        meta_trade_allowed=meta_trade_allowed,
        regime=regime,
        mr_regime=mr_regime,
        breakout_regime=breakout_regime,
        vwap_window=vwap_window,
        slow_window=slow_window,
        vwap_exit=vwap_exit,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        dd_threshold=dd_threshold,
        cooldown_bars=cooldown_bars
    )
    
    return signals, limit_flags
