"""Default configuration values for volatility regime switch strategy."""

DEFAULT_REGIME_CONFIG = {
    "fast_window": 20,
    "slow_window": 100,
    "low_vol_threshold": 1.2,
    "extreme_vol_threshold": 2.5,
}

DEFAULT_STRATEGY_CONFIG = {
    "fast_window": 20,
    "slow_window": 100,
    "low_vol_threshold": 1.2,
    "extreme_vol_threshold": 2.5,
    "vwap_window": 1440,
    "vwap_entry": 0.02,
    "vwap_exit": 0.002,
    "donchian_window": 20,
    "take_profit_pct": 0.03,
}

DEFAULT_INTEGRATION_CONFIG = {
    "fast_window": 20,
    "slow_window": 100,
    "low_vol_threshold": 1.2,
    "extreme_vol_threshold": 2.5,
    "vwap_window": 1440,
    "vwap_entry": 0.02,
    "vwap_exit": 0.002,
    "donchian_window": 20,
    "vr_smooth_window": 20,
    "stop_loss_pct": 0.015,
    "take_profit_pct": 0.03,
    "dd_threshold": 0.10,
    "cooldown_bars": 500,
}
