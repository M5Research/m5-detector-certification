"""Volatility Regime Switch Strategy - Layer 1 Regime Detection."""

from strategies.vol_regime_switch.defaults import (
    DEFAULT_INTEGRATION_CONFIG,
    DEFAULT_REGIME_CONFIG,
    DEFAULT_STRATEGY_CONFIG,
)
from strategies.vol_regime_switch.regime_detector import detect_regime
from strategies.vol_regime_switch.regime_engine import generate_signals
from strategies.vol_regime_switch.strategy_modules import (
    donchian_channels,
    generate_raw_signals,
    rolling_vwap,
)

__all__ = [
    "detect_regime",
    "generate_raw_signals",
    "rolling_vwap",
    "donchian_channels",
    "generate_signals",
    "DEFAULT_REGIME_CONFIG",
    "DEFAULT_STRATEGY_CONFIG",
    "DEFAULT_INTEGRATION_CONFIG",
]
