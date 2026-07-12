"""Per-run strategy wrapper that binds an explicit config (parallel-safe; no global mutation).

The C++ engine calls `strategy_module.generate_signals(data)` with the data dict only. Passing a
StrategyView instance (which the engine sees as a module with a generate_signals attribute) lets us
inject a per-run config without mutating DEFAULT_INTEGRATION_CONFIG.
"""
from __future__ import annotations

import numpy as np

from strategies.vol_regime_switch.regime_engine import generate_signals


class StrategyView:
    def __init__(self, config: dict | None) -> None:
        self._config = config

    def generate_signals(self, data: dict) -> tuple[np.ndarray, np.ndarray]:
        return generate_signals(data, config=self._config)
