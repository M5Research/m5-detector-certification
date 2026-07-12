"""DRY bootstrap for the prebuilt _backtest_engine pybind extension.

Consolidates the DLL-path / search-path / import logic previously copy-pasted in
scripts/optimize_vol_regime_switch.py, and adds a config-bound run helper.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pytest
from backtest.utils import PROJECT_ROOT  # noqa: E402

from scripts.wp1.strategy_view import StrategyView


def _add_msys_dll_path() -> None:
    if sys.platform == "win32":
        msys_bin = Path(r"C:\msys64\ucrt64\bin")
        if msys_bin.is_dir():
            os.add_dll_directory(str(msys_bin))


def _engine_search_paths() -> list[Path]:
    paths: list[Path] = []
    build_root = PROJECT_ROOT / "backtest_project" / "build"
    if build_root.is_dir():
        for child in sorted(build_root.iterdir()):
            if child.is_dir() and child.name.startswith("lib."):
                if list(child.glob("_backtest_engine*.pyd")) or list(child.glob("_backtest_engine*.so")):
                    paths.append(child)
    inplace = PROJECT_ROOT / "backtest_project"
    if list(inplace.glob("_backtest_engine*.pyd")) or list(inplace.glob("_backtest_engine*.so")):
        paths.append(inplace)
    # build_cpp output dir used by the committed build
    build_cpp = PROJECT_ROOT / "backtest_project" / "build_cpp" / "engine"
    if build_cpp.is_dir() and (
        list(build_cpp.glob("_backtest_engine*.pyd")) or list(build_cpp.glob("_backtest_engine*.so"))
    ):
        paths.append(build_cpp)
    return paths


def import_engine() -> Any:
    """Import and return the _backtest_engine module, or skip the test if unavailable."""
    _add_msys_dll_path()
    for d in _engine_search_paths():
        if str(d) not in sys.path:
            sys.path.insert(0, str(d))
    try:
        import _backtest_engine as engine_mod  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"_backtest_engine not built/importable: {exc}")
    return engine_mod


def build_engine_config(
    *,
    symbol: str,
    slippage_pct: float,
    taker_fee_pct: float,
    maker_slippage_pct: float = 0.0,
    maker_rebate_bps: float | None = -1.0,
    maker_fee_pct: float | None = None,
    limit_fill_probability: float = 1.0,
    limit_fill_fraction: float = 1.0,
    initial_capital: float = 10000.0,
    max_gap_allowed_mins: int = 60,
    data_path: str | None = None,
) -> dict:
    maker_fee = maker_fee_pct
    if maker_fee is None:
        maker_fee = (maker_rebate_bps or 0.0) / 10000.0
    return {
        "data_path": data_path or str(PROJECT_ROOT / "data" / "binance_futures"),
        "symbol": symbol,
        "timeframe": "1m",
        "initial_capital": initial_capital,
        "slippage_pct": slippage_pct,
        "maker_slippage_pct": maker_slippage_pct,
        "taker_fee_pct": taker_fee_pct,
        "maker_fee_pct": maker_fee,
        "limit_fill_probability": limit_fill_probability,
        "limit_fill_fraction": limit_fill_fraction,
        "max_gap_allowed_mins": max_gap_allowed_mins,
        "position_size_type": "leverage",
        "position_size_value": 1.0,
    }


class CppStrategyView:
    def __init__(self, view: StrategyView) -> None:
        self._view = view

    def generate_signals(self, data: dict):
        signals, use_limit_orders = self._view.generate_signals(data)
        return (signals, use_limit_orders)


def run_strategy(
    engine_mod: Any,
    engine_config: dict,
    *,
    symbol: str,
    start_ms: int,
    end_ms: int,
    config: dict | None,
) -> dict:
    """Construct a fresh engine, load data, run the strategy with an explicit config, return the report.

    A fresh engine per call is intentional: it is how cost is re-priced (engine cost is read once
    in the C++ constructor). `config` is bound via StrategyView so no module global is mutated.
    """
    engine = engine_mod.BacktestEngine(
        engine_config["data_path"], symbol, start_ms, end_ms, engine_config
    )
    engine.load_data()
    view = CppStrategyView(StrategyView(config))
    return engine.run(engine, view)

