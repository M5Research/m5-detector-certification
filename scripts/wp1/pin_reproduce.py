"""C-1: reproduce and checksum the headline maker result; emit live-config + mean-trade-duration meta.

Runs the pure-Python bar engine (scripts.wp1.py_engine.run_strategy_py) over the EXACT 90-day
window that produced the committed fixture (report___init___BTCUSDT_20260529_222435.json):

    start_time = 2026-02-28T21:24:30.446815+00:00
    end_time   = 2026-05-29T21:24:30.446815+00:00

at maker cost (slippage -0.5bp, taker_fee 1bp) with the default integration config, then writes a
checksummed fixture and metadata consumed by C0/C1. The literal "2026-01-01/2026-05-28" string from
the original spec yields 346 trades (more data) -- the committed +2.16% / 202-trade headline is the
90-day lookback, so that is what we pin here. See tests/wp1/test_py_engine.py for the parity guard.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
from backtest.utils import PROJECT_ROOT, dt_to_ms

# The exact fixture window (90-day lookback), reused from the parity test so the two stay in sync.
from tests.wp1.test_py_engine import _FIXTURE_END, _FIXTURE_START

import scripts._bootstrap  # noqa: F401  (prepends repo `src/` to sys.path)
from scripts.wp1.engine_loader import build_engine_config
from scripts.wp1.py_engine import run_strategy_py
from strategies.vol_regime_switch.defaults import DEFAULT_INTEGRATION_CONFIG

SYMBOL = "BTCUSDT"
OUT_DIR = PROJECT_ROOT / "backtest_results" / "wp1"


def _mean_trade_duration_bars(trades: list[dict]) -> float:
    durs = [
        (t["exit_time"] - t["entry_time"]) / 60000.0
        for t in trades
        if t["exit_time"] and t["entry_time"]
    ]
    return float(np.mean(durs)) if durs else float("nan")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    start_ms, end_ms = dt_to_ms(_FIXTURE_START), dt_to_ms(_FIXTURE_END)

    # config.maker.json values, set explicitly (slippage -0.5bp, taker_fee 1bp).
    engine_config = build_engine_config(symbol=SYMBOL, slippage_pct=-0.00005, taker_fee_pct=0.0001)
    strategy_config = dict(DEFAULT_INTEGRATION_CONFIG)  # regime_module_map defaults to 'shipped'

    report = run_strategy_py(
        SYMBOL, start_ms, end_ms, engine_config, strategy_config
    )

    fixture = {
        "window_start": _FIXTURE_START.isoformat(),
        "window_end": _FIXTURE_END.isoformat(),
        "symbol": SYMBOL,
        "engine_config": {
            k: engine_config[k] for k in ("slippage_pct", "taker_fee_pct", "initial_capital")
        },
        "strategy_config": dict(DEFAULT_INTEGRATION_CONFIG),
        "total_return": report["total_return"],
        "sharpe_ratio": report["sharpe_ratio"],
        "max_drawdown": report["max_drawdown"],
        "total_trades": report["total_trades"],
        "final_equity": report["final_equity"],
        "trades": report["trades"],
    }
    fixture_path = OUT_DIR / "fixture_maker_90d.json"
    fixture_path.write_text(json.dumps(fixture, indent=2, sort_keys=True), encoding="utf-8")
    sha = hashlib.sha256(fixture_path.read_bytes()).hexdigest()

    mean_dur = _mean_trade_duration_bars(report["trades"])
    meta = {
        "fixture_sha256": sha,
        "total_return": report["total_return"],
        "total_trades": report["total_trades"],
        "mean_trade_duration_bars": mean_dur,
        "window_start": _FIXTURE_START.isoformat(),
        "window_end": _FIXTURE_END.isoformat(),
        "live_config_keys": sorted(DEFAULT_INTEGRATION_CONFIG.keys()),
        "dead_config_keys": ["vwap_entry"],  # verified: never read by generate_signals
    }
    (OUT_DIR / "fixture_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(
        f"[C-1] total_return={report['total_return']:+.5f}  trades={report['total_trades']}  "
        f"mean_dur_bars={mean_dur:.1f}  sha={sha[:12]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
