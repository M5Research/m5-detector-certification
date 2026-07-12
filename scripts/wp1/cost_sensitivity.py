"""C1: re-price the pinned strategy across real Binance USD-M fee tiers; compute break-even.

Cost is re-priced by rebuilding the engine config per (slippage, fee) point and rerunning the
pure-Python bar engine (scripts.wp1.py_engine.run_strategy_py). The maker scenario uses
slippage_pct >= 0 (passive fills get adverse-or-zero improvement, never a free -0.5bp).

Periods:
  2024     -- full year 2024-01-01/2024-12-31 (OOS)
  2025     -- full year 2025-01-01/2025-12-31 (OOS)
  2026_90d -- the 90-day fixture window (in-sample control)
"""
from __future__ import annotations

import json

from backtest.utils import PROJECT_ROOT, dt_to_ms, parse_lookback

# The 90-day fixture window (in-sample control), reused from the parity test.
from tests.wp1.test_py_engine import _FIXTURE_END, _FIXTURE_START

import scripts._bootstrap  # noqa: F401  (prepends repo `src/` to sys.path)
from scripts.wp1.engine_loader import build_engine_config
from scripts.wp1.py_engine import run_strategy_py
from strategies.vol_regime_switch.defaults import DEFAULT_INTEGRATION_CONFIG

OUT_DIR = PROJECT_ROOT / "backtest_results" / "wp1"
SYMBOL = "BTCUSDT"

# (label, slippage_pct, taker_fee_pct). Real USD-M VIP0 (BNB-discounted) + half-spread on slippage.
COST_POINTS = [
    ("ideal_maker_-0.5bp", -0.00005, 0.0001),   # the original (impossible) assumption, for reference
    ("maker_real_1.8bp+0.5spread", 0.00005, 0.00018),
    ("maker_real_1.8bp+1.0spread", 0.00010, 0.00018),
    ("taker_real_4.5bp+0.5spread", 0.00005, 0.00045),
    ("taker_real_4.5bp+1.0spread", 0.00010, 0.00045),
    ("zero_cost", 0.0, 0.0),
]

# Period label -> (start_ms, end_ms). 2026_90d is the in-sample fixture window.
def _periods() -> dict[str, tuple[int, int]]:
    out: dict[str, tuple[int, int]] = {}
    for label, lookback in (("2024", "2024-01-01/2024-12-31"), ("2025", "2025-01-01/2025-12-31")):
        sd, ed = parse_lookback(lookback)
        out[label] = (dt_to_ms(sd), dt_to_ms(ed))
    out["2026_90d"] = (dt_to_ms(_FIXTURE_START), dt_to_ms(_FIXTURE_END))
    return out


def roundtrip_bps(slippage_pct: float, taker_fee_pct: float) -> float:
    # 2 fills per round trip; slippage and fee both apply per fill.
    return (abs(slippage_pct) + taker_fee_pct) * 2 * 10000.0


def break_even_cost_bps(points: list[dict]) -> float:
    """Linear interpolation of round-trip cost where total_return crosses 0 (NaN if no crossing)."""
    pts = sorted(points, key=lambda p: p["roundtrip_bps"])
    for a, b in zip(pts, pts[1:], strict=False):
        if (a["total_return"] > 0) != (b["total_return"] > 0):
            x0, y0 = a["roundtrip_bps"], a["total_return"]
            x1, y1 = b["roundtrip_bps"], b["total_return"]
            return float(x0 + (0.0 - y0) * (x1 - x0) / (y1 - y0))
    return float("nan")


def sweep_period(period_label: str, start_ms: int, end_ms: int, mapping: str = "shipped") -> dict:
    cfg = {**DEFAULT_INTEGRATION_CONFIG, "regime_module_map": mapping}
    rows = []
    for label, slip, fee in COST_POINTS:
        ec = build_engine_config(symbol=SYMBOL, slippage_pct=slip, taker_fee_pct=fee)
        rep = run_strategy_py(SYMBOL, start_ms, end_ms, ec, cfg)
        turnover = sum(t["qty"] for t in rep["trades"])
        fee_drag = sum(t["entry_fee"] + t["exit_fee"] for t in rep["trades"]) / ec["initial_capital"]
        rows.append({
            "label": label, "slippage_pct": slip, "taker_fee_pct": fee,
            "roundtrip_bps": roundtrip_bps(slip, fee),
            "total_return": rep["total_return"], "sharpe_ratio": rep["sharpe_ratio"],
            "max_drawdown": rep["max_drawdown"], "total_trades": rep["total_trades"],
            "turnover": turnover, "fee_drag": fee_drag,
        })
    return {
        "period": period_label, "mapping": mapping, "points": rows,
        "break_even_bps": break_even_cost_bps(rows),
    }


def _maker_real_return(points: list[dict]) -> float:
    return next(
        p["total_return"] for p in points if p["label"].startswith("maker_real_1.8bp+0.5")
    )


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    periods = _periods()
    for period_label, (start_ms, end_ms) in periods.items():
        for mapping in ("shipped", "blueprint"):
            res = sweep_period(period_label, start_ms, end_ms, mapping)
            out_path = OUT_DIR / f"cost_sensitivity_{period_label}_{mapping}.json"
            out_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
            be = res["break_even_bps"]
            print(
                f"[C1] {period_label:8s} {mapping:9s} break_even={be:.2f}bp  "
                f"maker_real_ret={_maker_real_return(res['points']):+.4f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
