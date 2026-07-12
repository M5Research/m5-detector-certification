"""Pure-Python port of the C++ `_backtest_engine` bar accounting engine.

This is a faithful, line-for-line reimplementation of the deterministic
market-on-open bar accounting found in:

  - backtest_project/engine/parquet_data_feed.cpp  (ParquetDataFeed::load)
  - backtest_project/engine/backtest_engine.cpp     (BacktestEngine::run, metrics)
  - backtest_project/engine/utils.h                 (validate_bar)

The goal is STRICT numerical parity with the committed C++ headline result so
the WP1 cost test can run without a C++ toolchain. There is NO order book /
microstructure on this path. Orders fill at the next bar's open with explicit
taker/maker fee and slippage knobs, plus deterministic passive fill probability
and partial-fill penalties for bar-level realism tests.

All arithmetic uses float64 and preserves the exact operation order of the C++
source (no -ffast-math style reordering).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
from backtest.utils import PROJECT_ROOT  # noqa: E402

import scripts._bootstrap  # noqa: F401  (prepends repo `src/` to sys.path)

__all__ = ["load_and_clean", "run_backtest", "run_strategy_py"]


_UINT64_MASK = (1 << 64) - 1


def _deterministic_uniform_01(timestamp: int, index: int) -> float:
    x = (int(timestamp) ^ ((int(index) + 1) * 0x9E3779B97F4A7C15)) & _UINT64_MASK
    x = (x + 0x9E3779B97F4A7C15) & _UINT64_MASK
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _UINT64_MASK
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _UINT64_MASK
    x = (x ^ (x >> 31)) & _UINT64_MASK
    return ((x >> 11) & ((1 << 53) - 1)) / float(1 << 53)


# ---------------------------------------------------------------------------
# Data feed: validation + cleaning (port of ParquetDataFeed::load + validate_bar)
# ---------------------------------------------------------------------------


def _validate_bar(ts: int, o: float, h: float, lo: float, c: float, v: float, row_index: int) -> None:
    """Port of engine::validate_bar (utils.h)."""
    if ts <= 0:
        raise ValueError(f"Invalid timestamp (<= 0) at row {row_index}")

    for val in (o, h, lo, c, v):
        if math.isnan(val) or math.isinf(val):
            raise ValueError(
                f"NaN or Inf detected in price/volume at row {row_index}, timestamp: {ts}"
            )

    if o <= 0.0 or h <= 0.0 or lo <= 0.0 or c <= 0.0:
        raise ValueError(
            f"Non-positive price detected at row {row_index}, timestamp: {ts}, "
            f"Open: {o}, High: {h}, Low: {lo}, Close: {c}"
        )

    if v < 0.0:
        raise ValueError(f"Negative volume detected at row {row_index}, timestamp: {ts}, Volume: {v}")

    # High must be >= Low, Open, Close ; Low must be <= Open, Close
    if h < lo or h < o or h < c or lo > o or lo > c:
        raise ValueError(
            f"Inconsistent OHLC price hierarchy at row {row_index}, timestamp: {ts}, "
            f"Open: {o}, High: {h}, Low: {lo}, Close: {c}"
        )


def _clean(
    ts_in: np.ndarray,
    open_in: np.ndarray,
    high_in: np.ndarray,
    low_in: np.ndarray,
    close_in: np.ndarray,
    volume_in: np.ndarray,
    start_ms: int,
    end_ms: int,
    max_gap_allowed_mins: int,
) -> dict:
    """Port of the cleaning loop in ParquetDataFeed::load.

    Operates on raw (already-concatenated) arrays so it can be unit-tested
    without parquet. Performs: range filter, per-bar validation, duplicate
    drop, out-of-order detection, and gap forward-fill.
    """
    ts_in = np.asarray(ts_in, dtype=np.int64)
    open_in = np.asarray(open_in, dtype=np.float64)
    high_in = np.asarray(high_in, dtype=np.float64)
    low_in = np.asarray(low_in, dtype=np.float64)
    close_in = np.asarray(close_in, dtype=np.float64)
    volume_in = np.asarray(volume_in, dtype=np.float64)

    timestamps: list[int] = []
    out_open: list[float] = []
    out_high: list[float] = []
    out_low: list[float] = []
    out_close: list[float] = []
    out_volume: list[float] = []
    trading_disabled: list[bool] = []

    prev_ts = 0
    prev_close = 0.0

    raw_length = len(ts_in)
    for i in range(raw_length):
        ts = int(ts_in[i])
        if ts < start_ms or ts > end_ms:
            continue  # Skip data outside filter range

        o = float(open_in[i])
        h = float(high_in[i])
        lo = float(low_in[i])
        c = float(close_in[i])
        v = float(volume_in[i])

        _validate_bar(ts, o, h, lo, c, v, i)

        if prev_ts == 0:
            # First valid bar in range
            timestamps.append(ts)
            out_open.append(o)
            out_high.append(h)
            out_low.append(lo)
            out_close.append(c)
            out_volume.append(v)
            trading_disabled.append(False)

            prev_ts = ts
            prev_close = c
        else:
            if ts == prev_ts:
                # Duplicate timestamp: keep first, discard later
                continue
            elif ts < prev_ts:
                raise ValueError(
                    f"Timestamps out of order at row {i}: current {ts} < previous {prev_ts}"
                )
            else:
                # Check for gaps
                diff = ts - prev_ts
                if diff > 60000:
                    gap_mins = (diff // 60000) - 1
                    disable_trading = gap_mins > max_gap_allowed_mins

                    # Forward fill the gap
                    for k in range(1, gap_mins + 1):
                        filled_ts = prev_ts + k * 60000
                        timestamps.append(filled_ts)
                        out_open.append(prev_close)
                        out_high.append(prev_close)
                        out_low.append(prev_close)
                        out_close.append(prev_close)
                        out_volume.append(0.0)
                        trading_disabled.append(disable_trading)

                # Push current bar
                timestamps.append(ts)
                out_open.append(o)
                out_high.append(h)
                out_low.append(lo)
                out_close.append(c)
                out_volume.append(v)
                trading_disabled.append(False)

                prev_ts = ts
                prev_close = c

    return {
        "timestamp": np.asarray(timestamps, dtype=np.int64),
        "open": np.asarray(out_open, dtype=np.float64),
        "high": np.asarray(out_high, dtype=np.float64),
        "low": np.asarray(out_low, dtype=np.float64),
        "close": np.asarray(out_close, dtype=np.float64),
        "volume": np.asarray(out_volume, dtype=np.float64),
        "trading_disabled": np.asarray(trading_disabled, dtype=bool),
    }


def _epoch_ms_to_year(ms: int) -> int:
    """Port of ParquetDataFeed::epoch_ms_to_year (UTC year of the epoch ms)."""
    seconds = ms // 1000 if ms >= 0 else -((-ms) // 1000)
    # gmtime semantics: floor division toward negative infinity for seconds.
    # ms are always positive in this dataset; keep it simple and UTC.
    return int(np.datetime64(int(seconds), "s").astype("datetime64[Y]").astype(int) + 1970)


def load_and_clean(
    data_path: str,
    symbol: str,
    start_ms: int,
    end_ms: int,
    max_gap_allowed_mins: int,
) -> dict:
    """Faithful port of ParquetDataFeed::load.

    Loads year-partitioned parquet files
    (``data_path/symbol=<S>/year=<Y>/part-0.parquet`` for each UTC year spanned
    by ``start_ms..end_ms``), concatenates them, then runs the cleaning loop.
    """
    import pyarrow.parquet as pq

    start_year = _epoch_ms_to_year(start_ms)
    end_year = _epoch_ms_to_year(end_ms)

    ts_chunks: list[np.ndarray] = []
    open_chunks: list[np.ndarray] = []
    high_chunks: list[np.ndarray] = []
    low_chunks: list[np.ndarray] = []
    close_chunks: list[np.ndarray] = []
    volume_chunks: list[np.ndarray] = []

    base = Path(data_path)
    found_any = False
    for year in range(start_year, end_year + 1):
        file_path = base / f"symbol={symbol}" / f"year={year}" / "part-0.parquet"
        if file_path.exists():
            found_any = True
            table = pq.read_table(str(file_path))
            for required in ("timestamp", "open", "high", "low", "close", "volume"):
                if required not in table.schema.names:
                    raise ValueError(
                        "Parquet schema is missing required columns. Schema must contain: "
                        "timestamp, open, high, low, close, volume."
                    )
            ts_chunks.append(table.column("timestamp").to_numpy(zero_copy_only=False))
            open_chunks.append(table.column("open").to_numpy(zero_copy_only=False))
            high_chunks.append(table.column("high").to_numpy(zero_copy_only=False))
            low_chunks.append(table.column("low").to_numpy(zero_copy_only=False))
            close_chunks.append(table.column("close").to_numpy(zero_copy_only=False))
            volume_chunks.append(table.column("volume").to_numpy(zero_copy_only=False))

    if not found_any:
        raise ValueError(
            f"No Parquet data files found for symbol: {symbol} in the range of year "
            f"{start_year} to {end_year} under base path: {data_path}"
        )

    ts_arr = np.concatenate(ts_chunks).astype(np.int64)
    open_arr = np.concatenate(open_chunks).astype(np.float64)
    high_arr = np.concatenate(high_chunks).astype(np.float64)
    low_arr = np.concatenate(low_chunks).astype(np.float64)
    close_arr = np.concatenate(close_chunks).astype(np.float64)
    volume_arr = np.concatenate(volume_chunks).astype(np.float64)

    return _clean(
        ts_arr,
        open_arr,
        high_arr,
        low_arr,
        close_arr,
        volume_arr,
        start_ms,
        end_ms,
        max_gap_allowed_mins,
    )


# ---------------------------------------------------------------------------
# Backtest engine: accounting loop + metrics (port of BacktestEngine)
# ---------------------------------------------------------------------------


def _is_funding_time(current_ts: int, prev_ts: int) -> bool:
    """Port of BacktestEngine::is_funding_time (8-hour boundary crossing)."""
    period_ms = 8 * 3600 * 1000
    return (current_ts // period_ms) > (prev_ts // period_ms)


def _calculate_max_drawdown(equity_curve: list[float]) -> float:
    """Port of BacktestEngine::calculate_max_drawdown."""
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    max_dd = 0.0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        if peak > 0.0:
            dd = (peak - eq) / peak
            if dd > max_dd:
                max_dd = dd
    return max_dd


def _calculate_sharpe(
    equity_curve: list[float], timestamps: np.ndarray, initial_capital: float
) -> float:
    """Port of BacktestEngine::calculate_sharpe.

    Resamples to daily equity (last value per UTC day), computes daily returns
    against a running previous-day equity (seeded with initial_capital), then
    annualizes with sqrt(365). Uses a population std (divide by N).
    """
    if not equity_curve:
        return 0.0

    day_ms = 24 * 3600 * 1000
    # std::map<int64_t,double> keeps keys sorted; later writes overwrite earlier
    # within the same day -> last equity value per day wins.
    daily_equity: dict[int, float] = {}
    for i in range(len(equity_curve)):
        day_key = int(timestamps[i]) // day_ms
        daily_equity[day_key] = equity_curve[i]

    if len(daily_equity) < 2:
        return 0.0

    daily_returns: list[float] = []
    prev_eq = initial_capital
    for day_key in sorted(daily_equity.keys()):
        eq = daily_equity[day_key]
        ret = (eq / prev_eq) - 1.0
        daily_returns.append(ret)
        prev_eq = eq

    s = 0.0
    for r in daily_returns:
        s += r
    mean = s / len(daily_returns)

    var_sum = 0.0
    for r in daily_returns:
        var_sum += (r - mean) * (r - mean)
    std = math.sqrt(var_sum / len(daily_returns))

    if std == 0.0:
        return 0.0

    return math.sqrt(365.0) * (mean / std)


def run_backtest(
    data: dict,
    signals: np.ndarray,
    engine_config: dict,
    funding_rate_cb: Callable[[int], float] | None = None,
    use_limit_orders: np.ndarray | None = None,
) -> dict:
    """Faithful port of BacktestEngine::run.

    Parameters mirror the C++ EngineConfig fields. ``funding_rate_cb`` default
    None reproduces the engine's default (funding == 0).
    """
    ts = np.asarray(data["timestamp"], dtype=np.int64)
    open_ = np.asarray(data["open"], dtype=np.float64)
    close = np.asarray(data["close"], dtype=np.float64)
    trading_disabled = np.asarray(data["trading_disabled"], dtype=bool)

    data_length = len(ts)
    signal_length = len(signals)
    if signal_length != data_length:
        raise ValueError(
            f"Signal length ({signal_length}) does not match data length ({data_length})"
        )

    # Config (mirror EngineConfig defaults)
    initial_capital = float(engine_config.get("initial_capital", 10000.0))
    slippage_pct = float(engine_config.get("slippage_pct", 0.0001))
    maker_slippage_pct = float(engine_config.get("maker_slippage_pct", 0.0))
    taker_fee_pct = float(engine_config.get("taker_fee_pct", 0.0004))
    
    # Mirror the C++ EngineConfig: maker_fee_pct is the per-fill rate used for
    # limit-routed orders. Negative values represent rebates. Keep the legacy
    # maker_rebate_bps alias only when maker_fee_pct is absent.
    if "maker_fee_pct" in engine_config:
        maker_fee_pct = float(engine_config["maker_fee_pct"])
    else:
        maker_rebate_bps = float(engine_config.get("maker_rebate_bps", 0.0))
        maker_fee_pct = maker_rebate_bps / 10000.0
    limit_fill_probability = min(
        max(float(engine_config.get("limit_fill_probability", 1.0)), 0.0), 1.0
    )
    limit_fill_fraction = min(max(float(engine_config.get("limit_fill_fraction", 1.0)), 0.0), 1.0)
    
    position_size_type = engine_config.get("position_size_type", "leverage")
    position_size_value = float(engine_config.get("position_size_value", 1.0))

    if use_limit_orders is None:
        use_limit_orders = np.zeros(data_length, dtype=bool)

    report: dict = {
        "total_return": 0.0,
        "final_equity": 0.0,
        "sharpe_ratio": 0.0,
        "max_drawdown": 0.0,
        "total_trades": 0,
        "trades": [],
        "equity_curve": np.array([], dtype=np.float64),
        "equity_timestamps": np.array([], dtype=np.int64),
    }

    if data_length == 0:
        report["error"] = "No data available to run simulation."
        return report

    cash = initial_capital
    position = 0.0
    entry_price = 0.0
    total_fees = 0.0
    total_funding = 0.0

    trades: list[dict] = []
    equity_curve: list[float] = []
    equity_timestamps: list[int] = []

    # active_trade fields. The C++ `active_trade` struct also carries
    # exit_time/exit_price/exit_fee/pnl/qty, but on this path only entry_time,
    # side, entry_price and entry_fee are read back (the closing trade dict is
    # built fresh from the close context), so we track only those four.
    at_entry_time = 0
    at_side = 0  # 1 BUY, -1 SELL
    at_entry_price = 0.0
    at_entry_fee = 0.0

    for i in range(data_length):
        current_ts = int(ts[i])

        # 1. Funding rate application
        if i > 0 and position != 0.0 and _is_funding_time(current_ts, int(ts[i - 1])):
            funding_rate = 0.0
            if funding_rate_cb is not None:
                funding_rate = funding_rate_cb(current_ts)
            funding_payment = position * close[i] * funding_rate
            cash -= funding_payment
            total_funding += funding_payment

        # 2. Track current equity before executing any new orders for this bar
        unrealized_pnl = position * (close[i] - entry_price)
        equity = cash + unrealized_pnl

        equity_curve.append(equity)
        equity_timestamps.append(current_ts)

        # 3. Process signal and execute market order at next bar's open
        sig = float(signals[i])

        target_pos = 0.0
        if not trading_disabled[i]:
            if position_size_type == "leverage":
                target_pos = (sig * equity) / close[i]
            else:
                target_pos = sig * position_size_value
        else:
            target_pos = position

        if target_pos != position and i < data_length - 1:
            trade_qty = target_pos - position
            is_limit = use_limit_orders[i]
            if is_limit:
                if limit_fill_probability <= 0.0:
                    continue
                if (
                    limit_fill_probability < 1.0
                    and _deterministic_uniform_01(current_ts, i) > limit_fill_probability
                ):
                    continue
                trade_qty *= limit_fill_fraction
                target_pos = position + trade_qty
                if trade_qty == 0.0 or target_pos == position:
                    continue
            side = 1 if trade_qty > 0 else -1

            exec_price = open_[i + 1]
            fill_ts = int(ts[i + 1])

            execution_slippage_pct = maker_slippage_pct if is_limit else slippage_pct
            if side == 1:
                fill_price = exec_price * (1.0 + execution_slippage_pct)
            else:
                fill_price = exec_price * (1.0 - execution_slippage_pct)

            total_trade_qty = abs(trade_qty)
            fee_pct = maker_fee_pct if is_limit else taker_fee_pct
            total_fee = total_trade_qty * fill_price * fee_pct

            is_reversal = (position > 0 and target_pos < 0) or (position < 0 and target_pos > 0)
            is_closing = target_pos == 0.0 and position != 0.0

            if is_closing or is_reversal:
                qty_closed = abs(position)
                pnl_factor = 1.0 if position > 0 else -1.0
                realized_pnl = qty_closed * (fill_price - entry_price) * pnl_factor
                exit_fee = qty_closed * fill_price * fee_pct

                closed_pnl = realized_pnl - at_entry_fee - exit_fee
                trades.append(
                    {
                        "entry_time": at_entry_time,
                        "exit_time": fill_ts,
                        "side": "BUY" if at_side == 1 else "SELL",
                        "qty": qty_closed,
                        "entry_price": at_entry_price,
                        "exit_price": fill_price,
                        "entry_fee": at_entry_fee,
                        "exit_fee": exit_fee,
                        "pnl": closed_pnl,
                    }
                )
                cash += realized_pnl - exit_fee
                total_fees += exit_fee

                position = 0.0
                entry_price = 0.0

            if is_reversal:
                position = target_pos
                entry_price = fill_price
                entry_fee = abs(target_pos) * fill_price * fee_pct

                at_entry_time = fill_ts
                at_side = 1 if target_pos > 0 else -1
                at_entry_price = fill_price
                at_entry_fee = entry_fee

                cash -= entry_fee
                total_fees += entry_fee
            elif not is_closing:
                if position == 0.0:
                    position = target_pos
                    entry_price = fill_price
                    entry_fee = abs(target_pos) * fill_price * fee_pct

                    at_entry_time = fill_ts
                    at_side = 1 if target_pos > 0 else -1
                    at_entry_price = fill_price
                    at_entry_fee = entry_fee

                    cash -= entry_fee
                    total_fees += entry_fee
                else:
                    # Position sizing increase: add to position, NO trade record
                    new_qty = position + trade_qty
                    total_cost = position * entry_price + trade_qty * fill_price
                    entry_price = total_cost / new_qty

                    at_entry_fee += total_fee
                    at_entry_price = entry_price

                    position = new_qty
                    cash -= total_fee
                    total_fees += total_fee

    # 4. Force close remaining position at the close of the final bar
    if position != 0.0:
        fill_price = close[data_length - 1]
        fill_ts = int(ts[data_length - 1])

        qty_closed = abs(position)
        pnl_factor = 1.0 if position > 0 else -1.0
        realized_pnl = qty_closed * (fill_price - entry_price) * pnl_factor
        exit_fee = qty_closed * fill_price * taker_fee_pct

        closed_pnl = realized_pnl - at_entry_fee - exit_fee
        trades.append(
            {
                "entry_time": at_entry_time,
                "exit_time": fill_ts,
                "side": "BUY" if at_side == 1 else "SELL",
                "qty": qty_closed,
                "entry_price": at_entry_price,
                "exit_price": fill_price,
                "entry_fee": at_entry_fee,
                "exit_fee": exit_fee,
                "pnl": closed_pnl,
            }
        )
        cash += realized_pnl - exit_fee
        total_fees += exit_fee

        position = 0.0
        entry_price = 0.0

        equity_curve[-1] = cash

    # 5. Compute performance metrics
    report["final_equity"] = cash
    report["total_return"] = (cash - initial_capital) / initial_capital
    report["max_drawdown"] = _calculate_max_drawdown(equity_curve)
    report["sharpe_ratio"] = _calculate_sharpe(equity_curve, ts, initial_capital)
    report["total_trades"] = int(len(trades))
    report["trades"] = trades
    report["equity_curve"] = np.asarray(equity_curve, dtype=np.float64)
    report["equity_timestamps"] = np.asarray(equity_timestamps, dtype=np.int64)

    return report


def run_strategy_py(
    symbol: str,
    start_ms: int,
    end_ms: int,
    engine_config: dict,
    strategy_config: dict | None,
) -> dict:
    """Mirror of the C++ flow: get_ohlcv -> generate_signals(dict) -> run(signals).

    Loads + cleans the data, generates signals via the vol_regime_switch
    strategy, then runs the pure-Python backtest engine.
    """
    from strategies.vol_regime_switch.regime_engine import generate_signals

    data_path = engine_config.get("data_path") or str(PROJECT_ROOT / "data" / "binance_futures")
    max_gap = int(engine_config.get("max_gap_allowed_mins", 60))

    data = load_and_clean(data_path, symbol, start_ms, end_ms, max_gap)
    signals, use_limit_orders = generate_signals(data, strategy_config)
    return run_backtest(data, signals, engine_config, use_limit_orders=use_limit_orders)
