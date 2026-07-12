"""C0: signal-level IC of VWAP-deviation / z-score vs forward returns, per regime and session.

Computes the strategy's own internal features causally (reusing regime_engine helpers), then the
Spearman IC vs H-bar forward returns, conditioned on regime (LOW/HIGH) and session (12-20 UTC).
"""
from __future__ import annotations

import glob
import json
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from backtest.utils import PROJECT_ROOT

from scripts.wp1.stats import block_bootstrap_ci, spearman_ic
from strategies.vol_regime_switch.defaults import DEFAULT_INTEGRATION_CONFIG
from strategies.vol_regime_switch.regime_detector import rolling_std_from_returns
from strategies.vol_regime_switch.regime_engine import ema, utc_hour_and_dow
from strategies.vol_regime_switch.strategy_modules import rolling_vwap

DATA_DIR = PROJECT_ROOT / "data" / "binance_futures" / "symbol=BTCUSDT"
OUT_DIR = PROJECT_ROOT / "backtest_results" / "wp1"


def _load_year(year: int) -> dict:
    files = sorted(glob.glob(str(DATA_DIR / f"year={year}" / "*.parquet")))
    if not files:
        raise FileNotFoundError(f"no parquet for {year}")
    t = pq.read_table(files[0]).to_pydict()
    return {k: np.asarray(t[k], dtype=np.float64) if k != "timestamp" else np.asarray(t[k], dtype=np.int64)
            for k in ("timestamp", "open", "high", "low", "close", "volume")}


def _features(data: dict, cfg: dict) -> dict:
    close = data["close"].astype(np.float64)
    log_close = np.log(np.clip(close, 1e-12, None))
    r = np.diff(log_close, prepend=log_close[0])
    fast = rolling_std_from_returns(r, int(cfg["fast_window"]))
    slow = rolling_std_from_returns(r, int(cfg["slow_window"]))
    vr = np.full_like(fast, np.nan)
    ok = (~np.isnan(slow)) & (slow > 0)
    vr[ok] = fast[ok] / slow[ok]
    vr_s = ema(vr, int(cfg["vr_smooth_window"]))
    regime = np.full(len(close), 2, dtype=np.int8)
    valid = ~np.isnan(vr_s)
    regime[valid & (vr_s <= cfg["low_vol_threshold"])] = 0
    regime[valid & (vr_s > cfg["low_vol_threshold"]) & (vr_s <= cfg["extreme_vol_threshold"])] = 1
    vwap = rolling_vwap(close, data["volume"].astype(np.float64), int(cfg["vwap_window"]))
    dev = np.full_like(close, np.nan)
    okv = (~np.isnan(vwap)) & (vwap > 0)
    dev[okv] = close[okv] / vwap[okv] - 1.0
    dev_std = rolling_std_from_returns(np.nan_to_num(dev), int(cfg["vwap_window"]))
    z = np.full_like(dev, np.nan)
    okz = (~np.isnan(dev_std)) & (~np.isnan(dev))
    z[okz] = dev[okz] / np.maximum(dev_std[okz], 1e-12)
    hour, _ = utc_hour_and_dow(data["timestamp"])
    return {"regime": regime, "deviation": dev, "zscore": z, "hour": hour, "close": close}


def _forward_return(close: np.ndarray, horizon: int) -> np.ndarray:
    fwd = np.full_like(close, np.nan)
    fwd[:-horizon] = close[horizon:] / close[:-horizon] - 1.0
    return fwd


def study_year(year: int, horizon: int, cfg: dict | None = None) -> dict:
    cfg = cfg or DEFAULT_INTEGRATION_CONFIG
    data = _load_year(year)
    f = _features(data, cfg)
    fwd = _forward_return(f["close"], horizon)
    in_session = (f["hour"] >= 12) & (f["hour"] < 20)
    # IC of mean-reversion signal: expect NEGATIVE IC of deviation vs fwd (high dev -> reverts down).
    out: dict[str, Any] = {"year": year, "horizon": horizon, "n": int(len(fwd))}
    for name, sig, mask in [
        ("deviation_all", f["deviation"], np.ones_like(fwd, bool)),
        ("zscore_session", f["zscore"], in_session),
        ("zscore_low_vol", f["zscore"], f["regime"] == 0),
        ("zscore_high_vol", f["zscore"], f["regime"] == 1),
    ]:
        s = np.where(mask, sig, np.nan)
        ic, n = spearman_ic(s, fwd)
        # bootstrap CI on per-bar sign-aligned product as a stability proxy
        prod = np.sign(np.nan_to_num(s)) * np.nan_to_num(fwd)
        _, lo, hi = block_bootstrap_ci(prod[mask & np.isfinite(fwd)], block=horizon, seed=year)
        out[name] = {"ic": ic, "n": n, "boot_lo": lo, "boot_hi": hi}
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = OUT_DIR / "fixture_meta.json"
    horizon = 60
    if meta_path.exists():
        mdur = json.loads(meta_path.read_text())["mean_trade_duration_bars"]
        if np.isfinite(mdur):
            horizon = max(1, int(round(mdur)))
    for year in (2021, 2022, 2023, 2024, 2025, 2026):
        try:
            res = study_year(year, horizon)
        except FileNotFoundError:
            continue
        (OUT_DIR / f"signal_ic_{year}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(f"[C0] {year} H={horizon}  dev_all IC={res['deviation_all']['ic']:+.4f}  "
              f"z_session IC={res['zscore_session']['ic']:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
