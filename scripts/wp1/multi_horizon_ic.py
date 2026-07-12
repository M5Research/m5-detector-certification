"""Multi-horizon IC scan: does an exploitable MR / momentum structure appear at LOWER frequency?

Pure-signal study (no P&L, no engine). Resamples 1m BTCUSDT OHLCV to 5m/15m/60m/240m/1440m and
computes, per (frequency, year), the Spearman IC of two CAUSAL signals vs the next-bar forward
return (bar t close -> bar t+1 close):

  * VWAP-deviation MR : dev_t = close_t / rolling_VWAP_t - 1, window ~= one trading day in real
                        time (window_bars = max(2, round(1440 / freq_minutes)); 20 bars for 1d).
                        Expect NEGATIVE IC if mean-reversion holds.
  * Return autocorr   : IC of the previous bar's return vs the next bar's return. A clean
                        MR(<0) / momentum(>0) structure diagnostic at that frequency.

Everything is causal: the signal at bar t uses only data <= t; the forward return is t -> t+1.
Each IC ships a circular block-bootstrap CI (block=10, n_boot=1000, seed=year) and the bars/year
(turnover context: fewer bars => lower turnover => lower cost drag).
"""
from __future__ import annotations

import glob
import json

import numpy as np
import pyarrow.parquet as pq
from backtest.utils import PROJECT_ROOT
from scipy import stats as scipy_stats

from scripts.wp1.stats import block_bootstrap_ci, spearman_ic

DATA_DIR = PROJECT_ROOT / "data" / "binance_futures" / "symbol=BTCUSDT"
OUT_DIR = PROJECT_ROOT / "backtest_results" / "wp1"

MINUTE_MS = 60_000
YEARS = (2021, 2022, 2023, 2024, 2025, 2026)
# (label, frequency in minutes)
FREQS: tuple[tuple[str, int], ...] = (
    ("5m", 5),
    ("15m", 15),
    ("60m", 60),
    ("240m", 240),
    ("1440m", 1440),
)
BOOT_BLOCK = 10
BOOT_NBOOT = 1000


def _load_year_1m(year: int) -> dict:
    files = sorted(glob.glob(str(DATA_DIR / f"year={year}" / "*.parquet")))
    if not files:
        raise FileNotFoundError(f"no parquet for {year}")
    t = pq.read_table(files[0]).to_pydict()
    out = {
        "timestamp": np.asarray(t["timestamp"], dtype=np.int64),
        "open": np.asarray(t["open"], dtype=np.float64),
        "high": np.asarray(t["high"], dtype=np.float64),
        "low": np.asarray(t["low"], dtype=np.float64),
        "close": np.asarray(t["close"], dtype=np.float64),
        "volume": np.asarray(t["volume"], dtype=np.float64),
    }
    order = np.argsort(out["timestamp"], kind="stable")
    return {k: v[order] for k, v in out.items()}


def resample_ohlcv(bars_1m: dict, freq_minutes: int) -> dict:
    """Causally aggregate 1m OHLCV into freq_minutes bars.

    Grouping key is floor(timestamp / freq_ms): all 1m bars whose open falls in the same
    freq-window collapse to one bar. open=first, high=max, low=min, close=last, volume=sum,
    timestamp=bar-open (the floored window start). Input is assumed time-sorted; we group by the
    integer bucket so partial windows at the series edges are kept as-is (causal, no padding).
    """
    ts = np.asarray(bars_1m["timestamp"], dtype=np.int64)
    if ts.size == 0:
        return {k: np.array([], dtype=(np.int64 if k == "timestamp" else np.float64))
                for k in ("timestamp", "open", "high", "low", "close", "volume")}
    freq_ms = int(freq_minutes) * MINUTE_MS
    bucket = ts // freq_ms
    # First index of each contiguous bucket run (input is time-sorted so buckets are non-decreasing).
    changes = np.empty(bucket.size, dtype=bool)
    changes[0] = True
    changes[1:] = bucket[1:] != bucket[:-1]
    starts = np.flatnonzero(changes)
    ends = np.empty_like(starts)
    ends[:-1] = starts[1:]
    ends[-1] = bucket.size
    o = bars_1m["open"]
    h = bars_1m["high"]
    low = bars_1m["low"]
    c = bars_1m["close"]
    v = bars_1m["volume"]
    n = starts.size
    out = {
        "timestamp": (bucket[starts] * freq_ms).astype(np.int64),  # floored window start
        "open": o[starts].astype(np.float64),
        "high": np.empty(n, dtype=np.float64),
        "low": np.empty(n, dtype=np.float64),
        "close": c[ends - 1].astype(np.float64),
        "volume": np.empty(n, dtype=np.float64),
    }
    # Reductions per group via reduceat (fast, vectorized).
    out["high"] = np.maximum.reduceat(h, starts)
    out["low"] = np.minimum.reduceat(low, starts)
    out["volume"] = np.add.reduceat(v, starts)
    return out


def rolling_vwap(close: np.ndarray, volume: np.ndarray, window: int) -> np.ndarray:
    """Causal rolling VWAP over `window` bars: sum(close*vol)/sum(vol), trailing & inclusive.

    NaN until `window` bars are available. Zero-volume windows yield NaN.
    """
    close = np.asarray(close, dtype=np.float64)
    volume = np.asarray(volume, dtype=np.float64)
    n = close.size
    out = np.full(n, np.nan, dtype=np.float64)
    if n < window or window < 1:
        return out
    pv = close * volume
    cum_pv = np.concatenate(([0.0], np.cumsum(pv)))
    cum_v = np.concatenate(([0.0], np.cumsum(volume)))
    # window ending at index i (inclusive): indices [i-window+1, i]
    end = np.arange(window - 1, n)
    start = end - window + 1
    num = cum_pv[end + 1] - cum_pv[start]
    den = cum_v[end + 1] - cum_v[start]
    with np.errstate(invalid="ignore", divide="ignore"):
        vwap = np.where(den > 0, num / den, np.nan)
    out[window - 1:] = vwap
    return out


def forward_return(close: np.ndarray) -> np.ndarray:
    """Next-bar forward return: r_fwd[t] = close[t+1]/close[t] - 1; last bar is NaN (causal)."""
    close = np.asarray(close, dtype=np.float64)
    fwd = np.full_like(close, np.nan)
    if close.size >= 2:
        fwd[:-1] = close[1:] / close[:-1] - 1.0
    return fwd


def prev_bar_return(close: np.ndarray) -> np.ndarray:
    """Previous-bar realized return: r_prev[t] = close[t]/close[t-1] - 1; first bar is NaN."""
    close = np.asarray(close, dtype=np.float64)
    prev = np.full_like(close, np.nan)
    if close.size >= 2:
        prev[1:] = close[1:] / close[:-1] - 1.0
    return prev


def _rank_product_series(signal: np.ndarray, fwd: np.ndarray) -> np.ndarray:
    """Per-bar standardized rank product over finite pairs; its mean == Spearman IC.

    Used to feed block_bootstrap_ci so the bootstrapped statistic IS the IC (mean of this series),
    rather than a sign-only proxy. Returns an empty array if fewer than 10 finite pairs.
    """
    s = np.asarray(signal, dtype=np.float64)
    f = np.asarray(fwd, dtype=np.float64)
    mask = np.isfinite(s) & np.isfinite(f)
    if mask.sum() < 10:
        return np.array([], dtype=np.float64)
    rs = scipy_stats.rankdata(s[mask])
    rf = scipy_stats.rankdata(f[mask])
    zs = (rs - rs.mean()) / rs.std()
    zf = (rf - rf.mean()) / rf.std()
    return zs * zf


def _ic_with_ci(signal: np.ndarray, fwd: np.ndarray, seed: int) -> dict:
    ic, n = spearman_ic(signal, fwd)
    series = _rank_product_series(signal, fwd)
    point, lo, hi = block_bootstrap_ci(
        series, block=BOOT_BLOCK, n_boot=BOOT_NBOOT, seed=seed
    )
    return {"ic": ic, "n": n, "boot_point": point, "boot_lo": lo, "boot_hi": hi}


def study_freq_year(bars_1m: dict, freq_minutes: int, year: int) -> dict:
    bars = resample_ohlcv(bars_1m, freq_minutes)
    close = bars["close"]
    volume = bars["volume"]
    n_bars = int(close.size)

    if freq_minutes >= 1440:
        vwap_window = 20
    else:
        vwap_window = max(2, int(round(1440 / freq_minutes)))

    vwap = rolling_vwap(close, volume, vwap_window)
    with np.errstate(invalid="ignore", divide="ignore"):
        dev = np.where((vwap > 0) & np.isfinite(vwap), close / vwap - 1.0, np.nan)

    fwd = forward_return(close)
    prev = prev_bar_return(close)

    vwap_dev = _ic_with_ci(dev, fwd, seed=year)
    autocorr = _ic_with_ci(prev, fwd, seed=year)

    return {
        "freq_minutes": freq_minutes,
        "year": year,
        "n_bars": n_bars,
        "vwap_window_bars": vwap_window,
        "vwap_dev_mr": vwap_dev,
        "return_autocorr": autocorr,
    }


def _fmt_cell(entry: dict | None) -> str:
    if entry is None:
        return "   --   "
    ic = entry["ic"]
    if not np.isfinite(ic):
        return "   nan  "
    return f"{ic:+.4f}"


def _print_table(title: str, results: dict, key: str) -> None:
    print(f"\n{title}")
    header = f"{'freq':>6} | " + " | ".join(f"{y:>8}" for y in YEARS)
    print(header)
    print("-" * len(header))
    for label, fmin in FREQS:
        cells = []
        for y in YEARS:
            entry = results.get((fmin, y))
            cells.append(_fmt_cell(entry[key] if entry else None))
        print(f"{label:>6} | " + " | ".join(f"{c:>8}" for c in cells))


def _print_bars_table(results: dict) -> None:
    print("\nBars / year (turnover context: fewer bars => lower turnover => lower cost drag)")
    header = f"{'freq':>6} | " + " | ".join(f"{y:>8}" for y in YEARS)
    print(header)
    print("-" * len(header))
    for label, fmin in FREQS:
        cells = []
        for y in YEARS:
            entry = results.get((fmin, y))
            cells.append(f"{entry['n_bars']:>8d}" if entry else f"{'--':>8}")
        print(f"{label:>6} | " + " | ".join(cells))


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[tuple[int, int], dict] = {}
    payload: dict = {
        "description": "Multi-horizon (5m-1d) Spearman IC scan of VWAP-deviation MR and "
                       "return-autocorrelation vs next-bar forward return. Causal, pure signal.",
        "config": {
            "freqs_minutes": [f for _, f in FREQS],
            "years": list(YEARS),
            "fwd_return": "bar t close -> bar t+1 close",
            "vwap_window_rule": "max(2, round(1440/freq_min)); 20 bars for 1d",
            "bootstrap": {"block": BOOT_BLOCK, "n_boot": BOOT_NBOOT, "seed": "year"},
        },
        "results": [],
    }

    for year in YEARS:
        try:
            bars_1m = _load_year_1m(year)
        except FileNotFoundError:
            continue
        for _label, fmin in FREQS:
            res = study_freq_year(bars_1m, fmin, year)
            results[(fmin, year)] = res
            payload["results"].append(res)

    (OUT_DIR / "multi_horizon_ic.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    _print_bars_table(results)
    _print_table("VWAP-deviation MR  IC  (expect NEGATIVE if mean-reversion holds)",
                 results, "vwap_dev_mr")
    _print_table("Return autocorrelation  IC  (NEGATIVE=MR, POSITIVE=momentum)",
                 results, "return_autocorr")
    print(f"\nwrote {OUT_DIR / 'multi_horizon_ic.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
