"""BTC-ETH relative-value IC study: lead-lag (H1) and spread mean-reversion (H2).

Pure-signal study (no P&L, no engine). Single-name 1m BTC directional/MR has no exploitable
OOS edge after cost; the professional next hypothesis is RELATIVE VALUE / market-neutral BTC-ETH,
where a directional dead-end can still hide a tradeable spread. This module tests two such
hypotheses, per year x {1m, 60m}, fully CAUSAL, each with a circular block-bootstrap CI
(block=10, n_boot=1000, seed=year). A pair trade pays 2x the single-leg cost, so the IC bar to
clear cost is HIGHER than for a single-name signal.

H1 -- Lead-lag ("ETH underreacts to BTC shocks", blueprint section 8.1):
  * IC( BTC_ret(t),  ETH_ret(t+1) )  -- does BTC lead ETH's NEXT bar?
  * IC( ETH_ret(t),  BTC_ret(t+1) )  -- reverse direction.
  * Conditional: restrict to bars where |BTC_ret(t)| is in its top decile that year, and report
    IC( sign(BTC_ret(t)), ETH_ret(t+1) ) -- does ETH follow BTC's BIG moves?
  ret is close-to-close log return. Positive IC => leader predicts laggard's next bar.

H2 -- Spread mean-reversion (stat-arb pair):
  * CAUSAL rolling-OLS hedge ratio beta(t) of log(ETH) on log(BTC) over a ~1-day window
    (1m: 1440 bars, 60m: 24 bars). spread(t) = log(ETH_close(t)) - beta(t)*log(BTC_close(t)).
  * CAUSAL rolling z-score of the spread over the same ~1-day window.
  * IC( z_spread(t), fwd_spread_ret(t->t+1) ) -- expect NEGATIVE (spread mean-reverts).
  * Mean-reversion half-life from a causal AR(1) fit per year:
    half_life = -ln(2)/ln(phi), reported in bars. AR(1) fit by hand with numpy lstsq
    (statsmodels NOT required).

Causality: every signal at bar t uses only data <= t; the forward target is t -> t+1. Both legs
are inner-joined on the shared 1m timestamp grid before any computation.

Run:  python -m scripts.wp1.relative_value_ic
"""
from __future__ import annotations

import glob
import json

import numpy as np
import pyarrow.parquet as pq
from backtest.utils import PROJECT_ROOT
from scipy import stats as scipy_stats

import scripts._bootstrap  # noqa: F401  (puts src/ on path; harmless if unused)
from scripts.wp1.multi_horizon_ic import resample_ohlcv
from scripts.wp1.stats import block_bootstrap_ci, spearman_ic

DATA_DIR = PROJECT_ROOT / "data" / "binance_futures"
BTC_DIR = DATA_DIR / "symbol=BTCUSDT"
ETH_DIR = DATA_DIR / "symbol=ETHUSDT"
OUT_DIR = PROJECT_ROOT / "backtest_results" / "wp1"

MINUTE_MS = 60_000
YEARS = (2023, 2024, 2025, 2026)
# (label, frequency in minutes, ~1-day window in bars at that frequency)
FREQS: tuple[tuple[str, int, int], ...] = (
    ("1m", 1, 1440),
    ("60m", 60, 24),
)
BOOT_BLOCK = 10
BOOT_NBOOT = 1000
TOP_DECILE = 0.90  # |BTC_ret| top-decile threshold for the conditional impulse test


# --------------------------------------------------------------------------------------------- #
# Data loading + inner join
# --------------------------------------------------------------------------------------------- #
def _load_year_1m(symbol_dir, year: int) -> dict:
    files = sorted(glob.glob(str(symbol_dir / f"year={year}" / "*.parquet")))
    if not files:
        raise FileNotFoundError(f"no parquet for {symbol_dir.name} {year}")
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


def inner_join_on_timestamp(btc: dict, eth: dict) -> tuple[dict, dict]:
    """Inner-join two time-sorted OHLCV dicts on their shared `timestamp` grid.

    Returns (btc_aligned, eth_aligned) carrying exactly the timestamps present in BOTH symbols,
    in ascending order. No interpolation, no fill: a relative-value signal must compare contemporaneous
    bars, so any unmatched timestamp on either side is dropped.
    """
    bt = btc["timestamp"]
    et = eth["timestamp"]
    common = np.intersect1d(bt, et, assume_unique=False)
    bi = np.searchsorted(bt, common)
    ei = np.searchsorted(et, common)
    btc_a = {k: v[bi] for k, v in btc.items()}
    eth_a = {k: v[ei] for k, v in eth.items()}
    return btc_a, eth_a


# --------------------------------------------------------------------------------------------- #
# Causal primitives
# --------------------------------------------------------------------------------------------- #
def log_return(close: np.ndarray) -> np.ndarray:
    """Close-to-close log return: r[t] = ln(close[t]/close[t-1]); first bar is NaN (causal)."""
    close = np.asarray(close, dtype=np.float64)
    r = np.full_like(close, np.nan)
    if close.size >= 2:
        r[1:] = np.log(close[1:] / close[:-1])
    return r


def lead(series: np.ndarray) -> np.ndarray:
    """Next-bar value: out[t] = series[t+1]; last bar is NaN. Used to build the t+1 target."""
    series = np.asarray(series, dtype=np.float64)
    out = np.full_like(series, np.nan)
    if series.size >= 2:
        out[:-1] = series[1:]
    return out


def rolling_ols(y: np.ndarray, x: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Causal rolling OLS of y on x WITH intercept over a trailing, inclusive `window` bars.

    Returns (alpha(t), beta(t)) for the fit  y = alpha + beta*x  over indices [t-window+1, t]:
        beta(t)  = cov(x,y) / var(x),   alpha(t) = mean(y) - beta(t)*mean(x).
    The intercept matters: the stat-arb spread is the OLS RESIDUAL  y - alpha - beta*x. Dropping
    alpha and using  y - beta*x  leaves a (beta-noise x price-level) term that swamps the true
    cointegrating residual when log-prices sit at a large level (e.g. ln(BTC) ~ 10-11) -- the
    estimated spread then fails to mean-revert. Both arrays are NaN until `window` bars exist and
    where var(x) == 0. Trailing-and-inclusive => uses only data <= t (no look-ahead).
    """
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    n = y.size
    alpha = np.full(n, np.nan, dtype=np.float64)
    beta = np.full(n, np.nan, dtype=np.float64)
    if n < window or window < 2:
        return alpha, beta
    cs_x = np.concatenate(([0.0], np.cumsum(x)))
    cs_y = np.concatenate(([0.0], np.cumsum(y)))
    cs_xx = np.concatenate(([0.0], np.cumsum(x * x)))
    cs_xy = np.concatenate(([0.0], np.cumsum(x * y)))
    end = np.arange(window - 1, n)
    start = end - window + 1
    w = float(window)
    sx = cs_x[end + 1] - cs_x[start]
    sy = cs_y[end + 1] - cs_y[start]
    sxx = cs_xx[end + 1] - cs_xx[start]
    sxy = cs_xy[end + 1] - cs_xy[start]
    cov = sxy - sx * sy / w
    var = sxx - sx * sx / w
    with np.errstate(invalid="ignore", divide="ignore"):
        b = np.where(var > 0, cov / var, np.nan)
        a = (sy - b * sx) / w
    alpha[window - 1:] = a
    beta[window - 1:] = b
    return alpha, beta


def rolling_beta(y: np.ndarray, x: np.ndarray, window: int) -> np.ndarray:
    """Causal rolling OLS slope (with intercept) of y on x. Convenience wrapper over rolling_ols."""
    _alpha, beta = rolling_ols(y, x, window)
    return beta


def rolling_zscore(series: np.ndarray, window: int) -> np.ndarray:
    """Causal rolling z-score over a trailing, inclusive `window` bars: (x - mean)/std (ddof=0).

    NaN until `window` bars exist and NaN where std == 0. Uses only data <= t.
    """
    series = np.asarray(series, dtype=np.float64)
    n = series.size
    out = np.full(n, np.nan, dtype=np.float64)
    valid = np.isfinite(series)
    if n < window or window < 2 or not valid.any():
        return out
    s = np.where(valid, series, 0.0)
    cs = np.concatenate(([0.0], np.cumsum(s)))
    cs2 = np.concatenate(([0.0], np.cumsum(s * s)))
    cv = np.concatenate(([0], np.cumsum(valid.astype(np.int64))))
    end = np.arange(window - 1, n)
    start = end - window + 1
    cnt = (cv[end + 1] - cv[start]).astype(np.float64)
    ssum = cs[end + 1] - cs[start]
    ssq = cs2[end + 1] - cs2[start]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(cnt > 0, ssum / cnt, np.nan)
        var = np.where(cnt > 0, ssq / cnt - mean * mean, np.nan)
        std = np.sqrt(np.maximum(var, 0.0))
        z = np.where((cnt == window) & (std > 0), (series[window - 1:] - mean) / std, np.nan)
    out[window - 1:] = z
    return out


def ar1_half_life(series: np.ndarray) -> tuple[float, float]:
    """Causal AR(1) fit  x[t] = a + phi*x[t-1] + e  by numpy lstsq over finite consecutive pairs.

    Returns (phi, half_life_in_bars). half_life = -ln(2)/ln(phi) for 0 < phi < 1 (mean-reverting).
    half_life is NaN (no finite mean reversion) when phi <= 0 or phi >= 1, or when too few pairs.
    statsmodels is NOT used.
    """
    x = np.asarray(series, dtype=np.float64)
    if x.size < 3:
        return float("nan"), float("nan")
    lhs = x[1:]
    rhs = x[:-1]
    mask = np.isfinite(lhs) & np.isfinite(rhs)
    if mask.sum() < 10:
        return float("nan"), float("nan")
    a_mat = np.column_stack([np.ones(mask.sum()), rhs[mask]])
    coef, *_ = np.linalg.lstsq(a_mat, lhs[mask], rcond=None)
    phi = float(coef[1])
    if phi <= 0.0 or phi >= 1.0:
        return phi, float("nan")
    half_life = float(-np.log(2.0) / np.log(phi))
    return phi, half_life


# --------------------------------------------------------------------------------------------- #
# IC + CI plumbing  (mirrors multi_horizon_ic: bootstrapped statistic IS the IC)
# --------------------------------------------------------------------------------------------- #
def _rank_product_series(signal: np.ndarray, fwd: np.ndarray) -> np.ndarray:
    s = np.asarray(signal, dtype=np.float64)
    f = np.asarray(fwd, dtype=np.float64)
    mask = np.isfinite(s) & np.isfinite(f)
    if mask.sum() < 10:
        return np.array([], dtype=np.float64)
    rs = scipy_stats.rankdata(s[mask])
    rf = scipy_stats.rankdata(f[mask])
    ss = rs.std()
    sf = rf.std()
    if ss == 0.0 or sf == 0.0:
        return np.array([], dtype=np.float64)
    zs = (rs - rs.mean()) / ss
    zf = (rf - rf.mean()) / sf
    return zs * zf


def _ic_with_ci(signal: np.ndarray, fwd: np.ndarray, seed: int) -> dict:
    ic, n = spearman_ic(signal, fwd)
    series = _rank_product_series(signal, fwd)
    point, lo, hi = block_bootstrap_ci(series, block=BOOT_BLOCK, n_boot=BOOT_NBOOT, seed=seed)
    return {"ic": ic, "n": n, "boot_point": point, "boot_lo": lo, "boot_hi": hi}


# --------------------------------------------------------------------------------------------- #
# H1 -- lead-lag
# --------------------------------------------------------------------------------------------- #
def study_lead_lag(btc_close: np.ndarray, eth_close: np.ndarray, seed: int) -> dict:
    btc_ret = log_return(btc_close)
    eth_ret = log_return(eth_close)
    eth_ret_next = lead(eth_ret)
    btc_ret_next = lead(btc_ret)

    btc_leads_eth = _ic_with_ci(btc_ret, eth_ret_next, seed=seed)
    eth_leads_btc = _ic_with_ci(eth_ret, btc_ret_next, seed=seed)

    # Conditional on a BTC impulse: |BTC_ret(t)| in its top decile that year.
    cond = {"ic": float("nan"), "n": 0, "boot_point": float("nan"),
            "boot_lo": float("nan"), "boot_hi": float("nan"), "threshold": float("nan")}
    finite = np.isfinite(btc_ret)
    if finite.sum() >= 10:
        thr = float(np.quantile(np.abs(btc_ret[finite]), TOP_DECILE))
        impulse = np.isfinite(btc_ret) & (np.abs(btc_ret) >= thr)
        sig = np.where(impulse, np.sign(btc_ret), np.nan)
        cond = _ic_with_ci(sig, eth_ret_next, seed=seed)
        cond["threshold"] = thr

    return {
        "btc_leads_eth": btc_leads_eth,
        "eth_leads_btc": eth_leads_btc,
        "btc_impulse_leads_eth": cond,
    }


# --------------------------------------------------------------------------------------------- #
# H2 -- spread mean-reversion
# --------------------------------------------------------------------------------------------- #
def study_spread_mr(btc_close: np.ndarray, eth_close: np.ndarray, window: int, seed: int) -> dict:
    log_btc = np.log(btc_close)
    log_eth = np.log(eth_close)

    alpha, beta = rolling_ols(log_eth, log_btc, window)     # causal hedge ratio (with intercept)
    spread = log_eth - alpha - beta * log_btc               # OLS residual; NaN during warmup
    z = rolling_zscore(spread, window)                      # causal z-score

    # Forward spread return t -> t+1: change in the (causal) spread level.
    fwd_spread_ret = np.full_like(spread, np.nan)
    if spread.size >= 2:
        fwd_spread_ret[:-1] = spread[1:] - spread[:-1]

    spread_mr = _ic_with_ci(z, fwd_spread_ret, seed=seed)

    phi, half_life = ar1_half_life(spread)

    return {
        "window_bars": window,
        "spread_mr": spread_mr,
        "ar1_phi": phi,
        "half_life_bars": half_life,
        "beta_median": float(np.nanmedian(beta)) if np.isfinite(beta).any() else float("nan"),
    }


# --------------------------------------------------------------------------------------------- #
# Per (freq, year) driver
# --------------------------------------------------------------------------------------------- #
def study_freq_year(btc_1m: dict, eth_1m: dict, freq_minutes: int, window: int, year: int) -> dict:
    if freq_minutes == 1:
        btc = btc_1m
        eth = eth_1m
    else:
        btc = resample_ohlcv(btc_1m, freq_minutes)
        eth = resample_ohlcv(eth_1m, freq_minutes)
        # Re-join after resampling: both share the same floored grid, but partial edges may differ.
        btc, eth = inner_join_on_timestamp(btc, eth)

    btc_close = btc["close"]
    eth_close = eth["close"]
    n_bars = int(btc_close.size)

    h1 = study_lead_lag(btc_close, eth_close, seed=year)
    h2 = study_spread_mr(btc_close, eth_close, window=window, seed=year)

    return {"freq_minutes": freq_minutes, "year": year, "n_bars": n_bars, "h1": h1, "h2": h2}


# --------------------------------------------------------------------------------------------- #
# Pretty printing
# --------------------------------------------------------------------------------------------- #
def _fmt(entry: dict | None) -> str:
    if entry is None:
        return "      --       "
    ic = entry["ic"]
    if not np.isfinite(ic):
        return "     nan       "
    lo = entry.get("boot_lo", float("nan"))
    hi = entry.get("boot_hi", float("nan"))
    if np.isfinite(lo) and np.isfinite(hi):
        return f"{ic:+.4f}[{lo:+.3f},{hi:+.3f}]"
    return f"{ic:+.4f}         "


def _print_h1(results: dict) -> None:
    print("\n" + "=" * 78)
    print("H1  LEAD-LAG IC  (positive => leader predicts laggard's NEXT bar; ret=log close-to-close)")
    print("=" * 78)
    for sub, title in (
        ("btc_leads_eth", "IC( BTC_ret(t), ETH_ret(t+1) )   does BTC lead ETH?"),
        ("eth_leads_btc", "IC( ETH_ret(t), BTC_ret(t+1) )   does ETH lead BTC?"),
        ("btc_impulse_leads_eth",
         "IC( sign(BTC_ret(t)), ETH_ret(t+1) ) | |BTC_ret| top-decile  (ETH follows BTC's big moves?)"),
    ):
        print(f"\n  {title}")
        for label, fmin, _w in FREQS:
            cells = []
            for y in YEARS:
                entry = results.get((fmin, y))
                cells.append(_fmt(entry["h1"][sub] if entry else None))
            row = "  |  ".join(f"{y}: {c}" for y, c in zip(YEARS, cells, strict=True))
            print(f"    {label:>4}  {row}")


def _print_h2(results: dict) -> None:
    print("\n" + "=" * 78)
    print("H2  SPREAD MEAN-REVERSION  IC( z_spread(t), fwd_spread_ret )  (expect NEGATIVE)")
    print("=" * 78)
    print("\n  Spread-MR IC")
    for label, fmin, _w in FREQS:
        cells = []
        for y in YEARS:
            entry = results.get((fmin, y))
            cells.append(_fmt(entry["h2"]["spread_mr"] if entry else None))
        row = "  |  ".join(f"{y}: {c}" for y, c in zip(YEARS, cells, strict=True))
        print(f"    {label:>4}  {row}")
    print("\n  AR(1) half-life of spread (bars)  /  phi  /  median hedge beta")
    for label, fmin, _w in FREQS:
        parts = []
        for y in YEARS:
            entry = results.get((fmin, y))
            if entry is None:
                parts.append(f"{y}:     --   ")
                continue
            h2 = entry["h2"]
            hl = h2["half_life_bars"]
            phi = h2["ar1_phi"]
            beta = h2["beta_median"]
            hl_s = f"{hl:8.1f}" if np.isfinite(hl) else "     nan"
            parts.append(f"{y}: hl={hl_s} phi={phi:+.4f} beta={beta:+.3f}")
        print(f"    {label:>4}  " + "  |  ".join(parts))


# --------------------------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------------------------- #
def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[tuple[int, int], dict] = {}
    payload: dict = {
        "description": "BTC-ETH relative-value IC study: H1 lead-lag and H2 spread mean-reversion. "
                       "Causal, pure signal, inner-joined on the shared 1m grid. A pair trade pays "
                       "2x the single-leg cost, so the IC bar to clear cost is higher.",
        "config": {
            "symbols": ["BTCUSDT", "ETHUSDT"],
            "years": list(YEARS),
            "freqs": [{"label": lbl, "minutes": fm, "window_bars": w} for lbl, fm, w in FREQS],
            "ret": "log close-to-close",
            "impulse_top_decile": TOP_DECILE,
            "spread": "log(ETH) - beta(t)*log(BTC), beta = causal rolling OLS over ~1-day window",
            "zscore": "causal rolling z over ~1-day window",
            "half_life": "-ln(2)/ln(phi) from causal AR(1) lstsq (statsmodels NOT used)",
            "bootstrap": {"block": BOOT_BLOCK, "n_boot": BOOT_NBOOT, "seed": "year"},
        },
        "results": [],
    }

    for year in YEARS:
        try:
            btc_1m = _load_year_1m(BTC_DIR, year)
            eth_1m = _load_year_1m(ETH_DIR, year)
        except FileNotFoundError:
            continue
        btc_1m, eth_1m = inner_join_on_timestamp(btc_1m, eth_1m)
        for _label, fmin, window in FREQS:
            res = study_freq_year(btc_1m, eth_1m, fmin, window, year)
            results[(fmin, year)] = res
            payload["results"].append(res)

    (OUT_DIR / "relative_value_ic.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    _print_h1(results)
    _print_h2(results)
    print(f"\nwrote {OUT_DIR / 'relative_value_ic.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
