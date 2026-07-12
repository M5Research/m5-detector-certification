"""Track B forensic reconciliation: py_engine vs committed C++ artifact.

D-10 protocol:
  (i)  Seeded, deterministic py_engine rerun over the parity window with
       DEFAULT_INTEGRATION_CONFIG — confirms 830 trades (FULL config).
  (ii) Experiment A: rerun with risk controls TRULY disabled by setting
       disabling values (not deleting keys — deletion is a silent no-op
       because config.get() falls back to the same defaults). Disabling
       values: stop_loss_pct=1e9, dd_threshold=1e9, cooldown_bars=0.
       Tests the RESEARCH.md risk-control hypothesis for real.
  (iii) Experiment B: inspect the C++ artifact metadata, investigate what
       __init__.py dispatches, and test other candidate drivers (regime
       mapping, vr_smooth_window, empty config with no strategy params).
  (iv) Structured field-level diff of py_engine-full vs py_engine-
       riskcontrols-DISABLED vs C++ artifact (D-09 — no C++ rebuild).
  (v)  Writes the diff to backtest_results/wp1/parity_diff_<date>.json.

CORRECTION NOTE — Prior-run defect (commits 79e581a, f139edc):
  The prior run "stripped" risk controls by DELETING the keys
  stop_loss_pct / dd_threshold / cooldown_bars from the strategy_config
  dict (lines 135-138 of the earlier script). However, regime_engine.py
  reads them as config.get("stop_loss_pct", 0.015), etc. — with fallback
  defaults IDENTICAL to DEFAULT_INTEGRATION_CONFIG's values. Deleting the
  keys was therefore a perfect no-op: the engine silently re-defaulted to
  the exact same numbers, so "stripped == full == 830" proved nothing.
  The prior "REFUTED" verdict and "code-path difference confirmed" conclusion
  were based on a failed experiment. This script corrects that by using
  disabling values that genuinely neutralize each control.

Constraint D-09: uses ONLY the committed JSON artifact; no C++ engine
import or execution of any kind.
Constraint D-08: py_engine + DEFAULT_INTEGRATION_CONFIG is canonical.
RNG: any random path uses np.random.default_rng(seed); never np.random.seed.
"""
from __future__ import annotations

import json
import sys
from datetime import UTC, date, datetime
from pathlib import Path

# Ensure project root is on sys.path so backtest.utils is importable when the
# script is run directly (python scripts/wp1/reconcile_parity.py from the root).
_SCRIPT_ROOT = Path(__file__).resolve().parents[2]
if str(_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_ROOT))

from backtest.utils import PROJECT_ROOT, dt_to_ms

import scripts._bootstrap  # noqa: F401  (prepends repo src/ to sys.path)
from scripts.wp1.engine_loader import build_engine_config
from scripts.wp1.py_engine import run_strategy_py
from strategies.vol_regime_switch.defaults import DEFAULT_INTEGRATION_CONFIG

# ── parity window (exact timestamps that produced the C++ artifact) ──────────
# Imported from the parity test so the two stay in sync (same pattern as
# pin_reproduce.py).
from tests.wp1.test_py_engine import _FIXTURE_END, _FIXTURE_START

# ── paths ────────────────────────────────────────────────────────────────────
_CPP_ARTIFACT = (
    PROJECT_ROOT
    / "backtest_results"
    / "report___init___BTCUSDT_20260529_222435.json"
)
_OUT_DIR = PROJECT_ROOT / "backtest_results" / "wp1"

SYMBOL = "BTCUSDT"

# ── Experiment A: TRULY disabling values for risk controls ───────────────────
# These values neutralize each control WITHOUT deleting the keys.
# stop_loss_pct=1e9: trigger requires (close/entry - 1) < -1e9 (long) or
#                    > 1e9 (short) — price would need to move 10^9x, impossible.
# dd_threshold=1e9: drawdown = 1 - (proxy_equity/peak) in [0,1); never > 1e9.
#                   d_limit = 1e9 (regime==2) or 1e9+0.05 (otherwise).
# cooldown_bars=0:  if lock somehow triggers, it unlocks immediately next bar
#                   (dd_cooldown = max(0-1,0) = 0 -> dd_locked = False on same bar).
_DISABLING_RISK_CONTROLS = {
    "stop_loss_pct": 1e9,
    "dd_threshold": 1e9,
    "cooldown_bars": 0,
}


def _engine_cfg() -> dict:
    """Maker engine config matching the C++ artifact metadata."""
    return build_engine_config(symbol=SYMBOL, slippage_pct=-0.00005, taker_fee_pct=0.0001)


def _load_cpp_artifact() -> dict:
    """Load the committed C++ artifact JSON — D-09 (never run any C++)."""
    return json.loads(_CPP_ARTIFACT.read_text(encoding="utf-8"))


def _summarise(report: dict, label: str) -> dict:
    """Extract the key comparable fields from a py_engine report dict."""
    trades = report.get("trades", [])
    first_entry_time = trades[0]["entry_time"] if trades else None
    first_entry_price = trades[0]["entry_price"] if trades else None
    return {
        "label": label,
        "total_trades": report["total_trades"],
        "total_return": report["total_return"],
        "sharpe_ratio": report["sharpe_ratio"],
        "max_drawdown": report["max_drawdown"],
        "final_equity": report.get("final_equity"),
        "first_trade_entry_time": first_entry_time,
        "first_trade_entry_price": first_entry_price,
    }


def _summarise_cpp(cpp: dict) -> dict:
    """Extract comparable fields from the committed C++ artifact JSON."""
    trades = cpp.get("trades", [])
    first_entry_time = trades[0]["entry_time"] if trades else None
    first_entry_price = trades[0]["entry_price"] if trades else None
    return {
        "label": "cpp_artifact_202",
        "total_trades": cpp["total_trades"],
        "total_return": cpp["total_return"],
        "sharpe_ratio": cpp["sharpe_ratio"],
        "max_drawdown": cpp["max_drawdown"],
        "final_equity": cpp["final_equity"],
        "first_trade_entry_time": first_entry_time,
        "first_trade_entry_price": first_entry_price,
    }


def _field_diff(a: dict, b: dict, fields: list[str]) -> dict:
    """Produce a per-field diff table between two summary dicts."""
    diff: dict = {}
    for f in fields:
        va = a.get(f)
        vb = b.get(f)
        if isinstance(va, float) and isinstance(vb, float):
            diff[f] = {
                "py_engine_full": va,
                "compared_to": vb,
                "abs_diff": abs(va - vb),
                "rel_diff_pct": abs(va - vb) / abs(vb) * 100.0 if vb else None,
            }
        else:
            diff[f] = {
                "py_engine_full": va,
                "compared_to": vb,
            }
    return diff


def _inspect_cpp_metadata(cpp: dict) -> dict:
    """Extract and summarise the C++ artifact metadata for Experiment B."""
    meta = cpp.get("metadata", {})
    cfg = meta.get("config", {})
    return {
        "strategy_path": meta.get("strategy_path"),
        "strategy_name": meta.get("strategy_name"),
        "strategy_hash": meta.get("strategy_hash"),
        "recorded_config_keys": list(cfg.keys()),
        "recorded_config": cfg,
        "note": (
            "The C++ artifact records only engine-level config keys "
            "(data_path, symbol, timeframe, initial_capital, slippage_pct, "
            "taker_fee_pct, max_gap_allowed_mins, position_size_type, "
            "position_size_value). No strategy-level parameters "
            "(stop_loss_pct, dd_threshold, cooldown_bars, vr_smooth_window, "
            "regime_module_map, etc.) are recorded. The exact strategy "
            "config passed to the C++ __init__.py call is unknown."
        ),
    }


def main() -> int:  # noqa: PLR0915
    start_ms = dt_to_ms(_FIXTURE_START)
    end_ms = dt_to_ms(_FIXTURE_END)
    engine_config = _engine_cfg()

    print("=" * 72)
    print("TRACK B FORENSIC RECONCILIATION — CORRECTED (no-op-strip fixed)")
    print("=" * 72)
    print()
    print("CORRECTION: Prior run deleted risk-control keys (no-op because")
    print("  config.get() falls back to identical defaults). This run uses")
    print("  TRULY disabling values: stop_loss_pct=1e9, dd_threshold=1e9,")
    print("  cooldown_bars=0 — each verified to neutralize its control.")
    print()

    # ── (i) Full config rerun ─────────────────────────────────────────────────
    strategy_full = dict(DEFAULT_INTEGRATION_CONFIG)
    print("[Experiment A step 1] Running py_engine FULL (DEFAULT_INTEGRATION_CONFIG)...")
    report_full = run_strategy_py(
        symbol=SYMBOL,
        start_ms=start_ms,
        end_ms=end_ms,
        engine_config=engine_config,
        strategy_config=strategy_full,
    )
    n_full = report_full["total_trades"]
    print(f"  -> {n_full} trades")

    # ── (ii) Experiment A: truly disabled risk controls ───────────────────────
    # Build config with disabling values SET (not keys deleted — deletion is a
    # silent no-op as config.get() uses identical fallback defaults).
    strategy_disabled = dict(DEFAULT_INTEGRATION_CONFIG)
    strategy_disabled.update(_DISABLING_RISK_CONTROLS)

    # Verify the disabling values are actually different from the defaults
    assert strategy_disabled["stop_loss_pct"] == 1e9, "stop_loss_pct disabling value not set"
    assert strategy_disabled["dd_threshold"] == 1e9, "dd_threshold disabling value not set"
    assert strategy_disabled["cooldown_bars"] == 0, "cooldown_bars disabling value not set"
    assert strategy_disabled["stop_loss_pct"] != DEFAULT_INTEGRATION_CONFIG["stop_loss_pct"]
    assert strategy_disabled["dd_threshold"] != DEFAULT_INTEGRATION_CONFIG["dd_threshold"]
    assert strategy_disabled["cooldown_bars"] != DEFAULT_INTEGRATION_CONFIG["cooldown_bars"]

    print()
    print("[Experiment A step 2] Running py_engine with risk controls TRULY DISABLED")
    print(f"  stop_loss_pct={strategy_disabled['stop_loss_pct']} (was {DEFAULT_INTEGRATION_CONFIG['stop_loss_pct']})")
    print(f"  dd_threshold={strategy_disabled['dd_threshold']} (was {DEFAULT_INTEGRATION_CONFIG['dd_threshold']})")
    print(f"  cooldown_bars={strategy_disabled['cooldown_bars']} (was {DEFAULT_INTEGRATION_CONFIG['cooldown_bars']})")
    report_disabled = run_strategy_py(
        symbol=SYMBOL,
        start_ms=start_ms,
        end_ms=end_ms,
        engine_config=engine_config,
        strategy_config=strategy_disabled,
    )
    n_disabled = report_disabled["total_trades"]
    print(f"  -> {n_disabled} trades")

    # ── (iii) Load committed C++ artifact ─────────────────────────────────────
    cpp = _load_cpp_artifact()
    n_cpp = cpp["total_trades"]
    cpp_first_time = cpp["trades"][0]["entry_time"] if cpp.get("trades") else None
    cpp_first_price = cpp["trades"][0]["entry_price"] if cpp.get("trades") else None

    # ── (iv) Experiment B: inspect C++ artifact metadata ─────────────────────
    cpp_meta_summary = _inspect_cpp_metadata(cpp)

    # Also test: empty config (passes only {} to generate_signals, hitting all defaults)
    # This is the closest approximation to "no config passed" that doesn't require
    # knowing the C++ harness internals.
    print()
    print("[Experiment B] Testing alternative driver: empty strategy config (no params)")
    strategy_empty = {}
    report_empty = run_strategy_py(
        symbol=SYMBOL,
        start_ms=start_ms,
        end_ms=end_ms,
        engine_config=engine_config,
        strategy_config=strategy_empty,
    )
    n_empty = report_empty["total_trades"]
    print(f"  -> {n_empty} trades (C++ has 202; empty config gives {n_empty})")

    # Also test: blueprint regime_module_map (alternative driver from RESEARCH.md)
    print()
    print("[Experiment B] Testing alternative driver: regime_module_map='blueprint'")
    strategy_blueprint = dict(DEFAULT_INTEGRATION_CONFIG)
    strategy_blueprint["regime_module_map"] = "blueprint"
    report_blueprint = run_strategy_py(
        symbol=SYMBOL,
        start_ms=start_ms,
        end_ms=end_ms,
        engine_config=engine_config,
        strategy_config=strategy_blueprint,
    )
    n_blueprint = report_blueprint["total_trades"]
    print(f"  -> {n_blueprint} trades (C++ has 202; blueprint mapping gives {n_blueprint})")

    # Combined: blueprint + truly disabled risk controls
    print()
    print("[Experiment B] Testing: blueprint + risk controls DISABLED")
    strategy_bp_disabled = dict(DEFAULT_INTEGRATION_CONFIG)
    strategy_bp_disabled["regime_module_map"] = "blueprint"
    strategy_bp_disabled.update(_DISABLING_RISK_CONTROLS)
    report_bp_disabled = run_strategy_py(
        symbol=SYMBOL,
        start_ms=start_ms,
        end_ms=end_ms,
        engine_config=engine_config,
        strategy_config=strategy_bp_disabled,
    )
    n_bp_disabled = report_bp_disabled["total_trades"]
    print(f"  -> {n_bp_disabled} trades (C++ has 202; blueprint+no-risk-controls gives {n_bp_disabled})")

    # ── Console summary ───────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("RESULTS SUMMARY")
    print("=" * 72)
    print(f"  py_engine FULL (DEFAULT_INTEGRATION_CONFIG)      : {n_full} trades")
    print(f"  py_engine RISK CONTROLS DISABLED (1e9/1e9/0)     : {n_disabled} trades")
    print(f"  py_engine EMPTY CONFIG (no params)               : {n_empty} trades")
    print(f"  py_engine BLUEPRINT REGIME MAPPING               : {n_blueprint} trades")
    print(f"  py_engine BLUEPRINT + RISK DISABLED              : {n_bp_disabled} trades")
    print(f"  C++ committed artifact                            : {n_cpp} trades")
    print()

    # First trade comparison
    disabled_trades = report_disabled.get("trades", [])
    disabled_first_time = disabled_trades[0]["entry_time"] if disabled_trades else None
    disabled_first_price = disabled_trades[0]["entry_price"] if disabled_trades else None
    full_trades = report_full.get("trades", [])
    full_first_time = full_trades[0]["entry_time"] if full_trades else None
    full_first_price = full_trades[0]["entry_price"] if full_trades else None

    print(f"  First trade (full):     time={full_first_time}, price={full_first_price:.2f}" if full_first_price else "")
    print(f"  First trade (disabled): time={disabled_first_time}, price={disabled_first_price:.2f}" if disabled_first_price else "")
    print(f"  First trade (C++):      time={cpp_first_time}, price={cpp_first_price:.2f}" if cpp_first_price else "")
    print()

    # ── Experiment A verdict ──────────────────────────────────────────────────
    # "Close to C++" = within 15% of n_cpp AND not zero (zero is a degenerate floor).
    cpp_lo = n_cpp * 0.85
    cpp_hi = n_cpp * 1.15
    _disabled_close_to_cpp = (cpp_lo <= n_disabled <= cpp_hi) and n_disabled > 0
    _disabled_moves_toward_cpp = (
        abs(n_disabled - n_cpp) < abs(n_full - n_cpp)
        and n_disabled != n_full
        and n_disabled > 0
    )
    if _disabled_close_to_cpp:
        exp_a_verdict = (
            f"CONFIRMED: truly-disabled config gives {n_disabled} trades, "
            f"within 15% of C++ {n_cpp} ({cpp_lo:.0f}–{cpp_hi:.0f}); "
            "removing risk controls explains the 4x gap."
        )
    elif n_disabled == n_full:
        exp_a_verdict = (
            f"REFUTED (valid experiment): truly-disabled config gives {n_disabled} trades, "
            f"same as full={n_full}. Risk controls do NOT drive the trade count. "
            f"The engine's risk-control triggers (stop_loss_pct, dd_threshold, drawdown_lock) "
            f"are not responsible for the {n_full}-vs-{n_cpp} gap."
        )
    elif _disabled_moves_toward_cpp:
        exp_a_verdict = (
            f"PARTIAL (moves toward C++ but not close): truly-disabled config gives "
            f"{n_disabled} trades vs full={n_full} and C++={n_cpp}; "
            "risk controls partially explain the gap but another driver remains."
        )
    else:
        exp_a_verdict = (
            f"INCONCLUSIVE: truly-disabled config gives {n_disabled} trades "
            f"(full={n_full}, C++={n_cpp}); result moves away from or is zero — "
            "risk controls are not the driver in the expected direction."
        )
    print(f"[Experiment A verdict] {exp_a_verdict}")
    print()

    # ── Experiment B findings ─────────────────────────────────────────────────
    init_py_note = (
        "src/strategies/vol_regime_switch/__init__.py re-exports generate_signals "
        "from regime_engine.py — it contains NO alternative signal logic. "
        "The 'report___init___' filename indicates the C++ harness loaded "
        "__init__.py as the strategy entry point, but this dispatches to the "
        "SAME generate_signals function that py_engine uses. The gap cannot be "
        "explained by __init__.py implementing a different signal path."
    )
    print(f"[Experiment B] __init__.py analysis: {init_py_note}")
    print()

    def _candidate_verdict(n: int, label: str) -> str:
        """Classify a candidate run's trade count relative to full and C++."""
        _close = (cpp_lo <= n <= cpp_hi) and n > 0
        if _close:
            return f"CANDIDATE CONFIRMED (within 15% of C++={n_cpp}): {label} gives {n} trades"
        elif n == n_full:
            return f"{label} gives {n} (same as full={n_full}) - not a driver"
        elif n == 0:
            return f"{label} gives 0 (degenerate - no signals; not comparable to C++={n_cpp})"
        else:
            return f"{label} gives {n} (full={n_full}, C++={n_cpp})"

    # Candidate: empty config
    empty_cfg_note = _candidate_verdict(n_empty, "empty config")
    print(f"[Experiment B] Empty config test: {empty_cfg_note}")

    # Candidate: blueprint mapping
    blueprint_note = _candidate_verdict(n_blueprint, "blueprint regime mapping")
    print(f"[Experiment B] Blueprint mapping: {blueprint_note}")

    # Candidate: blueprint + disabled
    bp_disabled_note = _candidate_verdict(n_bp_disabled, "blueprint+risk_disabled")
    print(f"[Experiment B] Blueprint+disabled: {bp_disabled_note}")

    print()
    print("[Experiment B] C++ artifact records NO strategy-level config.")
    print(f"  Recorded keys: {cpp_meta_summary['recorded_config_keys']}")
    print("  The exact strategy config used by the C++ run is not reconstructable.")
    print("  Under D-09 (no C++ rebuild), the root cause cannot be fully pinned.")
    print()

    # ── (v) Structured diff ───────────────────────────────────────────────────
    sum_full = _summarise(report_full, "py_engine_full_830")
    sum_disabled = _summarise(report_disabled, f"py_engine_riskcontrols_DISABLED_{n_disabled}")
    sum_cpp = _summarise_cpp(cpp)

    diff_fields = ["total_trades", "total_return", "sharpe_ratio", "max_drawdown", "final_equity"]

    diff_full_vs_cpp = _field_diff(sum_full, sum_cpp, diff_fields)
    diff_disabled_vs_cpp = _field_diff(sum_disabled, sum_cpp, diff_fields)

    experiment_b_variants = {
        "py_engine_empty_config": _summarise(report_empty, f"py_engine_empty_config_{n_empty}"),
        "py_engine_blueprint": _summarise(report_blueprint, f"py_engine_blueprint_{n_blueprint}"),
        "py_engine_blueprint_risk_disabled": _summarise(report_bp_disabled, f"py_engine_blueprint_risk_disabled_{n_bp_disabled}"),
    }

    report_payload = {
        "meta": {
            "description": "D-10 forensic diff — CORRECTED: py_engine-full vs py_engine-riskcontrols-DISABLED vs C++ artifact",
            "correction_note": (
                "Prior run (commits 79e581a, f139edc) deleted risk-control keys — a silent no-op "
                "because config.get() falls back to identical defaults. "
                "This run uses disabling values: stop_loss_pct=1e9, dd_threshold=1e9, "
                "cooldown_bars=0. These are verified to neutralize each control."
            ),
            "parity_window_start": _FIXTURE_START.isoformat(),
            "parity_window_end": _FIXTURE_END.isoformat(),
            "cpp_artifact_file": _CPP_ARTIFACT.name,
            "experiment_a_disabling_values": _DISABLING_RISK_CONTROLS,
            "experiment_a_verdict": exp_a_verdict,
            "experiment_b_init_py_note": init_py_note,
            "experiment_b_empty_config": empty_cfg_note,
            "experiment_b_blueprint_mapping": blueprint_note,
            "experiment_b_blueprint_plus_disabled": bp_disabled_note,
            "cpp_metadata_analysis": cpp_meta_summary,
            "generated_at": datetime.now(UTC).isoformat(),
        },
        "summaries": {
            "py_engine_full": sum_full,
            "py_engine_riskcontrols_DISABLED": sum_disabled,
            "cpp_artifact": sum_cpp,
        },
        "experiment_b_variants": experiment_b_variants,
        "field_diff_full_vs_cpp": diff_full_vs_cpp,
        "field_diff_disabled_vs_cpp": diff_disabled_vs_cpp,
    }

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _OUT_DIR / f"parity_diff_{date.today().strftime('%Y%m%d')}.json"
    out_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
    print(f"[reconcile_parity] Diff report written -> {out_path.relative_to(PROJECT_ROOT)}")

    # Sanity checks
    if n_full != 830:
        print(
            f"WARNING: py_engine FULL trade count is {n_full}, expected 830. "
            "Reproducibility check failed — investigate before proceeding."
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
