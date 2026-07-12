"""Phase 5 figure generation driver: v2.0 validation + diagnosis report figures.

Generates all 5 figures for docs/research/v2.0-validation-diagnosis-report.md.

Figures:
  fig01_diagnosis_histogram.png  -- v1.0 RV-ratio histogram with 2.5x threshold
  fig02_epsilon_sq_grid.png      -- epsilon-squared vs 0.01 floor, both detectors
  fig03_regime_populations.png   -- regime populations per detector
  fig04_crisis_timelines.png     -- LUNA/FTX crisis timelines + HMM soft-prob overlay
  fig05_v1v2_comparison.png      -- v1.0 vs v2.0 paired epsilon-squared contrast

Integrity controls:
  - D-04/D-10: NO epsilon-squared computed; gate_analysis, regate_analysis,
    and predictability are NOT imported in this driver.
  - D-10: two-layer 2026 holdout guard (epoch-ms assert + ts_end.year < 2026).
  - D-15: provenance stamp emitted to stdout + JSON sidecar in docs/research/figures/.
  - Re-derivation is VISUALIZATION ONLY: RV labels for Figs 1/4/5 via
    frozen detectors. No new statistics are produced.

Fig 4 note (D-06 known-fallback): HMMDetector(k_regimes=2) is used DIRECTLY.
This is the ESTABLISHED Phase-3 outcome (D-06 intractability fired: 3-state EM
extrapolated 35.5 min > 30 min threshold). This is NOT a new decision.

Output:
    docs/research/figures/fig01_diagnosis_histogram.png
    docs/research/figures/fig02_epsilon_sq_grid.png
    docs/research/figures/fig03_regime_populations.png
    docs/research/figures/fig04_crisis_timelines.png
    docs/research/figures/fig05_v1v2_comparison.png
    docs/research/figures/provenance_<timestamp>.json
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# D-12: Force single-threaded BLAS BEFORE any statsmodels / matplotlib import
# Pitfall 1 (RESEARCH.md): OMP/MKL/OPENBLAS env vars first, then Agg backend.
# ---------------------------------------------------------------------------
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# ---------------------------------------------------------------------------
# Pitfall 1 (RESEARCH.md): matplotlib.use('Agg') MUST come before pyplot import.
# ---------------------------------------------------------------------------
import matplotlib  # noqa: E402 -- must come before pyplot
matplotlib.use("Agg")

import importlib.metadata  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import datetime as _dt  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

# ---------------------------------------------------------------------------
# Path bootstrap (mirrors validate_rolling_quantile.py lines 39-48)
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_REPO_SRC = _REPO_ROOT / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from backtest.utils import PROJECT_ROOT  # noqa: E402
import scripts._bootstrap  # noqa: F401, E402

# ---------------------------------------------------------------------------
# Import ONLY the needed symbols — NOT gate_analysis, NOT regate_analysis,
# NOT predictability (D-04/D-10: no epsilon-squared in this driver).
# ---------------------------------------------------------------------------
from strategies.vol_regime_switch.regime_detector import rolling_std_from_returns  # noqa: E402
from strategies.vol_regime_switch.rolling_quantile_detector import RollingQuantileDetector  # noqa: E402
from strategies.vol_regime_switch.hmm_detector import HMMDetector  # noqa: E402

# Module-level import so tests can monkeypatch gf.load_and_clean (05-PATTERNS.md contract).
from scripts.wp1.py_engine import load_and_clean  # noqa: E402

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
FIGURES_DIR = PROJECT_ROOT / "docs" / "research" / "figures"

# ---------------------------------------------------------------------------
# V1.0 static-path constants (for Fig 1 re-derivation)
# ---------------------------------------------------------------------------
FAST_WINDOW: int = 20
SLOW_WINDOW: int = 100
EXTREME_THRESHOLD: float = 2.5

# ---------------------------------------------------------------------------
# Result JSON paths (Fig 2, 3, 5 read these — no re-computation)
# ---------------------------------------------------------------------------
REGATE_DIR = PROJECT_ROOT / "backtest_results" / "regate"
WP1_DIR = PROJECT_ROOT / "backtest_results" / "wp1"
GATE_DIR = PROJECT_ROOT / "backtest_results" / "gate"

# ---------------------------------------------------------------------------
# Publication-quality rcParams (Pattern 4, RESEARCH.md)
# ---------------------------------------------------------------------------
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

# Colorblind-safe Okabe-Ito palette (Nature Methods recommended)
OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
}
REGIME_COLORS = {
    "LOW": OKABE_ITO["sky_blue"],
    "ELEVATED": OKABE_ITO["orange"],
    "EXTREME": OKABE_ITO["vermillion"],
    "v1_empty": "#AAAAAA",
}

# ---------------------------------------------------------------------------
# Git commit helper (D-15; mirrors validate_rolling_quantile.py lines 89-103)
# ---------------------------------------------------------------------------


def _get_git_commit() -> str:
    """Return the short HEAD hash, or a safe fallback string if git is unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    return "unavailable"


# ---------------------------------------------------------------------------
# Module-level provenance (static integrity fields populated at import time;
# dynamic fields updated by main()). test_provenance_stamp reads gf._PROVENANCE.
# ---------------------------------------------------------------------------
_PROVENANCE: dict = {
    "year_2026_loaded": False,
    "prereg_commit": "169fc20",
    "phase4_amendment_commit": "04b667e",
    "git_commit": _get_git_commit(),  # resolved at import time
    "run_date": None,       # populated by main()
    "matplotlib_version": None,
    "numpy_version": None,
}


# ---------------------------------------------------------------------------
# Helper: load the latest regate JSON matching a glob pattern
# ---------------------------------------------------------------------------


def _load_latest_json(directory: Path, pattern: str) -> dict:
    """Load the most recent JSON matching pattern in directory."""
    matches = sorted(directory.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No JSON matching '{pattern}' found in {directory}"
        )
    return json.loads(matches[-1].read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Figure 1: v1.0 RV-ratio histogram (re-derive from close)
# ---------------------------------------------------------------------------


def _plot_fig1(close: np.ndarray, figures_dir: Path) -> None:
    """Fig 1: fast20/slow100 RV-ratio distribution with 2.5x threshold line.

    Re-derives the exact v1.0 static-path unsmoothed vr = fast_vol / slow_vol
    (diagnose_v1_detector.py logic). Source: RESEARCH.md Code Examples Figure 1.
    """
    log_close = np.log(close)
    r = np.diff(log_close, prepend=log_close[0])
    fast_vol = rolling_std_from_returns(r, FAST_WINDOW)
    slow_vol = rolling_std_from_returns(r, SLOW_WINDOW)
    vr = np.full_like(fast_vol, np.nan)
    valid_slow = (~np.isnan(slow_vol)) & (slow_vol > 0.0)
    vr[valid_slow] = fast_vol[valid_slow] / slow_vol[valid_slow]
    vr_valid = vr[~np.isnan(vr)]

    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    ax.hist(
        vr_valid,
        bins=200,
        color=OKABE_ITO["sky_blue"],
        alpha=0.7,
        edgecolor="none",
        label="fast₂₀/slow₁₀₀ RV ratio",
    )
    ax.axvline(
        EXTREME_THRESHOLD,
        color=OKABE_ITO["vermillion"],
        linewidth=1.5,
        linestyle="--",
        label=f"v1.0 threshold ({EXTREME_THRESHOLD}×)",
    )
    # Annotation placed after hist so ylim is set
    ymax = ax.get_ylim()[1]
    ax.annotate(
        "100th percentile\n5.34σ above mean\nn=0 above threshold",
        xy=(EXTREME_THRESHOLD, ymax * 0.5),
        xytext=(EXTREME_THRESHOLD + 0.05, ymax * 0.6),
        fontsize=8,
        color=OKABE_ITO["vermillion"],
        arrowprops=dict(arrowstyle="->", color=OKABE_ITO["vermillion"], lw=0.8),
    )
    ax.set_xlabel("fast₂₀ / slow₁₀₀ RMS ratio (unsmoothed)")
    ax.set_ylabel("Bar count")
    ax.set_title(
        "Fig 1: v1.0 detector instrument failure — static 2.5× threshold never fires"
    )
    ax.legend(fontsize=8)
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / "fig01_diagnosis_histogram.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Fig 1 saved: fig01_diagnosis_histogram.png")


# ---------------------------------------------------------------------------
# Figure 2: epsilon-squared grid, both detectors
# ---------------------------------------------------------------------------


def _plot_fig2(figures_dir: Path) -> None:
    """Fig 2: epsilon-squared vs 0.01 floor across (W,q), both detectors.

    Reads per_wq[*].epsilon_sq from the two regate JSONs (NO re-computation).
    HMM bars annotated 'EXTREME structurally absent' (Pitfall 6 / D-01 carve-out).
    Source: CONTEXT.md primary values 0.000318 / 0.000532 / 0.001691 and
    HMM 0.0000401 / 0.000386 / 0.001701.
    """
    q_json = _load_latest_json(REGATE_DIR, "gate_report_*_quantile.json")
    h_json = _load_latest_json(REGATE_DIR, "gate_report_*_hmm.json")

    wq_labels = ["(W=60, q=5)", "(W=120, q=5)", "(W=240, q=15)"]
    q_eps = [row["epsilon_sq"] for row in q_json["per_wq"]]
    h_eps = [row["epsilon_sq"] for row in h_json["per_wq"]]

    x = np.arange(len(wq_labels))
    width = 0.32

    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    bars_q = ax.bar(
        x - width / 2,
        q_eps,
        width,
        label="Quantile (primary)",
        color=OKABE_ITO["sky_blue"],
        alpha=0.85,
        edgecolor="white",
        linewidth=0.4,
    )
    bars_h = ax.bar(
        x + width / 2,
        h_eps,
        width,
        label="HMM (robustness, EXTREME absent)",
        color=OKABE_ITO["orange"],
        alpha=0.85,
        edgecolor="white",
        linewidth=0.4,
        hatch="//",
    )

    ax.axhline(
        0.01,
        color=OKABE_ITO["vermillion"],
        linewidth=1.5,
        linestyle="--",
        label="ε² = 0.01 floor (pre-registered)",
    )

    # Annotate HMM bars with "EXTREME absent" note
    for bar in bars_h:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.00005,
            "EXTREME\nabsent",
            ha="center",
            va="bottom",
            fontsize=6,
            color=OKABE_ITO["orange"],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(wq_labels, fontsize=8)
    ax.set_ylabel("ε² (epsilon-squared)")
    ax.set_title("Fig 2: ε² vs pre-registered 0.01 floor — both detectors FAIL")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_ylim(0, 0.015)

    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / "fig02_epsilon_sq_grid.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Fig 2 saved: fig02_epsilon_sq_grid.png")


# ---------------------------------------------------------------------------
# Figure 3: regime populations per detector
# ---------------------------------------------------------------------------


def _plot_fig3(figures_dir: Path) -> None:
    """Fig 3: grouped bar chart of regime populations, both detectors.

    Reads population_stats from the two validation JSONs (NO re-computation).
    Primary: LOW 75.1% / ELEVATED 19.4% / EXTREME 5.4%.
    HMM: LOW 51.0% / ELEVATED 49.0% / EXTREME 0% (bar empty/hatched).
    """
    q_val = _load_latest_json(WP1_DIR, "rolling_quantile_validation_*.json")
    h_val = _load_latest_json(WP1_DIR, "hmm_validation_*.json")

    q_pop = q_val["population_stats"]
    h_pop = h_val["population_stats"]

    regimes = ["LOW", "ELEVATED", "EXTREME"]
    # Quantile detector
    q_fracs = [
        q_pop["low_frac"],
        q_pop["elevated_frac"],
        q_pop["extreme_frac"],
    ]
    # HMM detector
    h_fracs = [
        h_pop["low_frac"],
        h_pop["elevated_frac"],
        h_pop["extreme_frac"],
    ]

    x = np.arange(len(regimes))
    width = 0.32

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.bar(
        x - width / 2,
        [f * 100 for f in q_fracs],
        width,
        label="Quantile (primary)",
        color=[REGIME_COLORS["LOW"], REGIME_COLORS["ELEVATED"], REGIME_COLORS["EXTREME"]],
        alpha=0.85,
        edgecolor="white",
        linewidth=0.4,
    )
    hmm_bar_colors = [REGIME_COLORS["LOW"], REGIME_COLORS["ELEVATED"], REGIME_COLORS["v1_empty"]]
    ax.bar(
        x + width / 2,
        [f * 100 for f in h_fracs],
        width,
        label="HMM (robustness)",
        color=hmm_bar_colors,
        alpha=0.7,
        edgecolor="white",
        linewidth=0.4,
        hatch="//",
    )

    # Annotate EXTREME HMM bar as structurally empty
    extreme_idx = 2
    ax.text(
        extreme_idx + width / 2,
        0.5,
        "EXTREME\nstruct.\nabsent",
        ha="center",
        va="bottom",
        fontsize=6.5,
        color=REGIME_COLORS["v1_empty"],
    )

    ax.set_xticks(x)
    ax.set_xticklabels(regimes, fontsize=9)
    ax.set_ylabel("Fraction of valid bars (%)")
    ax.set_title("Fig 3: Regime populations — both detectors (2021–2025 BTCUSDT)")
    ax.legend(fontsize=8)

    # Add value labels on bars
    for i, (qf, hf) in enumerate(zip(q_fracs, h_fracs)):
        ax.text(
            i - width / 2, qf * 100 + 0.4, f"{qf*100:.1f}%", ha="center", va="bottom", fontsize=7
        )
        if hf > 0.001:
            ax.text(
                i + width / 2, hf * 100 + 0.4, f"{hf*100:.1f}%", ha="center", va="bottom", fontsize=7
            )

    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / "fig03_regime_populations.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Fig 3 saved: fig03_regime_populations.png")


# ---------------------------------------------------------------------------
# Figure 5: v1.0 vs v2.0 comparison (paired epsilon-squared profiles)
# ---------------------------------------------------------------------------


def _plot_fig5(figures_dir: Path) -> None:
    """Fig 5: v1.0 vs v2.0 paired epsilon-squared profiles (variant B, D-09 centerpiece).

    Reads epsilon_sq from:
      - v1.0 gate JSON (gate_report_20260601_*.json): EXTREME n=0, epsilon_sq ~0.001
      - v2.0 quantile regate JSON: EXTREME fires (5.4%), still FAIL everywhere

    Both plotted against the 0.01 floor with a clear v1.0 EXTREME n=0 callout.
    Source: RESEARCH.md Fig 5 variant B recommendation.
    """
    # v1.0 gate results
    v1_json = _load_latest_json(GATE_DIR, "gate_report_*.json")
    # v2.0 quantile regate results
    v2_json = _load_latest_json(REGATE_DIR, "gate_report_*_quantile.json")

    wq_labels = ["(W=60, q=5)", "(W=120, q=5)", "(W=240, q=15)"]
    v1_eps = [row["epsilon_sq"] for row in v1_json["per_wq"]]
    v2_eps = [row["epsilon_sq"] for row in v2_json["per_wq"]]

    x = np.arange(len(wq_labels))
    width = 0.32

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bars_v1 = ax.bar(
        x - width / 2,
        v1_eps,
        width,
        label="v1.0 (EXTREME n=0, instrument failure)",
        color=REGIME_COLORS["v1_empty"],
        alpha=0.85,
        edgecolor="white",
        linewidth=0.4,
        hatch="xx",
    )
    ax.bar(
        x + width / 2,
        v2_eps,
        width,
        label="v2.0 quantile (EXTREME 5.4%, corrected)",
        color=OKABE_ITO["sky_blue"],
        alpha=0.85,
        edgecolor="white",
        linewidth=0.4,
    )

    ax.axhline(
        0.01,
        color=OKABE_ITO["vermillion"],
        linewidth=1.5,
        linestyle="--",
        label="ε² = 0.01 floor (pre-registered)",
    )

    # v1.0 callout annotation on first bar
    ax.annotate(
        "v1.0: EXTREME n=0\n(instrument failure)",
        xy=(bars_v1[0].get_x() + bars_v1[0].get_width() / 2.0, v1_eps[0]),
        xytext=(-0.3, 0.004),
        fontsize=7.5,
        color=OKABE_ITO["vermillion"],
        ha="center",
        arrowprops=dict(arrowstyle="->", color=OKABE_ITO["vermillion"], lw=0.8),
    )

    ax.set_xticks(x)
    ax.set_xticklabels(wq_labels, fontsize=8)
    ax.set_ylabel("ε² (epsilon-squared)")
    ax.set_title(
        "Fig 5: v1.0 ↔ v2.0 contrast — null holds with corrected instrumentation (D-09)"
    )
    ax.legend(fontsize=8, loc="upper left")
    ax.set_ylim(0, 0.015)

    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / "fig05_v1v2_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Fig 5 saved: fig05_v1v2_comparison.png")


# ---------------------------------------------------------------------------
# Figure 4: LUNA/FTX crisis timelines with HMM 2-state soft-prob overlay
# ---------------------------------------------------------------------------


def _plot_fig4(close: np.ndarray, timestamps: np.ndarray, figures_dir: Path) -> None:
    """Fig 4: LUNA + FTX crisis timelines with HMM filtered P(high-var) overlay.

    Re-derives quantile regime labels via RollingQuantileDetector().fit(close).
    Re-fits HMM for filtered_probs_ (not stored in validation JSON).

    D-06 known-fallback: HMMDetector(k_regimes=2) is used DIRECTLY.
    This is the established Phase-3 outcome (3-state EM extrapolated 35.5 min
    > 30 min threshold). This is NOT a new decision — documented here as a
    known-fallback comment per RESEARCH.md Assumption A3.

    Source: RESEARCH.md Code Examples Figure 4; Open Question 1 (±4-week windows).
    """
    # Fit quantile detector for regime labels
    detector_q = RollingQuantileDetector()
    regime_q = detector_q.fit(close)

    # Re-fit HMM 2-state DIRECTLY (D-06 known-fallback: Phase-3 established intractability)
    # high_var_col = k_regimes - 1 = 1 (ELEVATED / highest-variance state in 2-state model)
    hmm_detector = HMMDetector(k_regimes=2)
    print(
        "  Fig 4: fitting HMMDetector(k_regimes=2) on full span "
        "(D-06 known-fallback; OMP_NUM_THREADS=1 already set)..."
    )
    hmm_detector.fit(close)
    high_var_col = hmm_detector.k_regimes - 1  # = 1 for 2-state

    # ±4-week context windows around each crisis (RESEARCH.md Open Question 1 recommendation)
    crises = [
        {
            "name": "LUNA",
            "event_label": "LUNA collapse\n2022-05-07–05-18",
            "plot_start": _dt.datetime(2022, 4, 9, tzinfo=_dt.UTC),
            "plot_end": _dt.datetime(2022, 6, 16, tzinfo=_dt.UTC),
        },
        {
            "name": "FTX",
            "event_label": "FTX collapse\n2022-11-08–11-12",
            "plot_start": _dt.datetime(2022, 10, 11, tzinfo=_dt.UTC),
            "plot_end": _dt.datetime(2022, 12, 10, tzinfo=_dt.UTC),
        },
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 3.5), sharey=False)

    for ax1, crisis in zip(axes, crises):
        plot_start_ms = int(crisis["plot_start"].timestamp() * 1000)
        plot_end_ms = int(crisis["plot_end"].timestamp() * 1000)

        mask = (timestamps >= plot_start_ms) & (timestamps <= plot_end_ms)
        ts_plot = timestamps[mask]
        regime_plot = regime_q[mask]
        fprobs_plot = hmm_detector.filtered_probs_[mask, :]

        if len(ts_plot) == 0:
            ax1.set_title(f"{crisis['name']}: no data in window")
            continue

        # Downsample to hourly (stride=60) for plotting
        stride = 60
        ts_ds = ts_plot[::stride]
        regime_ds = regime_plot[::stride]
        fprobs_ds = fprobs_plot[::stride, :]

        # Convert epoch-ms to datetime for x-axis labels
        ts_dt = [_dt.datetime.fromtimestamp(t / 1000, tz=_dt.UTC) for t in ts_ds]

        # Regime shading on primary axis
        for regime_label, color, name in [
            (0, OKABE_ITO["sky_blue"], "LOW"),
            (1, OKABE_ITO["orange"], "ELEVATED"),
            (2, OKABE_ITO["vermillion"], "EXTREME"),
        ]:
            ax1.fill_between(
                ts_dt,
                0,
                1,
                where=(regime_ds == regime_label),
                color=color,
                alpha=0.35,
                label=name,
                transform=ax1.get_xaxis_transform(),
            )
        ax1.set_ylim(0, 1)
        ax1.set_yticks([])
        ax1.set_ylabel("Regime (quantile detector)")

        # HMM filtered P(high-variance state) on twin axis
        ax2 = ax1.twinx()
        ax2.plot(
            ts_dt,
            fprobs_ds[:, high_var_col],
            color=OKABE_ITO["reddish_purple"],
            linewidth=0.8,
            alpha=0.85,
            label="HMM filtered P(high-var)",
        )
        ax2.set_ylabel("HMM P(high-var state)", color=OKABE_ITO["reddish_purple"])
        ax2.set_ylim(0, 1.05)
        ax2.axhline(
            0.9999,
            linestyle=":",
            color=OKABE_ITO["reddish_purple"],
            linewidth=0.6,
            alpha=0.5,
        )

        # x-axis formatting: show year and month
        import matplotlib.dates as mdates
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right")

        ax1.set_xlabel("Date (2022)")
        ax1.set_title(f"Fig 4: {crisis['event_label']}")

        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=7)

    fig.suptitle(
        "Fig 4: Crisis-timeline regime labels + HMM soft-prob overlay "
        "(D-06 2-state fallback)",
        fontsize=9,
    )
    fig.tight_layout()

    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / "fig04_crisis_timelines.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Fig 4 saved: fig04_crisis_timelines.png")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> int:
    """Load 2021-2025 BTC, emit all 5 figures, write D-15 provenance stamp."""
    global _PROVENANCE

    # -----------------------------------------------------------------------
    # D-10 holdout guard Layer 1 (epoch-ms boundary) — verbatim from
    # validate_rolling_quantile.py lines 121-133
    # -----------------------------------------------------------------------
    start_ms = int(
        _dt.datetime(2021, 1, 1, tzinfo=_dt.UTC).timestamp() * 1000
    )
    end_ms = int(
        _dt.datetime(2025, 12, 31, 23, 59, 59, tzinfo=_dt.UTC).timestamp() * 1000
    )
    holdout_boundary_ms = int(
        _dt.datetime(2026, 1, 1, tzinfo=_dt.UTC).timestamp() * 1000
    )
    assert end_ms < holdout_boundary_ms, (
        f"D-10 VIOLATION: data window end ({end_ms}) must be < 2026-01-01 "
        f"({holdout_boundary_ms}). The 2026 holdout must NOT be loaded."
    )

    data_path = str(PROJECT_ROOT / "data" / "binance_futures")
    symbol = "BTCUSDT"

    print(f"Loading {symbol} 2021-2025 (D-10 holdout: 2026 NOT loaded)...")
    data = load_and_clean(
        data_path=data_path,
        symbol=symbol,
        start_ms=start_ms,
        end_ms=end_ms,
        max_gap_allowed_mins=60,
    )

    close = data["close"]
    timestamps = data["timestamp"]
    N = len(close)

    # Confirm data span
    ts_start = _dt.datetime.fromtimestamp(timestamps[0] / 1000, tz=_dt.UTC)
    ts_end = _dt.datetime.fromtimestamp(timestamps[-1] / 1000, tz=_dt.UTC)
    print(f"Data span: {ts_start.date()} -> {ts_end.date()} ({N:,} bars)")

    # -----------------------------------------------------------------------
    # D-10 holdout guard Layer 2: last loaded bar must not be in 2026 —
    # verbatim from validate_rolling_quantile.py lines 159-162
    # -----------------------------------------------------------------------
    assert ts_end.year < 2026, (
        f"D-10 VIOLATION: last loaded bar is in {ts_end.year} "
        "(year=2026 partition was read). The 2026 holdout must remain UNTOUCHED."
    )

    # -----------------------------------------------------------------------
    # Ensure output directory exists (Pitfall 7, RESEARCH.md)
    # -----------------------------------------------------------------------
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Emit all figures
    # -----------------------------------------------------------------------
    print("\nGenerating figures...")

    print("Fig 1: v1.0 RV-ratio histogram...")
    _plot_fig1(close, FIGURES_DIR)

    print("Fig 2: epsilon-squared grid...")
    _plot_fig2(FIGURES_DIR)

    print("Fig 3: regime populations...")
    _plot_fig3(FIGURES_DIR)

    print("Fig 5: v1.0 vs v2.0 comparison...")
    _plot_fig5(FIGURES_DIR)

    print("Fig 4: LUNA/FTX crisis timelines with HMM soft-prob overlay...")
    _plot_fig4(close, timestamps, FIGURES_DIR)

    # -----------------------------------------------------------------------
    # D-15 provenance stamp
    # -----------------------------------------------------------------------
    try:
        matplotlib_version = importlib.metadata.version("matplotlib")
    except importlib.metadata.PackageNotFoundError:
        matplotlib_version = matplotlib.__version__
    try:
        numpy_version = importlib.metadata.version("numpy")
    except importlib.metadata.PackageNotFoundError:
        numpy_version = str(np.__version__)

    git_commit = _get_git_commit()
    run_date = datetime.now(tz=_dt.UTC).strftime("%Y%m%d_%H%M%S")

    provenance = {
        "run_date": run_date,
        "year_2026_loaded": False,
        "data_span": {
            "start": str(ts_start.date()),
            "end": str(ts_end.date()),
            "n_bars": N,
        },
        "prereg_commit": "169fc20",
        "phase4_amendment_commit": "04b667e",
        "phase4_amendment_ref": "01-PREREGISTRATION.md §14.5 Phase-4 bundle (commit 04b667e)",
        "git_commit": git_commit,
        "matplotlib_version": matplotlib_version,
        "numpy_version": numpy_version,
        "python_version": sys.version,
        "figures_emitted": [
            "fig01_diagnosis_histogram.png",
            "fig02_epsilon_sq_grid.png",
            "fig03_regime_populations.png",
            "fig04_crisis_timelines.png",
            "fig05_v1v2_comparison.png",
        ],
    }

    # Populate module-level _PROVENANCE so test_provenance_stamp can access it
    _PROVENANCE = provenance

    # Write JSON sidecar
    sidecar_path = FIGURES_DIR / f"provenance_{run_date}.json"
    sidecar_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    print()
    print("=" * 70)
    print(f"All 5 figures written to: {FIGURES_DIR}")
    print(f"Provenance sidecar: {sidecar_path}")
    print(f"year_2026_loaded: {provenance['year_2026_loaded']}")
    print(f"prereg_commit: {provenance['prereg_commit']}")
    print(f"git_commit: {git_commit}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
