"""Appendix A1: Detection efficiency robustness across window widths (D-11).

Plots P_det(delta) at the primary q=5 across all three window widths
(W = 60, 120, 240), demonstrating the zero-detection result is consistent
across the full (W,q) grid, not just the primary cell.

Reads frozen injection-grid JSONs from data/injection_runs/.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backtest.utils import PROJECT_ROOT  # noqa: E402
import scripts._bootstrap  # noqa: F401, E402

DELTA_GRID = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10]
PRIMARY_Q = 5
W_VALUES = [60, 120, 240]
W_COLORS = ["#2c7bb6", "#fdae61", "#d7191c"]
W_MARKERS = ["o", "s", "D"]


def load_injection_results(injection_dir: Path) -> dict:
    """Load injection grid JSONs into {(delta, q, W): data} dict."""
    results = {}
    for jp in injection_dir.glob("inj_*.json"):
        data = json.loads(jp.read_text(encoding="utf-8"))
        key = (data.get("delta", data.get("delta_target")), data["q"], data["W"])
        results[key] = data
    return results


def generate_regime_stability_figure(
    injection_dir: Path, output_dir: Path
) -> Path:
    """Generate Fig A1: P_det(delta) across all W at q=5.

    Since P_det = 0 everywhere, regime-level stratification is degenerate.
    Instead, this figure shows robustness across window widths, confirming
    the zero-detection result is not specific to a single W value.
    """
    results = load_injection_results(injection_dir)

    fig, ax = plt.subplots(figsize=(8, 5))

    for W, color, marker in zip(W_VALUES, W_COLORS, W_MARKERS):
        deltas = []
        p_dets = []
        ci_los = []
        ci_his = []
        for delta in DELTA_GRID:
            key = (delta, PRIMARY_Q, W)
            if key in results:
                d = results[key]
                deltas.append(d.get("delta", d.get("delta_target")))
                p_dets.append(d["P_det"])
                ci_los.append(d["ci_95_lo"])
                ci_his.append(d["ci_95_hi"])

        # Offset slightly for visibility (all curves are at y=0, so overlap)
        offset = (W_VALUES.index(W) - 1) * 0.002
        p_dets_offset = [p + offset for p in p_dets]

        ax.errorbar(deltas, p_dets_offset,
                    yerr=[ [max(0, p + o - lo) for p, lo, o in zip(p_dets, ci_los, [offset]*len(p_dets))],
                           [max(0, hi - p - o) for p, hi, o in zip(p_dets, ci_his, [offset]*len(p_dets))] ],
                    fmt=marker + "-", color=color, linewidth=1.5, markersize=6,
                    capsize=3, label=f"$W={W}$")

    ax.axhline(0.90, color="gray", linestyle="--", alpha=0.6,
               label="$P_{\\mathrm{det}} = 0.90$")

    ax.set_xlabel("$\\delta$ (injected autocorrelation)")
    ax.set_ylabel("$P_{\\mathrm{det}}$ (+ offset for visibility)")
    ax.set_title("Appendix A1: Detection Efficiency by Window Width ($q=5$)")
    ax.legend(loc="lower right")
    ax.set_xscale("log")
    ax.set_ylim(-0.05, 1.10)

    # Annotate the null result
    ax.text(0.5, 0.5,
            "$P_{\\mathrm{det}} = 0$ at all\n$(\\delta, W)$ for $q=5$\n"
            "($N_{\\mathrm{MC}} = 200$ per cell)",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=11, bbox=dict(boxstyle="round,pad=0.5",
                                   facecolor="lightyellow", alpha=0.9))

    out_path = output_dir / "figA1_regime_stability.png"
    fig.savefig(str(out_path))
    plt.close(fig)
    print(f"  Saved {out_path}")
    return out_path


if __name__ == "__main__":
    injection_dir = PROJECT_ROOT / "data" / "injection_runs"
    output_dir = (
        PROJECT_ROOT / "docs" / "research" / "calibrated-detector-exclusion"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    generate_regime_stability_figure(injection_dir, output_dir)
