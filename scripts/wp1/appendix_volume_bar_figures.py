"""Appendix A2: Volume-bar count vs. calendar time (D-11).

Plots the distribution of volume-bar counts per calendar day over the in-sample
period, showing how volume-time sampling clusters during high-activity periods.

Reads bar-count data from the frozen gauge invariance report JSON.
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


def generate_volume_bar_figure(gauge_report_path: str, output_dir: Path) -> Path:
    """Generate Fig A2: Volume-bar count vs. calendar time.

    If the gauge report contains per-gauge bar count data, plots distributions.
    Otherwise produces a schematic placeholder noting data availability.
    """
    report = json.loads(Path(gauge_report_path).read_text(encoding="utf-8"))

    fig, ax = plt.subplots(figsize=(8, 4))

    # Check for bar-count data in the gauge report
    bar_counts = report.get("per_gauge", {}).get("bar_counts")
    if bar_counts and all(k in bar_counts for k in ("clock", "volume", "intrinsic")):
        # Plot bar-count distributions per gauge
        positions = [1, 2, 3]
        counts = [
            bar_counts["clock"],
            bar_counts["volume"],
            bar_counts["intrinsic"],
        ]
        labels = ["Calendar-time\n(1-min)", "Volume-time\n($V$-bar)", "Intrinsic-time\n($\\Delta P$-bar)"]

        bars = ax.bar(positions, counts, color=["steelblue", "darkorange", "seagreen"],
                      edgecolor="white", linewidth=0.5)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Number of bars")
        ax.set_title("Appendix A2: Bar Count by Gauge Construction")

        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.02,
                    f"{count:,}", ha="center", va="bottom", fontsize=9)
    else:
        # Placeholder: no bar-count data in report
        ax.text(0.5, 0.5, "Bar-count data not available in gauge report.\n"
                "Re-run gauge pipeline with --save-bar-counts.",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color="gray")
        ax.set_title("Appendix A2: Volume-Bar Count vs. Calendar Time (placeholder)")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out_path = output_dir / "figA2_volume_bar_count.png"
    fig.savefig(str(out_path))
    plt.close(fig)
    print(f"  Saved {out_path}")
    return out_path


if __name__ == "__main__":
    gauge_path = str(
        PROJECT_ROOT
        / "backtest_results"
        / "gauge_invariance"
        / "gauge_report_20260624_221834.json"
    )
    output_dir = (
        PROJECT_ROOT / "docs" / "research" / "calibrated-detector-exclusion"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    generate_volume_bar_figure(gauge_path, output_dir)
