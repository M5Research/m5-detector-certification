"""Pre-registered gauge-invariance figures from stamped JSON report."""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
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

GAUGE_DIR = PROJECT_ROOT / "backtest_results" / "gauge_invariance"
Q_GRID = (2, 5, 15, 60)
GAUGE_NAMES = ("clock", "volume", "intrinsic")

__all__ = [
    "load_gauge_report",
    "generate_horizon_profile_figure",
    "generate_gauge_overlap_matrix",
    "generate_attenuation_plot",
    "generate_all_figures",
]


def load_gauge_report(path: str | None = None) -> dict:
    """Load latest or specified gauge invariance JSON report."""
    if path is None:
        candidates = sorted(GAUGE_DIR.glob("gauge_report_*.json"), key=lambda p: p.stat().st_mtime)
        if not candidates:
            raise FileNotFoundError(
                "No gauge_report_*.json found. Run: python scripts/wp1/gauge_invariance.py --run"
            )
        report_path = candidates[-1]
    else:
        report_path = Path(path)
    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)
    report["_path"] = str(report_path)
    return report


def _horizon_points(gauge_block: dict) -> tuple[list[int], list[float], list[float], list[float]]:
    profile = gauge_block.get("horizon_profile", {})
    primary = gauge_block.get("per_wq_primary_q", [])
    q_vals: list[int] = []
    medians: list[float] = []
    ci_lo: list[float] = []
    ci_hi: list[float] = []
    for q in Q_GRID:
        q_key = str(q)
        med = float(profile.get(q_key, np.nan))
        lo = hi = med
        for cell in primary:
            if int(cell["q"]) == q:
                lo = float(cell.get("ci_95_lo", med))
                hi = float(cell.get("ci_95_hi", med))
                med = float(cell.get("median_vr_dep", med))
                break
        q_vals.append(q)
        medians.append(med)
        ci_lo.append(lo)
        ci_hi.append(hi)
    return q_vals, medians, ci_lo, ci_hi


def generate_horizon_profile_figure(report: dict, out_path: Path) -> None:
    """Three-panel horizon profile: median |VR-1| vs q per gauge."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, gauge in zip(axes, GAUGE_NAMES, strict=True):
        block = report["gauges"][gauge]["fixed_bar"]
        q_vals, medians, ci_lo, ci_hi = _horizon_points(block)
        yerr = [
            [m - lo for m, lo in zip(medians, ci_lo, strict=True)],
            [hi - m for m, hi in zip(medians, ci_hi, strict=True)],
        ]
        ax.errorbar(q_vals, medians, yerr=yerr, fmt="o-", capsize=4)
        ax.set_title(gauge.capitalize())
        ax.set_xlabel("q")
        ax.set_xticks(list(Q_GRID))
    axes[0].set_ylabel("median |VR-1|")
    fig.suptitle("Gauge Invariance — Horizon Profile (fixed bar count)")
    verdict = report.get("verdict", "")
    prereg = report.get("prereg_commit", "")[:12]
    fig.text(0.5, 0.01, f"prereg={prereg}  verdict={verdict}", ha="center", fontsize=8)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _ci_overlap(lo_a: float, hi_a: float, lo_b: float, hi_b: float) -> bool:
    return not (hi_a < lo_b or hi_b < lo_a)


def generate_gauge_overlap_matrix(report: dict, out_path: Path) -> None:
    """Heatmap of pairwise CI overlap per q horizon."""
    pairs = [("clock", "volume"), ("clock", "intrinsic"), ("volume", "intrinsic")]
    matrix = np.zeros((len(Q_GRID), len(pairs)))
    for i, q in enumerate(Q_GRID):
        for j, (ga, gb) in enumerate(pairs):
            a = report["gauges"][ga]["fixed_bar"]["per_wq_primary_q"]
            b = report["gauges"][gb]["fixed_bar"]["per_wq_primary_q"]
            cell_a = next((c for c in a if int(c["q"]) == q), None)
            cell_b = next((c for c in b if int(c["q"]) == q), None)
            if cell_a and cell_b:
                matrix[i, j] = float(
                    _ci_overlap(
                        float(cell_a["ci_95_lo"]),
                        float(cell_a["ci_95_hi"]),
                        float(cell_b["ci_95_lo"]),
                        float(cell_b["ci_95_hi"]),
                    )
                )
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels([f"{a}-{b}" for a, b in pairs], rotation=30, ha="right")
    ax.set_yticks(range(len(Q_GRID)))
    ax.set_yticklabels([f"q={q}" for q in Q_GRID])
    ax.set_title("Gauge Overlap Matrix (CI overlap)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def generate_attenuation_plot(report: dict, out_path: Path) -> None:
    """|VR(2)-1| vs |VR(60)-1| across gauges (bid-ask bounce signature)."""
    q_short, q_long = 2, 60
    short_vals: list[float] = []
    long_vals: list[float] = []
    for gauge in GAUGE_NAMES:
        profile = report["gauges"][gauge]["fixed_bar"]["horizon_profile"]
        short_vals.append(float(profile.get(str(q_short), np.nan)))
        long_vals.append(float(profile.get(str(q_long), np.nan)))

    x = np.arange(len(GAUGE_NAMES))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width / 2, short_vals, width, label=f"q={q_short}")
    ax.bar(x + width / 2, long_vals, width, label=f"q={q_long}")
    ax.set_xticks(x)
    ax.set_xticklabels([g.capitalize() for g in GAUGE_NAMES])
    ax.set_ylabel("median |VR-1|")
    ax.set_title("Volume-Time Attenuation (short vs long horizon)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def generate_all_figures(report_path: str | None = None) -> list[Path]:
    """Generate all three pre-registered gauge figures."""
    report = load_gauge_report(report_path)
    run_id = report.get("run_id", datetime.utcnow().strftime("%Y%m%d_%H%M%S"))
    GAUGE_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [
        GAUGE_DIR / f"gauge_horizon_profile_{run_id}.png",
        GAUGE_DIR / f"gauge_overlap_matrix_{run_id}.png",
        GAUGE_DIR / f"gauge_attenuation_{run_id}.png",
    ]
    generate_horizon_profile_figure(report, outputs[0])
    generate_gauge_overlap_matrix(report, outputs[1])
    generate_attenuation_plot(report, outputs[2])
    return outputs


def main() -> int:
    paths = generate_all_figures()
    for p in paths:
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
