"""Thermodynamic profit bound figures from stamped JSON report."""
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
from scripts.wp1.thermodynamic_bound import (  # noqa: E402
    MAKER_RT_BPS,
    PRIMARY_Q,
    TAKER_RT_BPS,
    W_THERMO,
)

THERMO_DIR = PROJECT_ROOT / "backtest_results" / "thermodynamic_bound"

__all__ = ["load_thermo_report", "generate_bound_vs_cost_figure"]


def load_thermo_report(results_dir: Path | None = None) -> dict:
    """Load latest thermo_report_*.json under results_dir."""
    base = THERMO_DIR if results_dir is None else Path(results_dir)
    if ".." in base.parts:
        raise ValueError("results_dir must not contain '..'")
    resolved = base.resolve()
    try:
        resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise ValueError("results_dir must resolve under PROJECT_ROOT") from exc

    candidates = sorted(resolved.glob("thermo_report_*.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No thermo_report_*.json in {resolved}. "
            "Run: python scripts/wp1/thermodynamic_bound.py --run"
        )
    with open(candidates[-1], encoding="utf-8") as f:
        report = json.load(f)
    report["_path"] = str(candidates[-1])
    return report


def generate_bound_vs_cost_figure(
    report: dict,
    output_dir: Path | None = None,
) -> Path:
    """Grouped bar chart: Gaussian + KSG bounds vs transaction costs."""
    per_q = sorted(report["per_q_results"], key=lambda row: row["q"])
    q_vals = [row["q"] for row in per_q]
    gauss = [row["gaussian_gmax_bps"] for row in per_q]
    ksg = [row["ksg_gmax_bps"] for row in per_q]

    x = np.arange(len(q_vals))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, gauss, width, label="Observed Gaussian bound", color="#2c7bb6")
    ax.bar(x + width / 2, ksg, width, label="KSG directional check", color="#abd9e9")

    envelope_status = report.get("delta_star_info", {}).get("status")
    envelope_vals = [row.get("envelope_gaussian_gmax_bps") for row in per_q]
    if envelope_status not in (None, "injection_dir_missing") and any(
        v is not None for v in envelope_vals
    ):
        env_y = [v if v is not None else np.nan for v in envelope_vals]
        ax.plot(
            x,
            env_y,
            "k--",
            marker="o",
            label="δ*₉₀ envelope (90% detection limit)",
        )

    ax.axhspan(min(TAKER_RT_BPS), max(TAKER_RT_BPS), alpha=0.15, color="red", label="Taker RT cost")
    ax.axhspan(min(MAKER_RT_BPS), max(MAKER_RT_BPS), alpha=0.15, color="green", label="Maker RT cost")

    ymax = max(
        [v for v in gauss + ksg + [v for v in envelope_vals if v is not None]]
        + [max(TAKER_RT_BPS)]
    )
    ymin = min([0.0] + [v for v in gauss + ksg if v > 0])
    if ymax / max(ymin, 1e-6) > 100:
        ax.set_yscale("log")
    ax.set_ylabel("Maximum extractable profit (bps/trade)")
    ax.set_xlabel("q (VR horizon)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(q) for q in q_vals])
    ax.set_title("Thermodynamic Profit Bound vs Transaction Costs")
    ax.legend(loc="upper right", fontsize=8)

    verdict = report.get("headline_verdict", "unknown")
    ax.text(
        0.02,
        0.02,
        f"Verdict: {verdict.replace('_', ' ')} (q={PRIMARY_Q}, W={W_THERMO})",
        transform=ax.transAxes,
        fontsize=9,
    )

    run_date = report.get("run_id") or report.get("run_date") or "undated"
    out_dir = THERMO_DIR if output_dir is None else Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"thermo_bound_vs_cost_{run_date}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    rep = load_thermo_report()
    path = generate_bound_vs_cost_figure(rep)
    print(path)
