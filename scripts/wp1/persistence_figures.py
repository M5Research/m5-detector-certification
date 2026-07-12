"""Persistence null survival figures from stamped JSON report."""
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
from scripts.wp1.persistence_null import (  # noqa: E402
    MAX_HORIZON,
    PERSIST_DIR,
    arcsin_survival,
)

__all__ = ["load_persistence_report", "generate_survival_figure"]


def load_persistence_report(results_dir: Path | None = None) -> dict:
    """Load latest persistence_report_*.json under results_dir."""
    base = PERSIST_DIR if results_dir is None else Path(results_dir)
    if ".." in base.parts:
        raise ValueError("results_dir must not contain '..'")
    resolved = base.resolve()
    try:
        resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise ValueError("results_dir must resolve under PROJECT_ROOT") from exc

    candidates = sorted(resolved.glob("persistence_report_*.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No persistence_report_*.json in {resolved}. "
            "Run: python scripts/wp1/persistence_null.py --run"
        )
    with open(candidates[-1], encoding="utf-8") as f:
        report = json.load(f)
    report["_path"] = str(candidates[-1])
    return report


def generate_survival_figure(
    report: dict,
    output_dir: Path | None = None,
) -> Path:
    """Dual-panel survival figure: left linear, right log-log (D-21)."""
    t_grid = np.arange(1, report["max_horizon"] + 1)
    s_emp = np.array(report["S_empirical"], dtype=np.float64)
    s0 = arcsin_survival(t_grid)
    ci_lo = np.array(report["bootstrap_ci_lo"], dtype=np.float64)
    ci_hi = np.array(report["bootstrap_ci_hi"], dtype=np.float64)

    fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(12, 5))

    ks_p = report["bootstrap_p_value"]
    theta = report["theta"]
    tci = report["theta_ci"]
    title = (
        f"θ={theta:.3f} (95% CI [{tci[0]:.3f},{tci[1]:.3f}])\n"
        f"KS p={ks_p:.3f} — {report['verdict'].replace('_', ' ')}"
    )

    for ax, xscale, yscale in [
        (ax_lin, "linear", "linear"),
        (ax_log, "log", "log"),
    ]:
        lo = np.maximum(ci_lo, 1e-6) if yscale == "log" else ci_lo
        emp = np.maximum(s_emp, 1e-6) if yscale == "log" else s_emp
        ref = np.maximum(s0, 1e-6) if yscale == "log" else s0
        hi = np.maximum(ci_hi, 1e-6) if yscale == "log" else ci_hi

        ax.fill_between(t_grid, lo, hi, alpha=0.25, color="steelblue", label="Bootstrap CI (95%)")
        ax.plot(t_grid, emp, color="steelblue", lw=2, label="Empirical S(t)")
        ax.plot(t_grid, ref, color="tomato", ls="--", lw=1.5, label="Arcsin null S₀(t)")
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        ax.set_xlabel("Horizon t (bars)")
        ax.set_ylabel("Survival probability S(t)")
        ax.set_title(title)
        ax.legend(loc="upper right", fontsize=8)

    run_date = report.get("run_timestamp") or report.get("run_id") or "undated"
    out_dir = PERSIST_DIR if output_dir is None else Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"persistence_survival_{run_date}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    rep = load_persistence_report()
    path = generate_survival_figure(rep)
    print(path)
