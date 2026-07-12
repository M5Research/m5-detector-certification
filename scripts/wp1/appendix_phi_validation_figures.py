"""Appendix figure: achieved |VR(q)-1| vs target delta per horizon q.

Validates the phi-to-delta inversion (C10 / R1).  Forward-evaluates achieved
delta only at the phi values tabulated in phi_grid.json (fast subset run).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
import sys

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backtest.utils import PROJECT_ROOT  # noqa: E402
from scripts.wp1.signal_injector import phi_to_delta_mapping  # noqa: E402

INJECTION_DIR = PROJECT_ROOT / "data" / "injection_runs"
V4_FIGURE_DIR = PROJECT_ROOT / "docs" / "research" / "calibrated-detector-exclusion"
Q_COLORS = {2: "#1f77b4", 5: "#ff7f0e", 15: "#2ca02c", 60: "#d62728"}


def _load_unique_delta_q() -> list[dict]:
    rows: dict[tuple[float, int], float] = {}
    for name in ("phi_grid.json", "phi_grid_addendum.json"):
        path = INJECTION_DIR / name
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for item in payload.get("unique_delta_q", []):
            key = (float(item["delta"]), int(item["q"]))
            rows[key] = float(item["phi"])
    return [{"delta": d, "q": q, "phi": phi} for (d, q), phi in sorted(rows.items())]


def _forward_at_phis(phis: np.ndarray) -> dict[float, dict[int, float]]:
    """Forward mapping at tabulated phi values only (subset calibration)."""
    mapping = phi_to_delta_mapping(phis_to_test=phis, N=50_000)
    return {float(p): {q: float(mapping[float(p)][q]) for q in (2, 5, 15, 60)} for p in phis}


def generate_phi_validation_figure(output_dir: Path | None = None) -> Path:
    output_dir = output_dir or V4_FIGURE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = _load_unique_delta_q()
    phis = np.array(sorted({p["phi"] for p in pairs if p["phi"] > 0}))
    forward = _forward_at_phis(phis)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    all_targets: list[float] = []
    all_achieved: list[float] = []

    for q in sorted({p["q"] for p in pairs}):
        subset = [p for p in pairs if p["q"] == q]
        targets = np.array([p["delta"] for p in subset])
        achieved = np.array([forward[p["phi"]][q] for p in subset])
        all_targets.extend(targets.tolist())
        all_achieved.extend(achieved.tolist())
        ax.plot(
            targets,
            achieved,
            "o-",
            color=Q_COLORS.get(q, "gray"),
            linewidth=1.5,
            markersize=5,
            label=f"$q={q}$",
        )

    lim_lo = min(all_targets) * 0.5
    lim_hi = max(all_targets) * 1.2
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", alpha=0.4, linewidth=1, label="45° line")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Target $\delta = |\mathrm{VR}(q)-1|$")
    ax.set_ylabel(r"Achieved median $|\widehat{\mathrm{VR}}(q)-1|$")
    ax.set_title(r"Injection calibration: achieved vs.\ target $\delta$ per $q$")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    out_path = output_dir / "figA4_phi_validation.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")

    # Print key validation point for manuscript prose
    for p in pairs:
        if abs(p["delta"] - 0.10) < 1e-9 and p["q"] == 2:
            ach = forward[p["phi"]][2]
            print(f"q=2, target=0.10, achieved={ach:.4f}, phi={p['phi']:.4f}")
    return out_path


if __name__ == "__main__":
    generate_phi_validation_figure()
