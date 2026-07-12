"""Generate Appendix B simulation figure: i.i.d. null VR noise floor.

Four-panel figure:
  (a) q=2  (b) q=5  (c) q=15  (d) q=60
Each panel: histogram of per-window median |VR(q)-1| under i.i.d. Gaussian null,
with horizontal lines at the 0.001 closure floor and the 95th percentile.

Output: docs/research/figures/fig_simulation_null.png
"""
from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import matplotlib

matplotlib.use("Agg")

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.wp1.vr_significance import compute_rolling_vr_and_z  # noqa: E402
from strategies.vol_regime_switch.regime_population import (  # noqa: E402
    non_overlapping_samples,
)

FIGURES_DIR = _REPO_ROOT / "docs" / "research" / "figures"
CLOSURE_FLOOR = 0.001
W = 120
N_SIM = 50_000
SEED = 42
QS = (2, 5, 15, 60)


def _iid_median_vr_deps(rng: np.random.Generator, q: int) -> np.ndarray:
    r = rng.normal(0, 0.001, N_SIM)
    close = np.exp(np.cumsum(r))
    vr_arr, _ = compute_rolling_vr_and_z(close, W=W, q=q)
    regime = np.zeros(N_SIM, dtype=np.int8)
    pred_nl, _, retained_idx = non_overlapping_samples(vr_arr, regime, stride=W)
    return pred_nl[np.isfinite(pred_nl)]


def main() -> None:
    rng = np.random.default_rng(SEED)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "figure.dpi": 150,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.5))
    axes_flat = axes.ravel()

    for ax, q in zip(axes_flat, QS):
        deps = _iid_median_vr_deps(rng, q)
        p95 = float(np.percentile(deps, 95))
        ax.hist(deps, bins=40, color="#4C72B0", alpha=0.85, edgecolor="white")
        ax.axvline(
            CLOSURE_FLOOR,
            color="#C44E52",
            linestyle="--",
            linewidth=1.5,
            label=f"Closure floor ({CLOSURE_FLOOR})",
        )
        ax.axvline(
            p95,
            color="#55A868",
            linestyle="-.",
            linewidth=1.5,
            label=f"95th pct ({p95:.3f})",
        )
        ax.set_title(f"$q = {q}$")
        ax.set_xlabel("Per-window $|VR(q)-1|$")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(
        f"Synthetic i.i.d. null: median $|VR(q)-1|$ per $W={W}$-bar window "
        f"($N={N_SIM:,}$ bars, seed={SEED})",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    out = FIGURES_DIR / "fig_simulation_null.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
