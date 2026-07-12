"""Appendix A3: MI estimator validation (D-11).

Generates synthetic bivariate Gaussian data with known mutual information,
applies the KSG estimator, and plots recovery (bias + variance) against
the analytic ground truth across sample sizes.

No external data dependency — fully self-contained validation figure.
Uses the production ``ksg_mi_2d`` estimator (numba-accelerated marginals).
"""
from __future__ import annotations

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
from scripts.wp1.mutual_information import ksg_mi_2d  # noqa: E402

try:
    import numba
except ImportError:  # pragma: no cover
    numba = None  # type: ignore[assignment]

SEED = 43
SAMPLE_SIZES = [200, 500, 1000, 2000, 5000, 10000]
TRUE_MI_VALUES = [0.0, 0.2, 0.5, 1.0, 1.5]
N_REPEATS = 20
VALIDATION_K = 3


def _rho_from_mi(mi_true: float) -> float:
    if mi_true <= 0.0:
        return 0.0
    return float(np.sqrt(1.0 - np.exp(-2.0 * mi_true)))


if numba is not None:

    @numba.njit(cache=True)
    def _bivariate_gaussian_batch(
        n: int,
        rho: float,
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Draw n correlated standard normals with correlation rho."""
        np.random.seed(seed % (2**32 - 1))
        z1 = np.random.standard_normal(n)
        z2 = np.random.standard_normal(n)
        x = z1
        y = rho * z1 + np.sqrt(1.0 - rho * rho) * z2
        return x, y


def _generate_bivariate_gaussian(
    n: int,
    rho: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if numba is not None:
        seed = int(rng.integers(0, 2**31 - 1))
        return _bivariate_gaussian_batch(n, rho, seed)
    cov = np.array([[1.0, rho], [rho, 1.0]])
    samples = rng.multivariate_normal([0.0, 0.0], cov, size=n)
    return samples[:, 0], samples[:, 1]


def generate_mi_validation_figure(output_dir: Path) -> Path:
    """Generate Fig A3: KSG estimator recovery vs analytic MI."""
    import warnings

    warnings.filterwarnings("ignore")

    rng = np.random.default_rng(SEED)
    n_samples = len(SAMPLE_SIZES)
    n_mi_values = len(TRUE_MI_VALUES)

    bias = np.zeros((n_samples, n_mi_values))
    std_err = np.zeros((n_samples, n_mi_values))

    total = n_samples * n_mi_values * N_REPEATS
    done = 0
    for i, n in enumerate(SAMPLE_SIZES):
        for j, mi_true in enumerate(TRUE_MI_VALUES):
            rho = _rho_from_mi(mi_true)
            estimates = np.empty(N_REPEATS, dtype=np.float64)
            for r in range(N_REPEATS):
                x, y = _generate_bivariate_gaussian(n, rho, rng)
                estimates[r] = ksg_mi_2d(x, y, k=VALIDATION_K)
                done += 1
                if done % 50 == 0 or done == total:
                    print(f"  A3 KSG progress: {done}/{total}", flush=True)

            bias[i, j] = np.mean(estimates) - mi_true
            std_err[i, j] = np.std(estimates) / np.sqrt(N_REPEATS)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_mi_values))

    for j, (mi_true, c) in enumerate(zip(TRUE_MI_VALUES, colors)):
        ax1.errorbar(
            SAMPLE_SIZES,
            bias[:, j],
            yerr=2 * std_err[:, j],
            fmt="o-",
            color=c,
            capsize=3,
            markersize=5,
            label=f"$I(X;Y) = {mi_true}$",
        )
    ax1.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax1.set_xlabel("Sample size $n$")
    ax1.set_ylabel("Bias $\\hat{I} - I_{\\mathrm{true}}$ (nats)")
    ax1.set_title("KSG Estimator Bias vs Sample Size")
    ax1.set_xscale("log")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    n_idx = SAMPLE_SIZES.index(5000)
    ax2.errorbar(
        TRUE_MI_VALUES,
        bias[n_idx, :],
        yerr=2 * std_err[n_idx, :],
        fmt="o-",
        color="steelblue",
        capsize=4,
        markersize=7,
    )
    ax2.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax2.set_xlabel("True $I(X;Y)$ (nats)")
    ax2.set_ylabel("Bias $\\hat{I} - I_{\\mathrm{true}}$ (nats)")
    ax2.set_title(f"KSG Recovery at $n = 5000$ ($N = {N_REPEATS}$ repeats)")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle(
        "Appendix A3: KSG Mutual Information Estimator Validation",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()

    out_path = output_dir / "figA3_mi_validation.png"
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}", flush=True)
    return out_path


if __name__ == "__main__":
    output_dir = (
        PROJECT_ROOT / "docs" / "research" / "calibrated-detector-exclusion"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    generate_mi_validation_figure(output_dir)
