"""Phase 10 new figure generation driver: z_m2 horizon-profile + provenance.

Generates the SC3 Pre-Check B figure for the terminal-synthesis manuscript.

Figures:
  fig_zm2_horizon.png  -- median |VR(q)-1| vs aggregation horizon q,
                          with 95% CI band, 0.001 closure reference line,
                          and the LOW-regime noise floor series.

Integrity controls:
  - VISUALIZATION ONLY: reads frozen precheck_b_20260605_212924.json;
    does NOT load Parquet, does NOT import predictability / vr_significance /
    regate_analysis / gate_analysis.
  - D-10 HOLDOUT GUARD (JSON-based): asserts data_span['year_2026_loaded']
    is False before any plotting; raises AssertionError("D-10 VIOLATION ...").
  - D-15: provenance sidecar emitted to docs/research/figures/ carrying
    year_2026_loaded:false, prereg_commit:720c1d4, git_commit, library versions.
  - Pixel-deterministic: Agg backend + fixed x/y data from frozen JSON.

Output:
    docs/research/figures/fig_zm2_horizon.png
    docs/research/figures/provenance_<timestamp>.json
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# D-12: Force single-threaded BLAS BEFORE any statsmodels / matplotlib import.
# OMP/MKL/OPENBLAS env vars must be set first, then the Agg backend.
# ---------------------------------------------------------------------------
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# ---------------------------------------------------------------------------
# matplotlib.use('Agg') MUST come before pyplot import.
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
# Path bootstrap (mirrors generate_figures.py lines 64-72)
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_REPO_SRC = _REPO_ROOT / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

# ---------------------------------------------------------------------------
# Frozen artifact paths (visualization-only — no Parquet, no data loader)
# ---------------------------------------------------------------------------
PRECHECK_B_JSON = (
    _REPO_ROOT / "backtest_results" / "precheck" / "precheck_b_20260605_212924.json"
)
FIGURES_DIR = _REPO_ROOT / "docs" / "research" / "figures"

# Pre-registration freeze anchor (Phase 7, human-ratified)
PREREG_COMMIT = "720c1d4"

# Closure threshold (frozen S7)
CLOSURE_THRESHOLD = 0.001

# ---------------------------------------------------------------------------
# Publication-quality rcParams (mirrors generate_figures.py Pattern 4)
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

# ---------------------------------------------------------------------------
# Git commit helper (D-15)
# ---------------------------------------------------------------------------


def _get_git_commit() -> str:
    """Return the short HEAD hash, or a safe fallback string if git is unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(_REPO_ROOT),
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    return "unavailable"


# ---------------------------------------------------------------------------
# Module-level provenance (static integrity fields populated at import time;
# dynamic fields updated by main()). test_zm2_provenance_stamp reads gnf._PROVENANCE.
# ---------------------------------------------------------------------------
_PROVENANCE: dict = {
    "year_2026_loaded": False,
    "prereg_commit": PREREG_COMMIT,
    "source_json": str(PRECHECK_B_JSON),
    "git_commit": _get_git_commit(),  # resolved at import time
    "run_date": None,       # populated by main()
    "matplotlib_version": None,
    "numpy_version": None,
}


# ---------------------------------------------------------------------------
# Holdout guard (JSON-based — adapted from the Parquet epoch-ms guard)
# ---------------------------------------------------------------------------


def _load_and_guard_json(json_path: Path = PRECHECK_B_JSON) -> dict:
    """Load the frozen precheck_b JSON and assert the 2026 holdout was never loaded.

    Raises AssertionError with "D-10 VIOLATION" if data_span['year_2026_loaded'] is True.
    This is the single entry point for all JSON reads in this driver.
    """
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["data_span"]["year_2026_loaded"] is False, (
        "D-10 VIOLATION: data_span['year_2026_loaded'] is True — "
        "the 2026 holdout must never be loaded. "
        f"Source JSON: {json_path}"
    )
    return data


# ---------------------------------------------------------------------------
# Headline series (CR-01): the plotted median line uses the primary-cell
# per-window median |VR(q)-1| (median_vr_dep) — the SAME series the CI band
# centres on and the manuscript cites — NOT horizon_profile.
# ---------------------------------------------------------------------------


def _headline_medians(data: dict) -> tuple[list[int], np.ndarray]:
    """Return ``(q_values, medians)`` for the plotted headline series.

    The headline median line MUST use ``per_wq_primary_q[*].median_vr_dep`` (the
    primary-cell per-window median |VR(q)-1|) — the SAME series the 95% CI band
    centres on and the manuscript prose/caption cite — so the median marker sits
    at the centre of its own CI. Plotting ``horizon_profile`` here (a different,
    slightly smaller frozen series) makes the marker sit off-centre within its
    own CI band and contradicts the text (e.g. 0.15970 vs 0.16087 at q=5).
    """
    wq_by_q = {row["q"]: row for row in data["per_wq_primary_q"]}
    q_values = sorted(wq_by_q.keys())
    medians = np.array([wq_by_q[q]["median_vr_dep"] for q in q_values])
    return q_values, medians


# ---------------------------------------------------------------------------
# Figure: z_m2 horizon profile
# ---------------------------------------------------------------------------


def _plot_fig_zm2_horizon(data: dict, figures_dir: Path) -> None:
    """Plot median |VR(q)-1| horizon profile with CI band and closure reference line.

    D-T6 framing: the profile increases with q, which is the signature of
    finite-sample estimation noise, not bid-ask bounce.

    Args:
        data: the loaded + holdout-guarded precheck_b JSON dict.
        figures_dir: output directory for the PNG.
    """
    low_noise_floor = data["low_noise_floor"]

    # Headline series + x-axis: the primary-cell per-window medians (CR-01 — the
    # SAME series the CI band centres on and the manuscript cites).
    q_values, medians = _headline_medians(data)
    q_strs = [str(q) for q in q_values]

    # LOW-regime noise floor (string-keyed)
    floor_vals = np.array([low_noise_floor[q] for q in q_strs])

    # Per-window 95% CI from per_wq_primary_q, matched by q — centred on `medians`
    wq_by_q = {row["q"]: row for row in data["per_wq_primary_q"]}
    ci_lo = np.array([wq_by_q[q]["ci_95_lo"] for q in q_values])
    ci_hi = np.array([wq_by_q[q]["ci_95_hi"] for q in q_values])

    fig, ax = plt.subplots(figsize=(7, 4))

    # CI band
    ax.fill_between(
        q_values,
        ci_lo,
        ci_hi,
        alpha=0.25,
        color=OKABE_ITO["blue"],
        label="95% CI (block bootstrap, B=2000)",
    )

    # Headline median series
    ax.plot(
        q_values,
        medians,
        color=OKABE_ITO["blue"],
        marker="o",
        linewidth=1.8,
        label="Median |VR(q)−1| (W=120, 2021–2025)",
    )

    # LOW-regime noise floor
    ax.plot(
        q_values,
        floor_vals,
        color=OKABE_ITO["orange"],
        marker="s",
        linewidth=1.4,
        linestyle="--",
        label="LOW-regime noise floor",
    )

    # 0.001 closure threshold reference line
    ax.axhline(
        CLOSURE_THRESHOLD,
        color=OKABE_ITO["vermillion"],
        linewidth=1.2,
        linestyle=":",
        label=f"Closure threshold ({CLOSURE_THRESHOLD})",
    )

    ax.set_xticks(q_values)
    ax.set_xticklabels([str(q) for q in q_values])
    ax.set_xlabel("Aggregation horizon q (bars)")
    ax.set_ylabel("Median |VR(q)−1|")
    ax.set_title(
        "zₘ₂ horizon profile — increasing with q ⇒ finite-sample noise, "
        "not bid-ask bounce",
        pad=8,
    )
    ax.legend(loc="upper left", framealpha=0.9)

    out_path = figures_dir / "fig_zm2_horizon.png"
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig_zm2_horizon.png -> {out_path}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> int:
    """Render the z_m2 horizon-profile figure from the frozen precheck_b JSON.

    Returns 0 on success. Raises AssertionError (D-10 VIOLATION) if the JSON
    indicates that 2026 holdout data was loaded.
    """
    # Load JSON + holdout guard (raises D-10 VIOLATION if contaminated)
    data = _load_and_guard_json()

    # Ensure output directory exists
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("\nRendering z_m2 horizon-profile figure (visualization-only)...")
    _plot_fig_zm2_horizon(data, FIGURES_DIR)

    # D-15 provenance stamp
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
        "prereg_commit": PREREG_COMMIT,
        "source_json": str(PRECHECK_B_JSON),
        "git_commit": git_commit,
        "matplotlib_version": matplotlib_version,
        "numpy_version": numpy_version,
        "python_version": sys.version,
        "figures_emitted": ["fig_zm2_horizon.png"],
    }

    # Update module-level _PROVENANCE so test_zm2_provenance_stamp can access it
    global _PROVENANCE
    _PROVENANCE = provenance

    # Write JSON sidecar
    sidecar_path = FIGURES_DIR / f"provenance_{run_date}.json"
    sidecar_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    print()
    print("=" * 70)
    print(f"Figure written to: {FIGURES_DIR}")
    print(f"Provenance sidecar: {sidecar_path}")
    print(f"year_2026_loaded: {provenance['year_2026_loaded']}")
    print(f"prereg_commit: {provenance['prereg_commit']}")
    print(f"git_commit: {git_commit}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
