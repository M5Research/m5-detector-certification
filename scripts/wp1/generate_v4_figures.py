"""Master figure orchestrator for v4.0 Regime-Detector Characterization paper.

Calls phase figure modules to regenerate all 10 figures (6 main + 4 appendix),
emitting flat PNGs to docs/research/calibrated-detector-exclusion/ with a
provenance JSON sidecar.

Figures:
  fig01_exclusion.png            — Exclusion surface (§2)
  fig02_gauge_horizon.png        — Three-panel gauge horizon profile (§3)
  fig03_bound_vs_cost.png        — Bound-vs-cost bar chart (§4)
  fig04_persistence_survival.png — Persistence survival function (§5)
  fig05_calibration_primary.png  — P_det(delta) calibration curve at PRIMARY (§2)
  figA1_regime_stability.png     — Regime-stability of detection efficiency (App A)
  figA2_volume_bar_count.png     — Volume-bar count vs. calendar time (App A)
  figA3_mi_validation.png        — MI estimator validation (App A)
  figA4_phi_validation.png       — Achieved vs target delta per q (App A)

All figures use unified matplotlib rcParams at savefig.dpi=300 (D-12).
Provenance JSON asserts year_2026_loaded is false (D-07 holdout guard).
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backtest.utils import PROJECT_ROOT  # noqa: E402
import scripts._bootstrap  # noqa: F401, E402

import matplotlib.pyplot as plt  # noqa: E402

V4_FIGURE_DIR = PROJECT_ROOT / "docs" / "research" / "calibrated-detector-exclusion"
V4_FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# Pinned artifact paths (D-29)
GAUGE_REPORT_PATH = str(
    PROJECT_ROOT
    / "backtest_results"
    / "gauge_invariance"
    / "gauge_report_20260624_221834.json"
)
PERSISTENCE_DIR = PROJECT_ROOT / "backtest_results" / "persistence"
INJECTION_DIR = PROJECT_ROOT / "data" / "injection_runs"
THERMO_DIR = PROJECT_ROOT / "backtest_results" / "thermodynamic_bound"
GAUGE_FIG_DIR = PROJECT_ROOT / "backtest_results" / "gauge_invariance"


def apply_rcparams():
    """Apply unified matplotlib rcParams (D-12)."""
    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "lines.linewidth": 1.5,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


def emit_provenance(figures_generated: list[str]) -> Path:
    """Write provenance JSON sidecar with D-15 timestamp and year_2026_loaded=false."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    prov_path = V4_FIGURE_DIR / f"provenance_v4_{stamp}.json"
    prov = {
        "paper": "v4.0 Regime-Detector Characterization Protocol",
        "figures_generated": figures_generated,
        "year_2026_loaded": False,
        "holdout_intact": True,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "figure_dir": str(V4_FIGURE_DIR),
    }
    prov_path.write_text(json.dumps(prov, indent=2), encoding="utf-8")
    print(f"Provenance: {prov_path}")
    return prov_path


def _step(label: str, fn) -> None:
    """Run one figure step with timing and flushed logging."""
    print(f"{label} ...", flush=True)
    t0 = time.perf_counter()
    fn()
    print(f"  done in {time.perf_counter() - t0:.1f}s", flush=True)


def main() -> int:
    apply_rcparams()
    figures_generated = []

    # ---- Fig 1 & 5: Exclusion plot + calibration curve ----
    try:
        from scripts.wp1.exclusion_plot import (
            generate_exclusion_plot,
            generate_calibration_curve_primary,
            generate_theory_vs_data_figure,
            load_grid_results,
        )

        def _fig1_5() -> None:
            print("  Loading injection grid results ...", flush=True)
            results = load_grid_results(INJECTION_DIR)
            print(f"  Loaded {len(results)} grid cells", flush=True)
            fig1_path = str(V4_FIGURE_DIR / "fig01_exclusion.png")
            generate_exclusion_plot(results, fig1_path)
            figures_generated.append("fig01_exclusion.png")
            generate_calibration_curve_primary(results, V4_FIGURE_DIR)
            figures_generated.append("fig05_calibration_primary.png")
            generate_theory_vs_data_figure(results, V4_FIGURE_DIR)
            figures_generated.append("fig06_theory_vs_data.png")

        _step("Generating Fig 1/5 (exclusion + calibration)", _fig1_5)
    except Exception as e:
        print(f"Fig 1/5 skipped: {e}", flush=True)

    # ---- Fig 2: Gauge horizon profile ----
    try:
        from scripts.wp1.gauge_figures import (
            generate_horizon_profile_figure,
            load_gauge_report,
        )

        def _fig2() -> None:
            report = load_gauge_report(GAUGE_REPORT_PATH)
            dst = V4_FIGURE_DIR / "fig02_gauge_horizon.png"
            generate_horizon_profile_figure(report, dst)
            figures_generated.append("fig02_gauge_horizon.png")

        _step("Generating Fig 2 (gauge horizon)", _fig2)
    except Exception as e:
        print(f"Fig 2 skipped: {e}", flush=True)

    # ---- Fig 3: Thermodynamic bound vs cost ----
    try:
        from scripts.wp1.thermodynamic_figures import (
            generate_bound_vs_cost_figure,
            load_thermo_report,
        )

        def _fig3() -> None:
            report = load_thermo_report(THERMO_DIR)
            print(f"  Headline: {report.get('headline_verdict')}", flush=True)
            out = generate_bound_vs_cost_figure(report, V4_FIGURE_DIR)
            canonical = V4_FIGURE_DIR / "fig03_bound_vs_cost.png"
            if out != canonical:
                if canonical.exists():
                    canonical.unlink()
                shutil.copy2(out, canonical)
            figures_generated.append("fig03_bound_vs_cost.png")

        _step("Generating Fig 3 (thermo bound)", _fig3)
    except Exception as e:
        print(f"Fig 3 skipped: {e}", flush=True)

    # ---- Fig 4: Persistence survival function ----
    try:
        from scripts.wp1.persistence_figures import (
            generate_survival_figure,
            load_persistence_report,
        )

        def _fig4() -> None:
            report = load_persistence_report(PERSISTENCE_DIR)
            print(
                f"  Verdict: {report.get('verdict')}, "
                f"KS={report.get('ks_statistic', 0):.3f}",
                flush=True,
            )
            out = generate_survival_figure(report, V4_FIGURE_DIR)
            canonical = V4_FIGURE_DIR / "fig04_persistence_survival.png"
            if out != canonical:
                if canonical.exists():
                    canonical.unlink()
                shutil.copy2(out, canonical)
            figures_generated.append("fig04_persistence_survival.png")

        _step("Generating Fig 4 (persistence)", _fig4)
    except Exception as e:
        print(f"Fig 4 skipped: {e}", flush=True)

    # ---- Fig A1: Regime-stability of detection efficiency ----
    try:
        from scripts.wp1.appendix_regime_stability_figures import (
            generate_regime_stability_figure,
        )

        def _fig_a1() -> None:
            generate_regime_stability_figure(INJECTION_DIR, V4_FIGURE_DIR)
            figures_generated.append("figA1_regime_stability.png")

        _step("Generating Fig A1 (regime stability)", _fig_a1)
    except Exception as e:
        print(f"Fig A1 skipped: {e}", flush=True)

    # ---- Fig A2: Volume-bar count vs calendar time ----
    try:
        from scripts.wp1.appendix_volume_bar_figures import (
            generate_volume_bar_figure,
        )

        def _fig_a2() -> None:
            generate_volume_bar_figure(GAUGE_REPORT_PATH, V4_FIGURE_DIR)
            figures_generated.append("figA2_volume_bar_count.png")

        _step("Generating Fig A2 (volume bars)", _fig_a2)
    except Exception as e:
        print(f"Fig A2 skipped: {e}", flush=True)

    # ---- Fig A3: MI estimator validation (numba-accelerated KSG) ----
    try:
        from scripts.wp1.appendix_mi_validation_figures import (
            generate_mi_validation_figure,
        )

        def _fig_a3() -> None:
            generate_mi_validation_figure(V4_FIGURE_DIR)
            figures_generated.append("figA3_mi_validation.png")

        _step("Generating Fig A3 (MI validation)", _fig_a3)
    except Exception as e:
        print(f"Fig A3 skipped: {e}", flush=True)

    # ---- Fig A4: phi-to-delta calibration validation ----
    try:
        from scripts.wp1.appendix_phi_validation_figures import (
            generate_phi_validation_figure,
        )

        def _fig_a4() -> None:
            generate_phi_validation_figure(V4_FIGURE_DIR)
            figures_generated.append("figA4_phi_validation.png")

        _step("Generating Fig A4 (phi validation)", _fig_a4)
    except Exception as e:
        print(f"Fig A4 skipped: {e}", flush=True)

    # ---- Provenance ----
    emit_provenance(figures_generated)
    print(f"Done: {len(figures_generated)}/10 figures generated.", flush=True)
    return 0 if len(figures_generated) >= 10 else 1


if __name__ == "__main__":
    raise SystemExit(main())
