"""Exclusion plot: detection-efficiency curves, delta*_90, and LIGO-style exclusion plot."""
from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_REPO_SRC = _REPO_ROOT / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from backtest.utils import PROJECT_ROOT
import scripts._bootstrap

def _load_inj_json(j_path: Path) -> tuple[tuple[float, int, int], dict]:
    with open(j_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    delta = data.get("delta", data.get("delta_target"))
    if delta is None:
        raise KeyError(f"{j_path} missing delta/delta_target")
    return (delta, data["q"], data["W"]), data


def load_grid_results(injection_dir: Path) -> dict:
    paths = sorted(injection_dir.glob("inj_*.json"))
    results: dict = {}
    if not paths:
        return results
    with ThreadPoolExecutor(max_workers=min(8, len(paths))) as pool:
        for key, data in pool.map(_load_inj_json, paths):
            results[key] = data
    return results

def compute_pdet_per_grid_point(results: dict) -> dict:
    """Return organized P_det data per (q,W) cell."""
    cells = {}
    for (delta, q, W), data in results.items():
        cell_key = (q, W)
        if cell_key not in cells:
            cells[cell_key] = {"deltas": [], "P_det": [], "ci_lo": [], "ci_hi": []}
        cells[cell_key]["deltas"].append(delta)
        cells[cell_key]["P_det"].append(data["P_det"])
        cells[cell_key]["ci_lo"].append(data["ci_95_lo"])
        cells[cell_key]["ci_hi"].append(data["ci_95_hi"])

    for key in cells:
        order = np.argsort(cells[key]["deltas"])
        cells[key]["deltas"] = np.array(cells[key]["deltas"])[order]
        cells[key]["P_det"] = np.array(cells[key]["P_det"])[order]
        cells[key]["ci_lo"] = np.array(cells[key]["ci_lo"])[order]
        cells[key]["ci_hi"] = np.array(cells[key]["ci_hi"])[order]

    return cells

def compute_delta_star_90(delta_grid: np.ndarray, P_det: np.ndarray) -> float:
    from scipy.interpolate import PchipInterpolator

    if np.max(P_det) < 0.90:
        return float('nan')

    interp = PchipInterpolator(delta_grid, P_det)
    fine_delta = np.logspace(np.log10(delta_grid.min()), np.log10(delta_grid.max()), 1000)
    fine_P = interp(fine_delta)
    mask = fine_P >= 0.90
    if not np.any(mask):
        return float('nan')
    return float(fine_delta[mask][0])

def generate_exclusion_plot(results: dict, output_path: str = "exclusion_plot.png") -> str:
    cells = compute_pdet_per_grid_point(results)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: P_det(delta) curves
    ax1 = axes[0]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(cells)))

    for (cell_key, data), color in zip(cells.items(), colors):
        q, W = cell_key
        deltas = data["deltas"]
        pdet = data["P_det"]
        ci_lo = data["ci_lo"]
        ci_hi = data["ci_hi"]

        ax1.semilogx(deltas, pdet, '-o', color=color, label=f"q={q}, W={W}", markersize=4)
        ax1.fill_between(deltas, ci_lo, ci_hi, color=color, alpha=0.15)

        ds = compute_delta_star_90(deltas, pdet)
        if not np.isnan(ds):
            ax1.axvline(ds, color=color, linestyle=':', alpha=0.7)

    ax1.axhline(0.90, color='red', linestyle='--', linewidth=1, label="90% detection")
    ax1.axvline(IID_FLOOR_Q5_W120, color='orange', linestyle='-.', linewidth=1.2,
                alpha=0.85, label=r"i.i.d. median floor ($q=5$)")
    ax1.axvline(CONFIRMATORY_DELTA_MAX, color='gray', linestyle=':', linewidth=1,
                alpha=0.8, label="confirmatory $\\delta_{\\max}$")
    ax1.set_xlabel("Signal strength delta = |VR(q)-1|")
    ax1.set_ylabel("Detection efficiency P_det")
    ax1.set_ylim(-0.02, 1.02)
    ax1.legend(fontsize=8)
    ax1.set_title("Detection Efficiency Curves")

    # Panel 2: Exclusion plot (delta*_90 surface)
    ax2 = axes[1]
    unique_qs = sorted(set(k[0] for k in cells.keys()))
    unique_Ws = sorted(set(k[1] for k in cells.keys()))
    delta_star = np.full((len(unique_Ws), len(unique_qs)), np.nan)
    for i, W in enumerate(unique_Ws):
        for j, q in enumerate(unique_qs):
            key = (q, W)
            if key in cells:
                delta_star[i, j] = compute_delta_star_90(
                    cells[key]["deltas"], cells[key]["P_det"]
                )

    ax2.set_xticks(range(len(unique_qs)))
    ax2.set_xticklabels([str(q) for q in unique_qs])
    ax2.set_yticks(range(len(unique_Ws)))
    ax2.set_yticklabels([str(W) for W in unique_Ws])
    ax2.set_xlabel("q (VR horizon)")
    ax2.set_ylabel("W (window size)")
    ax2.set_title("delta*_90 (minimum detectable signal)")

    if np.all(np.isnan(delta_star)):
        ax2.imshow(
            np.zeros_like(delta_star, dtype=np.float64),
            aspect='auto',
            cmap='Greys',
            origin='lower',
            vmin=0,
            vmax=1,
            alpha=0.08,
        )
        ax2.text(
            0.5,
            0.5,
            "No cell reached 90% detection\nwithin the tested delta grid",
            transform=ax2.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            color="black",
        )
        for spine in ax2.spines.values():
            spine.set_linestyle("--")
            spine.set_alpha(0.5)
    else:
        masked_delta_star = np.ma.masked_invalid(delta_star)
        cmap = plt.cm.viridis.copy()
        cmap.set_bad(color="#f2f2f2")
        im = ax2.imshow(masked_delta_star, aspect='auto', cmap=cmap, origin='lower')
        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label("delta*_90")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return output_path

IID_FLOOR_Q5_W120 = 0.145  # median |VR-1| under i.i.d. null (N=50k, W=120, q=5)
CONFIRMATORY_DELTA_MAX = 0.10
EXPLORATORY_PROBE_DELTA = 0.15
PRIMARY_Q = 5
PRIMARY_W = 120
ADDENDUM_GRID_START_INDEX = 96
ADDENDUM_DELTAS = (0.15, 0.20, 0.30, 0.50)
ADDENDUM_QS = (2, 5, 15, 60)
ADDENDUM_WS = (60, 120, 240)


class MissingInjectionCellsError(KeyError):
    """Actionable table-generation failure for absent injection artifacts."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return self.message


def _addendum_grid_index(delta: float, q: int, W: int) -> int | None:
    """Return current signal_injection.py addendum grid index, if known."""
    try:
        d_idx = ADDENDUM_DELTAS.index(float(delta))
        q_idx = ADDENDUM_QS.index(int(q))
        w_idx = ADDENDUM_WS.index(int(W))
    except ValueError:
        return None
    return (
        ADDENDUM_GRID_START_INDEX
        + d_idx * len(ADDENDUM_QS) * len(ADDENDUM_WS)
        + q_idx * len(ADDENDUM_WS)
        + w_idx
    )


def _format_missing_injection_cells(missing: list[tuple[float, int, int]]) -> str:
    has_exploratory = any(float(delta) > CONFIRMATORY_DELTA_MAX for delta, _q, _W in missing)
    heading = (
        "Missing exploratory injection cells:"
        if has_exploratory
        else "Missing injection cells:"
    )
    lines = [heading]
    for delta, q, W in missing:
        lines.append(f"- delta={delta:g} q={q} W={W}")

    commands: list[str] = []
    for delta, q, W in missing:
        idx = _addendum_grid_index(delta, q, W)
        if idx is None:
            continue
        commands.append(
            "python scripts/wp1/signal_injection.py --addendum "
            "--from-precomputed data/injection_runs/precomputed.npz "
            f"--start {idx} --end {idx + 1} --workers 1"
        )

    if commands:
        lines.append("Run the missing addendum cells, for example:")
        lines.extend(f"  {cmd}" for cmd in commands)
    else:
        lines.append(
            "Run the corresponding scripts/wp1/signal_injection.py cell(s), "
            "then rerun this table command."
        )
    return "\n".join(lines)


def generate_calibration_curve_primary(
    results: dict, output_dir: Path
) -> Path:
    """Generate Fig 5: P_det(delta) calibration curve at PRIMARY cell (D-09).

    Extracts only the (q=5, W=120) cell from the injection grid results,
    plots P_det vs delta on log scale with Clopper-Pearson 95% CI error bars,
    annotates delta*_90, and saves to output_dir/fig05_calibration_primary.png.
    """
    cells = compute_pdet_per_grid_point(results)
    key = (PRIMARY_Q, PRIMARY_W)

    if key not in cells:
        raise KeyError(
            f"PRIMARY cell (q={PRIMARY_Q}, W={PRIMARY_W}) not found in "
            f"injection grid results. Available cells: {sorted(cells.keys())}"
        )

    data = cells[key]
    deltas = data["deltas"]
    pdet = data["P_det"]
    ci_lo = data["ci_lo"]
    ci_hi = data["ci_hi"]
    ds90 = compute_delta_star_90(deltas, pdet)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.fill_between(deltas, ci_lo, ci_hi, alpha=0.2, color="steelblue",
                    label="95% CI (Clopper--Pearson)")
    ax.semilogx(deltas, pdet, "o-", color="steelblue", linewidth=2,
                markersize=6, label="$P_{\\mathrm{det}}(\\delta)$")
    ax.axhline(0.90, color="gray", linestyle="--", alpha=0.7,
               label="$P_{\\mathrm{det}} = 0.90$")

    if not np.isnan(ds90):
        ax.axvline(ds90, color="crimson", linestyle=":", linewidth=1.5,
                   alpha=0.8)
        ax.annotate(
            f"$\\delta^*_{{90}} = {ds90:.4f}$",
            xy=(ds90, 0.90),
            xytext=(ds90 * 1.5, 0.82),
            arrowprops=dict(arrowstyle="->", color="crimson", alpha=0.7),
            fontsize=9,
            color="crimson",
        )
    else:
        tested_max = float(np.max(deltas))
        ax.axvline(CONFIRMATORY_DELTA_MAX, color="gray", linestyle=":", linewidth=1.2,
                   alpha=0.8, label="confirmatory $\\delta_{\\max}=0.10$")
        ax.axvline(IID_FLOOR_Q5_W120, color="darkorange", linestyle="-.", linewidth=1.5,
                   alpha=0.85, label="i.i.d. median floor")
        if tested_max > CONFIRMATORY_DELTA_MAX:
            ax.axvline(tested_max, color="purple", linestyle=":", linewidth=1.2,
                       alpha=0.8, label=f"exploratory probe $\\delta={tested_max:g}$")
        ax.annotate(
            rf"$\delta^*_{{90}} > {tested_max:g}$ (within tested grid)",
            xy=(tested_max, 0.05),
            xytext=(0.42, 0.62),
            textcoords="axes fraction",
            arrowprops=dict(arrowstyle="->", color="crimson", alpha=0.7),
            fontsize=9,
            color="crimson",
        )
    ax.set_xlabel("$\\delta$ (injected autocorrelation)")
    ax.set_ylabel("$P_{\\mathrm{det}}$")
    ax.set_title(
        f"Detection Efficiency at Primary Cell "
        f"$(q={PRIMARY_Q}, W={PRIMARY_W})$"
    )
    ax.legend(loc="upper left", fontsize=8)
    ax.set_ylim(-0.02, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "fig05_calibration_primary.png"
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")
    return out_path


def generate_theory_vs_data_figure(results: dict, output_dir: Path) -> Path:
    """Fig 6: P_det at primary cell with structural reference lines."""
    cells = compute_pdet_per_grid_point(results)
    key = (PRIMARY_Q, PRIMARY_W)
    data = cells[key]
    deltas, pdet = data["deltas"], data["P_det"]
    ci_lo, ci_hi = data["ci_lo"], data["ci_hi"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(deltas, ci_lo, ci_hi, alpha=0.2, color="steelblue")
    ax.semilogx(deltas, pdet, "o-", color="steelblue", linewidth=2, markersize=6,
                label=r"$P_{\mathrm{det}}(\delta)$ (confirmatory)")
    ax.axhline(0.90, color="gray", linestyle="--", alpha=0.7)
    ax.axvline(IID_FLOOR_Q5_W120, color="darkorange", linestyle="-.", linewidth=1.5,
               label=r"i.i.d. median floor $\approx 0.145$")
    ax.axvline(CONFIRMATORY_DELTA_MAX, color="gray", linestyle=":", linewidth=1.2,
               label=r"confirmatory $\delta_{\max}=0.10$")
    tested_max = float(np.max(deltas))
    if tested_max > CONFIRMATORY_DELTA_MAX:
        ax.axvline(tested_max, color="purple", linestyle=":", linewidth=1.2,
                   label=rf"exploratory probe $\delta={tested_max:g}$")
    ax.set_xlabel(r"$\delta = |\mathrm{VR}(q)-1|$")
    ax.set_ylabel(r"$P_{\mathrm{det}}$")
    ax.set_title(r"Primary-cell firing conditions at $(W,q)=(120,5)$")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc="lower right", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    output_dir = Path(output_dir)
    out_path = output_dir / "fig06_theory_vs_data.png"
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")
    return out_path


def compute_z_trend_table(
    results: dict,
    *,
    q: int = PRIMARY_Q,
    W: int = PRIMARY_W,
    delta_values: tuple[float, ...] = (0.0005, 0.001, 0.01, 0.05, 0.10),
) -> list[dict]:
    """Recompute Table tab:z-trend median/max Z_q from per-draw holm_ordered arrays."""
    missing = [
        (float(delta), int(q), int(W))
        for delta in delta_values
        if (delta, q, W) not in results
    ]
    if missing:
        raise MissingInjectionCellsError(_format_missing_injection_cells(missing))

    rows = []
    for delta in delta_values:
        key = (delta, q, W)
        data = results[key]
        z_vals = []
        for draw in data.get("per_draw", []):
            ordered = draw.get("holm_ordered") or []
            z_q = None
            for item in ordered:
                if int(item.get("q", -1)) == q:
                    z_q = float(item.get("median_z_m2", item.get("z_m2", 0.0)))
                    break
            if z_q is not None:
                z_vals.append(z_q)
        if not z_vals:
            raise ValueError(f"No holm_ordered z values in {key}")
        rows.append({
            "delta": delta,
            "median_z": float(np.median(z_vals)),
            "max_z": float(np.max(z_vals)),
            "P_det": float(data.get("P_det", 0.0)),
        })
    return rows


def print_z_trend_table(rows: list[dict]) -> None:
    print(f"{'delta':>8} {'median_Z':>10} {'max_Z':>10} {'P_det':>8}")
    for row in rows:
        print(
            f"{row['delta']:8g} {row['median_z']:10.3f} "
            f"{row['max_z']:10.3f} {row['P_det']:8.1f}"
        )


def main() -> int:
    injection_dir = PROJECT_ROOT / "data" / "injection_runs"
    output_dir = PROJECT_ROOT / "backtest_results" / "injection"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading grid results from {injection_dir}...")
    if not injection_dir.exists():
        print("Injection directory does not exist. Please run signal_injection.py first.")
        return 1

    results = load_grid_results(injection_dir)

    n_points = len(results)
    print(f"Loaded {n_points} grid points")
    
    if n_points == 0:
        print("No grid points found.")
        return 1

    cells = compute_pdet_per_grid_point(results)
    print(f"\nDelta*_90 table:")
    print(f"{'q':>4} {'W':>4} {'d*_90':>10} {'P_det(max)':>12}")
    for (q, W), data in cells.items():
        ds = compute_delta_star_90(data["deltas"], data["P_det"])
        ds_str = f"{ds:.6f}" if not np.isnan(ds) else "NaN"
        print(f"  {q:>4} {W:>4} {ds_str:>10} {data['P_det'][-1]:>12.4f}")

    plot_path = str(output_dir / "exclusion_plot.png")
    generate_exclusion_plot(results, plot_path)
    print(f"\nExclusion plot saved: {plot_path}")
    return 0

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Exclusion plot and table utilities")
    parser.add_argument(
        "--z-trend-table",
        action="store_true",
        help="Print confirmatory z-trend table for primary cell (Table tab:z-trend)",
    )
    parser.add_argument(
        "--exploratory-primary-table",
        action="store_true",
        help="Print exploratory primary-cell bracket (Table tab:exploratory-primary)",
    )
    cli_args = parser.parse_args()

    if cli_args.z_trend_table:
        injection_dir = PROJECT_ROOT / "data" / "injection_runs"
        results = load_grid_results(injection_dir)
        rows = compute_z_trend_table(results)
        print_z_trend_table(rows)
        raise SystemExit(0)

    if cli_args.exploratory_primary_table:
        injection_dir = PROJECT_ROOT / "data" / "injection_runs"
        results = load_grid_results(injection_dir)
        rows = compute_z_trend_table(
            results,
            delta_values=(0.15, 0.20, 0.30),
        )
        print("Exploratory primary cell (W=120, q=5):")
        print_z_trend_table(rows)
        raise SystemExit(0)

    raise SystemExit(main())
