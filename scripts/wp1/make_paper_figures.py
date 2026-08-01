"""Regenerate the manuscript's figures from the frozen artifacts.

The paper carries four figures. Before this script, exactly one of them
(`fig01_exclusion`) had a generator anywhere in either repository, and that
generator wrote to `docs/research/calibrated-detector-exclusion/`, a path that
exists only in the private development repository. `fig02_transport_forest` and
`fig03_repair_power` had no generator at all, so the committed PNGs could not
be checked against the artifacts they claim to plot, and could drift from them
silently — which is exactly what happened when the transport margin was
corrected.

    python scripts/wp1/make_paper_figures.py            # all data-driven figures
    python scripts/wp1/make_paper_figures.py --only fig02_transport_forest
    python scripts/wp1/make_paper_figures.py --out /tmp/check   # compare, don't overwrite

`fig00_certification_pipeline` is a hand-drawn conceptual schematic with no
data behind it, so it is not regenerated here and is listed as such.

Every figure is drawn from committed artifacts only. No simulation, no market
data, no network access.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_REPO_SRC = _REPO_ROOT / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from backtest.utils import PROJECT_ROOT  # noqa: E402

import scripts._bootstrap  # noqa: F401, E402

FIGURE_DIR = PROJECT_ROOT / "paper" / "figures"
INJECTION_DIR = PROJECT_ROOT / "data" / "injection_runs"
GAUGE_MDE = (
    PROJECT_ROOT / "backtest_results" / "gauge_invariance" / "gauge_report_mde_margin.json"
)
ETH_ARTIFACT = (
    PROJECT_ROOT
    / "backtest_results"
    / "asset_replication"
    / "eth_replication_20260625_224506.json"
)
REPAIR_ARTIFACT = (
    PROJECT_ROOT
    / "backtest_results"
    / "reference_repair"
    / "recentered_reference_repair_20260710.json"
)

CONFIRMATORY_MAX_DELTA = 0.10
PAIR_ORDER = ("clock-volume", "clock-intrinsic", "volume-intrinsic")
PAIR_LABEL = {
    "clock-volume": "calendar–volume",
    "clock-intrinsic": "calendar–event-time",
    "volume-intrinsic": "volume–event-time",
}

plt.rcParams.update({
    "savefig.dpi": 300,
    "figure.dpi": 110,
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_cells() -> list[dict]:
    return [_load(p) for p in sorted(INJECTION_DIR.glob("inj_*.json"))]


# ---------------------------------------------------------------------------
# fig01 — power surface
# ---------------------------------------------------------------------------


def fig01_exclusion(out_dir: Path) -> Path:
    cells = _load_cells()
    by_wq: dict[tuple[int, int], dict[float, float]] = {}
    for c in cells:
        by_wq.setdefault((c["W"], c["q"]), {})[float(c["delta_target"])] = float(c["P_det"])

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(10.5, 4.2))

    qs = sorted({q for _, q in by_wq})
    colors = dict(zip(qs, plt.cm.viridis(np.linspace(0.05, 0.85, len(qs)))))
    styles = {60: ":", 120: "-", 240: "--"}

    for (w, q), table in sorted(by_wq.items()):
        deltas = sorted(table)
        ax_l.plot(
            deltas, [table[d] for d in deltas],
            marker="o", markersize=3, linewidth=1.2,
            color=colors[q], linestyle=styles.get(w, "-"),
            label=f"q={q}, W={w}",
        )

    ax_l.axvline(CONFIRMATORY_MAX_DELTA, color="0.35", linestyle=":", linewidth=1.1)
    ax_l.annotate(
        "preregistered grid ends",
        xy=(CONFIRMATORY_MAX_DELTA, 0.52), xytext=(-6, 0),
        textcoords="offset points", rotation=90, va="center", ha="right",
        fontsize=7.5, color="0.35",
    )
    ax_l.set_xscale("log")
    ax_l.set_xlabel(r"injected amplitude $\delta$")
    ax_l.set_ylabel(r"fire rate $P_{\rm det}$")
    ax_l.set_ylim(-0.04, 1.04)
    ax_l.set_title("Detector response surface", fontsize=10)
    ax_l.legend(fontsize=6.2, ncol=2, framealpha=0.9)

    # Right: delta_90 where the executed grid identifies it.
    ws = sorted({w for w, _ in by_wq})
    grid = np.full((len(qs), len(ws)), np.nan)
    for i, q in enumerate(qs):
        for j, w in enumerate(ws):
            table = by_wq.get((w, q), {})
            hit = [d for d in sorted(table) if table[d] >= 0.90]
            if hit:
                grid[i, j] = hit[0]

    masked = np.ma.masked_invalid(grid)
    cmap = plt.cm.magma_r.copy()
    cmap.set_bad("0.88")
    im = ax_r.imshow(masked, cmap=cmap, aspect="auto", norm=matplotlib.colors.LogNorm())
    ax_r.set_xticks(range(len(ws)), [str(w) for w in ws])
    ax_r.set_yticks(range(len(qs)), [str(q) for q in qs])
    ax_r.set_xlabel("window $W$")
    ax_r.set_ylabel("horizon $q$")
    ax_r.set_title(r"$\delta_{90}$ where the executed grid identifies it", fontsize=10)
    ax_r.grid(False)
    for i in range(len(qs)):
        for j in range(len(ws)):
            value = grid[i, j]
            ax_r.text(
                j, i, "—" if np.isnan(value) else f"{value:g}",
                ha="center", va="center", fontsize=8,
                color="0.35" if np.isnan(value) else "white",
            )
    fig.colorbar(im, ax=ax_r, label=r"$\delta_{90}$", fraction=0.046)

    fig.tight_layout()
    path = out_dir / "fig01_exclusion.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# fig02 — transport equivalence forest
# ---------------------------------------------------------------------------


def _forest_rows(comparisons: list[dict]) -> list[dict]:
    rows = []
    for q in sorted({int(c["q"]) for c in comparisons}):
        for pair in PAIR_ORDER:
            for c in comparisons:
                if int(c["q"]) == q and c["pair"] == pair:
                    rows.append(c)
    return rows


def _draw_forest(ax, comparisons: list[dict], epsilon: float, title: str) -> None:
    rows = _forest_rows(comparisons)
    ys = np.arange(len(rows))[::-1]

    ax.axvspan(-epsilon, epsilon, color="#cfe3f7", alpha=0.55, zorder=0,
               label=rf"equivalence margin $\pm{epsilon:.6f}$")
    ax.axvline(0.0, color="0.45", linewidth=0.9, zorder=1)

    for y, c in zip(ys, rows):
        lo = c.get("ci_90_lo")
        hi = c.get("ci_90_hi")
        if lo is None or hi is None:
            half = 1.6448536269514722 * float(c["se_diff"])
            lo, hi = float(c["diff"]) - half, float(c["diff"]) + half
        certified = bool(c.get("certified", c.get("holm_adjusted", 1.0) < 0.05))
        color = "#1a7f37" if certified else "#a4402a"
        ax.plot([lo, hi], [y, y], color=color, linewidth=1.7, solid_capstyle="round", zorder=3)
        ax.plot([c["diff"]], [y], marker="o", markersize=4.6, color=color, zorder=4)

    ax.set_yticks(ys)
    ax.set_yticklabels(
        [f"q={int(c['q'])}  {PAIR_LABEL[c['pair']]}" for c in rows], fontsize=7.2
    )
    ax.set_xlabel(r"transport discrepancy $\widehat{\Delta}_D$ (median $|VR-1|$)")
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", visible=False)


def fig02_transport_forest(out_dir: Path) -> Path:
    btc = _load(GAUGE_MDE)
    epsilon = float(btc["tost"]["epsilon"])

    eth = _load(ETH_ARTIFACT)["in_sample"]["gauge"]["tost"]
    eth_comparisons = json.loads(json.dumps(eth["per_q_comparisons"]))
    # The ETH artifact records the superseded 0.02 margin. Its verdicts are
    # unchanged under the preregistered margin (checked: the certified set is
    # identical), but the plotted band must be the preregistered one so both
    # panels are drawn against the same rule.
    from scipy.stats import norm

    from scripts.wp1.gauge_invariance import _holm_adjust

    raw = [
        max(
            1.0 - norm.cdf((c["diff"] + epsilon) / c["se_diff"]),
            norm.cdf((c["diff"] - epsilon) / c["se_diff"]),
        )
        for c in eth_comparisons
    ]
    for c, adj in zip(eth_comparisons, _holm_adjust(raw), strict=True):
        c["holm_adjusted"] = float(adj)
        c["certified"] = bool(adj < 0.05)

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11.0, 4.6), sharex=True)
    _draw_forest(ax_l, btc["tost"]["per_q_comparisons"], epsilon, "BTCUSDT")
    _draw_forest(ax_r, eth_comparisons, epsilon, "ETHUSDT")

    handles, labels = ax_l.get_legend_handles_labels()
    certified = plt.Line2D([], [], color="#1a7f37", linewidth=1.8, marker="o", markersize=4.6)
    uncertified = plt.Line2D([], [], color="#a4402a", linewidth=1.8, marker="o", markersize=4.6)
    fig.legend(
        handles + [certified, uncertified],
        labels + ["certified (Holm-adjusted TOST)", "not certified"],
        loc="lower center", ncol=3, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.06),
    )

    fig.tight_layout()
    path = out_dir / "fig02_transport_forest.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# fig03 — repair power
# ---------------------------------------------------------------------------


def fig03_repair_power(out_dir: Path) -> Path:
    artifact = _load(REPAIR_ARTIFACT)
    cells = artifact["cells"]
    names = [n for n in ("W60_q2", "W120_q2", "W240_q2", "W60_q5", "W120_q5", "W240_q5")
             if n in cells]

    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.0), sharex=True, sharey=True)
    for ax, name in zip(axes.ravel(), names):
        cell = cells[name]
        recentered = {float(k): v for k, v in cell["recentered_pdet"].items()}
        original = {float(k): v for k, v in cell.get("orig_pdet", {}).items()}

        ds = sorted(recentered)
        ax.plot(ds, [recentered[d] for d in ds], marker="o", markersize=3.4,
                linewidth=1.5, color="#1a7f37", label="recentered")
        if original:
            do = sorted(original)
            ax.plot(do, [original[d] for d in do], marker="s", markersize=3.0,
                    linewidth=1.2, linestyle="--", color="#8a8a8a", label="frozen reference")

        ax.axhline(0.90, color="0.4", linestyle=":", linewidth=1.0)
        d90 = cell.get("cmde_recentered_d90")
        if d90 is not None:
            ax.axvline(d90, color="#1a7f37", linestyle=":", linewidth=1.0)
        ax.set_xscale("log")
        ax.set_ylim(-0.04, 1.04)

        w, q = name.replace("W", "").split("_q")
        size = cell["oos_size"]
        flag = "" if size <= 0.05 else "  ⚠ above nominal"
        ax.set_title(
            f"W={w}, q={q}   cMDE {cell.get('cmde_orig_d90')}→{d90}\n"
            f"OOS size {size:.2f}{flag}",
            fontsize=8.5,
        )

    for ax in axes[-1]:
        ax.set_xlabel(r"injected amplitude $\delta$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$P_{\rm det}$")
    axes[0, 0].legend(fontsize=7, loc="upper left")

    fig.suptitle(
        "Recentered-reference repair (exploratory, post-freeze constants)", fontsize=10.5
    )
    fig.tight_layout()
    path = out_dir / "fig03_repair_power.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


BUILDERS = {
    "fig01_exclusion": fig01_exclusion,
    "fig02_transport_forest": fig02_transport_forest,
    "fig03_repair_power": fig03_repair_power,
}
NOT_DATA_DRIVEN = ("fig00_certification_pipeline",)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=FIGURE_DIR)
    parser.add_argument("--only", action="append", choices=sorted(BUILDERS))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    names = args.only or sorted(BUILDERS)
    for name in names:
        path = BUILDERS[name](args.out)
        print(f"wrote {path.relative_to(PROJECT_ROOT) if args.out == FIGURE_DIR else path}"
              f"  ({path.stat().st_size // 1024} KiB)")
    for name in NOT_DATA_DRIVEN:
        print(f"skipped {name}: hand-drawn schematic, no artifact behind it")


if __name__ == "__main__":
    main()
