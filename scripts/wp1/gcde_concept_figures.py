"""Generate conceptual GCDE figures for the method paper."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT_DIR = Path("docs/research/gauge-calibrated-detector-exclusion/figures")

BLUE = "#2F5D8C"
GREEN = "#3F7F5F"
RED = "#8C3F3F"
GRAY = "#F2F4F6"
TEXT = "#202428"


def _box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    subtitle: str = "",
    *,
    edge: str = BLUE,
    face: str = GRAY,
    title_size: float = 10,
    subtitle_size: float = 7.5,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        linewidth=1.3,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h * 0.60, title, ha="center", va="center", fontsize=title_size, color=TEXT)
    if subtitle:
        ax.text(
            x + w / 2,
            y + h * 0.30,
            subtitle,
            ha="center",
            va="center",
            fontsize=subtitle_size,
            color=TEXT,
            wrap=True,
        )


def _arrow(ax: plt.Axes, x1: float, y1: float, x2: float, y2: float, color: str = BLUE) -> None:
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops={"arrowstyle": "->", "linewidth": 1.4, "color": color, "shrinkA": 3, "shrinkB": 3},
    )


def generate_pipeline_figure(out_dir: Path = OUT_DIR) -> Path:
    """Write the GCDE admissibility pipeline figure."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fig00_gcde_pipeline.png"
    fig, ax = plt.subplots(figsize=(11.6, 3.8))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    labels = [
        ("Detector", "frozen output"),
        ("Sensitivity", "can it detect?"),
        ("Size", "does it false fire?"),
        ("Gauge", "does clock matter?"),
        ("Information", "informs target?"),
        ("Work", "cost-exhausted?"),
        ("Disposition", "admit or abstain"),
    ]
    x0 = 0.02
    w = 0.125
    gap = 0.016
    y = 0.42
    h = 0.26
    for i, (title, subtitle) in enumerate(labels):
        x = x0 + i * (w + gap)
        _box(
            ax,
            x,
            y,
            w,
            h,
            title,
            subtitle,
            edge=GREEN if i == len(labels) - 1 else BLUE,
            title_size=9.4,
            subtitle_size=7.1,
        )
        if i < len(labels) - 1:
            _arrow(ax, x + w, y + h / 2, x + w + gap, y + h / 2)

    ax.text(
        0.5,
        0.86,
        "GCDE admits detector outputs only after the declared gates are measured",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=TEXT,
    )
    ax.text(
        0.5,
        0.18,
        "GCDE certifies detector labels for a declared target and clock; it does not estimate the regime itself.",
        ha="center",
        va="center",
        fontsize=10,
        color=TEXT,
    )
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def generate_failure_fingerprint_figure(out_dir: Path = OUT_DIR) -> Path:
    """Write the failure-fingerprint decision tree figure."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fig05_failure_fingerprint_tree.png"
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    _box(ax, 0.37, 0.80, 0.26, 0.12, "Detector claim", "declared target and cost", edge=BLUE)
    branches = [
        (0.07, 0.52, "Weak signal", "Underpowered"),
        (0.30, 0.52, "Gauge fails", "Gauge-uncertified"),
        (0.53, 0.52, "Wrong target", "Target-mismatched"),
        (0.76, 0.52, "Cost-exhausted\ninformation", "Info-excluded"),
    ]
    for x, y, cause, disposition in branches:
        _box(ax, x, y, 0.17, 0.12, cause, "", edge=BLUE)
        _box(ax, x, 0.25, 0.17, 0.12, disposition, "", edge=RED, face="#FFF4F4")
        _arrow(ax, 0.50, 0.80, x + 0.085, y + 0.12)
        _arrow(ax, x + 0.085, y, x + 0.085, 0.37, color=RED)

    ax.text(
        0.5,
        0.08,
        "A failed detector claim becomes a diagnostic fingerprint, not a generic null result.",
        ha="center",
        va="center",
        fontsize=10,
        color=TEXT,
    )
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def generate_static_online_figure(out_dir: Path = OUT_DIR) -> Path:
    """Write the static-to-online GCDE schematic."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fig06_static_online_gcde.png"
    fig, ax = plt.subplots(figsize=(9.8, 3.6))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    labels = [
        ("Static\nGCDE", "certificate"),
        ("Monitoring", "rolling evidence"),
        ("Online\nGCDE", "sequential gates"),
        ("Admission", "may speak"),
        ("Revocation", "must abstain"),
    ]
    x0 = 0.06
    w = 0.14
    gap = 0.055
    y = 0.43
    h = 0.23
    for i, (title, subtitle) in enumerate(labels):
        edge = GREEN if i == 3 else RED if i == 4 else BLUE
        face = "#F4FFF7" if i == 3 else "#FFF4F4" if i == 4 else GRAY
        x = x0 + i * (w + gap)
        _box(ax, x, y, w, h, title, subtitle, edge=edge, face=face, title_size=9.5, subtitle_size=7.2)
        if i < len(labels) - 1:
            _arrow(ax, x + w, y + h / 2, x + w + gap, y + h / 2, color=BLUE)

    ax.text(
        0.5,
        0.84,
        "Online GCDE turns a static certificate into monitored admission and revocation",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=TEXT,
    )
    ax.text(
        0.5,
        0.17,
        "Admission is dynamic, but cross-gauge transport still requires its own certificate.",
        ha="center",
        va="center",
        fontsize=10,
        color=TEXT,
    )
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def generate_all(out_dir: Path = OUT_DIR) -> list[Path]:
    """Generate all conceptual GCDE figures."""
    return [
        generate_pipeline_figure(out_dir),
        generate_failure_fingerprint_figure(out_dir),
        generate_static_online_figure(out_dir),
    ]


def main() -> int:
    for path in generate_all():
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
