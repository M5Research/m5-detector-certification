"""Patch Table tab:exploratory-primary in 02b-power-theorem.tex from injection JSON."""
from __future__ import annotations

import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import scripts._bootstrap  # noqa: F401
from scripts.wp1.exclusion_plot import compute_z_trend_table, load_grid_results

TEX = _REPO / "docs/research/calibrated-detector-exclusion/sections/02b-power-theorem.tex"
INJ = _REPO / "data/injection_runs"


def main() -> int:
    results = load_grid_results(INJ)
    rows = compute_z_trend_table(results, delta_values=(0.15, 0.2, 0.30))
    tex_lines = []
    for row in rows:
        d = row["delta"]
        z = row["median_z"]
        p = row["P_det"]
        ci_hi = 0.018 if p == 0 else 1.0  # approximate upper bound at N_MC=200
        z_fmt = f"${z:+.2f}$" if z >= 0 else f"${z:.2f}$"
        tex_lines.append(
            f"{d:g} & {z_fmt} & {int(p)} & $[0,\\,{ci_hi:.3f}]$ \\\\"
        )
    block = "\n".join(tex_lines)
    text = TEX.read_text(encoding="utf-8")
    pattern = (
        r"(\\midrule\n)"
        r"(?:[0-9.]+ & .*?\\\\\n)+"
        r"(\\bottomrule)"
    )
    new_text, n = re.subn(
        pattern,
        r"\1" + block + "\n" + r"\2",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if n != 1:
        print("Could not find exploratory-primary table to patch", file=sys.stderr)
        return 1
    TEX.write_text(new_text, encoding="utf-8")
    print(f"Patched {TEX}")
    for row in rows:
        print(f"  delta={row['delta']} median_Z5={row['median_z']:.3f} P_det={row['P_det']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
