"""Recompute the gauge TOST under the preregistered MDE margin.

Pre-registration v4.0 §4.6 freezes the equivalence margin as

    epsilon = MDE(q, W, n_eff, alpha=0.05, power=0.80)

with the MDE convention of the Pre-Check B cascade. For the frozen 2021-2025
sample that evaluates to 0.019748090241536544, and the original 2026-06-12
gauge run implemented exactly that value.

The published 2026-06-24 regeneration replaced it with the constant 0.02 — the
value the manuscript prose reports as a rounding of the margin. That
regeneration also fixed a real and serious TOST defect: every Holm-adjusted
p-value in the 2026-06-12 artifact is 1.0, including comparisons that should be
overwhelmingly equivalent. The bug fix was necessary. The margin change rode
along with it undeclared.

This script separates the two. It reuses the frozen per-comparison differences
and standard errors from the published artifact and recomputes only the TOST
p-values and the Holm correction under the preregistered margin. It runs no new
simulation, resamples nothing, and touches no bootstrap: given the same input
artifact it is a pure function.

    python scripts/wp1/migrate_gauge_mde_margin.py

Writes backtest_results/gauge_invariance/gauge_report_mde_margin.json beside
the published artifact. The published artifact is not modified.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from scipy.stats import norm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_REPO_SRC = _REPO_ROOT / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from backtest.utils import PROJECT_ROOT  # noqa: E402

import scripts._bootstrap  # noqa: F401, E402
from scripts.wp1.gauge_invariance import _holm_adjust  # noqa: E402
from scripts.wp1.vr_significance import compute_mde_vr  # noqa: E402

GAUGE_DIR = PROJECT_ROOT / "backtest_results" / "gauge_invariance"
PUBLISHED = GAUGE_DIR / "gauge_report_20260624_221834.json"
DEFAULT_OUT = GAUGE_DIR / "gauge_report_mde_margin.json"

# Non-overlapping window count of the clock (calendar) gauge at W_PRIMARY=120
# over the frozen 2021-2025 sample, as recorded in the gauge artifacts.
CLOCK_N_NL_PRIMARY = 20126
ALPHA = 0.05


def tost_p(diff: float, se: float, epsilon: float) -> float:
    """Two one-sided tests for H0: |theta| >= epsilon against H1: |theta| < epsilon."""
    if se <= 0.0 or not math.isfinite(se):
        return 0.0 if abs(diff) < epsilon else 1.0
    p_lower = 1.0 - norm.cdf((diff + epsilon) / se)
    p_upper = norm.cdf((diff - epsilon) / se)
    return float(max(p_lower, p_upper))


def recompute(published: dict, epsilon: float) -> dict:
    """Return a new report with the TOST block recomputed at `epsilon`."""
    out = json.loads(json.dumps(published))
    tost = out["tost"]
    comparisons = tost["per_q_comparisons"]

    raw = [tost_p(c["diff"], c["se_diff"], epsilon) for c in comparisons]
    adjusted = _holm_adjust(raw)

    for comparison, p_raw, p_adj in zip(comparisons, raw, adjusted, strict=True):
        comparison["p_tost_raw"] = float(p_raw)
        comparison["holm_adjusted"] = float(p_adj)
        comparison["certified"] = bool(p_adj < ALPHA)
        comparison["within_margin"] = bool(abs(comparison["diff"]) < epsilon)

    tost["epsilon"] = float(epsilon)
    tost["holm_adjusted"] = [float(p) for p in adjusted]
    tost["holm_family_size"] = len(comparisons)
    tost["margin_provenance"] = {
        "rule": "preregistration v4.0 §4.6",
        "formula": "epsilon = MDE(q, W, n_eff, alpha=0.05, power=0.80)",
        "mde_convention": "vr_significance.compute_mde_vr (Pre-Check B cascade)",
        "n_nl": CLOCK_N_NL_PRIMARY,
        "n_nl_source": "clock (calendar) gauge, non-overlapping windows at W_PRIMARY=120",
        "epsilon": float(epsilon),
        "superseded_constant": 0.02,
        "note": (
            "Recomputed from the frozen per-comparison diff and se_diff in "
            "gauge_report_20260624_221834.json. No new simulation; no "
            "resampling. See prereg/DEVIATIONS.md D1."
        ),
    }

    certified_q2 = sorted(
        c["pair"] for c in comparisons if int(c["q"]) == 2 and c["certified"]
    )
    tost["verdict_summary"] = {
        "certified_pairs_q2": certified_q2,
        "n_certified_q2": len(certified_q2),
        "n_pairs_q2": sum(1 for c in comparisons if int(c["q"]) == 2),
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=PUBLISHED)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    epsilon = float(
        compute_mde_vr(CLOCK_N_NL_PRIMARY, alpha=0.05, power=0.80)["mde_vr_departure"]
    )
    published = json.loads(args.source.read_text(encoding="utf-8"))
    migrated = recompute(published, epsilon)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n": the repository pins LF via .gitattributes, and Python's
    # default text mode would write CRLF on Windows.
    args.out.write_text(
        json.dumps(migrated, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print(f"preregistered margin : {epsilon!r}")
    print(f"superseded constant  : 0.02")
    print(f"wrote                : {args.out.relative_to(PROJECT_ROOT)}")
    print()
    print(f"{'pair':<18} {'q':>3} {'diff':>10} {'se':>10} {'holm p':>10}  certified")
    for c in migrated["tost"]["per_q_comparisons"]:
        print(
            f"{c['pair']:<18} {int(c['q']):>3} {c['diff']:>10.6f} "
            f"{c['se_diff']:>10.6f} {c['holm_adjusted']:>10.5f}  "
            f"{'yes' if c['certified'] else 'no'}"
        )
    summary = migrated["tost"]["verdict_summary"]
    print()
    print(
        f"q=2 certified: {summary['n_certified_q2']}/{summary['n_pairs_q2']} "
        f"({', '.join(summary['certified_pairs_q2']) or 'none'})"
    )


if __name__ == "__main__":
    main()
