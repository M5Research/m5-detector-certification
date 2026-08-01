"""Recompute the injection cells' binomial intervals as exact Clopper-Pearson.

The `ci_95_lo` / `ci_95_hi` fields in data/injection_runs/*.json were computed
with Beta(k+1, n-k+1) for both bounds. That is the Bayesian posterior interval
under a uniform prior, not the exact Clopper-Pearson interval, which uses
different shape parameters for each bound:

    lo = B^-1(alpha/2;   k,   n-k+1),   = 0 if k == 0
    hi = B^-1(1-alpha/2; k+1, n-k  ),   = 1 if k == n

The manuscript labels these intervals Clopper-Pearson in the power table and
in exclusion_plot.py, and Proposition 1 asserts they are exact at any N_mc. So
the exact interval is the one that belongs in the artifacts.

The correction is small and runs against the paper's own claim: for the k=0,
n=200 case that dominates the grid, the upper bound moves from 0.018185 to
0.018275, i.e. the published exclusion was marginally *tighter* than the
stated method licenses.

This script rewrites only the two interval fields and records that it did so
in the artifact. Every other field, including P_det, n_fires and all 200
per-draw records, is untouched: the decision statistic does not depend on the
interval.

    python scripts/wp1/migrate_cp_intervals.py --check   # report, write nothing
    python scripts/wp1/migrate_cp_intervals.py           # apply
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_REPO_SRC = _REPO_ROOT / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from backtest.utils import PROJECT_ROOT  # noqa: E402

import scripts._bootstrap  # noqa: F401, E402
from scripts.wp1.signal_injector import clopper_pearson_ci  # noqa: E402

INJECTION_DIR = PROJECT_ROOT / "data" / "injection_runs"
ALPHA = 0.05


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="report the changes without writing them",
    )
    args = parser.parse_args()

    paths = sorted(INJECTION_DIR.glob("inj_*.json"))
    if not paths:
        raise SystemExit(f"no injection cells under {INJECTION_DIR}")

    changed = 0
    for path in paths:
        cell = json.loads(path.read_text(encoding="utf-8"))
        lo, hi = clopper_pearson_ci(cell["n_fires"], cell["N_mc"], alpha=ALPHA)
        old_lo = cell.get("ci_95_lo")
        old_hi = cell.get("ci_95_hi")

        if old_lo == lo and old_hi == hi:
            continue
        changed += 1
        if changed <= 4:
            print(
                f"{path.name}: [{old_lo:.6f}, {old_hi:.6f}] -> [{lo:.6f}, {hi:.6f}]"
            )

        if args.check:
            continue

        cell["ci_95_lo"] = lo
        cell["ci_95_hi"] = hi
        cell["ci_schema"] = {
            "method": "clopper_pearson_exact",
            "alpha": ALPHA,
            "migrated_from": "beta_uniform_prior",
            "script": "scripts/wp1/migrate_cp_intervals.py",
            "note": (
                "Only ci_95_lo and ci_95_hi were recomputed. P_det, n_fires "
                "and all per-draw records are unchanged from the frozen run."
            ),
        }
        # newline="\n": the repository pins LF via .gitattributes, and Python's
        # default text mode would write CRLF on Windows.
        path.write_text(
            json.dumps(cell, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )

    verb = "would change" if args.check else "changed"
    print(f"\n{verb} {changed} of {len(paths)} cells")


if __name__ == "__main__":
    main()
