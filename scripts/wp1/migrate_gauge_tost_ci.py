"""Backfill 90% TOST CI fields into a frozen gauge report.

The current gauge driver writes ``se_diff``, ``ci_90_lo``, ``ci_90_hi``, and
``ci_within_margin`` for each pairwise TOST row. The ratified 20260612 report
predates that schema, but it contains the per-gauge 95% median CIs required to
reconstruct the same values deterministically.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backtest.utils import PROJECT_ROOT  # noqa: E402

DEFAULT_REPORT = (
    PROJECT_ROOT
    / "backtest_results"
    / "gauge_invariance"
    / "gauge_report_20260612_110425.json"
)
W_PRIMARY = 120
_NORM_Z_90 = 1.6448536269514722
_NORM_Z_95 = 1.959963984540054


def _median_row_for_q(pipeline: dict, q: int) -> dict:
    for cell in pipeline.get("per_wq_primary_q", []):
        if int(cell["q"]) == q:
            return cell
    for cell in pipeline.get("per_wq", []):
        if int(cell.get("W", W_PRIMARY)) == W_PRIMARY and int(cell["q"]) == q:
            return cell
    raise KeyError(f"no gauge median row for q={q}")


def _median_se_for_q(pipeline: dict, q: int) -> float:
    cell = _median_row_for_q(pipeline, q)
    ci_lo = float(cell["ci_95_lo"])
    ci_hi = float(cell["ci_95_hi"])
    if not (math.isfinite(ci_lo) and math.isfinite(ci_hi) and ci_hi >= ci_lo):
        n_nl = max(1, int(cell.get("n_nl", 0)))
        return float(1.0 / math.sqrt(n_nl))
    return float((ci_hi - ci_lo) / (2.0 * _NORM_Z_95))


def backfill_tost_ci(report: dict) -> dict:
    fixed_bar = {
        gauge: report["gauges"][gauge]["fixed_bar"]
        for gauge in ("clock", "volume", "intrinsic")
    }
    se_by_q_gauge: dict[tuple[int, str], float] = {}
    for row in report["tost"]["per_q_comparisons"]:
        q = int(row["q"])
        ga, gb = row["pair"].split("-")
        se_a = se_by_q_gauge.setdefault((q, ga), _median_se_for_q(fixed_bar[ga], q))
        se_b = se_by_q_gauge.setdefault((q, gb), _median_se_for_q(fixed_bar[gb], q))
        se_diff = float(math.hypot(se_a, se_b))
        diff = float(row["diff"])
        epsilon = float(row["epsilon"])
        ci_lo = float(diff - _NORM_Z_90 * se_diff)
        ci_hi = float(diff + _NORM_Z_90 * se_diff)
        row["se_diff"] = se_diff
        row["ci_90_lo"] = ci_lo
        row["ci_90_hi"] = ci_hi
        row["ci_within_margin"] = bool(ci_lo >= -epsilon and ci_hi <= epsilon)

    report["tost"]["ci_schema"] = {
        "source": "backfilled_from_frozen_95pct_median_ci",
        "script": "scripts/wp1/migrate_gauge_tost_ci.py",
        "norm_z_90": _NORM_Z_90,
        "norm_z_95": _NORM_Z_95,
    }
    return report


def main(path: Path = DEFAULT_REPORT) -> int:
    report_path = Path(path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    backfill_tost_ci(report)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Backfilled TOST CI fields: {report_path}")
    return 0


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_REPORT
    raise SystemExit(main(target))
