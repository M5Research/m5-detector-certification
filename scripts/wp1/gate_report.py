"""Apply the pre-registered C0/C1 kill-criterion and emit the GO/NO-GO verdict markdown.

Refined kill-criterion (per the WP1 gate corrections):

  NO-GO if, for the OOS years (2024 AND 2025), ANY of:
    * return <= 0 at `maker_real_1.8bp+0.5spread`, OR
    * break-even < MIN_BREAK_EVEN_BPS, OR
    * the SESSION z-score IC (`zscore_session`) is sign-unstable across 2024/2025/2026
      (the strategy trades the session z-score, so this is the gating signal).

`deviation_all` IC stability is reported as an INFORMATIONAL line, not a hard blocker.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

from backtest.utils import PROJECT_ROOT

WP1 = PROJECT_ROOT / "backtest_results" / "wp1"
REPORT = PROJECT_ROOT / "docs" / "superpowers" / "reports" / "2026-05-30-wp1-c0c1-gate.md"
MIN_BREAK_EVEN_BPS = 2.0
# IC sign-stability is judged over these years (OOS + in-sample), keyed by int year in the IC JSONs.
IC_STABILITY_YEARS = (2024, 2025, 2026)


def _all_same_sign(values: list[float]) -> bool:
    return bool(values) and (all(v > 0 for v in values) or all(v < 0 for v in values))


def evaluate(ic_by_year: dict, cost_by_period: dict, mapping: str = "shipped") -> dict:
    """Return {'verdict','reasons','info','mapping'} for one regime_module_map.

    `cost_by_period` is keyed by (period_label, mapping) with period_label in {"2024","2025",...}.
    `ic_by_year` is keyed by int year.
    """
    reasons: list[str] = []
    info: list[str] = []

    for period_label in ("2024", "2025"):
        cost = cost_by_period.get((period_label, mapping))
        if cost is None:
            reasons.append(f"missing cost sweep for {period_label}/{mapping}")
            continue
        maker = next(
            (p for p in cost["points"] if p["label"].startswith("maker_real_1.8bp+0.5")), None
        )
        if maker is None:
            reasons.append(f"{period_label}: no maker_real_1.8bp+0.5spread point in sweep")
        elif maker["total_return"] <= 0:
            reasons.append(
                f"{period_label}: return <= 0 at real maker cost ({maker['total_return']:+.4f})"
            )
        be = cost["break_even_bps"]
        if be != be:  # NaN -> never positive at any tested cost
            reasons.append(f"{period_label}: break-even is NaN (return never positive)")
        elif be < MIN_BREAK_EVEN_BPS:
            reasons.append(f"{period_label}: break-even {be:.2f}bp < {MIN_BREAK_EVEN_BPS}bp")

    # HARD blocker: the SESSION z-score IC (the traded signal) must be sign-stable across the years.
    z_session = [
        ic_by_year[y]["zscore_session"]["ic"]
        for y in IC_STABILITY_YEARS
        if y in ic_by_year and "zscore_session" in ic_by_year[y]
    ]
    if z_session and not _all_same_sign(z_session):
        reasons.append(
            "zscore_session IC sign-unstable across "
            f"{'/'.join(str(y) for y in IC_STABILITY_YEARS)}: "
            f"{[round(v, 4) for v in z_session]}"
        )

    # INFORMATIONAL: deviation_all IC stability (not a hard blocker).
    dev_all = [
        ic_by_year[y]["deviation_all"]["ic"]
        for y in IC_STABILITY_YEARS
        if y in ic_by_year and "deviation_all" in ic_by_year[y]
    ]
    if dev_all:
        stable = "stable" if _all_same_sign(dev_all) else "UNSTABLE"
        info.append(
            f"deviation_all IC {stable} across "
            f"{'/'.join(str(y) for y in IC_STABILITY_YEARS)}: {[round(v, 4) for v in dev_all]}"
        )

    return {
        "verdict": "NO-GO" if reasons else "GO",
        "reasons": reasons,
        "info": info,
        "mapping": mapping,
    }


def _load() -> tuple[dict, dict]:
    ic: dict = {}
    for f in glob.glob(str(WP1 / "signal_ic_*.json")):
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        ic[d["year"]] = d
    cost: dict = {}
    for f in glob.glob(str(WP1 / "cost_sensitivity_*.json")):
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        cost[(d["period"], d["mapping"])] = d
    return ic, cost


def _maker_real_return(cost: dict | None) -> float | None:
    if cost is None:
        return None
    p = next((p for p in cost["points"] if p["label"].startswith("maker_real_1.8bp+0.5")), None)
    return None if p is None else p["total_return"]


def _fmt(v: float | None) -> str:
    if v is None:
        return "n/a"
    if v != v:  # NaN
        return "NaN"
    return f"{v:+.4f}" if abs(v) < 100 else f"{v:.4g}"


def main() -> int:
    ic, cost = _load()
    mappings = ("shipped", "blueprint")
    results = {m: evaluate(ic, cost, m) for m in mappings}

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = ["# WP1 C0/C1 Gate — GO/NO-GO verdict", ""]
    lines.append(
        "Kill-criterion: NO-GO if, for the OOS years (2024 AND 2025), the return at "
        "`maker_real_1.8bp+0.5spread` is <= 0, OR break-even < 2.0 bp, OR the traded session "
        "z-score IC (`zscore_session`) is sign-unstable across 2024/2025/2026."
    )
    lines.append("")

    # Per-mapping verdicts.
    for m in mappings:
        r = results[m]
        lines.append(f"## Mapping: {m} -> **{r['verdict']}**")
        lines.append("")
        lines.append("Blockers:")
        if r["reasons"]:
            lines += [f"- {x}" for x in r["reasons"]]
        else:
            lines.append("- none (all criteria passed)")
        if r["info"]:
            lines.append("")
            lines.append("Informational:")
            lines += [f"- {x}" for x in r["info"]]
        lines.append("")

    # OOS maker-real return + break-even table.
    lines.append("## OOS maker-real returns & break-evens")
    lines.append("")
    lines.append("| Period | Mapping | maker_real_1.8bp+0.5spread return | break-even (bp) |")
    lines.append("|---|---|---|---|")
    for period_label in ("2024", "2025"):
        for m in mappings:
            c = cost.get((period_label, m))
            be = c["break_even_bps"] if c else None
            lines.append(
                f"| {period_label} | {m} | {_fmt(_maker_real_return(c))} | "
                f"{('NaN' if (be is not None and be != be) else (f'{be:.2f}' if be is not None else 'n/a'))} |"
            )
    lines.append("")

    # Session z-score IC table (the gating signal).
    lines.append("## Session z-score IC (gating signal)")
    lines.append("")
    lines.append("| Year | zscore_session IC | deviation_all IC (info) |")
    lines.append("|---|---|---|")
    for y in IC_STABILITY_YEARS:
        if y in ic:
            zs = ic[y].get("zscore_session", {}).get("ic")
            da = ic[y].get("deviation_all", {}).get("ic")
            lines.append(f"| {y} | {_fmt(zs)} | {_fmt(da)} |")
    lines.append("")

    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(
        "\n".join(
            f"{m}: {results[m]['verdict']} ({len(results[m]['reasons'])} blockers)" for m in mappings
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
