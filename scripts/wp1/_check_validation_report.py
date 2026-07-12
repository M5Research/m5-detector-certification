"""Guard script: assert that 03-VALIDATION.md contains all required marker strings
AND that each reported headline number also appears as a value in the latest
backtest_results/wp1/hmm_validation_*.json (anti-fabrication check).

Usage:
    .venv\\Scripts\\python.exe scripts/wp1/_check_validation_report.py

Exits 0 and prints OK on full success.
Exits 1 and prints a failure description otherwise.
"""
from __future__ import annotations

import glob
import json
import math
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

_REPORT_PATH = (
    _REPO_ROOT
    / ".planning"
    / "phases"
    / "03-hmm-detector-alternative"
    / "03-VALIDATION.md"
)

# ---------------------------------------------------------------------------
# 1. Required marker strings
# ---------------------------------------------------------------------------
# These must all be present in 03-VALIDATION.md (acceptance criteria).
_REQUIRED_MARKERS: list[str] = [
    # D-02 limitation (case-sensitive per acceptance criterion)
    "causal-given-parameters",
    # "NOT truly" phrase (acceptance criterion: "NOT truly" case-insensitive)
    # We check case-insensitively in code below, but also include the canonical form.
    "rolling-quantile",       # PRIMARY instrument named
    "ROBUSTNESS",             # HMM role named
    # Integrity references
    "169fc20",                # pre-reg freeze commit
    "14.5",                   # §14.5 amendment reference
    # Source JSON provenance
    "hmm_validation",         # source JSON filename referenced
    # Crisis table section
    "LUNA",
    "FTX",
    "EXTREME",
    # Fallback outcome
    "hmm_fallback",
    "2-state",
    # Convergence + variance separation + floored count
    "top5_spread",            # or top-5; checked by broader string below
    "ratio_elev_low",
    "floored_bar_count",
    # 2026 holdout
    "year_2026_loaded",
    # Scope fence
    "epsilon",                # scope-fence disclaimer must mention epsilon
]


def _check_markers(text: str) -> list[str]:
    """Return list of required markers NOT found in the report."""
    missing = []
    for marker in _REQUIRED_MARKERS:
        if marker not in text:
            missing.append(marker)
    # Case-insensitive check for "NOT truly"
    if "not truly" not in text.lower():
        missing.append("NOT truly (case-insensitive)")
    return missing


# ---------------------------------------------------------------------------
# 2. Headline numbers that must appear in the JSON
# ---------------------------------------------------------------------------
# Each entry is a human-readable label + the string representation that
# must appear in the report AND be traceable to a JSON value.
#
# We check: does the stringified JSON value appear verbatim (or as a
# recognizable substring) in the report text?
#
# For floats we format with up to 4 significant digits and also check
# for the truncated integer portion, to allow for minor rounding in prose.

def _float_repr_variants(v: float) -> list[str]:
    """Return several string forms of a float for loose substring matching."""
    if math.isnan(v) or math.isinf(v):
        return ["NaN", "nan", "N/A", "n/a"]
    variants = []
    # Full repr as it appears in JSON
    variants.append(str(v))
    # Rounded forms
    variants.append(f"{v:.4f}")
    variants.append(f"{v:.2f}")
    # Integer part (for large numbers like llf ~ -1152101)
    int_val = int(v)
    variants.append(str(int_val))
    # Comma-formatted integer (e.g. "−1,152,101" or "1,152,101")
    # Handle negative: strip sign, format with commas, add sign back
    abs_val = abs(int_val)
    formatted_abs = f"{abs_val:,}"
    variants.append(formatted_abs)
    if int_val < 0:
        variants.append(f"-{formatted_abs}")
        # Also with em-dash (−) that markdown may use
        variants.append(f"−{formatted_abs}")
    # Also the absolute value as-is for llf (report may show "1,152,101" without sign context)
    return variants


def _find_json() -> Path | None:
    """Return the most recent hmm_validation_*.json or None."""
    pattern = str(_REPO_ROOT / "backtest_results" / "wp1" / "hmm_validation_*.json")
    matches = sorted(glob.glob(pattern))
    return Path(matches[-1]) if matches else None


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _check_headline_numbers(report_text: str, data: dict) -> list[str]:
    """Return list of headline numbers whose value is NOT found in the report."""
    failures = []

    def _require(label: str, value: object) -> None:
        """Assert that at least one string representation of value is in report."""
        if value is None:
            return
        if isinstance(value, float):
            variants = _float_repr_variants(value)
        elif isinstance(value, bool):
            # booleans: check "true"/"false" (JSON canonical) and Python forms
            variants = ["true" if value else "false", str(value)]
        elif isinstance(value, int):
            # Also include comma-formatted integer (e.g. 2415148 -> "2,415,148")
            variants = [str(value), f"{value:,}"]
        else:
            variants = [str(value)]

        if not any(v in report_text for v in variants):
            failures.append(
                f"{label}: JSON value {value!r} not found in report "
                f"(tried: {variants[:3]})"
            )

    diag = data.get("hmm_diagnostics", {})
    crisis = data.get("crisis_validation", {})
    pop = data.get("population_stats", {})
    span = data.get("data_span", {})
    prereg = data.get("preregistration", {})

    # Data span
    _require("data_span.n_bars", span.get("n_bars"))
    _require("data_span.start", span.get("start"))
    _require("data_span.end", span.get("end"))

    # Pre-registration
    _require("preregistration.prereg_commit", prereg.get("prereg_commit"))
    _require("preregistration.pinned_seed", prereg.get("pinned_seed"))

    # EM diagnostics
    _require("hmm_diagnostics.llf", diag.get("llf"))
    _require("hmm_diagnostics.top5_spread", diag.get("top5_spread"))
    _require("hmm_diagnostics.spread_fraction", diag.get("spread_fraction"))
    _require("hmm_diagnostics.floored_bar_count", diag.get("floored_bar_count"))
    _require("hmm_diagnostics.hmm_fallback", diag.get("hmm_fallback"))
    _require("hmm_diagnostics.regime_label_identical", diag.get("regime_label_identical"))
    _require("hmm_diagnostics.omp_num_threads", diag.get("omp_num_threads"))

    sigma2 = diag.get("sigma2_sorted", [])
    if len(sigma2) >= 2:
        _require("hmm_diagnostics.sigma2_sorted[0]", sigma2[0])
        _require("hmm_diagnostics.sigma2_sorted[1]", sigma2[1])
    _require("hmm_diagnostics.ratio_elev_low", diag.get("ratio_elev_low"))
    # ratio_ext_elev is NaN — we just check "NaN" or "N/A" is in report
    _require("hmm_diagnostics.ratio_ext_elev (NaN)", diag.get("ratio_ext_elev"))

    _require("hmm_diagnostics.timing_seconds", diag.get("timing_seconds"))
    _require("hmm_diagnostics.smoke_test_timing_seconds", diag.get("smoke_test_timing_seconds"))

    # Crisis validation — LUNA
    luna = crisis.get("luna", {})
    _require("crisis.luna.total_bars", luna.get("total_bars"))
    _require("crisis.luna.extreme_bar_count", luna.get("extreme_bar_count"))
    _require("crisis.luna.peak_filtered_p_extreme", luna.get("peak_filtered_p_extreme"))
    _require("crisis.luna.nonoverlapping_extreme_count_stride60",
             luna.get("nonoverlapping_extreme_count_stride60"))

    # Crisis validation — FTX
    ftx = crisis.get("ftx", {})
    _require("crisis.ftx.total_bars", ftx.get("total_bars"))
    _require("crisis.ftx.extreme_bar_count", ftx.get("extreme_bar_count"))
    _require("crisis.ftx.peak_filtered_p_extreme", ftx.get("peak_filtered_p_extreme"))
    _require("crisis.ftx.nonoverlapping_extreme_count_stride60",
             ftx.get("nonoverlapping_extreme_count_stride60"))

    # Population
    _require("population_stats.n_valid", pop.get("n_valid"))
    _require("population_stats.low_frac (first 4 digits)", pop.get("low_frac"))
    _require("population_stats.extreme_frac", pop.get("extreme_frac"))
    _require("population_stats.sparse", pop.get("sparse"))

    # Library versions
    libs = data.get("library_versions", {})
    _require("library_versions.statsmodels", libs.get("statsmodels"))

    # Git commit
    _require("git_commit (driver run)", data.get("git_commit"))

    return failures


# ---------------------------------------------------------------------------
# 3. Scope-fence check: no epsilon-squared or VR *values* computed
# ---------------------------------------------------------------------------

def _check_scope_fence(text: str) -> list[str]:
    """Return failures if the report contains computed ε² or VR values."""
    violations = []
    # Allow the word "epsilon" only in scope-fence disclaimers (not as a computed value)
    # A computed value would look like "epsilon_sq = 0.003" or "epsilon-squared = 0.003"
    # We look for numeric patterns near "epsilon" that would indicate a computed result.
    import re
    # Flag: "epsilon" followed within 30 chars by a decimal number NOT in a disclaimer context
    for match in re.finditer(r"epsilon.{0,30}=\s*[\d]+\.[\d]+", text, re.IGNORECASE):
        context = text[max(0, match.start() - 80):match.end() + 40]
        # Allow only if it's inside a scope-fence disclaimer or "not computed" statement
        if "not computed" not in context.lower() and "no epsilon" not in context.lower():
            violations.append(
                f"Possible computed epsilon-squared value found: ...{match.group()}..."
            )
    return violations


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    errors: list[str] = []

    # ---- Load report ----
    if not _REPORT_PATH.exists():
        print(f"ERROR: Report file not found: {_REPORT_PATH}", file=sys.stderr)
        return 1
    report_text = _REPORT_PATH.read_text(encoding="utf-8")

    # ---- Load JSON ----
    json_path = _find_json()
    if json_path is None:
        print(
            "ERROR: No hmm_validation_*.json found in backtest_results/wp1/",
            file=sys.stderr,
        )
        return 1
    data = _load_json(json_path)

    # ---- Check 1: required markers ----
    missing_markers = _check_markers(report_text)
    if missing_markers:
        errors.append(
            "MARKER CHECK FAILED — required strings missing from 03-VALIDATION.md:\n"
            + "\n".join(f"  - {m}" for m in missing_markers)
        )

    # ---- Check 2: headline numbers from JSON ----
    number_failures = _check_headline_numbers(report_text, data)
    if number_failures:
        errors.append(
            f"NUMBER PROVENANCE CHECK FAILED — {len(number_failures)} headline value(s) "
            f"not found in report (sourced from {json_path.name}):\n"
            + "\n".join(f"  - {f}" for f in number_failures)
        )

    # ---- Check 3: scope fence ----
    scope_violations = _check_scope_fence(report_text)
    if scope_violations:
        errors.append(
            "SCOPE FENCE VIOLATION — possible computed epsilon-squared values:\n"
            + "\n".join(f"  - {v}" for v in scope_violations)
        )

    # ---- Result ----
    if errors:
        for err in errors:
            print(f"ERROR: {err}\n", file=sys.stderr)
        return 1

    print(
        f"OK: 03-VALIDATION.md passes all checks.\n"
        f"  - {len(_REQUIRED_MARKERS) + 1} marker strings present\n"
        f"  - All headline numbers traceable to {json_path.name}\n"
        f"  - Scope fence: no computed epsilon-squared values detected"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
