"""
_check_front_sections.py
Verification script for Plan 10-03 Task 2.
Checks acceptance criteria for the manuscript front half.
"""
import pathlib
import re
import sys

TEX = pathlib.Path("docs/research/multi-layer-falsification/main.tex")

def main() -> int:
    t = TEX.read_text(encoding="utf-8")

    # Strip commas for number checks (handles LaTeX 2{,}415{,}148)
    n = t.replace(",", "").replace("{}", "")

    failures = []

    # Section headings
    if "Introduction" not in t:
        failures.append("Missing Introduction section")
    if "Falsification Framework" not in t:
        failures.append("Missing Falsification Framework section")
    if "Dependent Variable" not in t:
        failures.append("Missing Dependent Variable section")

    # Protocol box: exactly 8 enumerated items
    enum_match = re.search(r"\\begin\{enumerate\}(.*?)\\end\{enumerate\}", t, re.DOTALL)
    if not enum_match:
        failures.append("No enumerate block found")
    else:
        items = re.findall(r"\\item\s", enum_match.group(1))
        if len(items) != 8:
            failures.append(f"Protocol box has {len(items)} items; expected 8")
        else:
            print(f"Protocol box: {len(items)} steps (OK)")

    # Bars count and freeze hash
    if "2415148" not in n:
        failures.append("Bar count 2415148 not found in body (after stripping commas)")
    if "720c1d4" not in t:
        failures.append("Freeze hash 720c1d4 not found")

    # Single-asset HARD constraint
    if "BTCUSDT" not in t:
        failures.append("BTCUSDT not mentioned")
    if "HARD" not in t:
        failures.append("Single-asset HARD constraint not stated")
    if "by claim" not in t and "claim only" not in t:
        failures.append("Generalisation-by-claim wording missing")

    # Retired title absent; no positive-branch leakage
    if "When Bitcoin Becomes Predictable" in t:
        failures.append("RETIRED TITLE present in body!")
    if "onset curve" in t.lower():
        failures.append("Positive-branch leakage: 'onset curve'")
    if "half-life" in t.lower():
        failures.append("Positive-branch leakage: 'half-life'")
    if "OOS forecast" in t:
        failures.append("Positive-branch leakage: 'OOS forecast'")

    # Holdout never-loaded framing
    if "never loaded" not in t and "never been loaded" not in t:
        failures.append("Holdout never-loaded framing missing")

    # Blind-safe: author names must NOT appear in body
    if "Diego Urdaneta" in t:
        failures.append("BLIND VIOLATION: Diego Urdaneta in body")
    if "Claudio Martel" in t:
        failures.append("BLIND VIOLATION: Claudio Martel in body")

    # Minimum line count
    lines = t.splitlines()
    print(f"main.tex line count: {len(lines)}")
    if len(lines) < 120:
        failures.append(f"main.tex too short: {len(lines)} lines (min 120)")

    if failures:
        print("FAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("FRONT_SECTIONS_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
