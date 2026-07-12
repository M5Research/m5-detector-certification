"""Summarize free HistData EURUSD minute bid/ask rows for GCDE cost gates."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections.abc import Iterable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _normalize_paths(paths: Path | Sequence[Path]) -> list[Path]:
    if isinstance(paths, Path):
        raw_paths = [paths]
    else:
        raw_paths = list(paths)

    resolved: list[Path] = []
    for raw in raw_paths:
        path = Path(raw)
        if path.is_dir():
            resolved.extend(sorted(path.glob("*.csv")))
        elif any(char in str(path) for char in "*?[]"):
            resolved.extend(sorted(path.parent.glob(path.name)))
        else:
            resolved.append(path)
    return sorted(dict.fromkeys(resolved))


def _iter_filtered_rows(
    paths: Iterable[Path],
    *,
    start: datetime | None,
    end: datetime | None,
) -> Iterable[tuple[Path, dict[str, str]]]:
    for path in paths:
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                ts = _parse_timestamp(row["timestamp"])
                if start is not None and ts < start:
                    continue
                if end is not None and ts > end:
                    continue
                yield path, row


def summarize_minute_bidask(
    path: Path | Sequence[Path],
    *,
    start: str | None = None,
    end: str | None = None,
) -> dict[str, Any]:
    paths = _normalize_paths(path)
    if not paths:
        raise ValueError("no CSV paths matched")
    missing = [str(csv_path) for csv_path in paths if not csv_path.exists()]
    if missing:
        raise FileNotFoundError(f"missing CSV files: {missing}")

    start_dt = _parse_timestamp(start) if start else None
    end_dt = _parse_timestamp(end) if end else None

    spreads: list[float] = []
    mids: list[float] = []
    tick_counts: list[int] = []
    first_timestamp: str | None = None
    last_timestamp: str | None = None
    used_files: set[Path] = set()

    for csv_path, row in _iter_filtered_rows(paths, start=start_dt, end=end_dt):
        timestamp = row["timestamp"]
        first_timestamp = first_timestamp or timestamp
        last_timestamp = timestamp
        used_files.add(csv_path)
        spreads.append(float(row["spread"]))
        mids.append(float(row["mid"]))
        tick_counts.append(int(row.get("n_ticks", 0) or 0))

    if not spreads:
        raise ValueError("no rows remained after applying the date filter")
    if any(mid <= 0.0 or not math.isfinite(mid) for mid in mids):
        raise ValueError("mid prices must be positive and finite")
    spread_bps = [10_000.0 * spread / mid for spread, mid in zip(spreads, mids, strict=True)]
    return {
        "schema_version": 1,
        "artifact": "eurusd_histdata_cost_gate_summary",
        "source_file": str(paths[0].as_posix()) if len(paths) == 1 else None,
        "source_files": [str(csv_path.as_posix()) for csv_path in paths],
        "used_source_files": [str(csv_path.as_posix()) for csv_path in sorted(used_files)],
        "source": "HistData Generic ASCII tick quotes aggregated to one-minute last bid/ask",
        "cost_gate_executable": True,
        "n_minutes": len(spreads),
        "n_ticks": int(sum(tick_counts)),
        "first_timestamp": first_timestamp,
        "last_timestamp": last_timestamp,
        "span_filter": {
            "start": start,
            "end": end,
            "inclusive": True,
        },
        "spread": {
            "median": float(statistics.median(spreads)),
            "mean": float(statistics.mean(spreads)),
            "max": float(max(spreads)),
        },
        "spread_bps": {
            "median": float(statistics.median(spread_bps)),
            "mean": float(statistics.mean(spread_bps)),
            "max": float(max(spread_bps)),
        },
        "note": (
            "Unlike daily reference rates, these rows contain bid and ask quotes, "
            "so GCDE's all-in cost term can be instantiated without a "
            "frictionless macro-data exception."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_paths", type=Path, nargs="+")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("backtest_results/eurusd/eurusd_histdata_cost_gate_summary.json"),
    )
    args = parser.parse_args(argv)

    artifact = summarize_minute_bidask(args.csv_paths, start=args.start, end=args.end)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
