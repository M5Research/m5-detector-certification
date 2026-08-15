"""Download and normalize free HistData EURUSD Generic ASCII tick data.

HistData M1 bar files are bid-only. The Generic ASCII tick files include
DateTime,Bid,Ask,Volume, which is the minimum free-data route that keeps a
spread observable for GCDE cost-gate work.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import zipfile
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote

# `requests` is imported inside download_month() rather than here. Everything
# else in this module — URL construction, hidden-form payload extraction, tick
# row parsing — is offline string handling, and its tests should not need a
# network library installed. `requests` is an optional 'download' extra.

DEFAULT_SYMBOL = "EURUSD"
DEFAULT_OUT_DIR = Path("data/external/histdata/EURUSD/tick")
HISTDATA_BASE = "https://www.histdata.com/download-free-forex-data/"
HISTDATA_TZ = timezone(timedelta(hours=-5), name="EST")


def histdata_download_page_url(symbol: str, year: int, month: int) -> str:
    """Return the HistData Generic ASCII tick-data page URL for one month."""
    if month < 1 or month > 12:
        raise ValueError("month must be in 1..12")
    clean_symbol = symbol.replace("/", "").upper()
    path = f"/ascii/tick-data-quotes/{clean_symbol}/{year}/{month:02d}"
    return f"{HISTDATA_BASE}?{quote(path)}"


def _parse_tick_line(raw: str) -> dict[str, float | str] | None:
    line = raw.strip()
    if not line:
        return None
    parts = [part.strip() for part in line.split(",")]
    if len(parts) < 3:
        raise ValueError("HistData tick rows must contain DateTime,Bid,Ask")
    stamp = parts[0]
    bid = float(parts[1])
    ask = float(parts[2])
    if bid <= 0.0 or ask < bid:
        raise ValueError(f"quotes must satisfy ask >= bid > 0 for {stamp}")
    volume = float(parts[3]) if len(parts) >= 4 and parts[3] else 0.0
    dt = _parse_histdata_timestamp(stamp).replace(tzinfo=HISTDATA_TZ).astimezone(UTC)
    return {
        "timestamp": dt.isoformat(timespec="microseconds").replace("+00:00", "Z"),
        "bid": bid,
        "ask": ask,
        "spread": ask - bid,
        "mid": (bid + ask) / 2.0,
        "volume": volume,
    }


def parse_tick_csv_lines(
    lines: Iterable[str],
    max_rows: int | None = None,
) -> list[dict[str, float | str]]:
    """Parse HistData tick CSV rows and require Bid/Ask spread observability."""
    rows: list[dict[str, float | str]] = []
    for raw in lines:
        parsed = _parse_tick_line(raw)
        if parsed is None:
            continue
        rows.append(parsed)
        if max_rows is not None and len(rows) >= max_rows:
            break
    return rows


def aggregate_tick_rows_to_minutes(
    lines: Iterable[str],
    max_rows: int | None = None,
) -> list[dict[str, float | str | int]]:
    """Aggregate HistData tick quotes to one-minute last-quote bid/ask bars."""
    rows: list[dict[str, float | str | int]] = []
    current_minute: str | None = None
    last_quote: dict[str, float | str] | None = None
    n_ticks = 0
    parsed_count = 0

    def flush() -> None:
        if current_minute is None or last_quote is None:
            return
        rows.append(
            {
                "timestamp": current_minute,
                "bid": float(last_quote["bid"]),
                "ask": float(last_quote["ask"]),
                "spread": float(last_quote["spread"]),
                "mid": float(last_quote["mid"]),
                "n_ticks": int(n_ticks),
            }
        )

    for raw in lines:
        parsed = _parse_tick_line(raw)
        if parsed is None:
            continue
        parsed_count += 1
        minute = str(parsed["timestamp"])[:16] + ":00.000000Z"
        if current_minute is None:
            current_minute = minute
            n_ticks = 0
        elif minute != current_minute:
            flush()
            current_minute = minute
            n_ticks = 0
        last_quote = parsed
        n_ticks += 1
        if max_rows is not None and parsed_count >= max_rows:
            break
    flush()
    return rows


def _parse_histdata_timestamp(value: str) -> datetime:
    for fmt in ("%Y%m%d %H%M%S%f", "%Y.%m.%d %H:%M:%S.%f", "%Y%m%d %H%M%S"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    raise ValueError(f"unsupported HistData timestamp format: {value!r}")


def extract_download_payload(html: str) -> dict[str, str]:
    """Extract the hidden POST payload from HistData's final download page."""
    form_match = re.search(
        r'<form[^>]+id=["\']file_down["\'][^>]*>(?P<body>.*?)</form>',
        html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if form_match is None:
        raise ValueError("HistData final page did not contain the file_down form")

    body = form_match.group("body")
    payload: dict[str, str] = {}
    for input_match in re.finditer(r"<input\b(?P<attrs>[^>]*)>", body, flags=re.IGNORECASE):
        attrs = input_match.group("attrs")
        name_match = re.search(r'name=["\'](?P<name>[^"\']+)["\']', attrs, flags=re.IGNORECASE)
        value_match = re.search(r'value=["\'](?P<value>[^"\']*)["\']', attrs, flags=re.IGNORECASE)
        if name_match is not None and value_match is not None:
            payload[name_match.group("name")] = value_match.group("value")

    required = {"tk", "date", "datemonth", "platform", "timeframe", "fxpair"}
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"HistData file_down form missing fields: {missing}")
    return payload


def _iter_first_csv(zip_bytes: bytes) -> Iterator[str]:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        csv_names = [name for name in zf.namelist() if name.lower().endswith(".csv")]
        if not csv_names:
            raise ValueError("downloaded zip did not contain a CSV file")
        with zf.open(csv_names[0]) as handle:
            for line in handle:
                yield line.decode("utf-8", errors="replace")


def download_month_archive(
    symbol: str,
    year: int,
    month: int,
    timeout: int = 60,
) -> bytes:
    """Download one immutable HistData vendor archive."""
    try:
        import requests
    except ModuleNotFoundError as exc:  # pragma: no cover - install-path guard
        raise ModuleNotFoundError(
            "download_month_archive() needs the optional 'download' extra: "
            'pip install -e ".[download]"'
        ) from exc

    url = histdata_download_page_url(symbol, year, month)
    with requests.Session() as session:
        page = session.get(url, timeout=timeout)
        page.raise_for_status()
        payload = extract_download_payload(page.text)
        response = session.post(
            "https://www.histdata.com/get.php",
            data=payload,
            headers={"Referer": url},
            timeout=timeout,
        )
        response.raise_for_status()
    if not response.content.startswith(b"PK"):
        raise ValueError("HistData download response was not a zip file")
    return response.content


def parse_month_archive(
    zip_bytes: bytes,
    *,
    max_rows: int | None = None,
    aggregate: str = "tick",
) -> list[dict[str, float | str | int]]:
    """Parse one HistData Generic ASCII tick archive."""
    lines = _iter_first_csv(zip_bytes)
    if aggregate == "tick":
        rows = parse_tick_csv_lines(lines, max_rows=max_rows)
    elif aggregate == "minute":
        rows = aggregate_tick_rows_to_minutes(lines, max_rows=max_rows)
    else:
        raise ValueError("aggregate must be 'tick' or 'minute'")
    if not rows:
        raise ValueError("HistData archive contained no parseable rows")
    return rows


def download_month(
    symbol: str,
    year: int,
    month: int,
    timeout: int = 60,
    max_rows: int | None = None,
    aggregate: str = "tick",
) -> list[dict[str, float | str | int]]:
    """Download and parse one HistData Generic ASCII tick month."""
    archive = download_month_archive(symbol, year, month, timeout)
    return parse_month_archive(archive, max_rows=max_rows, aggregate=aggregate)


def write_rows_csv(rows: list[dict[str, float | str | int]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["timestamp", "bid", "ask", "spread", "mid"]
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_normalized_csv(path: Path) -> dict[str, object]:
    """Validate one normalized minute partition without loading it wholesale."""
    previous: datetime | None = None
    seen: set[str] = set()
    duplicates = 0
    monotonic = True
    quotes_valid = True
    rows = 0
    ticks = 0
    first: str | None = None
    last: str | None = None
    weekend_rows = 0
    max_gap_minutes = 0.0
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            stamp = row["timestamp"]
            current = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
            if current.tzinfo is None or current.utcoffset() != timedelta(0):
                monotonic = False
            if previous is not None:
                if current <= previous:
                    monotonic = False
                max_gap_minutes = max(max_gap_minutes, (current - previous).total_seconds() / 60)
            previous = current
            if stamp in seen:
                duplicates += 1
            seen.add(stamp)
            bid, ask = float(row["bid"]), float(row["ask"])
            quotes_valid &= bid > 0.0 and ask >= bid
            rows += 1
            ticks += int(row.get("n_ticks", 1))
            weekend_rows += int(current.weekday() >= 5)
            first = first or stamp
            last = stamp
    return {
        "path": path.as_posix(),
        "sha256": _sha256(path),
        "row_count": rows,
        "tick_count": ticks,
        "first_timestamp": first,
        "last_timestamp": last,
        "monotonic_utc": monotonic,
        "ask_gte_bid_positive": quotes_valid,
        "duplicate_timestamps": duplicates,
        "weekend_rows": weekend_rows,
        "max_gap_minutes": max_gap_minutes,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    parser.add_argument("--year", type=int, default=2022)
    parser.add_argument("--months", type=int, nargs="+", default=[10, 11, 12])
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--aggregate", choices=["tick", "minute"], default="tick")
    parser.add_argument(
        "--manifest",
        type=Path,
        help="optional compact JSON manifest containing vendor and normalized hashes",
    )
    args = parser.parse_args(argv)

    reports: list[dict[str, object]] = []
    for month in args.months:
        archive = download_month_archive(args.symbol, args.year, month)
        raw_path = args.out_dir / "raw" / f"{args.symbol.upper()}_{args.year}_{month:02d}_tick.zip"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_bytes(archive)
        rows = parse_month_archive(
            archive,
            max_rows=args.max_rows,
            aggregate=args.aggregate,
        )
        suffix = "tick_bidask" if args.aggregate == "tick" else "m1_bidask"
        out_path = args.out_dir / f"{args.symbol.upper()}_{args.year}_{month:02d}_{suffix}.csv"
        write_rows_csv(rows, out_path)
        print(f"Wrote {out_path} ({len(rows)} rows)")
        report = validate_normalized_csv(out_path)
        report.update(
            {
                "source_url": histdata_download_page_url(args.symbol, args.year, month),
                "raw_path": raw_path.as_posix(),
                "raw_sha256": _sha256(raw_path),
                "raw_bytes": raw_path.stat().st_size,
            }
        )
        reports.append(report)
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(
            json.dumps({"source": "HistData Generic ASCII tick", "partitions": reports}, indent=2)
            + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
