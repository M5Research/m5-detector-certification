"""Acquire and normalize the frozen BTCUSDT USD-M one-minute holdout."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path

BASE_URL = "https://data.binance.vision/data/futures/um/monthly/klines"
DEFAULT_DIR = Path("data/external/binance/BTCUSDT/1m")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_kline_archive(payload: bytes) -> list[dict[str, float | int]]:
    """Parse Binance monthly kline CSV and fail on malformed chronology."""
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        names = [name for name in zf.namelist() if name.lower().endswith(".csv")]
        if len(names) != 1:
            raise ValueError("Binance archive must contain exactly one CSV")
        with zf.open(names[0]) as handle:
            reader = csv.reader(io.TextIOWrapper(handle, encoding="utf-8"))
            rows: list[dict[str, float | int]] = []
            previous: int | None = None
            for fields in reader:
                if not fields:
                    continue
                try:
                    stamp = int(fields[0])
                except ValueError:  # modern archives may include a header
                    if fields[0].lower() in {"open_time", "timestamp"}:
                        continue
                    raise
                if len(fields) < 7:
                    raise ValueError("Binance kline row has fewer than seven fields")
                if previous is not None and stamp - previous != 60_000:
                    raise ValueError("Binance one-minute timestamps are not contiguous")
                previous = stamp
                open_, high, low, close, volume = map(float, fields[1:6])
                if min(open_, high, low, close) <= 0.0 or volume < 0.0:
                    raise ValueError("Binance kline contains invalid OHLCV")
                if high < max(open_, close) or low > min(open_, close):
                    raise ValueError("Binance kline violates OHLC bounds")
                rows.append(
                    {
                        "timestamp": stamp,
                        "open": open_,
                        "high": high,
                        "low": low,
                        "close": close,
                        "volume": volume,
                        "close_time": int(fields[6]),
                    }
                )
    if not rows:
        raise ValueError("Binance archive contained no kline rows")
    return rows


def normalize_archive(archive: Path, checksum: Path, out_csv: Path) -> dict[str, object]:
    """Verify a vendor checksum and emit a deterministic normalized partition."""
    payload = archive.read_bytes()
    actual = _sha256_bytes(payload)
    expected = checksum.read_text(encoding="utf-8").split()[0].lower()
    if actual != expected:
        raise ValueError(f"vendor checksum mismatch for {archive.name}")
    rows = parse_kline_archive(payload)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {
        "source_url": f"{BASE_URL}/BTCUSDT/1m/{archive.name}",
        "raw_path": archive.as_posix(),
        "raw_sha256": actual,
        "vendor_checksum_verified": True,
        "normalized_path": out_csv.as_posix(),
        "normalized_sha256": _sha256_path(out_csv),
        "row_count": len(rows),
        "first_timestamp": rows[0]["timestamp"],
        "last_timestamp": rows[-1]["timestamp"],
    }


def _download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310
        path.write_bytes(response.read())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--months", nargs="+", default=["2026-06", "2026-07"])
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("evidence/data/btcusdt-binance-2026-06_07-manifest.json"),
    )
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args(argv)

    reports: list[dict[str, object]] = []
    for month in args.months:
        stem = f"BTCUSDT-1m-{month}.zip"
        archive = args.out_dir / stem
        checksum = args.out_dir / f"{stem}.CHECKSUM"
        url = f"{BASE_URL}/BTCUSDT/1m/{stem}"
        if not args.no_download:
            _download(url, archive)
            _download(f"{url}.CHECKSUM", checksum)
        normalized = args.out_dir / f"BTCUSDT-1m-{month}.csv"
        reports.append(normalize_archive(archive, checksum, normalized))

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(
            {
                "source": "Binance USD-M public data archive",
                "instrument": "BTCUSDT",
                "interval": "1m",
                "partitions": reports,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
