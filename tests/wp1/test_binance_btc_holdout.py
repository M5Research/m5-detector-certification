"""BTCUSDT holdout acquisition and normalization tests."""
from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import pytest


def _archive(rows: str) -> bytes:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as zf:
        zf.writestr("BTCUSDT-1m-2026-06.csv", rows)
    return payload.getvalue()


def test_parse_binance_klines_requires_monotonic_minute_rows() -> None:
    from scripts.wp1.download_binance_btc import parse_kline_archive

    rows = parse_kline_archive(
        _archive(
            "1780272000000,100,101,99,100.5,3,1780272059999,0,2,0,0,0\n"
            "1780272060000,100.5,102,100,101,4,1780272119999,0,3,0,0,0\n"
        )
    )

    assert len(rows) == 2
    assert rows[0]["timestamp"] == 1780272000000
    assert rows[1]["volume"] == 4.0


def test_normalize_archive_verifies_vendor_checksum(tmp_path: Path) -> None:
    from scripts.wp1.download_binance_btc import normalize_archive

    raw = _archive("1780272000000,100,101,99,100.5,3,1780272059999,0,2,0,0,0\n")
    archive = tmp_path / "BTCUSDT-1m-2026-06.zip"
    archive.write_bytes(raw)
    expected = hashlib.sha256(raw).hexdigest()
    checksum = archive.with_suffix(".zip.CHECKSUM")
    checksum.write_text(f"{expected}  {archive.name}\n", encoding="utf-8")

    report = normalize_archive(archive, checksum, tmp_path / "normalized.csv")

    assert report["vendor_checksum_verified"] is True
    assert report["row_count"] == 1
    assert len(report["normalized_sha256"]) == 64

    checksum.write_text(f"{'0' * 64}  {archive.name}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        normalize_archive(archive, checksum, tmp_path / "bad.csv")
