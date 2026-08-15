"""Tests for free EURUSD HistData ingestion helpers."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_histdata_tick_url_uses_generic_ascii_tick_bid_ask() -> None:
    from scripts.wp1.download_histdata_eurusd import histdata_download_page_url

    url = histdata_download_page_url("EURUSD", 2022, 10)

    assert "histdata.com/download-free-forex-data" in url
    assert "ascii" in url
    assert "tick-data-quotes" in url
    assert "EURUSD" in url
    assert "2022" in url
    assert "10" in url


def test_parse_histdata_tick_csv_requires_bid_and_ask() -> None:
    from scripts.wp1.download_histdata_eurusd import parse_tick_csv_lines

    rows = parse_tick_csv_lines(["20221003 000000000,0.98000,0.98012,0"])

    assert rows[0]["timestamp"] == "2022-10-03T05:00:00.000000Z"
    assert rows[0]["bid"] == 0.98
    assert rows[0]["ask"] == 0.98012
    assert rows[0]["spread"] > 0.0


def test_aggregate_tick_rows_to_minutes_keeps_last_quote_spread() -> None:
    from scripts.wp1.download_histdata_eurusd import aggregate_tick_rows_to_minutes

    rows = aggregate_tick_rows_to_minutes(
        [
            "20221003 000000000,0.98000,0.98012,0",
            "20221003 000030000,0.98002,0.98015,0",
            "20221003 000100000,0.98020,0.98031,0",
        ]
    )

    assert len(rows) == 2
    assert rows[0]["timestamp"] == "2022-10-03T05:00:00.000000Z"
    assert rows[0]["bid"] == 0.98002
    assert rows[0]["ask"] == 0.98015
    assert rows[0]["n_ticks"] == 2
    assert rows[1]["timestamp"] == "2022-10-03T05:01:00.000000Z"


def test_validate_normalized_rows_reports_integrity_and_hash(tmp_path: Path) -> None:
    from scripts.wp1.download_histdata_eurusd import (
        validate_normalized_csv,
        write_rows_csv,
    )

    path = tmp_path / "EURUSD_2022_10_m1_bidask.csv"
    rows = [
        {
            "timestamp": "2022-10-03T05:00:00.000000Z",
            "bid": 0.98,
            "ask": 0.9801,
            "spread": 0.0001,
            "mid": 0.98005,
            "n_ticks": 2,
        },
        {
            "timestamp": "2022-10-03T05:01:00.000000Z",
            "bid": 0.9802,
            "ask": 0.9803,
            "spread": 0.0001,
            "mid": 0.98025,
            "n_ticks": 3,
        },
    ]
    write_rows_csv(rows, path)

    report = validate_normalized_csv(path)

    assert report["monotonic_utc"] is True
    assert report["ask_gte_bid_positive"] is True
    assert report["duplicate_timestamps"] == 0
    assert report["row_count"] == 2
    assert report["tick_count"] == 5
    assert len(report["sha256"]) == 64


def test_extract_download_payload_from_histdata_final_page() -> None:
    from scripts.wp1.download_histdata_eurusd import extract_download_payload

    html = """
    <form id="file_down" name="file_down" method="POST" action="/get.php">
      <input type="hidden" name="tk" value="abc123" />
      <input type="hidden" name="date" value="2022" />
      <input type="hidden" name="datemonth" value="202210" />
      <input type="hidden" name="platform" value="ASCII" />
      <input type="hidden" name="timeframe" value="T" />
      <input type="hidden" name="fxpair" value="EURUSD" />
    </form>
    """

    payload = extract_download_payload(html)

    assert payload == {
        "tk": "abc123",
        "date": "2022",
        "datemonth": "202210",
        "platform": "ASCII",
        "timeframe": "T",
        "fxpair": "EURUSD",
    }
