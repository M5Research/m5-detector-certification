"""Tests for EURUSD HistData cost-gate summary artifacts."""
from __future__ import annotations

from pathlib import Path


def test_summarize_minute_bidask_reports_spread_bps(tmp_path: Path) -> None:
    from scripts.wp1.eurusd_cost_gate_summary import summarize_minute_bidask

    path = tmp_path / "eurusd_m1.csv"
    path.write_text(
        "\n".join(
            [
                "timestamp,bid,ask,spread,mid,n_ticks",
                "2022-10-03T00:00:00.000000,0.98000,0.98010,0.00010,0.98005,3",
                "2022-10-03T00:01:00.000000,0.98020,0.98030,0.00010,0.98025,4",
            ]
        ),
        encoding="utf-8",
    )

    artifact = summarize_minute_bidask(path)

    assert artifact["cost_gate_executable"] is True
    assert artifact["n_minutes"] == 2
    assert artifact["n_ticks"] == 7
    assert artifact["spread_bps"]["median"] > 1.0


def test_summarize_minute_bidask_filters_multiple_files(tmp_path: Path) -> None:
    from scripts.wp1.eurusd_cost_gate_summary import summarize_minute_bidask

    first = tmp_path / "EURUSD_2021_05_m1_bidask.csv"
    second = tmp_path / "EURUSD_2021_06_m1_bidask.csv"
    first.write_text(
        "\n".join(
            [
                "timestamp,bid,ask,spread,mid,n_ticks",
                "2021-05-29T19:31:00.000000,1.10000,1.10010,0.00010,1.10005,1",
                "2021-05-29T19:32:00.000000,1.10000,1.10020,0.00020,1.10010,2",
            ]
        ),
        encoding="utf-8",
    )
    second.write_text(
        "\n".join(
            [
                "timestamp,bid,ask,spread,mid,n_ticks",
                "2021-06-01T00:00:00.000000,1.10100,1.10110,0.00010,1.10105,3",
            ]
        ),
        encoding="utf-8",
    )

    artifact = summarize_minute_bidask(
        [second, first],
        start="2021-05-29T19:32:00",
        end="2021-06-01T00:00:00",
    )

    assert artifact["n_minutes"] == 2
    assert artifact["n_ticks"] == 5
    assert artifact["first_timestamp"] == "2021-05-29T19:32:00.000000"
    assert artifact["last_timestamp"] == "2021-06-01T00:00:00.000000"
    assert len(artifact["used_source_files"]) == 2
