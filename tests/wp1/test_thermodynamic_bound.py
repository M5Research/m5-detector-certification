import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from backtest.utils import PROJECT_ROOT

import scripts.wp1.thermodynamic_bound as thermo_bound
from scripts.wp1.thermodynamic_bound import (
    Q_GRID,
    TAKER_RT_BPS,
    VERDICT_DEMON,
    compute_thermodynamic_bound,
    load_delta_star_primary,
    vr_departure_to_gaussian_mi,
    vr_to_mutual_information,
    _verdict_for_gmax,
)
from scripts.wp1.thermodynamic_figures import generate_bound_vs_cost_figure

N_BOOT_TEST = 15
N_CLOSE = 1500


def _random_walk_close(n: int, seed: int = 43) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 100.0 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))


def _cost_schedule() -> dict:
    return {
        "taker_fill_bps": 4.5,
        "maker_fill_bps": 1.8,
        "taker_rt_bps": [9.0, 10.0],
        "maker_rt_bps": [3.6, 5.6],
    }


def test_vr_to_mi_gaussian():
    out = vr_to_mutual_information(0.01, q=5, method="gaussian")
    rho = 0.01 / (2.0 * (1.0 - 1.0 / 5))
    expected = -0.5 * np.log(1.0 - rho**2)
    assert out["method"] == "gaussian"
    assert out["mi_nats"] > 0
    assert 0 < out["rho"] < 1
    assert abs(out["mi_nats"] - expected) < 1e-9


def test_gaussian_rho_clamp():
    mi = vr_departure_to_gaussian_mi(5.0, q=2)
    assert np.isfinite(mi)
    assert mi >= 0.0


def test_vr_to_mi_ksg():
    close = _random_walk_close(5000)
    out = vr_to_mutual_information(
        0.0, q=5, method="ksg", close=close, n_boot=N_BOOT_TEST
    )
    assert out["method"] == "ksg_signs_forward"
    assert out["mi_nats"] >= 0
    assert np.isfinite(out["mi_stderr"])


def test_compute_thermodynamic_bound():
    close = _random_walk_close(N_CLOSE)
    report = compute_thermodynamic_bound(
        close, Q_GRID, _cost_schedule(), injection_dir=None, n_boot=N_BOOT_TEST
    )
    assert "per_q_results" in report
    assert "headline_verdict" in report
    assert report["headline_q"] == 5
    assert len(report["per_q_results"]) == 4
    assert report["headline_verdict"] in ("demon_runs_at_loss", "bound_exceeds_costs")
    for row in report["per_q_results"]:
        assert "gaussian_gmax_bps" in row
        assert "ksg_gmax_bps" in row
        assert row["gaussian_gmax_bps"] >= 0
        assert row["ksg_gmax_bps"] >= 0


def _write_mock_injection(tmp_path: Path) -> Path:
    deltas = [0.001, 0.01, 0.05, 0.10]
    pdets = [0.2, 0.5, 0.92, 0.98]
    for delta, pdet in zip(deltas, pdets, strict=True):
        payload = {
            "delta": delta,
            "q": 5,
            "W": 120,
            "P_det": pdet,
            "ci_95_lo": max(pdet - 0.05, 0.0),
            "ci_95_hi": min(pdet + 0.05, 1.0),
        }
        path = tmp_path / f"inj_d{delta}_q5_w120.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
    return tmp_path


def test_exclusion_gap_with_mock_injection():
    close = _random_walk_close(N_CLOSE)
    inj_dir = PROJECT_ROOT / "data" / "_test_injection_thermo"
    if inj_dir.exists():
        shutil.rmtree(inj_dir)
    inj_dir.mkdir(parents=True)
    try:
        _write_mock_injection(inj_dir)
        report = compute_thermodynamic_bound(
            close, Q_GRID, _cost_schedule(), injection_dir=inj_dir, n_boot=N_BOOT_TEST
        )
        assert report["delta_star_info"]["status"] == "ok"
        assert report["exclusion_gap_summary"]["headline_gap_bps"] is not None
        assert any(row["exclusion_gap_bps"] is not None for row in report["per_q_results"])
    finally:
        if inj_dir.exists():
            shutil.rmtree(inj_dir)


def test_chain_consistent_across_q():
    close = _random_walk_close(N_CLOSE)
    report = compute_thermodynamic_bound(
        close, Q_GRID, _cost_schedule(), injection_dir=None, n_boot=N_BOOT_TEST
    )
    for row in report["per_q_results"]:
        assert np.isfinite(row["gaussian_gmax_bps"])
        assert np.isfinite(row["ksg_gmax_bps"])
        assert row["gaussian_gmax_bps"] >= 0
        assert row["ksg_gmax_bps"] >= 0


def test_cost_comparison_logic():
    assert _verdict_for_gmax(0.5) == VERDICT_DEMON
    assert _verdict_for_gmax(5.0) == VERDICT_DEMON
    assert _verdict_for_gmax(max(TAKER_RT_BPS) + 1.0) != VERDICT_DEMON


def test_report_verdict_uses_sign_pair_ceiling_not_gaussian(monkeypatch):
    close = _random_walk_close(N_CLOSE)

    def fake_horizon_profile(close_arr, W, q_grid):
        _ = close_arr, W
        return {q: 0.50 for q in q_grid}

    def fake_sign_pair_mi(vr_departure, q, method="gaussian", close=None, seed=43, n_boot=None):
        _ = vr_departure, q, close, seed, n_boot
        assert method == "ksg"
        return {"mi_nats": 0.0005, "mi_stderr": 0.0, "method": "ksg_signs_forward"}

    monkeypatch.setattr(thermo_bound, "compute_vr_horizon_profile", fake_horizon_profile)
    monkeypatch.setattr(thermo_bound, "vr_to_mutual_information", fake_sign_pair_mi)

    report = compute_thermodynamic_bound(
        close, Q_GRID, _cost_schedule(), injection_dir=None, n_boot=N_BOOT_TEST
    )

    assert report["headline_verdict"] == VERDICT_DEMON
    for row in report["per_q_results"]:
        assert row["gaussian_gmax_bps"] > max(TAKER_RT_BPS)
        assert row["ksg_gmax_bps"] < min(TAKER_RT_BPS)
        assert row["verdict_gaussian_vs_taker"] != VERDICT_DEMON
        assert row["verdict_sign_pair_vs_taker"] == VERDICT_DEMON
        assert row["verdict_vs_taker"] == row["verdict_sign_pair_vs_taker"]


def test_graceful_degradation_no_injection():
    close = _random_walk_close(N_CLOSE)
    report = compute_thermodynamic_bound(
        close,
        Q_GRID,
        _cost_schedule(),
        injection_dir=Path("/nonexistent"),
        n_boot=N_BOOT_TEST,
    )
    assert "missing" in report["delta_star_info"]["status"]
    for row in report["per_q_results"]:
        assert row["gaussian_gmax_bps"] is not None
        assert row["ksg_gmax_bps"] is not None


def test_load_delta_star_primary_missing(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    info = load_delta_star_primary(empty)
    assert info["status"] == "injection_dir_missing"


def test_figure_smoke(tmp_path):
    close = _random_walk_close(N_CLOSE)
    report = compute_thermodynamic_bound(
        close, Q_GRID, _cost_schedule(), injection_dir=None, n_boot=N_BOOT_TEST
    )
    report["run_id"] = "test_run"
    out = generate_bound_vs_cost_figure(report, output_dir=tmp_path)
    assert out.suffix == ".png"
    assert out.exists()
    assert out.stat().st_size > 1000


def test_figure_with_envelope(tmp_path):
    close = _random_walk_close(N_CLOSE)
    report = compute_thermodynamic_bound(
        close, Q_GRID, _cost_schedule(), injection_dir=None, n_boot=N_BOOT_TEST
    )
    report["run_id"] = "test_run"
    report["delta_star_info"] = {"delta_star_90": 0.05, "status": "ok"}
    for row in report["per_q_results"]:
        row["envelope_gaussian_gmax_bps"] = row["gaussian_gmax_bps"] + 2.0
    out = generate_bound_vs_cost_figure(report, output_dir=tmp_path)
    assert out.exists()
    assert out.stat().st_size > 1000
