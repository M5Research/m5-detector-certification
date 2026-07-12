import numpy as np

from scripts.wp1.iid_floor import simulate_iid_vr_floor


def test_iid_floor_returns_ordered_quantiles():
    rows = simulate_iid_vr_floor(
        n_paths=8,
        n_obs=2_000,
        W_grid=(120,),
        q_grid=(5,),
        seed=43,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["W"] == 120
    assert row["q"] == 5
    assert row["delta_iid_p50"] > 0.0
    assert row["delta_iid_p95"] >= row["delta_iid_p50"]
    assert row["n_samples_total"] > 0


def test_iid_floor_is_deterministic_for_seed():
    kwargs = {
        "n_paths": 4,
        "n_obs": 1_000,
        "W_grid": (60,),
        "q_grid": (2, 5),
        "seed": 123,
    }
    rows_a = simulate_iid_vr_floor(**kwargs)
    rows_b = simulate_iid_vr_floor(**kwargs)
    assert rows_a == rows_b
    assert all(np.isfinite(row["delta_iid_p50"]) for row in rows_a)
