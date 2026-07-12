import pytest

from scripts.wp1.exclusion_plot import compute_z_trend_table


def _injection_cell(delta: float, q: int = 5, W: int = 120) -> dict:
    return {
        "delta_target": delta,
        "q": q,
        "W": W,
        "P_det": 0.0,
        "per_draw": [
            {
                "holm_ordered": [
                    {
                        "q": q,
                        "median_z_m2": 0.0,
                    }
                ]
            }
        ],
    }


def test_missing_exploratory_primary_cells_have_actionable_message() -> None:
    results = {
        (0.15, 5, 120): _injection_cell(0.15),
    }

    with pytest.raises(KeyError) as excinfo:
        compute_z_trend_table(results, delta_values=(0.15, 0.20, 0.30))

    message = str(excinfo.value)
    assert "Missing exploratory injection cells" in message
    assert "delta=0.2 q=5 W=120" in message
    assert "delta=0.3 q=5 W=120" in message
    assert "python scripts/wp1/signal_injection.py --addendum" in message
    assert "--start 112 --end 113" in message
    assert "--start 124 --end 125" in message
