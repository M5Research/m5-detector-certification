"""BTC VR detector information under the stationary-block reference."""

from __future__ import annotations

import numpy as np


def test_stationary_trigger_information_has_no_permutation_fields() -> None:
    from scripts.wp1.vr_detector_stationary_block import stationary_trigger_information

    labels = np.array([0, 1, 0, 1] * 50, dtype=np.int8)
    target = labels.copy()
    row = stationary_trigger_information(
        labels,
        target,
        spec_hash="ab" * 32,
        alpha=0.025,
        n_boot=99,
        gate_label="btc-vr-q2-information",
    )

    assert "permutation_p" not in row
    assert row["reference"] == "stationary_block"
    assert row["e_value"] == 0.5 * row["p_value"] ** -0.5
    assert row["e_value_threshold"] == 40.0
    assert row["gate_state"] in {"pass", "fail", "invalid"}
