"""Pin the polars behaviour the frozen D-01 warmup length depends on.

`rolling_quantile_detector` documents its warmup boundary as a consequence of
three specific polars behaviours, observed in 1.41.2:

  1. `fill_nan(None)` turns NaN into a polars null;
  2. `min_samples=None` means "require a full window", so any null inside the
     window yields a null result;
  3. `interpolation="linear"` is requested explicitly, because the polars
     default is `"nearest"` and the frozen spec is linear.

Together these fix the first valid bar at `rv_window + pct_window - 2`. That
number is load-bearing: it determines `n_warmup`, and therefore the bar counts
recorded in the Pre-Check artifacts and the label populations the whole
cascade is computed over.

pyproject pins `polars>=1.41,<2`, which is a range, so a future patch release
inside that range could change any of the three without anyone noticing. These
tests fail loudly if that happens, instead of silently shifting a frozen
warmup boundary.
"""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

# Deliberately NOT marked `artifact`: this module pins a polars behaviour that
# the frozen warmup length depends on, but it reads no committed artifact.
# `pytest -m artifact` must keep meaning "validates the frozen artifacts".


def test_fill_nan_none_makes_nan_a_null() -> None:
    s = pl.Series([1.0, float("nan"), 3.0]).fill_nan(None)
    assert s.null_count() == 1
    assert s.to_list()[0] == 1.0


def test_min_samples_none_requires_a_full_window() -> None:
    """A window containing a null must produce a null, not a partial quantile."""
    s = pl.Series([float("nan"), 1.0, 2.0, 3.0, 4.0]).fill_nan(None)
    out = s.rolling_quantile(
        quantile=0.5,
        interpolation="linear",
        window_size=3,
        min_samples=None,
        center=False,
    ).to_list()

    # Windows ending at index 0, 1, 2 all contain the leading null.
    assert out[0] is None
    assert out[1] is None
    assert out[2] is None
    # First window free of nulls is [1.0, 2.0, 3.0], ending at index 3.
    assert out[3] == pytest.approx(2.0)
    assert out[4] == pytest.approx(3.0)


def test_interpolation_linear_is_not_the_default() -> None:
    """The frozen spec requests linear explicitly; polars defaults to nearest.

    If these ever coincide, the explicit argument stops being load-bearing and
    the docstring's warning is stale — worth knowing either way.
    """
    s = pl.Series([1.0, 2.0, 3.0, 4.0])
    linear = s.rolling_quantile(
        quantile=0.75, interpolation="linear", window_size=4, min_samples=None
    ).to_list()[-1]
    nearest = s.rolling_quantile(
        quantile=0.75, interpolation="nearest", window_size=4, min_samples=None
    ).to_list()[-1]
    assert linear == pytest.approx(3.25)
    assert nearest != pytest.approx(linear)


def test_d01_warmup_boundary_is_rv_window_plus_pct_window_minus_two() -> None:
    """The exact D-01 identity, reproduced from polars primitives.

    With `rv_window` leading NaN bars in the RV series (the RV warmup) and a
    percentile window of `pct_window`, the first bar carrying a non-null
    quantile is at index `rv_window + pct_window - 2`.
    """
    rv_window, pct_window = 5, 4
    n = 40
    rv = np.arange(n, dtype=np.float64)
    # compute_rv leaves rv_window - 1 leading NaNs.
    rv[: rv_window - 1] = np.nan

    q = (
        pl.Series(rv)
        .fill_nan(None)
        .rolling_quantile(
            quantile=0.75,
            interpolation="linear",
            window_size=pct_window,
            min_samples=None,
            center=False,
        )
        .to_numpy()
    )

    first_valid = int(np.argmax(~np.isnan(q)))
    assert first_valid == rv_window + pct_window - 2
    assert np.isnan(q[first_valid - 1])
    assert not np.isnan(q[first_valid])
