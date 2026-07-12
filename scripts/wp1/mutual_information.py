"""KSG k-NN mutual information estimator for thermodynamic profit bound."""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma

import scripts._bootstrap  # noqa: F401

try:
    import numba
except ImportError:  # pragma: no cover
    numba = None  # type: ignore[assignment]

_NUMBA_COUNT_THRESHOLD = 8_000

KSG_K = 5
KSG_BOOTSTRAP_RESAMPLES = 500
BOOT_SEED = 43
JITTER_SCALE = 1e-10
BOOTSTRAP_PAIR_SAMPLE_SIZE = 50_000
_EPS = 1e-15
_EPS_RADIUS_FLOOR = 1e-10

__all__ = [
    "KSG_K",
    "KSG_BOOTSTRAP_RESAMPLES",
    "BOOT_SEED",
    "JITTER_SCALE",
    "BOOTSTRAP_PAIR_SAMPLE_SIZE",
    "estimate_mi_from_returns",
    "ksg_mi_2d",
    "build_forward_sign_pairs",
    "build_forward_lag_pairs",
    "build_sign_pairs",
    "build_lag_pairs",
]


def _validate_returns_and_q(r: np.ndarray, q: int) -> np.ndarray:
    r_arr = np.asarray(r, dtype=np.float64)
    if r_arr.ndim != 1:
        raise ValueError(f"returns must be 1-D, got shape {r_arr.shape}")
    if q <= 0:
        raise ValueError(f"q must be positive, got {q}")
    if q >= len(r_arr):
        raise ValueError(f"q must be < len(r)={len(r_arr)}, got {q}")
    if not np.all(np.isfinite(r_arr)):
        raise ValueError("returns must be finite (no NaN or Inf)")
    return r_arr


def build_lag_pairs(r: np.ndarray, q: int) -> tuple[np.ndarray, np.ndarray]:
    """Build lag-q return pairs per D-17: x = r[q:], y = r[:-q]."""
    r_arr = _validate_returns_and_q(r, q)
    x = r_arr[q:]
    y = r_arr[:-q]
    assert len(x) == len(y) == len(r_arr) - q
    return x, y


def build_sign_pairs(r: np.ndarray, q: int) -> tuple[np.ndarray, np.ndarray]:
    """Build lag-q sign pairs per D-16/D-17."""
    r_arr = _validate_returns_and_q(r, q)
    s = np.sign(r_arr)
    return s[q:], s[:-q]


def build_forward_lag_pairs(r: np.ndarray, q: int) -> tuple[np.ndarray, np.ndarray]:
    """Build causal signal -> forward-return pairs: x = r[:-q], y = r[q:]."""
    x_future, y_past = build_lag_pairs(r, q)
    return y_past, x_future


def build_forward_sign_pairs(r: np.ndarray, q: int) -> tuple[np.ndarray, np.ndarray]:
    """Build causal sign signal -> forward sign pairs."""
    x_future, y_past = build_sign_pairs(r, q)
    return y_past, x_future


def _apply_marginal_jitter(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Break exact ties in marginal coordinates before KSG tree build."""
    x_j = x.copy()
    y_j = y.copy()
    if len(np.unique(x_j)) < len(x_j) or len(np.unique(y_j)) < len(y_j):
        x_j = x_j + JITTER_SCALE * rng.standard_normal(len(x_j))
        y_j = y_j + JITTER_SCALE * rng.standard_normal(len(y_j))
    return x_j, y_j


if numba is not None:

    @numba.njit(cache=True)
    def _marginal_neighbor_counts_1d(values: np.ndarray, radii: np.ndarray) -> np.ndarray:
        """Chebyshev (L-inf) neighbor counts in 1D for KSG marginal terms."""
        n = values.shape[0]
        counts = np.empty(n, dtype=np.float64)
        for i in range(n):
            ri = radii[i]
            vi = values[i]
            cnt = 0
            for j in range(n):
                if abs(values[j] - vi) <= ri:
                    cnt += 1
            counts[i] = cnt - 1.0
        return counts


def _marginal_neighbor_counts(
    values: np.ndarray,
    radii: np.ndarray,
    tree: cKDTree,
) -> np.ndarray:
    """Marginal KSG neighbor counts; numba for moderate n, scipy for large n."""
    n = len(values)
    if numba is not None and n <= _NUMBA_COUNT_THRESHOLD:
        return _marginal_neighbor_counts_1d(values, radii)

    col = values.reshape(-1, 1)
    workers = -1 if n >= 2_000 else 1
    return np.fromiter(
        (
            len(neighbors) - 1
            for neighbors in tree.query_ball_point(col, radii, p=np.inf, workers=workers)
        ),
        dtype=np.float64,
        count=n,
    )


def ksg_mi_2d(x: np.ndarray, y: np.ndarray, k: int = KSG_K) -> float:
    """KSG Type-I mutual information for 2D pairs (nats)."""
    x_col = np.asarray(x, dtype=np.float64).reshape(-1, 1)
    y_col = np.asarray(y, dtype=np.float64).reshape(-1, 1)
    if len(x_col) != len(y_col):
        raise ValueError("x and y must have the same length")
    n = len(x_col)
    if n <= k:
        raise ValueError(f"need at least k+1 samples, got n={n}, k={k}")

    jitter_seed = int(np.bitwise_xor(x_col.view(np.uint64).sum(), y_col.view(np.uint64).sum())) % (2**32)
    rng = np.random.default_rng(jitter_seed)
    x_col, y_col = _apply_marginal_jitter(x_col.ravel(), y_col.ravel(), rng)
    x_col = x_col.reshape(-1, 1)
    y_col = y_col.reshape(-1, 1)

    xy = np.hstack([x_col, y_col])
    tree_xy = cKDTree(xy)
    dists, _ = tree_xy.query(xy, k=k + 1, p=np.inf)
    eps = dists[:, -1]

    tree_x = cKDTree(x_col)
    tree_y = cKDTree(y_col)
    radii = np.maximum(eps - _EPS, _EPS_RADIUS_FLOOR)
    nx = _marginal_neighbor_counts(x_col.ravel(), radii, tree_x)
    ny = _marginal_neighbor_counts(y_col.ravel(), radii, tree_y)

    mi = digamma(k) + digamma(n) - np.mean(digamma(nx + 1) + digamma(ny + 1))
    return float(max(mi, 0.0))


def _bootstrap_stderr(
    x: np.ndarray,
    y: np.ndarray,
    k: int,
    n_boot: int,
    seed: int,
) -> float:
    n_pairs = len(x)
    rng = np.random.default_rng(seed)
    sample_size = min(n_pairs, BOOTSTRAP_PAIR_SAMPLE_SIZE)

    mi_boots = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.choice(n_pairs, size=sample_size, replace=True)
        mi_boots[b] = ksg_mi_2d(x[idx], y[idx], k=k)

    return float(np.std(mi_boots, ddof=1))


def _discrete_mi_from_signs(x: np.ndarray, y: np.ndarray) -> float:
    """Exact plug-in MI for sign pairs in {-1, 0, 1}."""
    x_idx = (np.asarray(x, dtype=np.int8) + 1).astype(np.int64)
    y_idx = (np.asarray(y, dtype=np.int8) + 1).astype(np.int64)
    counts = np.zeros((3, 3), dtype=np.float64)
    for xi, yi in zip(x_idx, y_idx, strict=True):
        counts[xi, yi] += 1.0

    total = counts.sum()
    if total <= 0.0:
        return 0.0

    pxy = counts / total
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    mi = 0.0
    for i in range(3):
        for j in range(3):
            p = pxy[i, j]
            if p > 0.0 and px[i] > 0.0 and py[j] > 0.0:
                mi += p * np.log(p / (px[i] * py[j]))
    return float(max(mi, 0.0))


def _bootstrap_stderr_signs(
    x: np.ndarray,
    y: np.ndarray,
    n_boot: int,
    seed: int,
) -> float:
    n_pairs = len(x)
    if n_boot <= 1 or n_pairs == 0:
        return 0.0

    rng = np.random.default_rng(seed)
    sample_size = min(n_pairs, BOOTSTRAP_PAIR_SAMPLE_SIZE)
    mi_boots = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.choice(n_pairs, size=sample_size, replace=True)
        mi_boots[b] = _discrete_mi_from_signs(x[idx], y[idx])
    return float(np.std(mi_boots, ddof=1))


def estimate_mi_from_returns(
    r: np.ndarray,
    q: int,
    k: int = KSG_K,
    n_boot: int = KSG_BOOTSTRAP_RESAMPLES,
    seed: int = BOOT_SEED,
    method: str = "signs",
    orientation: str = "serial",
) -> tuple[float, float]:
    """Estimate lag-q MI from returns with bootstrap stderr (pair-row resampling)."""
    _validate_returns_and_q(r, q)
    if orientation not in {"serial", "forward"}:
        raise ValueError(
            f"orientation must be 'serial' or 'forward', got {orientation!r}"
        )
    if method == "signs":
        if orientation == "forward":
            x, y = build_forward_sign_pairs(r, q)
        else:
            x, y = build_sign_pairs(r, q)
        n_pairs = len(x)
        sample_size = min(n_pairs, BOOTSTRAP_PAIR_SAMPLE_SIZE)
        if n_pairs > BOOTSTRAP_PAIR_SAMPLE_SIZE:
            rng = np.random.default_rng(seed)
            idx = rng.choice(n_pairs, size=sample_size, replace=False)
            x_est = x[idx]
            y_est = y[idx]
        else:
            x_est = x
            y_est = y
        mi = _discrete_mi_from_signs(x_est, y_est)
        stderr = _bootstrap_stderr_signs(x, y, n_boot=n_boot, seed=seed)
        return mi, stderr
    elif method == "raw":
        if orientation == "forward":
            x, y = build_forward_lag_pairs(r, q)
        else:
            x, y = build_lag_pairs(r, q)
    else:
        raise ValueError(f"method must be 'signs' or 'raw', got {method!r}")

    n_pairs = len(x)
    sample_size = min(n_pairs, BOOTSTRAP_PAIR_SAMPLE_SIZE)
    if n_pairs > BOOTSTRAP_PAIR_SAMPLE_SIZE:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n_pairs, size=sample_size, replace=False)
        x_est = x[idx]
        y_est = y[idx]
    else:
        x_est = x
        y_est = y

    mi = ksg_mi_2d(x_est, y_est, k=k)
    stderr = _bootstrap_stderr(x, y, k=k, n_boot=n_boot, seed=seed)
    return mi, stderr
