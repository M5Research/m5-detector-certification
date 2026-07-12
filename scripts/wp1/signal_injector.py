"""Signal injection: AR(1) autocorrelation injection, phi-to-delta mapping, GARCH fitting."""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

__all__ = [
    "inject_ar1_autocorrelation",
    "fit_garch_sigma_t",
    "hash_combine",
    "clopper_pearson_ci",
    "phi_to_delta_mapping",
    "invert_phi_delta_mapping",
]

_PHI_DELTA_CACHE = {}

def hash_combine(master_seed: int, mc_index: int, grid_hash: int) -> int:
    import hashlib
    m = hashlib.sha256()
    m.update(str(master_seed).encode())
    m.update(str(mc_index).encode())
    m.update(str(grid_hash).encode())
    return int(m.hexdigest()[:16], 16)

def clopper_pearson_ci(n_fires: int, N_mc: int, alpha: float = 0.05) -> tuple[float, float]:
    from scipy.stats import beta
    lo = beta.ppf(alpha / 2, n_fires + 1, N_mc - n_fires + 1) if n_fires > 0 else 0.0
    hi = beta.ppf(1 - alpha / 2, n_fires + 1, N_mc - n_fires + 1) if n_fires < N_mc else 1.0
    return (float(lo), float(hi))

def fit_garch_sigma_t(returns: np.ndarray) -> np.ndarray:
    try:
        from arch import arch_model
        # arch scales internally better if we multiply by 100
        am = arch_model(returns * 100.0, mean='Constant', vol='GARCH', p=1, q=1, dist='normal', rescale=True)
        res = am.fit(disp='off')
        
        # Check convergence and EWMA fallback trigger
        alpha_1 = res.params.get('alpha[1]', 0.0)
        beta_1 = res.params.get('beta[1]', 0.0)
        
        if res.convergence_flag != 0 or (alpha_1 + beta_1 >= 0.999):
            raise ValueError("GARCH convergence failed or near integrated")
            
        sigma_t = res.conditional_volatility / 100.0
        return np.asarray(sigma_t)
        
    except Exception:
        # EWMA fallback (D-21): lambda=0.94, gamma=0.06
        lam = 0.94
        gamma = 0.06
        sigma2 = np.zeros_like(returns)
        sigma2[0] = np.var(returns)
        for t in range(1, len(returns)):
            sigma2[t] = gamma * (returns[t-1]**2) + lam * sigma2[t-1]
        return np.sqrt(sigma2)

def inject_ar1_autocorrelation(
    returns: np.ndarray, 
    phi: float, 
    seed: int, 
    sigma_t: np.ndarray | None = None,
    *,
    allow_negative_phi: bool = False,
) -> np.ndarray:
    if allow_negative_phi:
        if not (-1.0 < phi < 1.0):
            raise ValueError(f"phi must be in (-1, 1), got {phi}")
    elif not (0.0 <= phi < 1.0):
        raise ValueError(f"phi must be in [0, 1), got {phi}")
    
    if sigma_t is None:
        sigma_t = fit_garch_sigma_t(returns)
        
    rng = np.random.default_rng(seed)
    eta = rng.normal(0, 1, len(returns))
    
    eps = np.zeros_like(returns)
    eps[0] = eta[0]
    
    scale = np.sqrt(1.0 - phi**2)
    for t in range(1, len(returns)):
        eps[t] = phi * eps[t-1] + scale * eta[t]
        
    return returns + sigma_t * eps

def phi_to_delta_mapping(
    rng_seed: int = 43, 
    N: int = 500_000, 
    q_grid: tuple[int, ...] = (2, 5, 15, 60), 
    W_primary: int = 120, 
    phis_to_test: np.ndarray | None = None
) -> dict:
    global _PHI_DELTA_CACHE
    cache_key = (rng_seed, N, q_grid, W_primary, tuple(phis_to_test) if phis_to_test is not None else None)
    
    if cache_key in _PHI_DELTA_CACHE:
        return _PHI_DELTA_CACHE[cache_key]
        
    import scripts.wp1.vr_significance as vr_significance
    from strategies.vol_regime_switch.regime_population import non_overlapping_samples
    
    if phis_to_test is None:
        phis_to_test = np.linspace(0, 0.999, 100)
        
    rng = np.random.default_rng(rng_seed)
    sigma_btc = 0.001
    base_returns = rng.normal(0, sigma_btc, N)
    
    # We use constant sigma_t for the calibration
    sigma_t = np.full(N, sigma_btc)
    
    mapping = {}
    for phi in phis_to_test:
        phi_float = float(phi)
        mapping[phi_float] = {}
        
        # Inject AR(1) autocorrelation
        r_inj = inject_ar1_autocorrelation(base_returns, phi_float, seed=rng_seed, sigma_t=sigma_t)
        
        # Reconstruct price
        close = np.exp(np.cumsum(r_inj))
        
        # We don't have regimes for synthetic i.i.d. data, pass all 0s
        regime = np.zeros(N, dtype=int)
        
        for q in q_grid:
            vr_arr, _ = vr_significance.compute_rolling_vr_and_z(close, W=W_primary, q=q)
            pred_nl, _, _ = non_overlapping_samples(vr_arr, regime, stride=W_primary)
            
            delta = float(np.median(pred_nl))
            mapping[phi_float][q] = delta
            
    _PHI_DELTA_CACHE[cache_key] = mapping
    return mapping

def invert_phi_delta_mapping(delta_target: float, q: int, mapping: dict | None = None) -> float:
    if mapping is None:
        mapping = phi_to_delta_mapping()
        
    from scipy.interpolate import PchipInterpolator
    
    phis = sorted(mapping.keys())
    deltas = [mapping[p][q] for p in phis]
    
    # Ensure strict monotonicity for PchipInterpolator inversion
    # delta is a function of phi. We want phi as a function of delta.
    # Since we want delta -> phi, delta must be strictly increasing.
    
    clean_deltas = []
    clean_phis = []
    last_d = -np.inf
    for p, d in zip(phis, deltas):
        if p == 0.0:
            continue  # phi=0 inherits VR sampling noise; not a valid anchor
        if d > last_d:
            clean_deltas.append(d)
            clean_phis.append(p)
            last_d = d
            
    if len(clean_deltas) < 2:
        return float(clean_phis[0]) if clean_phis else 0.0

    d_min, d_max = clean_deltas[0], clean_deltas[-1]
    p_min, p_max = clean_phis[0], clean_phis[-1]

    if delta_target <= 0.0:
        return 0.0
    if delta_target < d_min:
        # Below the i.i.d. VR sampling floor: scale linearly from the origin.
        return float(p_min * (delta_target / d_min))
    if delta_target > d_max:
        return float(min(p_max * (delta_target / d_max), 0.99))

    interpolator = PchipInterpolator(clean_deltas, clean_phis, extrapolate=False)
    return float(interpolator(delta_target))
