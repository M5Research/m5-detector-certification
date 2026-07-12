"""Signal injection orchestrator: runs full injection grid with multiprocessing.
Supports --precompute-only (save data+GARCH to .npz), --from-precomputed (load .npz
instead of parquet), --start/--end (process grid subset), and --phi-only (φ lookup
only — no BTC data, no MC cascade).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from functools import partial
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_REPO_SRC = _REPO_ROOT / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from backtest.utils import PROJECT_ROOT
import scripts._bootstrap
from scripts.wp1.gate_analysis import WQ_GRID
from scripts.wp1.precheck_b import run_precheck_b_cascade, _gate_guard
from scripts.wp1.py_engine import load_and_clean
from scripts.wp1.signal_injector import (
    inject_ar1_autocorrelation,
    invert_phi_delta_mapping,
    phi_to_delta_mapping,
    hash_combine,
    fit_garch_sigma_t,
    clopper_pearson_ci,
)

DELTA_GRID = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10]
# Post-freeze positive-control extension (above i.i.d. VR floor ~0.14 at q=5).
# Run separately: python scripts/wp1/signal_injection.py --addendum
DELTA_GRID_ADDENDUM = [0.15, 0.20, 0.30, 0.50]
ADDENDUM_DELTA_SET = frozenset(DELTA_GRID_ADDENDUM)
Q_GRID = (2, 5, 15, 60)
N_MC = 200
BOOT_SEED = 43
INJECTION_DIR = PROJECT_ROOT / "data" / "injection_runs"
PREREG_PATH_09 = ".planning/phases/11-signal-injection-the-ligo-calibration/09-PREREGISTRATION.md"
PRECOMPUTED_NPZ = INJECTION_DIR / "precomputed.npz"

_PHI_LOOKUP: dict[tuple[float, int], float] | None = None


def load_phi_lookup(injection_dir: Path | None = None) -> dict[tuple[float, int], float]:
    """Load precomputed phi values from phi_grid*.json (avoids 15-30 min recalibration)."""
    global _PHI_LOOKUP
    if _PHI_LOOKUP is not None:
        return _PHI_LOOKUP
    injection_dir = injection_dir or INJECTION_DIR
    lookup: dict[tuple[float, int], float] = {}
    for fname in ("phi_grid.json", "phi_grid_addendum.json"):
        path = injection_dir / fname
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for item in data.get("unique_delta_q", []):
            lookup[(float(item["delta"]), int(item["q"]))] = float(item["phi"])
        for gp in data.get("grid_points", []):
            key = (float(gp["delta"]), int(gp["q"]))
            lookup.setdefault(key, float(gp["phi"]))
    _PHI_LOOKUP = lookup
    return lookup


def resolve_phi(delta: float, q: int, injection_dir: Path | None = None) -> float:
    lookup = load_phi_lookup(injection_dir)
    key = (float(delta), int(q))
    if key in lookup:
        return lookup[key]
    return float(invert_phi_delta_mapping(delta, q))


# Documented v4.0 instrument-characterization freeze commit (matches the
# addendum pre-registration and the §3 gauge provenance stamp).  The runtime
# gate-guard additionally resolves and stamps the actual prereg + code commits.
FREEZE_COMMIT = "1dc5c82"


def resolve_provenance_commits(prereg_path: str = PREREG_PATH_09) -> dict[str, str]:
    """Enforce the D-09 gate-guard and resolve provenance commit hashes.

    Fails closed (``SystemExit``) if the pre-registration is missing or does not
    predate this run.  Returns freeze/prereg/code commit hashes stamped into every
    injection-run JSON so reviewers can verify commit-predates-run integrity.
    """
    prereg_results = _gate_guard([prereg_path])
    prereg_commit = prereg_results[prereg_path]
    try:
        code_commit = subprocess.run(
            ["git", "log", "--format=%H", "-1"],
            capture_output=True,
            text=True,
            cwd=str(_REPO_ROOT),
        ).stdout.strip()
    except Exception:
        code_commit = ""
    return {
        "freeze_commit": FREEZE_COMMIT,
        "prereg_commit": prereg_commit,
        "code_commit": code_commit,
    }


def build_grid_points(*, include_addendum: bool = False):
    """Return ordered grid-point dicts (96 preregistered; +48 with --addendum)."""
    deltas = list(DELTA_GRID)
    if include_addendum:
        deltas.extend(DELTA_GRID_ADDENDUM)
    W_GRID = [W for W, q in WQ_GRID]
    grid_points = []
    for delta in deltas:
        for q in Q_GRID:
            for W in W_GRID:
                gp = {
                    "delta": delta,
                    "q": q,
                    "W": W,
                    "grid_hash": hash_combine(43, int(W), int(q) * 1000 + int(delta * 100000)),
                }
                grid_points.append(gp)
    return grid_points


def process_grid_point(
    gp: dict,
    r_real: np.ndarray,
    sigma_t: np.ndarray,
    timestamps: np.ndarray,
    N_mc: int,
    injection_dir: Path,
    provenance_commits: dict[str, str] | None = None,
) -> dict:
    delta = gp["delta"]
    q = gp["q"]
    W = gp["W"]
    grid_hash = gp["grid_hash"]

    out_path = injection_dir / f"inj_d{delta}_q{q}_W{W}.json"
    tmp_path = out_path.with_suffix(".json.tmp")

    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text())
            prov = existing.get("provenance", {})
            needs_restamp = prov.get("freeze_commit") == "bypassed_for_execution"
            if existing.get("N_mc") == N_mc and existing.get("injected") and not needs_restamp:
                return {"skipped": True, "path": str(out_path)}
        except (json.JSONDecodeError, KeyError):
            pass

    phi = resolve_phi(delta, q, injection_dir)

    per_draw = []
    n_fires = 0
    cell_label = f"delta={delta} q={q} W={W}"
    for mc_idx in range(N_mc):
        if mc_idx % 10 == 0:
            print(f"  [{cell_label}] MC draw {mc_idx+1}/{N_mc} ...", flush=True)
        mc_seed = hash_combine(BOOT_SEED, mc_idx, grid_hash)

        try:
            r_inj = inject_ar1_autocorrelation(r_real, phi, mc_seed, sigma_t)
            close_inj = np.exp(np.cumsum(r_inj))

            result = run_precheck_b_cascade(
                close_inj, timestamps, mc_seed, diagnostics=False
            )
            fired = result["cascade_fired"]
            n_fires += int(fired)

            per_draw.append({
                "mc_idx": int(mc_idx),
                "seed": mc_seed,
                "cascade_fired": bool(fired),
                "holm_ordered": result["holm_correction"]["ordered"],
            })
        except Exception as e:
            # D-15: Failed draws count as non-detections
            per_draw.append({
                "mc_idx": int(mc_idx),
                "seed": mc_seed,
                "cascade_fired": False,
                "error": str(e),
            })

    P_det = n_fires / N_mc
    ci_lo, ci_hi = clopper_pearson_ci(n_fires, N_mc)

    output = {
        "injected": True,
        "phi": float(phi),
        "delta_target": delta,
        "q": q,
        "W": W,
        "grid_hash": grid_hash,
        "N_mc": N_mc,
        "n_fires": n_fires,
        "P_det": P_det,
        "ci_95_lo": ci_lo,
        "ci_95_hi": ci_hi,
        "per_draw": per_draw,
        "provenance": {
            "freeze_commit": (provenance_commits or {}).get("freeze_commit", FREEZE_COMMIT),
            "prereg_commit": (provenance_commits or {}).get("prereg_commit", ""),
            "code_commit": (provenance_commits or {}).get("code_commit", ""),
            "boot_seed": BOOT_SEED,
            "exploratory_addendum": delta in ADDENDUM_DELTA_SET,
            "library_versions": {"numpy": np.__version__},
        },
    }

    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    tmp_path.replace(out_path)

    return {"completed": True, "path": str(out_path), "P_det": P_det}


# ---------------------------------------------------------------------------
# Precompute: load data + fit GARCH once, save to .npz
# ---------------------------------------------------------------------------

def precompute_and_save(npz_path: Path) -> Path:
    """Load BTCUSDT 2021-2025 parquet, fit GARCH, save r_real/sigma_t/timestamps to .npz.

    This is the memory-heavy step — do it ONCE.  The resulting .npz is ~50-100 MB
    and loads in seconds with negligible memory.
    """
    import datetime as _dt

    print("=" * 60)
    print("PRECOMPUTE: Loading 2021-2025 BTCUSDT parquet + fitting GARCH")
    print("This is the heavy step.  It may take 10-30 minutes on a slow PC.")
    print("=" * 60)

    start_ms = int(_dt.datetime(2021, 1, 1, tzinfo=_dt.UTC).timestamp() * 1000)
    end_ms = int(_dt.datetime(2025, 12, 31, 23, 59, 59, tzinfo=_dt.UTC).timestamp() * 1000)
    data_path = str(PROJECT_ROOT / "data" / "binance_futures")
    symbol = "BTCUSDT"

    t0 = time.time()
    print(f"Loading parquet for {symbol} 2021-2025 ...")
    df = load_and_clean(
        data_path=data_path,
        symbol=symbol,
        start_ms=start_ms,
        end_ms=end_ms,
        max_gap_allowed_mins=60,
    )
    close = df["close"]
    log_close = np.log(close)
    r_real = np.empty_like(close)
    r_real[0] = 0.0
    r_real[1:] = np.diff(log_close)
    timestamps = df["timestamp"]
    n_bars = len(close)
    print(f"  Loaded {n_bars:,} bars in {time.time() - t0:.1f}s")

    t0 = time.time()
    print("Fitting GARCH(1,1) on returns ...")
    sigma_t = fit_garch_sigma_t(r_real)
    print(f"  GARCH fit done in {time.time() - t0:.1f}s")

    npz_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    np.savez_compressed(
        str(npz_path),
        r_real=r_real,
        sigma_t=sigma_t,
        timestamps=timestamps,
        n_bars=n_bars,
    )
    file_mb = npz_path.stat().st_size / (1024 * 1024)
    print(f"  Saved {npz_path} ({file_mb:.1f} MB) in {time.time() - t0:.1f}s")
    print("PRECOMPUTE DONE.  You can now run injection grid in small batches.")
    return npz_path


# ---------------------------------------------------------------------------
# Worker data (loaded from .npz or from raw parquet)
# ---------------------------------------------------------------------------

_WORKER_DATA = {}


def worker_init_from_npz(npz_path: str):
    """Load precomputed arrays from .npz (fast, low memory)."""
    import os as _os
    import warnings as _w

    _w.filterwarnings("ignore")
    pid = _os.getpid()
    print(f"Worker {pid} loading precomputed data from {npz_path} ...")
    data = np.load(npz_path)
    _WORKER_DATA["r_real"] = data["r_real"]
    _WORKER_DATA["sigma_t"] = data["sigma_t"]
    _WORKER_DATA["timestamps"] = data["timestamps"]
    print(f"Worker {pid} ready ({len(_WORKER_DATA['r_real']):,} bars loaded).")


def worker_init():
    """Original full-loader: parquet -> clean -> GARCH (heavy, used without --from-precomputed)."""
    import os as _os
    import warnings as _w
    import datetime as _dt
    from scripts.wp1.py_engine import load_and_clean
    from scripts.wp1.signal_injector import fit_garch_sigma_t

    _w.filterwarnings("ignore")
    pid = _os.getpid()
    print(f"Worker {pid} initializing data (parquet + GARCH) ...")

    start_ms = int(_dt.datetime(2021, 1, 1, tzinfo=_dt.UTC).timestamp() * 1000)
    end_ms = int(_dt.datetime(2025, 12, 31, 23, 59, 59, tzinfo=_dt.UTC).timestamp() * 1000)
    data_path = str(PROJECT_ROOT / "data" / "binance_futures")
    symbol = "BTCUSDT"

    df = load_and_clean(
        data_path=data_path,
        symbol=symbol,
        start_ms=start_ms,
        end_ms=end_ms,
        max_gap_allowed_mins=60,
    )
    close = df["close"]
    log_close = np.log(close)
    r_real = np.empty_like(close)
    r_real[0] = 0.0
    r_real[1:] = np.diff(log_close)
    timestamps = df["timestamp"]

    sigma_t = fit_garch_sigma_t(r_real)

    _WORKER_DATA["r_real"] = r_real
    _WORKER_DATA["sigma_t"] = sigma_t
    _WORKER_DATA["timestamps"] = timestamps
    print(f"Worker {pid} ready.")


def process_grid_point_worker(
    gp: dict, N_mc: int, injection_dir: Path, provenance_commits: dict[str, str] | None = None
) -> dict:
    r_real = _WORKER_DATA["r_real"]
    sigma_t = _WORKER_DATA["sigma_t"]
    timestamps = _WORKER_DATA["timestamps"]
    return process_grid_point(
        gp, r_real, sigma_t, timestamps, N_mc, injection_dir, provenance_commits
    )


def run_phi_only(grid_subset: list[dict], injection_dir: Path, *, include_addendum: bool) -> int:
    """Calibrate φ→δ mapping and write φ for each grid point (no injection / cascade)."""
    print("=" * 60)
    print("PHI-ONLY: calibrating mapping on synthetic i.i.d. returns")
    print("No BTC parquet, GARCH, or MC draws. Mapping takes ~15-30 min once.")
    print("=" * 60)

    t0 = time.time()
    mapping = phi_to_delta_mapping()
    elapsed = time.time() - t0
    print(f"Mapping ready in {elapsed:.1f}s")

    phi_by_delta_q: dict[tuple[float, int], float] = {}
    rows = []
    n_zero = 0
    for gp in grid_subset:
        key = (gp["delta"], gp["q"])
        if key not in phi_by_delta_q:
            phi_by_delta_q[key] = float(invert_phi_delta_mapping(gp["delta"], gp["q"], mapping))
        phi = phi_by_delta_q[key]
        if phi == 0.0:
            n_zero += 1
        rows.append({
            "delta": gp["delta"],
            "q": gp["q"],
            "W": gp["W"],
            "phi": phi,
            "grid_hash": gp["grid_hash"],
        })

    injection_dir.mkdir(parents=True, exist_ok=True)
    out_path = injection_dir / ("phi_grid_addendum.json" if include_addendum else "phi_grid.json")
    payload = {
        "phi_only": True,
        "include_addendum": include_addendum,
        "n_grid_points": len(rows),
        "n_unique_delta_q": len(phi_by_delta_q),
        "n_phi_zero": n_zero,
        "mapping_seconds": elapsed,
        "grid_points": rows,
        "unique_delta_q": [
            {"delta": d, "q": q, "phi": phi_by_delta_q[(d, q)]}
            for (d, q) in sorted(phi_by_delta_q.keys())
        ],
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out_path} ({len(rows)} grid points, {n_zero} with phi=0)")

    print("\nUnique (delta, q) -> phi:")
    for (d, q), phi in sorted(phi_by_delta_q.items()):
        print(f"  delta={d:g} q={q} -> phi={phi:.6f}")

    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Signal injection grid orchestrator (low-memory batch mode supported)"
    )
    parser.add_argument(
        "--precompute-only",
        action="store_true",
        help="Load data + fit GARCH, save %s, then exit." % PRECOMPUTED_NPZ.name,
    )
    parser.add_argument(
        "--from-precomputed",
        type=str,
        default=None,
        metavar="NPZ",
        help="Load r_real/sigma_t/timestamps from .npz instead of parquet.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="First grid point index to process (0-based, inclusive).",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=-1,
        help="Last grid point index to process (0-based, exclusive). -1 = all.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of multiprocessing workers (default: INJECTION_WORKERS env or 1).",
    )
    parser.add_argument(
        "--addendum",
        action="store_true",
        help="Append post-freeze DELTA_GRID_ADDENDUM (0.15, 0.20, 0.30, 0.50).",
    )
    parser.add_argument(
        "--phi-only",
        action="store_true",
        help="Only calibrate φ→δ and write phi_grid.json (no BTC data / MC cascade).",
    )
    args = parser.parse_args()

    # ---- precompute-only --------------------------------------------------
    if args.precompute_only:
        precompute_and_save(PRECOMPUTED_NPZ)
        return 0

    # ---- grid -------------------------------------------------------------
    grid_points = build_grid_points(include_addendum=args.addendum)
    total = len(grid_points)
    start_idx = max(0, args.start)
    end_idx = total if args.end < 0 else min(total, args.end)
    grid_subset = grid_points[start_idx:end_idx]

    print(f"Grid points: {start_idx}-{end_idx-1} ({len(grid_subset)}/{total})")
    for i, gp in enumerate(grid_subset, start_idx):
        already = (INJECTION_DIR / f"inj_d{gp['delta']}_q{gp['q']}_W{gp['W']}.json").exists()
        tag = " (exists)" if already else ""
        print(f"  [{i}] delta={gp['delta']} q={gp['q']} W={gp['W']}{tag}")

    if not grid_subset:
        print("No grid points to process in this range.")
        return 0

    if args.phi_only:
        return run_phi_only(grid_subset, INJECTION_DIR, include_addendum=args.addendum)

    # ---- D-09 gate-guard (enforce pre-registration predates run) ----------
    # Runs BEFORE any data load; fails closed if the pre-reg is missing or does
    # not predate this run.  Resolved commits are stamped into every JSON.
    print("D-09 gate-guard: verifying pre-registration integrity ...", flush=True)
    provenance_commits = resolve_provenance_commits()
    print(
        "D-09 gate-guard PASSED. "
        f"freeze_commit={provenance_commits['freeze_commit']}, "
        f"prereg_commit={provenance_commits['prereg_commit'][:12]}..., "
        f"code_commit={provenance_commits['code_commit'][:12]}...",
        flush=True,
    )

    # ---- data source ------------------------------------------------------
    npz_path = args.from_precomputed or str(PRECOMPUTED_NPZ)
    use_npz = args.from_precomputed is not None or PRECOMPUTED_NPZ.exists()

    n_workers = args.workers if args.workers is not None else int(os.environ.get("INJECTION_WORKERS", "1"))
    if n_workers > len(grid_subset):
        n_workers = len(grid_subset)

    worker = partial(
        process_grid_point_worker,
        N_mc=N_MC,
        injection_dir=INJECTION_DIR,
        provenance_commits=provenance_commits,
    )

    if n_workers <= 1:
        # Sequential — load once, loop
        if use_npz:
            print(f"Loading precomputed data from {npz_path} ...")
            data = np.load(npz_path)
            _WORKER_DATA["r_real"] = data["r_real"]
            _WORKER_DATA["sigma_t"] = data["sigma_t"]
            _WORKER_DATA["timestamps"] = data["timestamps"]
            print(f"Loaded {len(_WORKER_DATA['r_real']):,} bars.")
        else:
            print("Loading data from parquet (sequential mode)...")
            worker_init()

        results = []
        for i, gp in enumerate(grid_subset):
            idx = start_idx + i
            print(f"Grid point {idx+1}/{total} (index {idx}): delta={gp['delta']} q={gp['q']} W={gp['W']}")
            results.append(worker(gp))
    else:
        # Multiprocessing pool
        print(f"Starting pool with {n_workers}/{len(grid_subset)} workers ...")
        init_fn = worker_init
        if use_npz:
            init_fn = partial(worker_init_from_npz, npz_path)
        with Pool(processes=n_workers, initializer=init_fn) as pool:
            results = pool.map(worker, grid_subset)

    completed = [r for r in results if "completed" in r]
    skipped = [r for r in results if "skipped" in r]
    print(f"Batch complete: {len(completed)} completed, {len(skipped)} skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
