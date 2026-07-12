# Public Replication Runbook

Reproduction runbook for **"Certifying Regime Detectors Before Use"** (M5 Research).
Run commands from the `m5-detector-certification/` root unless stated otherwise.

## 0. What reproduces, and from what

- **The paper's values, tables, and figures reproduce fully offline** from the frozen
  artifacts under `data/injection_runs/` and `backtest_results/`. No market data required.
- **Regenerating those artifacts from scratch** requires the raw Binance/HistData inputs
  (not redistributed — see §5) and the download scripts.

The package is self-contained: every script resolves paths to this folder
(`PROJECT_ROOT = m5-detector-certification/`), so a standalone clone reads its own artifacts.

## 1. Environment

Preferred, if `uv` is available:

```powershell
uv sync --frozen --extra dev
```

Fallback with standard Python (>=3.11):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

No install of the package itself is needed: `conftest.py` and the `pyproject.toml`
`pythonpath` put `.` and `src/` on the path, so a plain checkout runs.

## 2. Build the manuscript

```powershell
Set-Location paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
Select-String -Path main.log -Pattern 'undefined|Overfull|LaTeX Error|Emergency stop|Fatal error|Warning: Citation'
Set-Location ..
```

Expected: `paper/main.pdf` (24 pages); no undefined references/citations, no LaTeX
errors, no overfull boxes.

## 3. Run the verification suite

```powershell
python -m pytest -q          # full suite; slow/data-marked tests auto-skip
```

Expected: all selected tests pass; tests marked `data` skip without local raw data.
The suite validates every frozen artifact's shape, provenance, and reported values,
plus the analysis-helper behavior. The documented per-gate tests are:

```powershell
python -m pytest `
  tests/wp1/test_phase15_artifact_gate.py `
  tests/wp1/test_empirical_vr_null.py `
  tests/wp1/test_gauge_invariance.py `
  tests/wp1/test_thermodynamic_bound.py `
  tests/wp1/test_vr_detector_mi.py `
  tests/wp1/test_holdout_confirmatory.py `
  tests/wp1/test_harmonized_benchmark.py `
  tests/wp1/test_eth_replication.py `
  tests/wp1/test_eurusd_cost_gate_summary.py `
  tests/wp1/test_histdata_eurusd.py
```

(`tests/wp1/test_online_gcde.py` covers a legacy sequential-replay driver retained for
provenance; the current manuscript states online monitoring via the anytime-valid
e-value proposition, §3.4, not the scripted replay.)

## 4. Verify included evidence paths

```powershell
Test-Path data/injection_runs/precomputed.npz
Test-Path backtest_results/empirical_vr_null/empirical_vr_null_20260623_230753.json
Test-Path backtest_results/gauge_invariance/gauge_report_20260624_221834.json
Test-Path backtest_results/thermodynamic_bound/vr_detector_mi.json
Test-Path backtest_results/reference_repair/recentered_reference_repair_20260710.json
Test-Path paper/CLAIM_TRACEABILITY.md
```

All should return `True`.

## 5. Regenerate comparable artifacts (requires raw data)

These commands require separately obtained raw data in the expected local layout.
Write to new output paths first; do not overwrite frozen artifacts unless the
manuscript and `paper/CLAIM_TRACEABILITY.md` are updated together.

```powershell
python scripts/wp1/empirical_vr_null.py --n-boot 100 --nulls circular-block --cells 120:2,120:5 --out backtest_results/empirical_vr_null/empirical_vr_null_rebuild.json
python scripts/wp1/gauge_invariance.py --run
python scripts/wp1/thermodynamic_bound.py --run
python scripts/wp1/vr_detector_mi.py --symbol BTCUSDT --start 2021-01-01 --end 2025-12-31 --W 120 --q-values 2 5 --n-boot 500 --n-perm 1000 --out backtest_results/thermodynamic_bound/vr_detector_mi_rebuild.json
python scripts/wp1/holdout_confirmatory.py --confirm-jef-draft
python scripts/wp1/eth_replication.py --symbol ETHUSDT
python scripts/wp1/harmonized_benchmark.py --out backtest_results/harmonized_benchmark/harmonized_benchmark_rebuild.json --symbol BTCUSDT --start 2022-10-01 --end 2022-12-31 --detectors rolling_quantile hmm vr_cascade --W 120 --q 5 --hmm-regimes 2
```

The BTC/ETH runs expect local Binance USD-M parquet under
`data/binance_futures/symbol=<SYMBOL>/year=YYYY/part-0.parquet`. The EUR/USD cost-gate
sample uses HistData Generic ASCII bid/ask ticks fetched with
`scripts/wp1/download_histdata_eurusd.py` (see `paper/REPRODUCTION.md`).

The **repair** result (§4.8) needs no raw data: it is exact offline re-thresholding of
the frozen per-draw `median_z_m2` recorded in the injection cells; the recentering
constants are post-freeze/exploratory and a preregistered confirmatory rerun is required
before any deployment certificate.

## 6. Claim boundary

The package supports audit of the paper's bounded method claims. It does not claim
Bitcoin inefficiency, trading profitability, detector optimality, or a completed
production certificate across every detector, asset, clock, and cost schedule. The one
`admissible` disposition (§4.8) is an exploratory, post-freeze result for a bounded
q=2 scientific claim, not a confirmatory or economic certificate.
