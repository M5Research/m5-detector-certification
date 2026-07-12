# Reproduction Notes

This folder contains the submission-style GCDE method paper and the evidence map used to keep claims bounded.

## Environment

Run commands from the repository root unless a section says otherwise. The manuscript build requires a LaTeX distribution with `pdflatex` and `bibtex`. Evidence checks use the repository's Python environment and data files under `data/` and `backtest_results/`.

Raw Binance futures data are used only through local repository artifacts. Data redistribution remains subject to the original venue and vendor terms.

## Build Manuscript

Run from `docs/research/gauge-calibrated-detector-exclusion/`:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
Select-String -Path main.log -Pattern 'undefined|Overfull|LaTeX Error|Emergency stop|Fatal error|Warning: Citation'
```

Expected output:

- `main.pdf`
- no undefined citations or references
- no LaTeX errors, emergency stops, fatal errors, citation warnings, or overfull boxes
- underfull box warnings may remain and are not treated as submission blockers

## Evidence Artifacts

The manuscript uses repository-local artifacts only.

| Gate | Artifact |
|---|---|
| Sensitivity | `data/injection_runs/inj_d0.15_q2_W120.json`, `inj_d0.15_q5_W120.json`, `inj_d0.2_q5_W120.json`, `inj_d0.3_q5_W120.json` |
| Empirical-null size diagnostic | `backtest_results/empirical_vr_null/empirical_vr_null_20260623_230753.json` |
| Gauge defect | `backtest_results/gauge_invariance/gauge_report_20260624_221834.json` |
| Raw information scale diagnostic | `backtest_results/thermodynamic_bound/thermo_report_20260624_082448.json` |
| VR detector-contingent MI | `backtest_results/thermodynamic_bound/vr_detector_mi.json` |
| Holdout disposition example | `backtest_results/holdout/holdout_confirmatory_20260623_175305.json` |
| ETHUSDT external replication | `backtest_results/asset_replication/eth_replication_20260625_224506.json` |
| Benchmark schema validation | `backtest_results/harmonized_benchmark/harmonized_benchmark_smoke.json` |
| Native-frequency benchmark | `backtest_results/harmonized_benchmark/harmonized_benchmark_q4_2022_btc_minimal_hmm.json` |
| Online sequential replay | `backtest_results/online_gcde/online_gcde_replay.json` |
| EURUSD cost-gate sample | `backtest_results/eurusd/eurusd_histdata_cost_gate_matched_20210529_20251231_summary.json` |

## Verification Tests

Run the focused artifact/test suite from the repository root:

```powershell
python -m pytest tests/wp1/test_phase15_artifact_gate.py tests/wp1/test_empirical_vr_null.py tests/wp1/test_gauge_invariance.py tests/wp1/test_thermodynamic_bound.py tests/wp1/test_holdout_confirmatory.py tests/wp1/test_eth_replication.py tests/wp1/test_harmonized_benchmark.py tests/wp1/test_vr_detector_mi.py tests/wp1/test_histdata_eurusd.py tests/wp1/test_eurusd_cost_gate_summary.py tests/wp1/test_online_gcde.py
```

These tests check artifact shape, provenance expectations, and gate-specific helper behavior. They do not replace high-resolution GCDE deployment certification.

## Artifact Regeneration

The manuscript is written against the frozen artifacts listed above. To regenerate comparable artifacts, use the repository scripts with pre-declared parameters and write to a new output path rather than overwriting the frozen artifacts.

```powershell
python scripts/wp1/empirical_vr_null.py --n-boot 100 --nulls circular-block --cells 120:2,120:5 --out backtest_results/empirical_vr_null/empirical_vr_null_rebuild.json
python scripts/wp1/gauge_invariance.py --run
python scripts/wp1/thermodynamic_bound.py --run
python scripts/wp1/vr_detector_mi.py --symbol BTCUSDT --start 2021-01-01 --end 2025-12-31 --W 120 --q-values 2 5 --n-boot 500 --n-perm 1000 --out backtest_results/thermodynamic_bound/vr_detector_mi.json
python scripts/wp1/holdout_confirmatory.py --confirm-jef-draft
python scripts/wp1/eth_replication.py --symbol ETHUSDT
python scripts/wp1/harmonized_benchmark.py --smoke --out backtest_results/harmonized_benchmark/harmonized_benchmark_schema_rebuild.json
python scripts/wp1/harmonized_benchmark.py --out backtest_results/harmonized_benchmark/harmonized_benchmark_q4_2022_btc_minimal_hmm.json --symbol BTCUSDT --start 2022-10-01 --end 2022-12-31 --detectors rolling_quantile hmm vr_cascade --W 120 --q 5 --hmm-regimes 2 --hmm-em-iter 10 --hmm-search-reps 1
python scripts/wp1/online_gcde.py --npz data/injection_runs/precomputed.npz --mode both --start 2021-06-01 --end 2025-12-31 --window-days 60 --stride-days 7 --demo-cell 120:2 --real-cell 120:5 --demo-delta 0.15 --sequential-control maxT --n-boot 4999 --n-mc 500 --n-perm 2000 --out backtest_results/online_gcde/online_gcde_replay.json
```

For production GCDE certificates, increase the null bootstrap count to the pre-declared Monte Carlo precision target and extend the real benchmark to the full declared detector set. EURUSD robustness should use HistData Generic ASCII tick bid/ask files, not FRED or ECB daily reference rates, because daily reference series do not carry an executable spread for the GCDE cost gate.

The ETHUSDT replication expects local Binance USD-M parquet partitions under `data/binance_futures/symbol=ETHUSDT/year=YYYY/part-0.parquet`. The driver keeps 2026 data out of the in-sample layers and loads the matched partial 2026 block only inside the replication holdout section. The BTC injection grid is not reused as an ETH calibration surface.

To fetch and summarize the BTC/ETH-matched free EURUSD bid/ask span:

```powershell
5..12 | ForEach-Object { python scripts/wp1/download_histdata_eurusd.py --symbol EURUSD --year 2021 --months $_ --aggregate minute --out-dir data/external/histdata/EURUSD/minute_matched }
2022..2025 | ForEach-Object { $y = $_; 1..12 | ForEach-Object { python scripts/wp1/download_histdata_eurusd.py --symbol EURUSD --year $y --months $_ --aggregate minute --out-dir data/external/histdata/EURUSD/minute_matched } }
python scripts/wp1/eurusd_cost_gate_summary.py data/external/histdata/EURUSD/minute_matched --start 2021-05-29T19:32:00 --end 2025-12-31T23:59:00 --out backtest_results/eurusd/eurusd_histdata_cost_gate_matched_20210529_20251231_summary.json
```

The cost-gate appendix uses the matched-span summary, not the earlier short development artifact. FX session availability means the first and last available quotes inside the calendar filter need not equal the exact calendar endpoints.

For a quick online replay smoke check, use:

```powershell
python scripts/wp1/online_gcde.py --mode both --smoke --n-boot 19 --n-mc 20 --n-perm 50 --out backtest_results/online_gcde/online_gcde_replay_smoke.json
```

## Claim Boundary

The manuscript claims a new method: the GCDE admissibility operator for detector-output claims. It does not claim Bitcoin inefficiency, trading profitability, detector optimality, or a completed detector-specific economic admissibility certificate.

For exact claim status, use `CLAIM_TRACEABILITY.md` before editing the manuscript.
