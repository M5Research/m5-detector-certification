# Certifying Regime Detectors Before Use — Reproduction Package

Reproduction code, frozen artifacts, and the manuscript for the paper
**"Certifying Regime Detectors Before Use"** (M5 Research). The manuscript source
and compiled PDF are under [`paper/`](paper/).

## What the paper is

Value-at-risk forecasts are backtested before they are trusted; regime detectors —
whose labels and triggers enter inference and allocation just as directly — face no
analogous pre-use validation. The paper proposes a **certification protocol** that
treats a frozen detector as a measurement instrument and, for a single pre-declared
claim, measures five operating characteristics — power, empirical size, cross-clock
transport, detector-output information, and net information value — then returns an
auditable certificate with a disposition and a gate-level failure profile.

The worked instrument is a Holm-corrected Lo–MacKinlay variance-ratio cascade on
BTCUSDT one-minute perpetual futures. It is an example instrument, not the contribution.

## Two levels of reproduction

**1. Paper results from frozen artifacts — fully offline, no market data.**
Every number, table, and figure in the paper traces to a version-pinned JSON artifact
under `backtest_results/` or `data/injection_runs/`. The analysis and figure scripts
recompute the paper's values from those frozen artifacts, so no market data is required.

```bash
pip install -r requirements.txt
pytest -m artifact           # validate every frozen artifact's shape and values
```

**2. Regenerating the frozen artifacts from raw data — requires vendor data.**
The artifacts were produced from Binance USD-M futures bars and HistData EUR/USD ticks,
which are subject to vendor redistribution terms and are **not** included here. Exact
regeneration commands (with the download scripts and pre-declared parameters) are in
[`paper/REPRODUCTION.md`](paper/REPRODUCTION.md).

## Claim → artifact → script

| Paper section | Result | Frozen artifact | Script |
|---|---|---|---|
| §4.2 Power | 96-cell exclusion grid (δ≤0.10 all silent); disclosed exploratory positive controls; cMDE bracket | `data/injection_runs/inj_d*_q*_W*.json` | `scripts/wp1/signal_injection.py`, `scripts/wp1/gate_analysis.py` |
| §4.3 Size | empirical circular-block null; zero false alarms; mis-centered asymptotic reference | `backtest_results/empirical_vr_null/empirical_vr_null_20260623_230753.json` | `scripts/wp1/empirical_vr_null.py` |
| §4.4 Transport | equivalence certified at q=2, fails at q=5 (all three clock pairs) | `backtest_results/gauge_invariance/gauge_report_20260624_221834.json` | `scripts/wp1/gauge_invariance.py` |
| §4.5 Information & net value | detector-output MI `I_D`; per-epoch vs per-trigger net-value sign flip | `backtest_results/thermodynamic_bound/vr_detector_mi.json` (raw-scale diagnostic: `thermo_report_20260624_082448.json`) | `scripts/wp1/vr_detector_mi.py` |
| §4.6 2026 window | temporally-disjoint holdout disposition | `backtest_results/holdout/holdout_confirmatory_20260623_175305.json` | `scripts/wp1/holdout_confirmatory.py` |
| §4.8 Repair | recentering lowers cMDE 0.15→0.02 (q=2), 0.30→0.10 (q=5) | `backtest_results/reference_repair/recentered_reference_repair_20260710.json` | exact offline re-thresholding of frozen per-draw `median_z_m2` (see note) |
| §5 Three families | rolling-quantile / HMM / VR-cascade dispositions | `backtest_results/harmonized_benchmark/harmonized_benchmark_q4_2022_btc_minimal_hmm.json` | `scripts/wp1/harmonized_benchmark.py` |
| App. B ETH | ETHUSDT external replication | `backtest_results/asset_replication/eth_replication_20260625_224506.json` | `scripts/wp1/eth_replication.py` |
| App. C EUR/USD | cost-gate spread observability | `backtest_results/eurusd/eurusd_histdata_cost_gate_matched_20210529_20251231_summary.json` | `scripts/wp1/eurusd_cost_gate_summary.py` |

**E-value certificate (§3.4).** The gate e-values (E = 2.17 at q=2, E = 15.8 at q=5) are
the a-priori calibrator `f_κ(p) = κ·p^(κ−1)` at κ = ½ applied to the information-gate
permutation p-values (p = 0.053 at q=2, p = 0.001 at q=5) recorded in `vr_detector_mi.json`.

**Repair (§4.8).** No new simulation is needed: because each injection cell stores its
per-draw, per-horizon `median_z_m2`, the recentered decision rule is applied by exact
offline re-thresholding of the already-frozen draws.

## Layout

```
m5-detector-certification/
├── paper/                              manuscript (main.tex, main.pdf), figures, REPRODUCTION.md, CLAIM_TRACEABILITY.md
├── src/strategies/vol_regime_switch/   the three detectors (VR cascade, HMM, rolling quantile)
├── scripts/wp1/                        analysis, gate, and figure scripts
├── backtest/                           minimal path/time helpers (backtest.utils)
├── data/injection_runs/                frozen signal-injection grid (104 cells) + precompute
├── backtest_results/                   frozen gate artifacts (size, transport, MI, holdout, repair, benchmark, …)
├── tests/wp1/                          artifact-validation and unit tests
├── requirements.txt
└── pyproject.toml
```

No install step is required to run the tests or scripts: `conftest.py` and the
`pyproject.toml` `pythonpath` put `.` and `src/` on the path, so a plain checkout works.

## Data access

Raw market data are **not redistributed** (vendor terms):

- **Binance USD-M futures** (BTCUSDT / ETHUSDT, 1-minute): obtain locally; the loaders
  expect `data/binance_futures/symbol=<SYMBOL>/year=YYYY/part-0.parquet`.
- **HistData EUR/USD** ticks: fetch with `scripts/wp1/download_histdata_eurusd.py`
  (commands in `paper/REPRODUCTION.md`).

The frozen JSON artifacts and the injection `.npz` precompute (derived statistics, not
raw quotes) **are** included, so the paper's results reproduce with no raw data.

## Provenance and claim boundaries

- [`paper/CLAIM_TRACEABILITY.md`](paper/CLAIM_TRACEABILITY.md) maps every manuscript claim
  to its supporting artifact and a status (supported / partial / method-defined).
- [`paper/REPRODUCTION.md`](paper/REPRODUCTION.md) gives exact manuscript-build and
  artifact-regeneration commands.

Confirmatory versus exploratory provenance is distinguished per table cell. The paper
makes no claim of Bitcoin inefficiency, trading profitability, or detector optimality.

## License

- **Code:** MIT License — see [`LICENSE`](LICENSE).
- **Manuscript, figures, and documentation:** CC BY 4.0 — see [`paper/LICENSE`](paper/LICENSE), subject to journal policy.
- **Raw third-party market data:** excluded; redistribution depends on the exchange or
  vendor terms.
