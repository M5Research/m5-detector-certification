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

The worked records are a Holm-corrected Lo–MacKinlay variance-ratio cascade on
BTCUSDT one-minute perpetual futures and a rolling-quantile detector on EUR/USD
bid/ask ticks. They are example instruments; the executable certification framework,
immutable lineage, and refusal precedence are the contribution.

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
| §4.4 Transport | historical v4 diagnostic: two of three q=2 pairs and no q=5 pairs certify | `backtest_results/gauge_invariance/gauge_report_mde_margin.json` | `scripts/wp1/gauge_invariance.py` |
| §4.5 Information & net value | protocol-v5 stationary-block inference; BTC q2/q5 fail | `evidence/gates/btcusdt-vr-q2-v5-r0/evidence.json` | `scripts/wp1/btc_successor.py` |
| §4.6 2026 window | temporally-disjoint holdout disposition | `backtest_results/holdout/holdout_confirmatory_20260623_175305.json` | `scripts/wp1/holdout_confirmatory.py` |
| §4.8 Repair | recentering lowers cMDE 0.15→0.02 (q=2), 0.30→0.10 (q=5) | `backtest_results/reference_repair/recentered_reference_repair_20260710.json` | exact offline re-thresholding of frozen per-draw `median_z_m2` (see note) |
| §5 Three families | rolling-quantile / HMM / VR-cascade dispositions | `backtest_results/harmonized_benchmark/harmonized_benchmark_q4_2022_btc_minimal_hmm.json` | `scripts/wp1/harmonized_benchmark.py` |
| App. B ETH | ETHUSDT external replication | `backtest_results/asset_replication/eth_replication_20260625_224506.json` | `scripts/wp1/eth_replication.py` |
| §6 EUR/USD | complete seven-gate rolling-quantile certificate; `size_distorted` | `certificates/eurusd-rq-v5-r0.json` | `scripts/wp1/eurusd_certificate.py` |
| Protocol-v5 BTC | complete successor certificate; `target_mismatched` | `certificates/btcusdt-vr-q2-v5-r0.json` | `scripts/wp1/btc_successor.py` |

**E-value certificate (§3.4).** Certificate-grade information inference uses a
stationary-block null with `B=49,999`, not IID permutation. At revision 0 the BTC
successor returns `E=0.801` at q=2 and `E=0.620` at q=5 (both fail the threshold 40),
while EUR/USD returns `E=111.803` (pass). The older `vr_detector_mi.json` is retained
only as `legacy_noncertificate` provenance.

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
├── specs/, certificates/, evidence/    protocol-v5 claim tuples, records, and gate evidence
├── provenance/                         upstream snapshot, acquisition, and certificate lineage
├── backtest_results/                   historical frozen diagnostics and legacy artifacts
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

The compact acquisition manifests, frozen JSON evidence, certificates, and injection
`.npz` precompute (derived statistics, not raw quotes) **are** included, so the paper's
reported results and provenance checks reproduce with no raw market data.

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
