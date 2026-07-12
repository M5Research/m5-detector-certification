# Artifact Ledger

This ledger records existing artifacts that can seed the GCDE paper. Paths are relative to the repository root.

## Verification Pass

Checked on 2026-06-26 against the repository working tree. All file-path artifacts cited in this ledger were found. Inline numeric values below were re-read from the cited JSON artifacts; rounded prose values match the source artifacts at the precision shown.

| Area | Path Status | Number Status | Notes |
|---|---|---|---|
| Sensitivity and injection-recovery | Verified | Verified | Four cited injection JSON files report the stated `N_mc`, `n_fires`, `P_det`, and exact 95% intervals. |
| Empirical null / size diagnostic | Verified | Verified with limitation | The manuscript now reports the focused `B=100` circular-block diagnostic; the broader multi-null smoke artifact remains a development harness and is not promoted as statistical validation. |
| Gauge defect | Verified | Verified | `q=5` TOST comparisons report the stated differences, 90% intervals, and Holm non-equivalence decisions. |
| Information and cost | Verified | Verified with limitation | Raw sign-pair bounds and VR trigger MI values match cited artifacts; the VR detector information is cost-exhausted under the 10 bps convention. |
| Holdout | Verified | Verified | Data span, bar count, primary-cell statistic, p-value, and `closed=false` match the holdout JSON. |
| ETHUSDT external replication | Verified | Verified with limitation | Same-venue ETHUSDT symbol-swap artifact matches the manuscript table; it is not pooled with BTCUSDT and has no ETH-specific injection grid. |
| Harmonized benchmark | Verified | Verified with limitation | Native-frequency BTC Q4 benchmark preserves 1-minute horizons; sparse HMM output is treated as execution diagnostic, not scientific evidence. |
| EURUSD cost-gate sample | Verified | Verified with limitation | BTC/ETH-matched HistData bid/ask sample proves spread observability for the cost gate; it is not a full second-asset detector certificate. |
| Online GCDE | Regenerable | Verified by tests | `scripts/wp1/online_gcde.py` and `tests/wp1/test_online_gcde.py` define max-statistic sequential calibration, state transitions, causal windows, and gauge-uncertified transport handling. |
| Prior manuscript sources | Verified | N/A | Prior TeX/Markdown sources exist and are framing/provenance sources, not new empirical evidence. |

## Sensitivity And Injection-Recovery

| Artifact | Status | Reuse In GCDE |
|---|---|---|
| `data/injection_runs/inj_d0.15_q2_W120.json` | Existing Monte Carlo injection cell | Demonstrates `P_det=1.0` at short horizon `q=2`. |
| `data/injection_runs/inj_d0.15_q5_W120.json` | Existing Monte Carlo injection cell | Demonstrates `P_det=0.0` at primary `q=5` first exploratory point. |
| `data/injection_runs/inj_d0.2_q5_W120.json` | Existing Monte Carlo injection cell | Extends the `q=5` silent region. |
| `data/injection_runs/inj_d0.3_q5_W120.json` | Existing Monte Carlo injection cell | Demonstrates `P_det=1.0` at higher `q=5` amplitude. |
| `scripts/wp1/signal_injection.py` | Existing injection orchestrator | Basis for a reusable detector response-surface estimator. |
| `scripts/wp1/exclusion_plot.py` | Existing plotting/table script | Should be reframed from exclusion plot to response-surface plot. |
| `tests/wp1/test_phase15_artifact_gate.py` | Existing artifact checks | Useful as provenance and schema guardrails. |

Current evidence anchors:

| Cell | Fires | `P_det` | 95% CI |
|---|---:|---:|---|
| `delta=0.15, q=2, W=120` | `200/200` | `1.0` | `[0.9818, 1.0]` |
| `delta=0.15, q=5, W=120` | `0/200` | `0.0` | `[0.0, 0.0182]` |
| `delta=0.20, q=5, W=120` | `0/200` | `0.0` | `[0.0, 0.0182]` |
| `delta=0.30, q=5, W=120` | `200/200` | `1.0` | `[0.9818, 1.0]` |

Interpretation for GCDE:

The detector has a finite-sample detectability frontier. The old "exclusion" narrative should become "horizon-specific detector response surface."

## Empirical Null And Size

| Artifact | Status | Reuse In GCDE |
|---|---|---|
| `backtest_results/empirical_vr_null/empirical_vr_null_20260623_230753.json` | Existing focused empirical-null artifact | Provides the manuscript's 100-draw circular-block diagnostic for `W=120`, `q=2` and `q=5`. |
| `backtest_results/empirical_vr_null/empirical_vr_null_fullshape_smoke_20260624.json` | Existing low-resolution multi-null harness | Exercises the six-cell, six-null artifact shape; retain for development traceability, not as manuscript statistical evidence. |
| `scripts/wp1/empirical_vr_null.py` | Existing null-reference runner | Basis for high-resolution size certification. |
| `tests/wp1/test_empirical_vr_null.py` | Existing tests | Preserve bootstrap-index and artifact-shape expectations. |

Focused manuscript diagnostic from `empirical_vr_null_20260623_230753.json`:

| Cell | `B` | Observed median `Z_m2` | Two-sided p | Holm p | Positive-gate p |
|---|---:|---:|---:|---:|---:|
| `W=120,q=2` | `100` | `-0.1819` | `0.0396` | `0.0792` | `0.9703` |
| `W=120,q=5` | `100` | `-0.4017` | `0.0495` | `0.0792` | `0.9604` |

Interpretation for GCDE:

Empirical size is part of the operator, not a hidden asymptotic assumption. The focused circular-block diagnostic is adequate to show the size gate implementation path in the scoped methods paper; a production admissibility certificate should rerun the pre-declared multi-null family at higher Monte Carlo resolution.

## Online Sequential Replay

| Artifact | Status | Reuse In GCDE |
|---|---|---|
| `scripts/wp1/online_gcde.py` | New online replay CLI and helper module | Implements max-statistic sequential calibration, dynamic admission states, causal replay windows, and deterministic JSON artifacts. |
| `tests/wp1/test_online_gcde.py` | New focused tests | Guards gate-level alpha budgeting, `max_j Z^*_{j,k}` threshold calibration, admission/revocation, no lookahead metadata, gauge absence handling, and schema determinism. |
| `backtest_results/online_gcde/online_gcde_replay.json` | Regenerated artifact | Two-panel replay: controlled dynamic admission path anchored to the precomputed `W=120,q=2,delta=0.15` injection cell and real BTC no-injection replay. |
| `backtest_results/online_gcde/online_gcde_replay.png` | Regenerated figure | State-trajectory figure for the manuscript's online GCDE section. |

Default command:

```powershell
python scripts/wp1/online_gcde.py `
  --npz data/injection_runs/precomputed.npz `
  --mode both `
  --start 2021-06-01 `
  --end 2025-12-31 `
  --window-days 60 `
  --stride-days 7 `
  --demo-cell 120:2 `
  --real-cell 120:5 `
  --demo-delta 0.15 `
  --sequential-control maxT `
  --n-boot 4999 `
  --n-mc 500 `
  --n-perm 2000 `
  --out backtest_results/online_gcde/online_gcde_replay.json
```

Interpretation for GCDE:

Online GCDE controls false detector admission over a declared monitoring horizon by calibrating one threshold per gate from the pathwise maximum of complete null replays. The controlled dynamic replay demonstrates online mechanics and is anchored to the precomputed injection evidence rather than presented as a new profitability or market-inefficiency result. The real BTC replay is allowed to remain `non_admitted`; conservative abstention is not a failure. The artifact scope is `clock_only` with `transport_status=gauge_uncertified`.

## Gauge Defect

| Artifact | Status | Reuse In GCDE |
|---|---|---|
| `backtest_results/gauge_invariance/gauge_report_20260624_221834.json` | Existing gauge report | Seed evidence for gauge non-certification across market clocks. |
| `scripts/wp1/gauge_invariance.py` | Existing gauge pipeline | Extend `Phi` beyond median `|VR-1|`. |
| `tests/wp1/test_gauge_invariance.py` | Existing tests | Preserve TOST and artifact expectations. |

Existing `q=5` gauge results:

| Pair | Difference | 90% CI | Equivalence |
|---|---:|---|---|
| calendar-volume | `0.0231` | `[0.0179, 0.0283]` | not certified |
| calendar-intrinsic | `0.0351` | `[0.0260, 0.0442]` | not certified |
| volume-intrinsic | `0.0120` | `[0.0023, 0.0217]` | not certified after Holm |

Interpretation for GCDE:

Gauge defect is a measured property of the detector-market pair, not a prose caveat.

## Information And Cost

| Artifact | Status | Reuse In GCDE |
|---|---|---|
| `backtest_results/thermodynamic_bound/thermo_report_20260624_082448.json` | Existing raw sign-pair information artifact | Use only as a scale diagnostic; detector-specific claims use the VR trigger MI artifact. |
| `backtest_results/thermodynamic_bound/vr_detector_mi.json` | New VR trigger MI artifact | Reports non-overlapping binary Holm-trigger MI for `q=2` and `q=5`, plus zero-entropy summaries for saturated Monte Carlo trigger cells. |
| `scripts/wp1/vr_detector_mi.py` | New detector MI runner | Computes detector-contingent MI for discrete VR trigger labels and writes a reproducible JSON artifact. |
| `tests/wp1/test_vr_detector_mi.py` | New tests | Guards zero-entropy behavior for saturated injection cells. |
| `scripts/wp1/thermodynamic_bound.py` | Existing information-cost script | Refactor toward detector-output inputs. |
| `scripts/wp1/mutual_information.py` | Existing MI estimators | Reuse estimator infrastructure. |
| `tests/wp1/test_thermodynamic_bound.py` | Existing tests | Keep cost-bound and graceful-degradation tests. |

Existing raw sign-pair result:

| Horizon `q` | Sign-pair gross bound |
|---:|---:|
| 2 | `19.06` bps |
| 5 | `20.49` bps |
| 15 | `12.52` bps |
| 60 | `7.35` bps |

Interpretation for GCDE:

This is not enough for a detector-specific economic claim. The paper treats it as a scale diagnostic and negative control, then reports `I(D_t; Y_{t+q})` for the discrete VR trigger itself.

Detector-contingent VR trigger result from `vr_detector_mi.json`:

| Horizon `q` | Labels | Triggers | MI (nats) | Net work at 10 bps |
|---:|---:|---:|---:|---:|
| 2 | `20126` | `74` | `0.0001281962` | `-8.7180` bps |
| 5 | `20126` | `115` | `0.0003463579` | `-6.5364` bps |

The saturated Monte Carlo trigger cells have zero path-level trigger entropy: `q=2,delta=0.15` is `200/200`, `q=5,delta=0.15` is `0/200`, `q=5,delta=0.20` is `0/200`, and `q=5,delta=0.30` is `200/200`. Within those cells, `I(D;Y) <= H(D)=0`.

## Holdout

| Artifact | Status | Reuse In GCDE |
|---|---|---|
| `backtest_results/holdout/holdout_confirmatory_20260623_175305.json` | Existing 2026 holdout artifact | Use as frozen-disposition example, not as discovery data. |
| `scripts/wp1/holdout_confirmatory.py` | Existing holdout script | Template for final holdout disposition matrix. |
| `tests/wp1/test_holdout_confirmatory.py` | Existing tests | Preserve holdout structure. |

Existing primary-cell holdout fields:

- Data span: 2026-01-01 to 2026-05-29.
- `n_bars=214355`.
- `observed_vr_dep=0.1575`.
- `median_z_m2=-0.3328`.
- `p_twotailed=0.7393`.
- `closed=false`.

Interpretation for GCDE:

The holdout helps show frozen claim disposition. It should not be used to tune the method.

## ETHUSDT External-Asset Replication

| Artifact | Status | Reuse In GCDE |
|---|---|---|
| `backtest_results/asset_replication/eth_replication_20260625_224506.json` | Existing same-venue symbol-swap artifact | Appendix-level external check that the BTCUSDT workflow can be rerun on ETHUSDT without pooling assets. |
| `scripts/wp1/eth_replication.py` | Existing replication driver | Reuses the WP1 analysis functions while keeping `primary_asset=BTCUSDT` and forbidding reuse of the BTC injection grid as ETH calibration. |
| `tests/wp1/test_eth_replication.py` | Existing focused tests | Guards BTC-primary scope, 2026 in-sample rejection, partial-holdout marking, and explicit injection-boundary metadata. |

Existing ETHUSDT summary:

| Layer | Value |
|---|---|
| In-sample span | 2021-05-29 19:32 UTC to 2025-12-31 23:59 UTC, `n_bars=2415148`, `year_2026_loaded=false` |
| Partial holdout span | 2026-01-01 00:00 UTC to 2026-05-29 20:34 UTC, `n_bars=214355`, `partial_holdout=true` |
| Primary cell | Median `|VR(5)-1|=0.1609`, mean `0.1832`, median `Z_m,5=-0.3981`, `p=0.6906`, `cascade_fired=false` |
| Gauge | `gauge equivalence not certified` |
| Information-cost | Raw sign-pair `q=5` gross bound `9.21` bps; no ETH-specific injection grid |
| Persistence | `KS=0.2605`, bootstrap `p=0.0` |
| Holdout primary cell | `|VR(5)-1|=0.1568`, median `Z_m,5=-0.3126`, `p=0.7546` |
| SHA256 | `CFD31CEBD10797FC12F120A593B99E22DC1820416A50E7357D08459B3A7155EF` |

Interpretation for GCDE:

The ETHUSDT artifact supports portability of the audit workflow to a second liquid Binance USD-M perpetual symbol. It does not create a pooled BTC-ETH claim, does not replace BTCUSDT as the primary demonstration, and does not provide an ETH-specific injection-recovery surface.

## Harmonized Benchmark

| Artifact | Status | Reuse In GCDE |
|---|---|---|
| `docs/research/gauge-calibrated-detector-exclusion/HARMONIZED_BENCHMARK_PROTOCOL.md` | Existing protocol, copied into the GCDE folder for submission packaging | Directly aligns with detector admissibility. |
| `backtest_results/harmonized_benchmark/harmonized_benchmark_smoke.json` | Smoke artifact only | Shows schema and disposition flow, not empirical evidence. |
| `backtest_results/harmonized_benchmark/harmonized_benchmark_q4_2022_btc_minimal_hmm.json` | Real native-frequency artifact | Q4 2022 BTCUSDT, 132,480 one-minute bars, no downsampling, rolling quantile valid, sparse HMM retained only as execution diagnostic, VR target-excluded from common volatility-state agreement. |
| `scripts/wp1/harmonized_benchmark.py` | Updated benchmark runner | Supports native-frequency time-domain windows, discrete detector-state MI, and explicit HMM benchmark profiles. |
| `tests/wp1/test_harmonized_benchmark.py` | Updated tests | Guards 1-minute window metadata, discrete soft-probability conversion, explicit HMM profile recording, and artifact shape. |
| `scripts/wp1/download_histdata_eurusd.py` | New EURUSD ingestion helper | Pulls HistData Generic ASCII tick files through the tokenized download form and normalizes bid, ask, spread, mid, and volume fields. |
| `tests/wp1/test_histdata_eurusd.py` | New tests | Guards HistData tick URL shape, hidden form payload extraction, and bid/ask row parsing. |

Current real Q4 2022 disposition:

```text
instrument_failure
```

The two-state task reports rolling quantile as `valid`, the bounded HMM profile as sparse/instrument-failed, and VR cascade as `excluded_from_agreement_headline`. Pairwise common-state agreement is therefore not interpreted. The HMM sparse-state output is not used as scientific evidence. Detector-contingent MI is estimated from discrete detector states; under the 10 bps convention, reported net available-work bounds are negative.

Interpretation for GCDE:

This is the right posture. Detector outputs that fail validity gates should not drive scientific conclusions, and serial-dependence detectors should not be forced into common volatility-state headlines.

## EURUSD Cost-Gate Sample

| Artifact | Status | Reuse In GCDE |
|---|---|---|
| `data/external/histdata/EURUSD/minute_matched/` | BTC/ETH-matched free-data sample | 56 HistData EURUSD monthly files filtered to `2021-05-29T19:32:00` through `2025-12-31T23:59:00`; available FX quotes aggregate 113,650,998 ticks to 1,659,616 one-minute bid/ask rows. |
| `backtest_results/eurusd/eurusd_histdata_cost_gate_matched_20210529_20251231_summary.json` | Cost-gate sample summary | Median observed spread is `0.2837` bps; mean spread is `0.4167` bps. |
| `scripts/wp1/eurusd_cost_gate_summary.py` | New summary runner | Verifies spread-bps observability for GCDE cost terms across one file, many files, directories, and date filters. |
| `tests/wp1/test_eurusd_cost_gate_summary.py` | New tests | Guards cost-gate summary fields. |

Interpretation for GCDE:

The EURUSD sample proves that free high-frequency FX data can carry executable bid/ask spreads for the cost gate over the same calendar evidence window used for the BTCUSDT/ETHUSDT in-sample artifacts. It does not certify EURUSD detector admissibility; a full appendix should execute the detector gates under the FX claim tuple.

## Prior Manuscript Sources

The current GCDE submission package does not require the old
`docs/research/calibrated-detector-exclusion` folder to compile or to reproduce
the bounded evidence map. The prior manuscript sources below are provenance
notes only; they identify where wording and framing originated before the GCDE
version became the authoritative submission folder.

| Legacy source | Reuse |
|---|---|
| Prior introduction draft | Measurement-instrument framing. |
| Prior calibrated-detector draft | Detector-agnostic objects and injection design. |
| Prior gauge draft | Gauge-defect definitions and evidence. |
| Prior information-ceiling draft | Available-work framing, with current limitation preserved. |
| Prior cross-detector draft | Benchmark posture, but avoid self-congratulatory yes/no framing. |
| `docs/research/v3-novelty-assessment.md` | Evidence that alpha/predictability routes should not be revived as novelty claims. |

## Artifact Gaps

| Gap | Required Output |
|---|---|
| Full gauge-indexed response surface | `P_det(delta; W, q, g, s)` across calendar, volume, and intrinsic gauges. |
| High-resolution size certificate | `alpha_D_star` under the declared dependence-preserving nulls with sufficient Monte Carlo resolution for production claims. |
| Detector-contingent MI | Broader `I(label_t; Y)`, `I(trigger_t; Y)`, `I(stat_bin_t; Y)` estimates beyond the VR q=2/q=5 trigger artifact and Q4 benchmark. |
| Real benchmark | Extend the Q4 non-smoke artifact to Markov-switching variance if retained. |
| EURUSD robustness | Execute the detector gates on HistData Generic ASCII tick bid/ask data; do not use FRED/ECB daily reference rates for cost-gate evidence. |
| Failure fingerprint | Per-cell counts for sign gate, multiplicity gate, empirical-size failure, gauge failure, target mismatch, and cost exhaustion. |
