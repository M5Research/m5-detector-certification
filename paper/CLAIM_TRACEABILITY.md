# Claim Traceability

Traceability for **"Certifying Regime Detectors Before Use"** (M5 Research). "GCDE" /
"gauge" naming in artifact fields is legacy provenance; the manuscript uses
"certification protocol" and "transport / cross-clock".

Status definitions:

- `supported`: directly supported by an existing frozen artifact or source file.
- `partial`: partly supported; the manuscript must preserve the stated limitation.
- `method-defined`: a definition, proposition, or positioning statement supported by the
  method rather than an empirical result.
- `exploratory`: supported only by post-freeze evidence; a preregistered confirmatory
  rerun is required before any strong or deployment claim.

## Evidence Checks

Paths are relative to the package root (`m5-detector-certification/`, or the repository root in
the private working copy). Values verified against the frozen artifacts.

| Evidence Area | Artifact | Verified Values |
|---|---|---|
| Injection response | `data/injection_runs/inj_d0.15_q2_W120.json` | `delta=0.15`, `q=2`, `W=120`, `N_mc=200`, `n_fires=200`, `P_det=1.0`, 95% CI `[0.9818, 1.0]`. |
| Injection response | `data/injection_runs/inj_d0.15_q5_W120.json` | `delta=0.15`, `q=5`, `n_fires=0`, `P_det=0.0`, 95% CI `[0.0, 0.0182]`. |
| Injection response | `data/injection_runs/inj_d0.2_q5_W120.json` | `delta=0.20`, `q=5`, `n_fires=0`, `P_det=0.0`. |
| Injection response | `data/injection_runs/inj_d0.3_q5_W120.json` | `delta=0.30`, `q=5`, `n_fires=200`, `P_det=1.0`. |
| Preregistered grid | `data/injection_runs/inj_d{0.0005..0.10}_q{2,5,15,60}_W{60,120,240}.json` | 96 confirmatory cells, all `P_det=0` (0 of 200 fires per cell). |
| Empirical null / size | `backtest_results/empirical_vr_null/empirical_vr_null_20260623_230753.json` | Circular-block diagnostic, `n_boot=100`, cells `W=120,q=2` and `W=120,q=5`; zero false alarms; two-cell Holm p-values `0.0792`; positive-gate p-values `0.9703`/`0.9604`; no Holm rejection at `alpha=0.05`. Nominal N(0,1) reference is mis-centered. |
| Transport (cross-clock) | `backtest_results/gauge_invariance/gauge_report_mde_margin.json` (margin-corrected; source artifact `gauge_report_20260624_221834.json` retained unmodified) | Margin `eps_g = MDE = 0.019748` (preregistration v4.0 §4.6). `q=2`: clock-volume Holm `5.7e-11` and volume-intrinsic Holm `2.3e-8` certified; clock-intrinsic Holm `0.0682` **not** certified (interval `[0.0105,0.0179]` inside the margin, TOST not met). `q=5`: clock-volume diff `0.0231` CI `[0.0179,0.0283]`, clock-intrinsic `0.0351` CI `[0.0260,0.0442]`, volume-intrinsic `0.0120` CI `[0.0023,0.0217]` — none certified. See `prereg/DEVIATIONS.md` D1. |
| Raw sign-pair scale | `backtest_results/thermodynamic_bound/thermo_report_20260624_082448.json` | Sign-pair gross bounds `q=2` 19.06 bps, `q=5` 20.49 bps, `q=15` 12.52 bps, `q=60` 7.35 bps (scale diagnostic, not detector evidence). |
| Legacy detector-contingent MI | `backtest_results/thermodynamic_bound/vr_detector_mi.json` | Retained byte-unchanged as `legacy_noncertificate`; its IID-permutation values are not certificate-grade evidence. |
| BTC successor certificate | `certificates/btcusdt-vr-q2-v5-r0.json`; `evidence/gates/btcusdt-vr-q2-v5-r0/evidence.json` | Complete; `target_mismatched`. Size `187/10000` passes, cMDE is `0.10`, calendar-volume transport fails, stationary-block information fails at both `q=2` (`E=0.801`) and `q=5` (`E=0.620`), and the cost-free value interval crosses zero. |
| EUR/USD rolling-quantile certificate | `certificates/eurusd-rq-v5-r0.json`; `evidence/gates/eurusd-rq-v5-r0/evidence.json` | Complete; `size_distorted`. Instrument and target pass; size is `554/10000` with upper exact bound `0.0601`; power and transport fail; information passes (`MI=0.1514`, 97.5% CI `[0.1245,0.1797]`, `E=111.803`); net value passes (point `0.1101`, 97.5% CI `[0.0831,0.1410]`). |
| 2026 holdout | `backtest_results/holdout/holdout_confirmatory_20260623_175305.json` | Span `2026-01-01..2026-05-29`, `n_bars=214355`, primary `W=120,q=5`, `observed_vr_dep=0.1575`, `median_z_m2=-0.333`, `p_twotailed=0.739`. |
| Reference repair | `backtest_results/reference_repair/recentered_reference_repair_20260710.json` | Exact offline re-thresholding of frozen per-draw `median_z_m2`: recentered cMDE `0.02` (`q=2`), `0.10` (`q=5`); out-of-sample size `0.02`/`0.07`; real-BTC recentered statistic negative (still abstains). Recentering constants are post-freeze (exploratory). |
| Three-family benchmark | `backtest_results/harmonized_benchmark/harmonized_benchmark_q4_2022_btc_minimal_hmm.json` | Real BTCUSDT Q4 2022, `n_bars=132480`, `bar_frequency=1min`, `downsampled=false`; rolling quantile `valid`, HMM `instrument_failure`, VR cascade `excluded_from_agreement_headline`; discrete detector-state net work negative under 10 bps. Schema companion: `harmonized_benchmark_smoke.json`. |
| ETHUSDT replication | `backtest_results/asset_replication/eth_replication_20260625_224506.json` | Symbol swap, `primary_asset=BTCUSDT`; in-sample `2021-05-29..2025-12-31`, `n_bars=2415148`, `year_2026_loaded=false`; primary median `|VR(5)-1|=0.1609`, `p=0.6906`, `cascade_fired=false`; transport not certified; `eth_specific_injection_grid=false`. |
| EUR/USD cost gate | `backtest_results/eurusd/eurusd_histdata_cost_gate_matched_20210529_20251231_summary.json` | HistData matched span, 56 monthly files; `1,659,616` one-minute bid/ask rows from `113,650,998` ticks; median spread `0.284` bps, mean `0.417` bps; `cost_gate_executable=true`. |

## Manuscript Claim Map

| ID | Manuscript Claim | Status | Evidence |
|---|---|---|---|
| C1 | A pre-use certification protocol (detector admissibility certification) treats a frozen detector as an instrument and measures five operating characteristics for one declared claim. | method-defined | `main.tex` §3; `CLAIMS.md`. |
| C2 | The protocol returns a disposition via a fixed precedence rule (instrument-failed, target-mismatched, incomplete, size-distorted, transport-uncertified, underpowered, information-unsupported, value-negative, else admissible). Missing required gates always produce incomplete. | method-defined | `main.tex` §3.2 precedence enumerate + certificate table; `src/certificate/disposition.py`. |
| C3 | The certified minimum detectable effect (cMDE, `delta_90`) is a named reporting standard for a detector's power. | method-defined | `main.tex` §3.3. |
| C4 | Casting each testing gate as an e-value gives static admission control with no multiplicity penalty (intersection-union) and anytime-valid admission (Ville). At revision 0 the stationary-block information threshold is 40: both BTC horizons fail and EUR/USD passes. | method-defined (propositions) + supported (computed) | `main.tex` §3.4, App. A proofs; protocol-v5 evidence JSONs. |
| C5 | The preregistered 96-cell grid excludes `delta <= 0.10` everywhere (0/200 per cell); disclosed post-freeze positive controls fire at `q=2,0.15` and `q=5,0.30`, silent at `q=5,0.15` and `q=5,0.20`. | supported | injection JSONs (Evidence Checks); provenance separated per cell. |
| C6 | The nominal-level rule is strongly conservative under a circular-block empirical null (zero false alarms in 100 replicates), a size distortion traceable to a mis-centered asymptotic reference. | supported | `empirical_vr_null_20260623_230753.json`. |
| C7 | Under the historical v4 transport artifact, two of three `q=2` pairs and none of the `q=5` pairs certify. The protocol-v5 BTC calendar-volume comparison fails; three of four EUR/USD comparisons fail. | supported | `gauge_report_mde_margin.json`; both protocol-v5 evidence JSONs. |
| C8 | Protocol-v5 stationary-block information fails at BTC `q=2` and `q=5`, while EUR/USD information passes. The old IID-permutation MI is legacy-only. | supported | both protocol-v5 evidence JSONs; `vr_detector_mi.json` labeled `legacy_noncertificate`. |
| C9 | Recentering the decision statistic on the empirical null lowers the historical cMDE ~tenfold at `W=120`, but that repair is `superseded_exploratory`. The frozen successor is complete and `target_mismatched` on previously unseen June/July 2026 BTC data. | exploratory repair + supported successor | `recentered_reference_repair_20260710.json`; `certificates/btcusdt-vr-q2-v5-r0.json`. |
| C10 | A three-family exhibit (rolling quantile, HMM, VR cascade) yields incomplete, instrument-failed, and target-mismatched dispositions — none of which is relative performance. The rolling-quantile negative value is a failure-profile observation only. | supported | `harmonized_benchmark_q4_2022_btc_minimal_hmm.json`; protocol v5 completeness rule. |
| C11 | A temporally disjoint 2026 window is used as a frozen-disposition example, not discovery data. | supported | `holdout_confirmatory_20260623_175305.json`. |
| C12 | A same-venue ETHUSDT symbol swap reproduces the conservative primary-cell disposition and the transport non-certification pattern. | partial | `eth_replication_20260625_224506.json`; external workflow-portability check only — do not pool assets or imply an ETH-specific injection surface. |
| C13 | The EUR/USD rolling-quantile certificate is the second asset-class/detector-family certificate. Development is 2021-05-30..2025-12-31 and the previously unseen evaluation span is 2026-01-01..2026-07-31. It is complete and `size_distorted`. | supported | `certificates/eurusd-rq-v5-r0.json`; `evidence/gates/eurusd-rq-v5-r0/evidence.json`. |
| C14 | The VR cascade is an example instrument, not the contribution and not a superiority claim. | supported | `main.tex` §4.1, §6; `CLAIMS.md`. |
| C15 | The paper claims no Bitcoin inefficiency, trading profitability, or detector optimality. | supported (non-claim) | `CLAIMS.md`; `README.md`. |

## Deployment Evidence Gaps

Claims most likely to be overstated if the manuscript were used as a full detector-specific
admissibility certificate without new artifacts:

| Gap | Required before a strong claim |
|---|---|
| Confirmatory repair | Preregistered rerun with the recentering constants frozen (C9 is currently exploratory). |
| Additional detector/asset revisions | Each new claim or evidentiary revision requires a newly frozen spec and the next geometric alpha slice; current certificates cannot be generalized beyond their tuples. |
| Clock-indexed sensitivity | The complete records execute frozen transport comparisons, but full response surfaces across all market clocks would be required for broader portability claims. |
| BTC economic value | The BTC successor freezes a cost-free scientific value estimand; any trading or economic claim requires a new cost-bearing specification and revision. |
| ETH certificate | ETH remains a workflow-portability check rather than a complete certificate; it must not be pooled with BTC or represented as certified. |
