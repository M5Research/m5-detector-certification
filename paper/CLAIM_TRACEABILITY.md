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
| Transport (cross-clock) | `backtest_results/gauge_invariance/gauge_report_20260624_221834.json` | `q=2`: all three scheme pairs certified equivalent (Holm). `q=5`: clock-volume diff `0.0231` CI `[0.0179,0.0283]`, clock-intrinsic `0.0351` CI `[0.0260,0.0442]`, volume-intrinsic `0.0120` CI `[0.0023,0.0217]` — none certified. |
| Raw sign-pair scale | `backtest_results/thermodynamic_bound/thermo_report_20260624_082448.json` | Sign-pair gross bounds `q=2` 19.06 bps, `q=5` 20.49 bps, `q=15` 12.52 bps, `q=60` 7.35 bps (scale diagnostic, not detector evidence). |
| Detector-contingent MI | `backtest_results/thermodynamic_bound/vr_detector_mi.json` | Non-overlapping Holm-trigger labels 2021-05-29..2025-12-31: `q=2` MI `1.282e-4` nats, `74/20126` triggers, net work `-8.72` bps, perm `p=0.053`; `q=5` MI `3.464e-4` nats, `115/20126` triggers, net work `-6.54` bps, perm `p=0.001`. |
| E-value certificate | (computed from the row above) | Calibrator `f_kappa(p)=kappa*p^(kappa-1)`, `kappa=1/2`: `E=2.17` at `q=2`, `E=15.8` at `q=5`; both below `1/alpha=20`. |
| 2026 holdout | `backtest_results/holdout/holdout_confirmatory_20260623_175305.json` | Span `2026-01-01..2026-05-29`, `n_bars=214355`, primary `W=120,q=5`, `observed_vr_dep=0.1575`, `median_z_m2=-0.333`, `p_twotailed=0.739`. |
| Reference repair | `backtest_results/reference_repair/recentered_reference_repair_20260710.json` | Exact offline re-thresholding of frozen per-draw `median_z_m2`: recentered cMDE `0.02` (`q=2`), `0.10` (`q=5`); out-of-sample size `0.02`/`0.07`; real-BTC recentered statistic negative (still abstains). Recentering constants are post-freeze (exploratory). |
| Three-family benchmark | `backtest_results/harmonized_benchmark/harmonized_benchmark_q4_2022_btc_minimal_hmm.json` | Real BTCUSDT Q4 2022, `n_bars=132480`, `bar_frequency=1min`, `downsampled=false`; rolling quantile `valid`, HMM `instrument_failure`, VR cascade `excluded_from_agreement_headline`; discrete detector-state net work negative under 10 bps. Schema companion: `harmonized_benchmark_smoke.json`. |
| ETHUSDT replication | `backtest_results/asset_replication/eth_replication_20260625_224506.json` | Symbol swap, `primary_asset=BTCUSDT`; in-sample `2021-05-29..2025-12-31`, `n_bars=2415148`, `year_2026_loaded=false`; primary median `|VR(5)-1|=0.1609`, `p=0.6906`, `cascade_fired=false`; transport not certified; `eth_specific_injection_grid=false`. |
| EUR/USD cost gate | `backtest_results/eurusd/eurusd_histdata_cost_gate_matched_20210529_20251231_summary.json` | HistData matched span, 56 monthly files; `1,659,616` one-minute bid/ask rows from `113,650,998` ticks; median spread `0.284` bps, mean `0.417` bps; `cost_gate_executable=true`. |

## Manuscript Claim Map

| ID | Manuscript Claim | Status | Evidence |
|---|---|---|---|
| C1 | A pre-use certification protocol (detector admissibility certification) treats a frozen detector as an instrument and measures five operating characteristics for one declared claim. | method-defined | `main.tex` §3; `CLAIMS.md`. |
| C2 | The protocol returns a disposition via a fixed precedence rule (instrument-failed, target-mismatched, size-distorted, transport-uncertified, underpowered, cost-dominated, incomplete, else admissible). | method-defined | `main.tex` §3.2 precedence enumerate + certificate table. |
| C3 | The certified minimum detectable effect (cMDE, `delta_90`) is a named reporting standard for a detector's power. | method-defined | `main.tex` §3.3. |
| C4 | Casting each testing gate as an e-value gives static admission control with no multiplicity penalty (intersection-union) and anytime-valid admission (Ville); computed info-gate e-values are `2.17` (q=2) and `15.8` (q=5). | method-defined (propositions) + supported (computed) | `main.tex` §3.4, App. A proofs; `vr_detector_mi.json`. |
| C5 | The preregistered 96-cell grid excludes `delta <= 0.10` everywhere (0/200 per cell); disclosed post-freeze positive controls fire at `q=2,0.15` and `q=5,0.30`, silent at `q=5,0.15` and `q=5,0.20`. | supported | injection JSONs (Evidence Checks); provenance separated per cell. |
| C6 | The nominal-level rule is strongly conservative under a circular-block empirical null (zero false alarms in 100 replicates), a size distortion traceable to a mis-centered asymptotic reference. | supported | `empirical_vr_null_20260623_230753.json`. |
| C7 | Cross-clock equivalence is certified at `q=2` (all three pairs) and not certified at `q=5`. | supported | `gauge_report_20260624_221834.json`. |
| C8 | Trigger information about forward-return signs is statistically resolvable, but its net value is convention-dependent, flipping sign between per-epoch and per-trigger cost attribution. | supported | `vr_detector_mi.json`. |
| C9 | Recentering the decision statistic on the empirical null lowers the cMDE ~tenfold, restores size, and yields the protocol's first `admissible` certificate for a bounded `q=2` claim; real BTC still abstains (powered exclusion). | exploratory | `recentered_reference_repair_20260710.json`; constants are post-freeze — confirmatory rerun required. |
| C10 | A three-family exhibit (rolling quantile, HMM, VR cascade) yields three different dispositions for three different reasons — none of which is relative performance. | supported | `harmonized_benchmark_q4_2022_btc_minimal_hmm.json`. |
| C11 | A temporally disjoint 2026 window is used as a frozen-disposition example, not discovery data. | supported | `holdout_confirmatory_20260623_175305.json`. |
| C12 | A same-venue ETHUSDT symbol swap reproduces the conservative primary-cell disposition and the transport non-certification pattern. | partial | `eth_replication_20260625_224506.json`; external workflow-portability check only — do not pool assets or imply an ETH-specific injection surface. |
| C13 | The EUR/USD appendix demonstrates bid/ask spread observability for the cost gate. | partial | `eurusd_histdata_cost_gate_matched_...summary.json`; a matched-span sample, not a full FX admissibility certificate. |
| C14 | The VR cascade is an example instrument, not the contribution and not a superiority claim. | supported | `main.tex` §4.1, §6; `CLAIMS.md`. |
| C15 | The paper claims no Bitcoin inefficiency, trading profitability, or detector optimality. | supported (non-claim) | `CLAIMS.md`; `README.md`. |

## Deployment Evidence Gaps

Claims most likely to be overstated if the manuscript were used as a full detector-specific
admissibility certificate without new artifacts:

| Gap | Required before a strong claim |
|---|---|
| Confirmatory repair | Preregistered rerun with the recentering constants frozen (C9 is currently exploratory). |
| High-resolution empirical size | `alpha_D_star(W,q,g)` under declared dependence-preserving nulls at deployment Monte Carlo resolution. |
| Clock-indexed sensitivity | Response surfaces across calendar, volume, and intrinsic clocks, not only the current non-clock injection cells. |
| Detector-contingent MI breadth | Estimates beyond the VR `q=2`/`q=5` trigger and Q4 BTC benchmark, with CIs and permutation nulls for declared assets/clocks. |
| Available decision work | Detector-contingent MI net of executable all-in cost scenarios (BTC uses a 10 bps convention; the EUR/USD sample proves observability, not full FX admissibility). |
| Second-asset certificate | Extend ETH beyond a workflow-portability check to a full declared-claim certificate. |
