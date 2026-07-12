# Harmonized Regime-Detector Benchmark Protocol

Protocol ID: `harmonized-regime-benchmark-mvp-v1`

Status: MVP benchmark protocol. It supersedes the exploratory interpretation of
the legacy cross-detector kappa only after a clean benchmark artifact is
generated under this protocol. Legacy `cross_detector_*.json` artifacts remain
exploratory.

## Research Question

Do regime detectors measure the same market state after the obvious sources of
disagreement are controlled?

The benchmark does not assume detector disagreement. It adjudicates three
possibilities:

- agreement becomes high after harmonization;
- disagreement remains after harmonization;
- some detectors fail validity gates and their outputs are not admissible as
  labels for agreement interpretation.

## Scope

- Asset and venue: BTCUSDT Binance USD-M perpetual futures.
- In-sample window: 2021-2025 only.
- Holdout: 2026 partition remains unloaded by in-sample benchmark scripts.
- Primary clock: calendar 1-minute bars.
- Primary detector set: RollingQuantile, HMM, Markov-switching variance model,
  and VR cascade.
- Deferred to v2: BOCPD, multi-asset replication, spot/futures venue panel, and
  non-calendar clocks as primary results.

## Harmonized Label Tasks

The benchmark reports two label tasks.

1. `2-state`: LOW versus HIGH. ELEVATED and EXTREME are collapsed into HIGH.
2. `3-state`: LOW, ELEVATED, EXTREME. HMM/MS-style state order is determined by
   ascending variance rank. Quantile-style state order follows its frozen
   thresholds. VR-derived labels are not admitted to the common volatility-state
   headline unless the VR mapping is explicitly marked eligible by a future
   protocol amendment.

Warmup labels are always `-1` and are excluded from all agreement calculations.
All agreement metrics use only timestamps where every compared detector has a
valid non-warmup label.

## Validity Gates

A detector is `valid` only if all gates pass for the task being reported.

- Convergence and determinism diagnostics pass.
- Every reported hard state has at least 50 non-overlapping observations at
  stride 60.
- HMM/MS-style adjacent variance-separation ratios are at least 1.10.
- Label mapping follows the frozen task mapping above; no post-hoc relabeling is
  permitted.

If a detector fails these gates, it is reported as `instrument_failure`, not as
evidence of kappa disagreement. If a detector targets a different object, such
as the VR cascade measuring serial dependence rather than volatility state, it
is reported as `excluded_from_agreement_headline` for common-state agreement.

## Metrics

The benchmark reports:

- state occupancy and transition matrices;
- run-length summaries;
- confusion matrices;
- Cohen's kappa;
- adjusted Rand index;
- normalized mutual information;
- variation of information;
- detector-contingent mutual information against forward return sign;
- gross and cost-adjusted information ceiling in basis points.

Cohen's kappa is secondary because it is sensitive to category imbalance.
Cluster-invariant metrics and the confusion matrix carry the main interpretation.

## Disposition Rules

The benchmark artifact classifies the run as:

- `harmonized_convergence` when all valid pairwise NMI values are at least 0.60;
- `persistent_disagreement` when all valid pairwise NMI values are at most 0.20;
- `mixed_family_structure` otherwise;
- `instrument_failure` when any detector fails frozen validity gates.

Manuscript conclusions must be determined after reading the benchmark artifact.
The paper must not pre-commit to "regime labels are unstable" before this test.

## Artifact Contract

The runner writes a JSON artifact with:

- `schema_version`;
- `benchmark.protocol_id`;
- `data_span` with `year_2026_loaded=false`;
- per-task detector summaries;
- per-task pairwise agreement metrics;
- per-task detector-contingent economic summaries;
- overall disposition;
- code provenance.

Smoke artifacts may use synthetic data to test the machinery, but they are not
submission-grade empirical evidence.
