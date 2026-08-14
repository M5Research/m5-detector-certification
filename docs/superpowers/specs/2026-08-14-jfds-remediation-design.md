# Production-Grade JFDS Remediation Design

**Date:** 2026-08-14  
**Repositories:** `m5-detector-certification`, `VolRegime-Engine`  
**Paper title:** *Certifying Regime Detectors Before Use*

## Objective

Deliver a production-grade, two-repository remediation in which a frozen detector-output claim is admitted only when every required gate is complete and passes. The principal scholarly contribution is an executable certification framework that binds power, size control, clock transport, target information, and decision value to one claim, one immutable lineage, and one machine-verifiable disposition.

Scientific honesty is a non-negotiable operating property of the framework. It is not the paper's headline contribution. The framework, its conjunction semantics, and its complete cross-asset demonstration are the headline.

## Repository Boundary

### `VolRegime-Engine`

The engine remains within its existing BTCUSDT and OHLCV milestone scope. It owns only:

- shared detector implementations and their tests;
- generic market-clock implementations and their tests;
- upstream release history for those shared implementations.

It must never contain EUR/USD data, bid/ask processing, certificate schemas, certificate inference, evidence records, manuscript changes, or submission metadata.

The upstream reconciliation branch starts from local commit `d7b0df956db61652b6355111d60a9b1a1866bc95`. That commit is preserved.

### `m5-detector-certification`

The certification repository exclusively owns:

- EUR/USD download, normalization, bid/ask validation, and partition manifests;
- detector adapters and alternative-clock threshold calibration;
- certificate schemas, inference, gate execution, and immutable evidence;
- protocol v5 and no-peeking records;
- BTC successor and EUR/USD certificate specifications;
- manuscript, claims ledger, traceability, reproduction, readiness, and submission records.

It is the authoritative submission and evidence package.

## Upstream Reconciliation

Eight detector files are duplicated across the repositories. Semantic comparison establishes that six differ only by line endings:

- `defaults.py`
- `realized_vol.py`
- `regime_detector.py`
- `regime_engine.py`
- `rolling_quantile_detector.py`
- `strategy_modules.py`

Two contain substantive certification-side safety changes that must be ported upstream:

- `hmm_detector.py` documents anchored full-sample parameter fitting accurately, rejects an all-nonfinite restart set, demotes NaN likelihoods before selection, and uses a runtime exception rather than an optimization-removable assertion for the statsmodels API guard.
- `regime_population.py` removes forced percentile-interval clamping and emits a diagnostic warning when bootstrap bias places the point estimate outside the percentile interval.

The engine receives focused regression tests for these behaviors before the implementation changes. Both repositories then run their full default, artifact, and slow suites as applicable. No copy is described as authoritative until semantic parity and both suites pass.

The shared market-clock implementation is engine-owned and asset-neutral. BTC constants and research orchestration remain outside the generic core. Certification-side EUR/USD adapters may supply tick-count activity and intrinsic-variance increments, but quote semantics and threshold calibration remain certification-only.

## Upstream Snapshot and Provenance

The certification repository vendors selected files only from a merged and tagged engine release. Before that event, no final snapshot manifest is created.

The manifest records:

- upstream repository URL;
- release tag and exact commit;
- selected relative paths and SHA-256 hashes;
- dependency versions relevant to the selected implementations;
- export timestamp and export tool version.

Parity tests verify the vendored files against the manifest and fail on additions, omissions, or byte divergence. The certification repository may wrap vendored code, but it may not edit it without a new upstream release and snapshot.

## Certificate Core

The certificate core uses strict Pydantic v2 models and an exported `certificate-v1` JSON Schema.

### `CertificateSpec`

Contains claim identity, evidence revision, parent revision, detector, instrument, target, schemes, horizon, signal/null families, margins, costs, required gates, preregistration commit, alpha allocation, and artifact chronology constraints.

The spec rejects unknown fields, non-finite numbers, duplicate required gates, invalid alpha allocations, and scope mutation under an existing evidence revision.

### `GateResult`

Contains gate name, state (`pass`, `fail`, `not_run`, or `invalid`), estimand, threshold, uncertainty, artifact references, and a reason. Non-finite estimands require an explicit invalid reason.

### `CertificateRecord`

Contains the deterministic spec hash, code/data/upstream hashes, all gate results, completeness, complete failure profile, final disposition, provenance, lineage, and supersession status.

Canonical JSON uses UTF-8, sorted keys, fixed separators, and rejected NaN/Infinity values. The SHA-256 spec hash is computed from the canonical semantic spec, not source formatting.

## Disposition and Completeness

Disposition precedence is implemented independently of detector and gate code:

1. `instrument_failed`
2. `target_mismatched`
3. `incomplete`
4. `size_distorted`
5. `transport_uncertified`
6. `underpowered`
7. `information_unsupported`
8. `value_negative`
9. `admissible`

Missing required gates always produce `incomplete`, even when an observed gate has already failed. Every observed failure remains in `failure_profile`.

A completed, scientifically withheld certificate is a valid certificate-run outcome and returns exit code 0. Schema failure, incomplete execution, integrity failure, invalid chronology, or attempted overwrite returns nonzero.

## Immutable Lineage and Error Spending

Evidentiary revision `k` receives `alpha_k = 0.05 * 2^(-(k+1))`. Conjunctive admission does not split alpha across gates. Metadata-only patches retain the evidence revision and consume no alpha. Any change to code, calibration, target, horizon, scheme, or claim scope creates a new evidentiary revision.

Existing specs, evidence artifacts, and certificate records are create-only. The runner uses exclusive creation and atomic finalization. Duplicate IDs, in-place replacement, parent cycles, changed scope under one revision, and artifacts predating preregistration are rejected.

## Runner and Evidence Flow

The public entry point is:

```text
python -m scripts.wp1.run_certificate --spec <spec.json> --out <new-artifact.json>
```

The runner performs, in order:

1. schema and canonical-hash validation;
2. upstream-manifest validation;
3. preregistration and repository-state validation;
4. revision-tree and alpha validation;
5. required-gate dispatch;
6. create-only artifact finalization;
7. mechanical completeness, failure-profile, and disposition classification.

Gate implementations return `GateResult`; they cannot choose the final disposition. Seeds for stochastic inference derive from the spec hash and a domain-separated gate label.

## Stationary-Block Information Inference

Certificate-grade information evidence operates on non-overlapping decision epochs. It reports plug-in and Miller-Madow-adjusted mutual information.

- Paired stationary-bootstrap indices form uncertainty intervals.
- Independently bootstrapped detector and target blocks form the independence null.
- The primary mean block length is `ceil(n^(1/3))`.
- Half and double block-length sensitivity runs are mandatory.
- A reversed gate conclusion produces `invalid: unstable_reference`.
- Production evidence uses `B = 49,999` and the finite-sample `+1/+1` correction.
- New certificate schemas and outputs contain no `permutation_p` field.

Legacy permutation artifacts remain byte-unchanged and are labeled `legacy_noncertificate` in external ledgers.

## Protocol and Evidence Ordering

Protocol v5 freezes completeness, precedence, stationary-block information inference, geometric error spending, the BTC successor claim, and the complete EUR/USD tuple and thresholds before new evidence is examined.

The amendment is committed and archived before downloading the new BTC or EUR/USD evaluation spans. An author no-peeking attestation and archive identifier are recorded. Evidence runners refuse a missing, dirty, chronologically invalid, undisclosed-amendment, or spec-inconsistent preregistration state.

Reduced synthetic fixtures may run before the freeze. Real holdout acquisition and certificate-grade evaluation may not.

## Confirmatory Certificates

### BTC successor

The current repaired `q=2` result becomes `superseded_exploratory`. The successor uses the frozen BTCUSDT, `W=120`, `q=2`, calendar/volume scope, AR(1) injection family, recentering constants, transport margin, and revision alpha. Size, power, and declared transport must all pass for admission; otherwise the classifier produces the withheld disposition.

### EUR/USD rolling quantile

Certification exclusively owns HistData tick acquisition and normalization. Monthly input streams are processed into bounded-memory partitions. Validation covers UTC monotonicity, `ask >= bid > 0`, duplicate policy, session/weekend gaps, tick counts, row totals, and raw/derived hashes.

The certificate freezes the 60-minute rolling RMS detector, 43,200-minute percentile window, 0.75/0.95 boundaries, strict EXTREME boundary, causal mode, and 60-minute decision horizon. Calendar, tick-count, and intrinsic-variance clocks are constructed through certification adapters over the tagged shared market-clock core.

Power, size, transport, information, and value gates follow the frozen numerical rules from protocol v5. The record is complete regardless of admission or withholding. No post-evaluation thresholds, targets, policies, margins, or costs may change.

## Manuscript Thesis

The paper's greatest point is the executable certification framework: no identified financial framework jointly binds power, size, transport, target information, and decision value to one frozen detector-output claim with deterministic refusal precedence and machine-verifiable lineage.

Supporting contributions are:

1. dependence-preserving information inference;
2. immutable, error-controlled recertification lineage;
3. a complete EUR/USD certificate demonstrating the full conjunction in a second asset class.

Gauge R&R and ISO 5725 are acknowledged as prior measurement-system practice. Novelty is bounded to the joint financial-detector certificate, not claimed as the invention of pre-use measurement or honest reporting.

## Error Handling and Failure Publication

Operational failures never silently become scientific failures. Schema, integrity, chronology, overwrite, and unstable-reference failures are explicit invalid executions. Measured scientific failures remain valid publishable certificate outcomes. Missing work produces `incomplete`, never an inferred pass or fail.

Large raw data remains external. Source URLs, checksums, normalized manifests, and compact derived evidence are committed. Logs must avoid local absolute paths and secrets.

## Verification Strategy

Implementation follows test-driven development. Coverage includes:

- detector safety regressions and semantic upstream parity;
- deterministic schema/hash behavior and strict field validation;
- precedence, missing-gate completeness, lineage mutation, and alpha budget;
- create-only artifacts and chronology guards;
- stationary-block determinism, autocorrelated-null type-I behavior, coupled-series power, and block sensitivity;
- absence of certificate-grade permutation fields;
- EUR/USD parsing, validation, clocks, causal alignment, injection, costs, and utility;
- admitted, withheld, incomplete, and invalid end-to-end fixtures;
- artifact-to-manuscript value traceability;
- clean LaTeX build, unresolved-reference checks, author-marker checks, and rendered-page inspection.

Reduced deterministic simulations run in pull-request CI. Full bootstrap and Monte Carlo regeneration runs in a dedicated reproducibility workflow.

## Delivery Sequence

1. Reconcile and verify the engine from `d7b0df9`.
2. Merge and tag the engine upstream release.
3. Export and verify the certification snapshot manifest.
4. Implement and verify the certificate core.
5. Commit and archive protocol v5.
6. Acquire unseen evaluation data.
7. Generate immutable BTC and EUR/USD evidence.
8. Update and audit the manuscript and supporting records.
9. Complete independent statistical/code review and submission readiness.

Steps that require remote merge, tag publication, archival services, external data acquisition, or independent human review are explicit release gates. Local implementation must not fabricate completion of those gates.
