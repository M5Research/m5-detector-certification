# Protocol v5 Amendment — Completeness, Stationary-Block Information, Lineage

**Status:** frozen before acquisition of repository-unseen BTC (2026-06-01 through 2026-07-31) and EUR/USD (2026-01-01 through 2026-07-31) evaluation spans.

**Parent protocol:** `prereg/PREREGISTRATION-v4.0-freeze-1dc5c82.md`

**Shared detector release:** `VolRegime-Engine` tag `jfds-detector-core-v1.0.0`, commit `b6911cf9812e9eafad269a44c92d77a5c95fabc3`.

**Archive identifier:** `https://github.com/M5Research/m5-detector-certification/tree/protocol-v5.0`

**Claim:** this amendment freezes completeness, disposition precedence, information inference, error spending, the BTC \(q=2\) successor claim, and the complete EUR/USD rolling-quantile claim tuple. No threshold, policy, margin, or target may change after evaluation data are examined.

## Author no-peeking attestation

I attest that this amendment was written and committed before downloading the evaluation spans named above, and that those spans were not inspected for certificate-grade evidence before this freeze. The immutable archive identifier is recorded above.

## 1. Completeness and disposition precedence

Required gates for a complete certificate: `instrument`, `target`, `size`, `transport`, `power`, `information`, `value`.

Disposition precedence, applied mechanically:

1. `instrument_failed`
2. `target_mismatched`
3. `incomplete`
4. `size_distorted`
5. `transport_uncertified`
6. `underpowered`
7. `information_unsupported`
8. `value_negative`
9. `admissible`

Missing required gates always produce `incomplete`, even when a measured gate already fails. All observed failures remain in the failure profile. Across-gate alpha is not split: admission is conjunctive.

A completed scientifically withheld certificate is a valid outcome. Operational failures (schema, integrity, chronology, overwrite, `invalid: unstable_reference`) are not scientific dispositions.

The current post-freeze repaired BTC \(q=2\) admission is `superseded_exploratory`. It is not the first admissible certificate of this protocol version.

The harmonized three-family benchmark is an explicitly non-certificate exhibit. Its rolling-quantile outcome is `incomplete`. Any negative value result is retained only as a failure-profile observation.

## 2. Stationary-block information inference

Certificate-grade information evidence:

- Sample detector outputs and targets at non-overlapping decision epochs.
- Report plug-in and Miller–Madow-adjusted mutual information.
- Form confidence intervals from paired stationary-bootstrap indices.
- Form the independence null by independently bootstrapping detector and target blocks.
- \(B=49{,}999\), finite-sample \(+1/+1\) correction.
- Seeds derived from the certificate spec hash and a domain-separated gate label.
- Primary mean block length \(\lceil n^{1/3}\rceil\).
- Repeat at half and double block length. A reversed gate conclusion yields `invalid: unstable_reference`.

`permutation_p` is forbidden in certificate-v1 schemas and outputs. Legacy permutation files remain byte-unchanged and are labeled `legacy_noncertificate`.

## 3. Geometric error spending and immutable lineage

Evidentiary revision \(k\) receives \(\alpha_k=0.05\cdot 2^{-(k+1)}\). The infinite sum is \(0.05\). Metadata-only patches retain \(k\) and consume no alpha. Changes to code, calibration, target, horizon, scheme, or claim scope require a new evidentiary revision. Existing artifacts are never overwritten.

Evidence runners refuse a missing, dirty, later-amended-without-disclosure, or spec-inconsistent preregistration commit.

## 4. BTC \(q=2\) successor claim (evidentiary revision 0)

Frozen tuple:

- Instrument: BTCUSDT, Binance USD-M.
- Detector: recentered VR cascade, \(W=120\), \(q=2\).
- Target: positive short-horizon serial-dependence scientific claim.
- Schemes: calendar and volume bars only.
- Recentering constants: existing constants, frozen before evaluation.
- Signal family: existing AR(1) injection family.
- Transport margin: \(0.019748\).
- \(\alpha_0=0.025\).
- Size: at least 10,000 null draws; exact binomial bounds; pass only if revision-adjusted upper false-event bound \(\le 0.025\).
- Power: at least 2,000 injection draws per declared amplitude cell; cMDE is the smallest amplitude whose revision-adjusted lower bound reaches 0.90.
- Transport: calendar–volume TOST on the single frozen comparison.

Admit only if size, power, and in-scope transport all pass. Otherwise report the mechanically generated withheld disposition. Evaluation span: 2026-06-01 through 2026-07-31, acquired only after this freeze.

## 5. EUR/USD rolling-quantile certificate (second asset-class / detector-family certificate)

### Data

- Source: HistData Generic ASCII EUR/USD tick quotes.
- Development/calibration: 2021-05-30 through 2025-12-31.
- Repository-unseen evaluation: 2026-01-01 through 2026-07-31.
- Stream monthly downloads into normalized partitions. Do not load entire tick months into memory.
- Validate: monotonic UTC timestamps, `ask >= bid > 0`, duplicates, weekend/session gaps, tick counts, row totals, raw/derived hashes.

### Frozen detector

- 60-minute rolling RMS volatility.
- 43,200-minute percentile window.
- LOW/ELEVATED boundary \(0.75\).
- EXTREME boundary \(0.95\).
- Causal rolling mode and strict EXTREME boundary.
- Decision horizon \(q=60\) minutes.
- Clocks: calendar, tick-count, and intrinsic-variance. Alternative-clock thresholds are calibrated on development data only; the resulting numbers are written into the certificate spec before evaluation.

### Power

- Six-hour multiplicative volatility bursts.
- Scale grid \(\{1.25, 1.5, 2, 3, 4\}\).
- 2,000 draws per cell.
- 60-minute response period allowed.
- Success: EXTREME occupancy of at least 50% over the remaining episode.
- cMDE: smallest scale whose revision-adjusted lower confidence bound reaches 0.90.

### Size

- Identical episode event on non-injected stationary-block null paths.
- Pass only if the revision-adjusted upper false-event bound is at most \(\alpha_0=0.025\).

### Transport

- Map alternative-clock labels causally back to calendar time.
- Test calendar–tick-count and calendar–intrinsic pairs.
- Overall disagreement at most 10%; EXTREME-indicator disagreement at most 5%.
- Dependence-aware 95% equivalence intervals; Holm adjustment across the four declared comparisons.

### Information

- Target: whether next-hour realized volatility exceeds its causal trailing-30-day 75th percentile.
- Evaluate at non-overlapping hourly decision epochs.
- Stationary-block information reference only, at the revision-specific evidence threshold.

### Value

- Exposure multipliers LOW/ELEVATED/EXTREME: \(Q=(1.0, 0.5, 0.0)\).
- Baseline: constant \(Q=1\).
- Utility: \(u_t=Q_t r_{t,t+60}-\frac{\gamma}{2}(Q_t r_{t,t+60})^2-c_t\).
- \(\gamma=1/\mathrm{Var}(r_{t,t+60})\) on development data; freeze the number before evaluation.
- Turnover cost: \(|\Delta Q_t|(\tfrac12\mathrm{spread}_t+1\,\mathrm{bp})\).
- Pass only if the revision-adjusted lower stationary-block confidence bound on utility improvement over baseline exceeds zero.

Generate a complete certificate regardless of outcome.
