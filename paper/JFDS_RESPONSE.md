# JFDS reviewer response

Title used everywhere: **Certifying Regime Detectors Before Use**.

The reviewer's former Table 6 is now Table~\ref{tab:families} (three-family exhibit). New tables were inserted for the certificate schema, power surface, transport, information, and the historical repaired record, which shifted numbering. The rolling-quantile row of that table is **incomplete**.

## Issue 1 — Completeness and rolling-quantile disposition

Missing required gates now produce `incomplete` even when a measured gate has already failed. Observed failures remain in the failure profile. The historical rolling-quantile exhibit had no power, size, or transport measurement, so its disposition remains `incomplete`. The negative value result is retained only as a failure-profile entry. The new EUR/USD rolling-quantile certificate is a separate, complete protocol-v5 record.

## Issue 2 — Permutation information evidence

Certificate-grade information inference is stationary-block, not IID permutation. New schemas and outputs contain no `permutation_p`. Legacy permutation files are byte-unchanged and labeled `legacy_noncertificate`.

## Issue 3 — Gauge R&R related work

NIST Gauge R&R and ISO 5725-1:2023 are cited as prior measurement-system practice. Novelty is bounded: no identified financial framework jointly binds power, size, transport, target information, and decision value to one frozen detector-output claim with refusal precedence.

## Issue 4 — Title capitalization

The title remains exactly `Certifying Regime Detectors Before Use` in LaTeX, package metadata, Zenodo, and supporting records. Local checks enforce capitalization without network dependence.

## Issue 5 — Post-hoc \(q=2\) admission

The repaired \(q=2\) record is `superseded_exploratory`. It is not defended as a first admissible certificate. A protocol-v5 successor with \(\alpha_0=0.025\), frozen recentering constants, calendar/volume scope, and previously repository-unseen 2026-06-01 through 2026-07-31 BTC data replaces it. The successor is complete and is withheld as `target_mismatched`.

## Issue 6 — Complete EUR/USD rolling-quantile certificate

EUR/USD is the second asset-class/detector-family certificate. Protocol v5 froze the detector, clocks, gates, and numerical rules before evaluation-span download. The completed certificate is withheld as `size_distorted`; its information and value gates pass, but refusal precedence prevents those downstream results from overriding empirical-size failure.

## Certificate lineage

| Record | Parent | Evidence revision | \(\alpha\) | Disposition |
| --- | --- | --- | --- | --- |
| Historical BTC VR cascade (v4 freeze) | — | pre-v5 | 0.05 nominal | withheld (size/transport/power/value profile) |
| Repaired \(q=2\) calendar-volume | historical BTC | exploratory / unallocated | none | `superseded_exploratory` |
| BTC \(q=2\) successor | repaired record | 0 | 0.025 | `target_mismatched` |
| EUR/USD rolling quantile | — | 0 | 0.025 | `size_distorted` |

The machine-readable lineage is `provenance/certificate-lineage.json`; artifact and
specification hashes are enforced by artifact tests.
