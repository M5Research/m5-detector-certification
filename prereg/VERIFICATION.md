# Pre-Registration Lineage and Freeze Verification

This directory publishes the frozen pre-registration protocol cited by
**"Certifying Regime Detectors Before Use"** (DOI
[10.2139/ssrn.7100358](https://doi.org/10.2139/ssrn.7100358)) so that the
freeze claims in the manuscript can be read and checked without access to the
private development repository.

## Freeze lineage

| Commit | Timestamp (UTC) | Content |
|---|---|---|
| `720c1d4` | 2026-06-05 09:12:33 | v3.0 pre-check pre-registration (no-HARKing freeze) |
| `dd44e7a` | 2026-06-05 13:22:28 | D-07 operationalization amendment (pre-result, §11.5-permitted) |
| `1dc5c82` | 2026-06-11 19:21:13 | v4.0 pre-registration freeze |

## Freeze-before-run ordering

| Run | Timestamp (UTC) | Cited freeze | Margin | Ordering |
|---|---|---|---|---|
| Pre-Check A | 2026-06-05 13:40:54 | `dd44e7a` | + 18m 26s | holds |
| Pre-Check B | 2026-06-05 21:29:24 | `dd44e7a` | + 8h 06m 56s | holds |
| Holdout confirmatory | 2026-06-23 17:53:08 | `1dc5c82` | + 11d 22h 31m 55s | holds |

Every cited run post-dates the freeze it depends on. The Pre-Check A margin is
deliberately reported at full precision rather than rounded: it is short
(under twenty minutes), and a reader is entitled to see that rather than
infer a comfortable gap.

## What this evidence does and does not establish

**It establishes** that the specification documents in this directory are the
ones committed at the stated hashes, that no computed result from the
2021–2025 sample appears in them, and that the commit ordering recorded in
the development repository is internally consistent with the run artifacts
under `backtest_results/`.

**It does not establish** third-party attestation of the ordering. Git
committer timestamps are self-reported: `GIT_COMMITTER_DATE` can be set to an
arbitrary value at commit time. Git ordering is therefore evidence supplied by
the authors, not an independent record, and readers should treat it as such.

This limitation is the reason for the archival deposit described below, and it
is stated here rather than left for a reader to discover.

## Archival deposit

This directory is archived at
[**10.6084/m9.figshare.33106934**](https://doi.org/10.6084/m9.figshare.33106934),
which fixes the document content under an independent DOI and makes the
protocol permanently readable.

That deposit is **not** a prospective pre-registration. It was made in July
2026, which post-dates the Pre-Check runs (2026-06-05) by seven weeks and the
holdout run (2026-06-23) by five, so the deposit date cannot function as the
pre-registration timestamp and is not offered as one. It establishes what the
frozen documents say and that they cannot be silently altered — nothing more.
The manuscript describes it on exactly those terms.

`.zenodo.json` carries the same deposit metadata in machine-readable form and
is retained for any future deposit.

## Independent verification

A reader with access to the development repository can reproduce the table
above:

```bash
for c in 720c1d4 dd44e7a 1dc5c82; do
  git log -1 --format="%h %ct %ci %s" "$c"
done
```

Run timestamps are recorded inside the artifacts themselves, e.g.
`backtest_results/holdout/holdout_confirmatory_20260623_175305.json`, whose
`provenance` block carries both `prereg_commit` and `run_utc`.

## Contents

| File | Purpose |
|---|---|
| `PREREGISTRATION-v3.0-freeze-720c1d4.md` | The frozen v3.0 protocol, numbers-free |
| `FREEZE_ANCHOR.txt` | Machine-readable commit hashes, unix times, and run times |
| `.zenodo.json` | Prepared deposit metadata |
| `VERIFICATION.md` | This file |
