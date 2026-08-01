# Exploratory Addendum Pre-Registration — δ-Grid Extension & Cross-Detector Characterization

**Authored:** 2026-06-17 (working date, not independently verifiable)  
**First committed:** 2026-06-22, commit `9daf993`, in the same commit that added
the first five exploratory result artifacts (`inj_d0.15_q2_W{60,120,240}.json`,
`inj_d0.15_q5_W{60,120}.json`).  
**Parent freeze:** v4.0 instrument-characterization freeze commit `1dc5c82`  
**Scope:** Post-freeze exploratory analyses only; confirmatory claims remain frozen at the original 96-cell grid.

> **Ordering disclosure.** An earlier version of this header read "Date:
> 2026-06-17 (committed before any exploratory run)". Git does not support
> that: there is no commit of this file on 2026-06-17, and its first commit
> also introduces five of its own eight result artifacts. **The commit
> ordering does not establish that this addendum predates its own runs.**
>
> What git does establish is that the confirmatory/exploratory boundary is
> intact: the 96 confirmatory cells were committed on 2026-06-15 (`58226a9`),
> a week before this document and its results appeared. Nothing in the
> manuscript treats these cells as confirmatory or as preregistered; they are
> reported as post-freeze exploratory throughout, which is the correct
> handling and does not depend on this document's date.
>
> This disclosure is stated here rather than left for a reader to find, on the
> same reasoning as the self-reported-git-timestamp limitation in
> `prereg/VERIFICATION.md`.

## Motivation

The confirmatory injection grid (δ ≤ 0.10) lies predominantly **below** the i.i.d. VR median sampling floor at q=5, W=120 (≈ 0.145). Zero detections there are partly mechanical. This addendum extends δ upward and applies the characterization protocol to standard detector families.

## Exploratory Estimand A — Extended δ grid

- **Cells:** δ ∈ {0.15, 0.20, 0.30, 0.50} × q ∈ {2,5,15,60} × W ∈ {60,120,240} (48 cells)
- **MC draws:** N_MC = 200 (same as confirmatory)
- **Primary cell:** (W,q) = (120,5)
- **Disposition:** Report P_det(δ) curves, first δ with P_det > 0, and bracket δ*_90 = min{δ : P_det(δ) ≥ 0.90} or state δ*_90 > δ_max if not reached
- **Driver:** `python scripts/wp1/signal_injection.py --addendum --from-precomputed data/injection_runs/precomputed.npz`
- **Provenance stamp:** separate JSON block labeled `exploratory_addendum: true`

## Exploratory Estimand B — Cross-detector protocol

Apply estimands (2)–(4) where applicable to:

1. **Markov-switching** (Hamilton 1989; statsmodels `MarkovRegression`, 2-state variance-switching)
2. **HMM regime detector** (Kim 1994 / statsmodels filtered probabilities; repository `HMMDetector`)

Estimand (1) injection exclusion applies only to the Holm-corrected VR cascade (primary instrument). Cross-detector rows in the comparison table report literature capability and exploratory label-level projections (gauge, persistence) where run.

## Firewall rules

- Exploratory outputs MUST NOT overwrite confirmatory JSON under `data/injection_runs/inj_d{≤0.10}_*.json`
- Manuscript labels: **confirmatory** (§2–§5 frozen results) vs **exploratory** (§8–§9 addendum)
- No retrospective re-labeling of confirmatory runs

## Exploratory Estimand C — Direction-specificity (negative phi)

- **Cells:** primary cell (W,q)=(120,5); delta in {0.10, 0.15, 0.20}; phi_inj = -phi(delta,q)
- **Disposition:** document P_det and median Z_q; expect zero detections via sign gate
- **Driver:** `python scripts/wp1/negative_phi_probe.py`
- **Artifact:** `data/injection_runs/negative_phi_probe_primary.json`

## Exploratory Estimand D — Robustness probes

- GARCH scaling: `scripts/wp1/garch_filter_sensitivity.py`
- MI bootstrap block length: `scripts/wp1/mi_bootstrap_sensitivity.py`

## Holdout

2026 data remain reserved; exploratory runs use 2021–2025 only.
