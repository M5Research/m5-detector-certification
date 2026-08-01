---
status: ratified
plan: 07-01
authored: 2026-06-05
ratified: 2026-06-05
ratified_date: 2026-06-05
ratified_by: execution-orchestrator (human-delegated)
frozen_before_results: true
---

> **Which state of this document is this?**
>
> This file is the document **as of amendment commit `dd44e7a`**, not as of the
> freeze commit `720c1d4` its filename names. The difference is a pure append:
> the final "Amendment A1" section (65 lines) was added by `dd44e7a`, and no
> line of the frozen text was modified. Recover the exact freeze state with:
>
> ```bash
> git show 720c1d4:.planning/phases/07-pre-registration-freeze/07-PREREGISTRATION.md
> ```
>
> The filename keeps the freeze hash because that is the anchor the manuscript
> cites; this note exists so the name cannot mislead. `VERIFICATION.md` records
> the amendment and its timing, and `FREEZE_ANCHOR.txt` carries both hashes.



# v3.0 Pre-Check Pre-Registration: Frozen Decision Protocol

> **INTEGRITY CONTROL (D-07/D-08/D-09/D-15).** This document specifies the
> frozen protocol for the two v3.0 pre-checks — **A** (the nested
> transition-vs-levels test) and **B** (the 1-minute M2 VR-significance test).
> It is written and human-ratified **BEFORE any v3.0 pre-check code touches
> 2021–2025 data.** The v3.0 NO-PEEKING scope covers **every computed VR value,
> every M2 z-statistic (`z_m2`), every Wald χ² statistic, every ΔR², every β_T,
> and every regime/transition count from real 2021–2025 BTCUSDT data** — none
> appears anywhere in this file. Phase 8 (Pre-Check A) and Phase 9 (Pre-Check B)
> consume these frozen constants and specs as-is and compute provisional results
> against them; the human then ratifies each verdict against this same frozen
> rule. No spec pivot, no threshold edit, and no free-form post-hoc call is
> permitted after the freeze commit. This pre-registration is the **primary
> no-HARKing control for the entire v3.0 milestone**, and its standalone git
> commit hash is the integrity anchor every downstream v3.0 result must record.

---

## 1. Data Span

*Reused-frozen from v1.0 (`00-PREREGISTRATION.md` §1) and v2.0 (`01-PREREGISTRATION.md` §1). NOT changed.*

- **In-scope for both pre-checks:** 2021-01-01 through 2025-12-31 inclusive (all
  available 1-minute OHLCV BTCUSDT bars,
  `data/binance_futures/symbol=BTCUSDT/year=2021` through `year=2025`).
- **2026 holdout (D-07):** The `year=2026` parquet partition is **UNTOUCHED** —
  not loaded, not inspected, not used in any pre-check computation. **Even a
  descriptive read of the 2026 partition voids the holdout guarantee.** The 2026
  partition was never loaded across v1.0 or v2.0 and must remain so through v3.0.
  No data of any year is loaded within Phase 7 at all.

---

## 2. Pinned (W, q) Grid and Primary Cell

*Reused-frozen grid from v1.0/v2.0 (`00-PREREGISTRATION.md` §2, `01-PREREGISTRATION.md` §2); confirmed in research `SUMMARY.md` line 42. NOT changed.*

```
WQ_GRID = {(60,5), (120,5), (240,15)}
```

| Setting | W (bars) | q (bars) | Role | W/q |
|---------|----------|----------|------|-----|
| (W=60, q=5)   | 60  | 5  | Robustness (fast) | 12.0 |
| **(W=120, q=5)** | **120** | **5** | **PRIMARY** | **24.0** |
| (W=240, q=15) | 240 | 15 | Robustness (slow) | 16.0 |

- **Named primary cell: `(120,5)`** — the 2-hour window, 5-minute horizon. This
  is named explicitly (not "the primary cell") and is the single most
  load-bearing parameter choice in this document. **Rationale:** the v3.0
  research worked-example uses stride W=120 / q=5 (`SUMMARY.md` line 33), and
  `(W=120, q=5)` is the v2.0 Phase-2 design-default **PRIMARY** of this exact
  triple (`01-PREREGISTRATION.md` §2, line 53, where `(W=120, q=5)` is labelled
  PRIMARY). The other two cells `(60,5)` and `(240,15)` are robustness only.
- **Units:** W and q are in **1-minute bars** (the dataset is 1-minute OHLCV
  throughout; not resampled horizons).
- Each pair satisfies the Lo–MacKinlay (1988) stability criterion (W/q ≥ 10
  non-overlapping q-blocks per window).

---

## 3. Pre-Check A Specification — Nested Transition-vs-Levels Test

**Estimand.** Whether the regime-**transition** event adds predictive content
for `|VR(q)−1|` over and above current and lagged regime **levels**.

- **LEVEL model:** `|VR(q)−1|` regressed on dummies for the current regime `R_t`
  plus lagged dummies for the prior regime `R_{t-1}`.
- **TRANSITION model:** the LEVEL model **plus** the transition indicator
  `T_t = 1[R_t != R_{t-1}]`.
- **Test:** **HAC-robust Wald χ²(1)** on the `T_t` coefficient — `statsmodels`
  `sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': L, 'use_correction': True})`
  then `.wald_test(R, scalar=True)` (API confirmed against statsmodels 0.14.6,
  `SUMMARY.md` line 28). **Wald, not LRT.**
- **Verbatim human-ratified kill-criterion** (`docs/research/v3-novelty-assessment.md`
  §5.A, lines 277/279, ratified 2026-06-04):
  > "regress `|VR(q)−1|` on `(R_t, R_{t-1})`, then add `T_t`. If a Wald/LR test
  > fails to reject that `T_t`'s coefficient is zero, direction A **collapses to
  > known regime-level effects**." The transition indicator is a derived variable
  > — a one-lag interaction of the regime labels `(R_t, R_{t-1})` — so the LEVEL
  > model with one lag already *nests* it; A's distinctiveness is contingent on
  > this nested test rejecting.
- **Newey–West lag rule (locked):** primary `L = floor(4*(n/100)^(2/9))`
  (≈ 13 at n ≈ 20,126 non-overlapping samples, stride W=120); report robustness
  at `L = floor(n^(1/3))` (`SUMMARY.md` line 33).
- **Dependent variable:** the rolling `|VR(q)−1|` produced by
  `compute_rolling_predictability(close, W, q)` (`scripts/wp1/predictability.py`
  lines 245–302; causal, warmup bars 0..W-2 = NaN) over the **existing Phase-2
  rolling-quantile regime labels** (`R_t`; label provenance per
  `01-PREREGISTRATION.md` §7.1 — 60-min RV, 30-day rolling percentile,
  p_elevated=0.75, p_extreme=0.95). **No detector is rebuilt; labels are reused.**

---

## 4. Pre-Check A — Degeneracy Guard + Fallback Parameterization

*Open-param PINs #4 (degeneracy cutoff) and #2 (fallback). Source: `SUMMARY.md` pitfall #1 (line 82) + gaps (lines 158, 160); REQ-precheck-prereg.*

- **Rank/condition degeneracy guard (PIN #4 — locked):** because `T_t` is a
  deterministic function of `(R_t, R_{t-1})`, the design matrix
  `X = [R_t dummies, R_{t-1} dummies, T_t]` can be rank-deficient before any test
  runs. The guard declares `X` **degenerate** if
  `np.linalg.matrix_rank(X) < X.shape[1]` (rank deficiency) **OR**
  `np.linalg.cond(X) > 1e10` (a conventional ill-conditioning condition-number
  cutoff). This check runs **before** any Wald test and its result (rank,
  condition number, degenerate flag) is recorded in the JSON output artifact
  (`np.linalg.matrix_rank` / `np.linalg.cond` per `SUMMARY.md` line 29).
- **Escalation/de-escalation fallback (PIN #2 — locked):** when, and **only
  when**, the degeneracy guard trips, the single pooled indicator `T_t` is
  replaced by two direction-split indicators
  `E_t = 1[R_t > R_{t-1}]` (escalation) and `D_t = 1[R_t < R_{t-1}]`
  (de-escalation), sourced from `SUMMARY.md` pitfall #1 and the design in
  REQ-precheck-prereg. The transition increment is then tested as a single joint
  HAC-robust Wald χ²(2) on `(E_t, D_t)` (see §8 — it remains exactly one member
  of the Holm family).
- **`n_transitions >= 100` split gate (locked):** the optional escalation/
  de-escalation split is used only when `n_transitions >= 100` per direction
  survive the minimum-separation filter (§5); otherwise the analysis collapses to
  the pooled `T_t` (`SUMMARY.md` line 160).

---

## 5. Pre-Check A — Minimum-Separation (Window-Contamination) Filter

*Open-param PIN #3 (minimum-separation length). Source: `SUMMARY.md` pitfall #1 (line 82, "post-transition window long enough to measure VR-departure without the next transition contaminating it") + `v3-novelty-assessment.md` §5.A feasibility-tension (iii), line 278.*

- **Minimum-separation length (locked):** two transition events are **too close**
  if separated by fewer than `W` bars (the DV window width). Events closer than
  `W` bars share overlapping `|VR(q)−1|` windows and contaminate one another.
  The filter retains, for each transition, only events whose post-transition
  window of length `W` does not overlap the next transition. The separation
  length is **tied to the DV window `W`** of the cell under test (so it is
  `W=60`, `W=120`, `W=240` for the three grid cells respectively).
- **Under-powered stopping rule (locked):** if fewer than **50** events survive
  this filter per transition type (stratum), that stratum is flagged
  **under-powered** per the stopping rule (`SUMMARY.md` line 159). The survival
  count per transition type is reported in the JSON artifact **before** any test.

---

## 6. Pre-Check B Specification — 1-min M2 VR-Significance

**Estimand.** Whether the HAC-robust M2 variance-ratio statistic departs
significantly from 1 at 1-minute resolution for BTCUSDT.

- **Horizons:** M2 VR-significance at `q ∈ {2,5,15,60}`. **`q=1` is EXCLUDED as
  degenerate** — `VR(1) == 1` identically by construction (`00-PREREGISTRATION.md`
  lines 58–60; `compute_rolling_predictability` raises `ValueError` for q < 2,
  `predictability.py` lines 276–279).
- **Estimator (activation, not new logic):** activate the `z_m2` M2 z-statistic
  that `_vr_m2_kernel(r, q)` already computes and discards in every prior gate
  (`scripts/wp1/predictability.py` lines 41–116; `z_m2` returned at line 115:
  `z_m2 = sqrt(nq) * (vr − 1) / sqrt(phi)`, the heteroskedasticity-robust M2
  asymptotic statistic). Two-tailed p-value via `scipy.stats.norm.cdf`
  (scipy 1.17.1, `SUMMARY.md` line 30).
- **Primary inference: NON-OVERLAPPING windows** (effective sample size =
  `floor(N/W)`) via the WR-01-corrected `non_overlapping_samples(series, regime,
  stride)` (`src/strategies/vol_regime_switch/regime_population.py` lines 20–58;
  strides on the original time-axis index). HAC / **circular block-bootstrap**
  confirmation via the `epsilon_sq_boot_ci(...)` pattern (block-size param,
  `seed=43`; `regime_population.py` lines 81–104) is reported **alongside** as
  CONFIRMATION. **Overlapping-sample t-tests are NOT the primary inference
  device.**
- **Reporting:** global sample **and** per-year breakdown.
- **Microstructure-bias acknowledgment (locked):** OHLCV-only data leaves
  residual **bid-ask-bounce** VR inflation, especially in high-vol regimes
  (`SUMMARY.md` pitfall #4, line 88; `v3-novelty-assessment.md` §5.B/§5.D). Pre-
  Check B therefore **gates Pre-Check A interpretation** and reports the **VR
  horizon profile** (`|VR(2)−1|` vs `|VR(60)−1|`) and the **LOW-regime noise
  floor** (median `|VR(q)−1|` during LOW-regime persistence windows).
- **Append-spike SC1 causal constraint (locked):** the rolling VR kernel MUST be
  causal — **no `pd.rolling(center=True)`**; the append-spike SC1 test
  (`|VR(q)−1|_t` at bar `t` unchanged when a bar is appended after `t`) is
  **mandatory in Phase 9** (`SUMMARY.md` pitfall #6, line 92).

---

## 7. Effect-Size-First Kill/Revival Thresholds

*Statistical significance alone is NEVER sufficient at N ≈ 2.4M. Stated verbatim from `SUMMARY.md` lines 47–50 + REQ-precheck-prereg. Effect sizes (γ̂ / β_T, 95% block-bootstrap CI, ΔR²) are reported FIRST, ahead of any p-value.*

- **A REVIVED** iff: **Wald rejects AND ΔR² ≥ 0.005 AND |β_T| ≥ 0.01 AND stable
  across ≥ 2 `(W,q)` cells.**
- **A KILLED** otherwise — i.e. Wald fails to reject, **OR** Wald rejects but
  (ΔR² < 0.005 AND |β_T| < 0.01).
- **B CLOSED** iff: **`|VR(q)−1| < 0.001` at ALL `q ∈ {2,5,15,60}`** (below the
  v2.0 effect-size floor). **Comparison anchor:** the v2.0 re-gate baseline
  max ε² = 0.001691 — already ~6× below the 0.01 floor
  (`docs/research/v2.0-validation-diagnosis-report.md` §3.2 / Executive Summary);
  cited here as a *published* comparison anchor, not a computed v3.0 result.

The thresholds **0.005** (ΔR²), **0.01** (|β_T|), and **0.001** (B-closure) are
frozen here and cannot change after the freeze commit.

---

## 8. Holm–Bonferroni Correction — LOCKED Trial Count

*Source: `SUMMARY.md` pitfall #5 (line 90, "trial count locked"). The denominators below are EXPLICIT integers and CANNOT change after the freeze commit; enlarging any denominator post-freeze to manufacture or dilute significance is a pre-registration violation.*

Pre-Check A and Pre-Check B are **distinct hypothesis families** (A: does the
transition increment add over levels; B: does VR depart from 1) with
**independent dispositions**, so each is Holm-corrected **within its own family**.

- **Pre-Check A Holm family — exactly `3` tests.** One transition test per
  `WQ_GRID` cell: the HAC-robust Wald χ²(1) on `T_t` at `(60,5)`, `(120,5)`,
  `(240,15)`. **Arithmetic:** 3 cells × 1 transition test/cell = **3**. If the
  §4 degeneracy fallback triggers at a cell, that cell's single pooled-`T_t` Wald
  χ²(1) is **replaced** by a single joint Wald χ²(2) on `(E_t, D_t)` — it remains
  **exactly one** member of the family, so the denominator stays **3**.
- **Pre-Check B Holm family — exactly `4` tests.** One M2 VR-significance test
  per horizon `q ∈ {2,5,15,60}`. **Arithmetic:** 4 horizons × 1 test/horizon =
  **4**.
- **Combined-family robustness view — `7` tests** (`= 3 + 4`), reported as a
  conservative cross-family check. All three integers — **3, 4, and 7** — are
  locked here and unchangeable after the freeze commit.

Both uncorrected and Holm-adjusted p-values are reported; the effect-size
verdict (§7) is primary regardless of p-values.

---

## 9. Disposition Rule

*Source: `SUMMARY.md` pitfall #7 (line 94) + REQ-precheck-prereg + STATE.md ("Positive A → NEW pre-registration, not extension").*

- **Positive A (A REVIVED) → a NEW, independently pre-registered study.** A
  revived Pre-Check A does **NOT** extend this document and does **NOT** trigger
  immediate onset-curve / OOS-forecast / half-life analysis. The terminal paper
  is **NOT** written until that separate, freshly pre-registered study concludes.
- **Both null (A KILLED and B CLOSED) → the terminal honest-null paper** (Phase
  10), framed as the methodological template.
- **Scope fence (explicit):** a significant pre-check becomes a **NEW**
  pre-registration, never a silently-extended study. This fence is the
  Elevation-of-Privilege control against scope creep past the frozen boundary.

---

## 10. Integrity Controls + Downstream Wiring

- **2026-holdout-untouched guard (D-07):** restated — the `year=2026` partition
  is never loaded, inspected, or used across v1.0 + v2.0 + v3.0 (§1). Two-layer:
  no Phase 8/9 code path loads year=2026, and even a descriptive read voids it.
- **D-15 provenance stamp:** every Phase 8/9 JSON output artifact carries the
  D-15 provenance stamp **AND** the git freeze commit hash of this document.
- **Freeze-before-run ordering:** this document is committed in its **OWN
  standalone commit** (exactly one file) **BEFORE** any Phase 8/9 pre-check code
  runs on 2021–2025 data. The git commit timestamp is machine-verifiable by a
  referee (`git log --format=%ct`) and is the no-HARKing anchor for the milestone.
- **D-09 gate-guard reuse (downstream wiring):** Phase 8 and Phase 9 reuse the
  proven `_gate_guard()` from `scripts/wp1/regate_analysis.py` (lines 84–184) by
  changing **only** the `PREREG_PATH` constant (currently line 85) to point at
  `.planning/phases/07-pre-registration-freeze/07-PREREGISTRATION.md`. The guard
  **FAILS CLOSED** (non-zero `SystemExit`) unless this pre-reg commit exists and
  its commit timestamp predates the run, and it stamps `prereg_commit` (the
  `git log --format=%H -1` hash) into every report. **This code change BELONGS TO
  Phase 8/9, NOT Phase 7 — Phase 7 changes no code and `regate_analysis.py` is
  not modified here.**
- **Output location (locked):** all Phase 8/9 pre-check artifacts are written to
  `backtest_results/precheck/` (`SUMMARY.md` line 74; mirrors the v1.0 `gate/`
  → v2.0 `regate/` milestone-isolation pattern).

---

## 11. NO-PEEKING Clause

*Adapted from `01-PREREGISTRATION.md` §14 / `00-PREREGISTRATION.md` §7.*

1. This file is written and human-ratified (Task 1–2 of Plan 07-01) and committed
   in its **own dedicated commit** BEFORE Phase 8 (Pre-Check A) or Phase 9
   (Pre-Check B) writes or runs any code on real 2021–2025 BTCUSDT data.
2. The v3.0 NO-PEEKING invariant covers **every** computed VR value, **every**
   M2 z-statistic (`z_m2`), **every** Wald χ² statistic, **every** ΔR², **every**
   β_T, and **every** regime/transition count from real 2021–2025 data — none
   appears in this file.
3. Phase 8 and Phase 9 depend on this freeze commit; they are blocked from
   touching 2021–2025 data until this commit exists and predates the run. Pre-
   check code may be developed and validated on **simulated data only** until
   then (`SUMMARY.md` pitfall #3, line 86).
4. **No spec or threshold in this document may be changed after any Phase 8/9
   result has been read** — doing so voids the pre-registration.
5. Any edit after authoring but BEFORE the freeze commit (or, exceptionally,
   before any Phase 8/9 result is read) must be recorded as a versioned amendment
   here, dated, with a no-peeking attestation that no computed 2021–2025 result
   had been read at amendment time.
6. The freeze-first ordering is **git-verifiable**: the commit hash of this
   document (recorded as `prereg_commit` in every Phase 8/9 report by the D-09
   gate-guard) must timestamp-predate the first pre-check run.

**The no-peeking invariant currently HOLDS:** as of authoring, no Phase 8/9
pre-check code has run on real 2021–2025 data; no VR value, no `z_m2`, no Wald
χ², no ΔR², no β_T, and no regime/transition count from real data has been
computed.

---

## 12. What This File Is Not

This document contains pre-committed specifications, thresholds, and integrity
controls **only**. It contains:

- **NO** computed VR value from real 2021–2025 data.
- **NO** Wald χ² statistic from a real run.
- **NO** ΔR² from a real run.
- **NO** β_T (transition coefficient) from a real run.
- **NO** M2 `z_m2` value from real data.
- **NO** regime count or transition count from real data.

It is a **specification**, not a result. **The absence of any computed number is
itself the integrity guarantee.** The only numbers permitted here are the frozen
spec constants (the `WQ_GRID`, the q-grid, the thresholds 0.005 / 0.01 / 0.001,
the locked Holm denominators 3 / 4 / 7, the lag-rule formula, the `1e10`
condition-number cutoff, the `n_transitions >= 100` / `< 50`-event gates) and the
v2.0 baseline ε² = 0.001691 cited as a *published* comparison anchor. All
computed pre-check values belong exclusively in the Phase 8/9 JSON artifacts
under `backtest_results/precheck/`.

---

## 13. Ratification Record

**Ratified:** 2026-06-05

**Decision:** ratify-as-proposed (no spec or threshold edits).

**Authority:** Human researcher — ratified-as-proposed via the GSD execute-phase
blocking-human checkpoint on 2026-06-05.

**Rationale:** This document freezes every Pre-Check A and Pre-Check B spec, the
effect-size kill/revival thresholds, the locked Holm trial count, the disposition
rule, and every integrity control BEFORE any v3.0 pre-check code touches
2021–2025 data. Every pinned value is sourced from a cited artifact — the
verbatim human-ratified kill-criteria in `v3-novelty-assessment.md` §5.A/§5.B,
the HIGH-confidence frozen values in research `SUMMARY.md`, and the proven v2.0
Phase-2 design-default primary `(120,5)` from `01-PREREGISTRATION.md` §2 — not
from any observed 2021–2025 statistic. The two immutable load-bearing choices —
the named primary cell `(120,5)` and the locked Holm denominators (3 / 4 / 7) —
were scrutinized at this checkpoint and ratified without edit. The freeze-first
commit ordering is the primary no-HARKing control for the entire v3.0 milestone:
the git commit hash of this file is the integrity reference every Phase 8/9
report records via the D-09 gate-guard. No tightening or loosening of any
threshold is warranted — all values are pre-specified to their design defaults.

**No-peeking invariant:** HELD at ratification. No Phase 8/9 pre-check code had
run on real 2021–2025 BTCUSDT data; no computed VR value, no M2 `z_m2`, no Wald
χ² statistic, no ΔR², no β_T, and no regime/transition count from real data had
been produced; Phases 8 and 9 had not executed at the time of ratification. The
no-peeking invariant is verified and confirmed.

---

## Appendix: Reference Formulas (for Phase 8/9 implementers)

*Recorded for traceability. These are formulas, NOT computed results.*

**Variance ratio (Lo–MacKinlay 1988, 1990 erratum):**
```
VR(q) = var_q / var_1
  var_1 = (1/(T-1)) * sum((r_t - mu)^2)
  var_q = (1/m)     * sum((R_t^q - q*mu)^2)
  m     = q*(T-q+1)*(1 - q/T)
  R_t^q = r_t + r_{t-1} + ... + r_{t-q+1}   (q-period overlapping return)
  T     = W (window size in bars)
```

**Predictability DV:**
```
predictability_t = |VR(q) - 1|
```

**M2 heteroskedasticity-robust z-statistic** (matches `predictability.py` line 115):
```
z_m2 = sqrt(nq) * (vr - 1) / sqrt(phi)
  nq  = W - 1                              (price increments in the W-bar window)
  phi = sum_{k=1}^{q-1} 4*(1 - k/q)^2 * delta_k   (M2 robust asymptotic variance)
```

**Newey–West HAC lag rule:**
```
L = floor(4*(n/100)^(2/9))     (primary)
L = floor(n^(1/3))             (robustness)
  n = number of non-overlapping samples (effective sample size, = floor(N/W))
```

**Holm–Bonferroni:** ordered p-values p_(1) ≤ ... ≤ p_(k); reject H_(i) while
p_(i) ≤ α / (k − i + 1). Family sizes locked in §8: k = 3 (Pre-Check A),
k = 4 (Pre-Check B), k = 7 (combined robustness).

---

## Amendment A1 (2026-06-05) — D-07 operationalization of '>= 2 (W,q) cells' [pre-result, §11.5-permitted]

**Permitting clause.** §11.5 of this document explicitly permits a pre-result versioned
amendment, provided it is dated, carries a no-peeking attestation confirming that no computed
2021–2025 result has been read at amendment time, and is recorded as a separate, later commit
whose timestamp still predates the first Phase 8 real-data run.

**Scope.** This amendment operationalizes **only** the phrase "stable across ≥ 2 (W,q) cells"
in the §7 revival rule. It alters NO frozen threshold. The following values are unchanged:
- Effect-size floor ΔR² ≥ **0.005** (frozen in §7)
- Effect-size floor |β_T| ≥ **0.01** (frozen in §7)
- Holm denominator for Pre-Check A = **3** (frozen in §8)
- Grid **WQ_GRID = {(60,5), (120,5), (240,15)}** (frozen in §2)
- Named primary cell **(W=120, q=5)** (frozen in §2)

None of these constants is loosened, tightened, or re-derived. The only thing this amendment
adds is the definition of how the three grid cells combine when evaluating "stable."

**D-07 — Operationalization of "stable across ≥ 2 (W,q) cells."**

A REVIVED requires BOTH:

(a) the **primary cell (120,5)** passes the full conjunction:
    Wald rejects ∧ ΔR² ≥ 0.005 ∧ |β_T| ≥ 0.01

AND

(b) **≥ 2 of the 3 grid cells** each independently pass that same full conjunction
    (Wald rejects ∧ ΔR² ≥ 0.005 ∧ |β_T| ≥ 0.01).

A KILLED otherwise (i.e., primary (120,5) fails the conjunction, OR fewer than 2 grid
cells pass the conjunction — including the case where the primary cell passes but both
robustness cells fail).

This is the **most conservative reading**: it simultaneously honors the named primary cell
(condition a) and the literal "≥ 2 cells" text (condition b), making A the hardest direction
for a referee to attack. No threshold is modified; only the combinatorial rule for the three
cells is defined.

**NO-PEEKING attestation.**

At the time this amendment was authored, no real 2021–2025 BTC result of any kind had been
read or computed. Specifically: no Pre-Check A regression output, no Wald χ² statistic, no
ΔR² value, no β_T (transition coefficient), no regime/transition count from real 2021–2025
BTCUSDT data, and no run of `precheck_a.py` against real data has been examined. The
rationale for this operationalization draws exclusively on the frozen spec text in this
document (§2, §7, §8) and the pre-registered design context in
`.planning/phases/08-pre-check-a-nested-transition-vs-levels/08-CONTEXT.md` — no observed
statistic, no preliminary result, and no data-derived quantity informed any aspect of the
D-07 rule stated above.

**NO-PEEKING invariant: HELD at amendment authoring.** The 2026 holdout remains untouched.

**Freeze-commit integrity.** Freeze commit `720c1d4` is left untouched. This amendment is
appended as a forward-only addition to this document — no existing frozen section has been
modified or reflowed. This amendment will be committed as a **separate, later commit** whose
timestamp still predates the first Phase 8 real-data run (`precheck_a.py` must not run on
real 2021–2025 data until this amendment commit exists and its timestamp predates the run —
the no-peeking gating invariant). The amendment commit hash will be recorded as
`phase8_amendment_ref` in the Phase 8 JSON output artifact and in `08-VERDICT.md`.

**Authored:** 2026-06-05
