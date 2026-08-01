# Deviations from the Frozen Protocol

This document lists every point at which the executed analysis departs from the
pre-registered protocol, and every point at which the manuscript attributed a
choice to the freeze that the freeze does not actually fix.

It exists because the difference between a declared deviation and one a
reviewer finds unaided is the difference between a rigorous study and a
suspicious one. Every item below is individually defensible; none of them
would be if a reader met it without warning.

Read alongside `PREREGISTRATION-v3.0-freeze-720c1d4.md`,
`PREREGISTRATION-v4.0-freeze-1dc5c82.md` and `VERIFICATION.md`.

Every item states: what was frozen, what was executed, what the impact on a
published claim is, and how it is resolved.

| ID | Subject | Changes a published number? |
|---|---|:--:|
| [D1](#d1-equivalence-margin) | Equivalence margin | **yes** |
| [D2](#d2-grid-structure-and-the-lo-mackinlay-guideline) | Grid structure | no |
| [D3](#d3-bootstrap-block-length) | Bootstrap block length | no |
| [D4](#d4-seed) | Seed | no |
| [D5](#d5-holm-family-size-for-the-transport-gate) | Holm family size | no |
| [D6](#d6-artifact-schema) | Artifact schema | no |
| [D7](#d7-sigma_t-source-never-disclosed) | `sigma_t` source | no |
| [D8](#d8-one-exploratory-cell-is-a-clone) | Cloned cell | no |
| [D9](#d9-addendum-scope-8-of-48-cells) | Addendum scope | no |
| [D10](#d10-binomial-interval-method) | Binomial interval method | **yes** |
| [D11](#d11-gate-guard-inactive-during-the-grid-run) | Gate guard | no |
| [D12](#d12-persistence-null-did-not-simulate-the-null) | Persistence null | **yes** |
| [D13](#d13-the-repaired-instrument-is-size-controlled-only-at-the-primary-window) | Repair size scope | no |
| [D14](#d14-the-48-repair-had-no-generator) | Repair had no generator | no |
| [D15](#d15-confidence-intervals-were-clamped-to-contain-their-point-estimate) | Clamped intervals | no |
| [D16](#d16-the-eth-replication-carries-the-same-superseded-margin) | ETH margin field | no |

---

## D1 — Equivalence margin

**Frozen.** v4.0 §4.6: `epsilon = MDE(q, W, n_eff, alpha=0.05, power=0.80)`,
with the MDE convention of the Pre-Check B cascade,
`MDE = (z_{1-alpha/2} + z_{power}) / sqrt(n_nl)`.

**Executed.** The original gauge run (`gauge_report_20260612_110425.json`,
2026-06-12) implemented the rule and recorded
`epsilon = 0.019748090241536544`, the MDE at the calendar gauge's
`n_nl = 20126` non-overlapping windows at `W = 120`. The published run
(`gauge_report_20260624_221834.json`, 2026-06-24) replaced it with a hardcoded
constant `PRACTICAL_TOST_EPSILON = 0.02`, and `compute_tost_epsilon()` was
rewritten to discard its argument and return that constant. A unit test was
added asserting the substitution (`eps == PRACTICAL_TOST_EPSILON` and
`eps != mde`).

**Impact.** The margins differ by 1.3%, and that decides one row:

| Margin | q=2 clock–intrinsic Holm p | Verdict |
|---|---:|---|
| 0.02 (rounded constant) | 0.0497 | certified |
| 0.019748 (preregistered MDE) | 0.0682 | **not certified** |

The q=2 transport verdict therefore moves from three of three scheme pairs
certified to two of three. The manuscript asserted the stronger version in five
places. No other row changes verdict, and no q=5 or q≥15 conclusion is
affected.

**Context that cuts both ways.** The 2026-06-24 regeneration was not gratuitous:
it fixed a real and serious TOST defect. Every Holm-adjusted p-value in the
2026-06-12 artifact is 1.0, including comparisons that should be overwhelmingly
equivalent (diff 0.0093 against se 0.0015). That fix was necessary and correct.
The margin substitution rode along with it, in the same regeneration, without
being recorded. Separately, the project's own earlier manuscript prose
described 0.019748 as the frozen margin "reported as 0.020 in rounded prose" —
so the intent was that 0.02 was a presentational rounding, and it became the
test threshold by accident of implementation.

**An ambiguity in the frozen rule, declared.** `MDE(q, W, n_eff)` is written as
a function of the cell, which could be read as one margin per pair. Read that
way, the event-time pairs have only 454 windows and their margin would be
`2.801585 / sqrt(454) = 0.1315` — about seven times wider, under which every
comparison certifies trivially and the gate stops discriminating anything. A
single margin anchored to the calendar gauge is the conservative reading, it is
what the original run implemented, and it is the reading adopted here. This
choice is a declared interpretation, not something the freeze settles.

**Resolution.** `compute_tost_epsilon()` restored to compute the MDE.
`scripts/wp1/migrate_gauge_mde_margin.py` recomputes the TOST and the Holm
correction from the frozen per-comparison differences and standard errors — no
new simulation, no resampling — and writes `gauge_report_mde_margin.json`
beside the published artifact, which is left unmodified. The manuscript's
transport table, its five headline statements, and the two certificate tables
are corrected. The unit test now asserts the preregistered behaviour.

**Open for author decision.** The first "admissible" certificate (§4.8) rested
in part on transport certifying across all three schemes at q=2. With two of
three, the manuscript now narrows that certificate's declared scheme scope to
calendar-and-volume rather than claiming a full transport pass. A marked
comment at that point in `main.tex` records the alternative — demoting the
disposition to transport-uncertified — as an explicit choice still to be
ratified.

---

## D2 — Grid structure and the Lo–MacKinlay guideline

**Frozen.** v4.0 §2 defines `WQ_GRID = {(60,5), (120,5), (240,15)}` — three
(W,q) pairs — and invokes the Lo–MacKinlay stability guideline `W/q >= 10`.

**Executed.** The full Cartesian product `W ∈ {60,120,240} × q ∈ {2,5,15,60}`,
i.e. 12 combinations, with the cascade evaluating all four horizons inside each
window.

**Cause.** The preregistration contradicts itself: §3.7 computes over
`WQ_GRID` (3 cells) while §3.8 fixes `HOLM_FAMILY_SIZE = 4`. A Holm family of 4
cannot be formed over a grid of 3. The Cartesian reading is the only one that
makes both sections executable.

**Impact.** Five of the twelve combinations fall below `W/q >= 10`: (60,60)
with ratio 1, (120,60) with 2, (240,60) with 4, (60,15) with 4, and (120,15)
with 8. Those rows sit outside the regime where variance-ratio sampling theory
is well behaved. All of them are silent (0/200), so no conclusion turns on
them, but their evidential weight is lower than the compliant rows'.

**Resolution.** Declared here and in a footnote to the manuscript's power
table.

---

## D3 — Bootstrap block length

**Frozen.** §6.4 fixes `PERSISTENCE_BOOTSTRAP_RESAMPLES = 500` and the circular
block rule. It does **not** fix the block length.

**Executed.** `block_size = 120`.

**Impact.** None on any result. The manuscript presented the block length as
part of the frozen claim tuple, which overstated what the freeze fixed.

**Resolution.** The manuscript now labels the block length a post-freeze
implementation choice.

---

## D4 — Seed

**Frozen.** §3.5 fixes `MASTER_SEED = 43` for the injection grid.

**Executed.** The injection grid uses 43. `persistence_report_20260612_214557.json`
records `seed = 42`.

**Impact.** None on any result; both runs are deterministic and the seed is
recorded in the artifact. The manuscript's sentence "all artifacts pin these
commits and seed 43" was simply false for that artifact.

**Resolution.** The manuscript now states seed 43 for the injection grid and
seed 42 for the persistence null.

---

## D5 — Holm family size for the transport gate

**Frozen.** v4.0 §8.2 requires Holm correction "within their comparison family"
but never fixes the integer. By contrast v3.0 §8 did fix its families
explicitly ("exactly 3", "exactly 4", "locked here and unchangeable") and
warned that enlarging a denominator post-freeze is a violation; v4.0 is weaker
on this specific control.

**Executed.** 12 — three scheme pairs times four horizons.

**Impact.** None on any verdict, and the direction is conservative: a larger
family makes certification harder, not easier.

**Resolution.** Declared. `nested_test.apply_holm` now raises if it is handed
more p-values than the frozen denominator, so the failure mode this control
exists to prevent cannot occur silently.

---

## D6 — Artifact schema

**Frozen.** v4.0 Appendix B requires `schema_version`, separate `injection_q`
and `q_vr`, `p_det`, `n_detected`, `n_failed`, and per-draw records carrying
`status`, `failure_reason`, `correct_sign` and `holm_min_p`.

**Executed.** No `schema_version`; `injection_q` and `q_vr` collapsed into a
single `q`; `P_det` and `n_fires` instead of `p_det` / `n_detected`; and
**no `n_failed` and no per-draw `status`**.

**Impact — the one that matters.** §3.8 rules that failed numerical draws count
as non-detections. Without `n_failed` or `status`, a cell reporting 0/200
because 200 draws genuinely did not detect is **indistinguishable in the
published artifact** from one reporting 0/200 because draws failed
numerically. The entire confirmatory result is a uniform 0/200, so this is
precisely the check a sceptical reader will want to run, and it cannot be run
from the artifacts as published.

**Mitigation available now.** The per-draw records are complete and each
carries `cascade_fired` and its own seed, so the draw count is verifiable
(`tests/wp1/test_frozen_artifacts.py` recounts all 104 cells and confirms
`n_fires` and `P_det`). What cannot be reconstructed is the failed-draw count.

**Resolution.** Declared. The grid is deterministic and takes about 5h28m of
CPU to regenerate (per `low_impact_run.log`); regenerating it with the
Appendix B fields emitted is the clean fix and is recommended, but it is not
done here because it would replace frozen evidence artifacts, which is an
author decision rather than a maintenance one.

---

## D7 — `sigma_t` source never disclosed

**Frozen.** §3.2 is imperative: "the result JSON and paper must record whether
GARCH or EWMA supplied `sigma_t`".

**Executed.** The field appears in no artifact, and the manuscript describes
the fallback rule without ever saying which branch ran.

**Impact.** None on the values, but a required disclosure is missing.

**Resolution.** Open. Determining it requires re-running one cell and
inspecting, or reading the run logs; it should be added to the text and, if
D6's regeneration happens, to the artifacts.

---

## D8 — One exploratory cell is a clone

**Fact.** `data/injection_runs/inj_d0.15_q5_W240.json` carries
`provenance.w_invariant_clone = true` and
`cloned_from = inj_d0.15_q5_W120.json`. Its per-draw block is identical to the
source cell's. It is the only clone among the 104 cells.

**Impact.** The manuscript's power table presented `δ=0.15, q=5,
W ∈ {60,120,240}, 0/200 each` as three runs. Two are runs; one is a declared
clone. The 96 confirmatory cells are all genuinely independent — verified by
comparing per-draw blocks across W within each (δ,q) group.

**Resolution.** Footnoted in the manuscript's power table, and
`tests/wp1/test_frozen_artifacts.py` asserts that exactly this one cell is a
clone, so a second one cannot appear unnoticed.

---

## D9 — Addendum scope: 8 of 48 cells

**Registered.** The exploratory addendum specifies
`δ ∈ {0.15, 0.20, 0.30, 0.50} × q ∈ {2,5,15,60} × W ∈ {60,120,240}` = 48 cells.

**Executed.** 8. `δ = 0.50` never ran.

**Impact.** The manuscript is accurate about what was executed and honest that
the response is "unbracketed above on the executed grid", but did not say that
48 were planned. The unexecuted cells are at the higher amplitudes, where
detection is more likely, so the executed subset is conservative with respect
to the exclusion claim and uninformative about the upper brackets.

**Resolution.** Stated in the manuscript's power-table notes.

---

## D10 — Binomial interval method

**Frozen / claimed.** The manuscript labels the injection intervals
Clopper–Pearson, and Proposition 1 asserts they are exact at any `N_mc`.

**Executed.** `clopper_pearson_ci` used `Beta(k+1, n-k+1)` for **both** bounds.
That is the Bayesian posterior interval under a uniform `Beta(1,1)` prior. The
exact Clopper–Pearson interval uses different shapes per bound:
`lo = B⁻¹(α/2; k, n−k+1)`, `hi = B⁻¹(1−α/2; k+1, n−k)`.

**Impact.** For the `k=0, n=200` case that dominates the grid, the upper bound
was 0.018185 where the exact value is 0.018275. The published interval was
marginally **tighter** than the stated method licenses — the error ran in
favour of the paper's own exclusion claim, which is the direction that most
deserves disclosure.

**Resolution.** The function now computes the exact interval, verified against
the closed form `1 − 0.025^(1/200)`. All 104 cells are migrated by
`scripts/wp1/migrate_cp_intervals.py`, which rewrites only the two interval
fields and stamps `ci_schema` in each cell recording the migration. `P_det`,
`n_fires` and every per-draw record are untouched. The manuscript's power table
moves from `[0.0000, 0.0182]` to `[0.0000, 0.0183]` and from `[0.9818, 1.0000]`
to `[0.9817, 1.0000]`.

---

## D11 — Gate guard inactive during the grid run

Recorded in full in `VERIFICATION.md` under "Gate-guard provenance for the
injection grid". In summary: the 96-cell grid ran on 2026-06-14 (run log
15:28:59–20:57:43, "ALL DONE: 96/96") with the pre-registration gate guard
imported but never invoked, because `get_git_commit()` returned the literal
string `bypassed_for_execution`, which was stamped into
`provenance.freeze_commit`.

The artifacts went through three generations in the development repository:

| Commit | Date | numpy | `freeze_commit` |
|---|---|---|---|
| `58226a9` | 2026-06-15 16:45 | 2.4.4 | `bypassed_for_execution` |
| `fa1d20d` | 2026-06-17 09:26 | 2.4.6 | `bypassed_for_execution` |
| `31e3913` | 2026-06-22 17:35 | 2.4.4 | `1dc5c82` |

The published artifacts are the third generation. **Across all three, not one
of the 96 cells changed its `P_det` or `n_fires`.** The `library_versions`
field does change, which is the signature of genuine re-execution rather than a
hand-edited provenance block.

**Impact.** For this grid, freeze-before-run ordering rests on the run log and
commit dates rather than on a runtime assertion captured at the moment of the
run. The provenance blocks were written after the fact.

---

## D12 — Persistence null did not simulate the null

**Claimed.** The persistence diagnostic reports a bootstrap p-value for the
observed first-passage survival curve against the arcsine law, i.e. against the
martingale-difference hypothesis.

**Executed.** `_bootstrap_ks_null` generated its reference distribution by
circular block-resampling **the observed returns**. That preserves the observed
dependence structure within each block rather than imposing H0, so `KS_boot` is
the distribution of the statistic under the observed process, not under the
martingale.

**Impact, measured.** Run on the same simulated data, the two generators
behave completely differently:

| Reference | AR(1) φ=0.4 (H0 false) | GARCH, no sign dependence (H0 true) |
|---|---|---|
| sign-flip (correct) | mean p = 0.024, rejects 5/6 | 0/6 false rejections |
| block-observed (legacy) | mean p = 0.442, rejects **0/6** | 0/6 false rejections |

The legacy reference has essentially **no power** against the alternative the
arcsine law is a hypothesis about. Under a plain random walk the two agree,
because an iid series already satisfies H0, which is why the existing unit
tests never caught it.

The published artifact nonetheless reports `bootstrap_p_value = 0.0` and
`persistence_deviation_detected`. Both facts hold at once: the legacy test
cannot detect sign dependence, yet it rejected on real BTC data. The rejection
must therefore come from structure that block resampling at `L = 120` destroys
— i.e. behaviour at scales beyond the block length — and not from the
hypothesis under test. The published p-value does not mean what its name says.

**Scope.** The persistence diagnostic is secondary. It supports no gate in the
certificate and no headline claim; it appears as a diagnostic and in the ETH
appendix.

**Resolution.** The default null is now `sign_flip`
(`r*_t = s_t · r_t`, `s_t` iid Rademacher), which preserves `|r_t|` and hence
volatility clustering while destroying sign dependence — exactly the
martingale-difference hypothesis. The p-value now uses the `(1+k)/(1+B)`
convention, so it can no longer be exactly zero. The legacy generator remains
reachable as `null="block_observed"` so the published artifact stays
reproducible. Calibration and power tests are added, including one that
demonstrates the two side by side.

**Open.** `persistence_report_20260612_214557.json` was produced under the
legacy null and has **not** been regenerated: that needs the BTC parquet data,
which is not redistributable in this package. Its `bootstrap_p_value` should
be read as the legacy quantity until it is rerun.

---

## D13 — The repaired instrument is size-controlled only at the primary window

**Found while writing the missing generator for §4.8**
(`scripts/wp1/reference_repair.py`), not previously recorded anywhere.

The repair's out-of-sample false-alarm rate, read from the frozen artifact:

| Cell | OOS size |
|---|---:|
| W60, q=2 | 0.12 |
| **W120, q=2** | **0.02** |
| W240, q=2 | 0.08 |
| W60, q=5 | 0.01 |
| W120, q=5 | 0.07 |
| W240, q=5 | 0.06 |

The manuscript quotes 0.02 for the admissible certificate. That is the W120
q=2 cell and it is correct. But at the other two q=2 windows the OOS size is
0.12 and 0.08, both above the nominal 0.05.

**Impact.** No published number is wrong. What was understated is how much
work the certificate's `W=120` scoping does: the repaired instrument is *not*
size-controlled across the window grid, only at the window the certificate
declares. A reader could reasonably have assumed the repair generalised.

**Resolution.** Recorded here, asserted in
`tests/wp1/test_reference_repair.py`, and added to the manuscript's
admissible-certificate scope row.

---

## D14 — The §4.8 repair had no generator

**Claimed.** `README.md` describes the repair as an "exact offline
re-thresholding of the frozen per-draw `median_z_m2`", i.e. fully reproducible
with no new simulation.

**Executed.** No script producing
`recentered_reference_repair_20260710.json` existed in either repository. This
was the sharpest asymmetry in the package: every negative result was
reproducible from versioned code, and the single positive one — the protocol's
first admissible certificate — was not.

**Resolution, and its limit.** `scripts/wp1/reference_repair.py` now implements
the repair: empirical-null estimation from the δ=0.0005 calibration cell,
recentring, one-sided re-thresholding of the frozen draws, cMDE and δ₅₀
extraction, and the split-half out-of-sample size. It is covered by tests
including an end-to-end reproduction of a known threshold on synthetic data.

It cannot regenerate the frozen artifact from this repository, because the
per-draw `median_z_m2` values it consumes are not published: the injection
cells carry only `cascade_fired`, `holm_ordered`, `mc_idx` and `seed`, and
`precomputed.npz` holds the injection inputs rather than the per-draw
statistic. This is the same root cause as D6. Emitting `median_z_m2` per draw
when the grid is next regenerated closes D6 and D14 together.

What can be checked today is checked: `reference_repair.py --verify` audits
every relation internal to the frozen artifact — that each cell's cMDE and δ₅₀
are exactly the thresholds implied by its own P_det table, that the amplitude
grids agree, that the recentring is not a no-op, and that the repair never
made power worse. All six cells pass. That is not proof the artifact was
produced by this procedure, and the tool says so rather than implying
otherwise.

---

## D15 — Confidence intervals were clamped to contain their point estimate

**Executed.** Three bootstrap routines forced `lo <= point <= hi` after
computing percentile bounds: `nested_test.beta_t_boot_ci`,
`vr_significance.median_vr_dep_boot_ci` and
`regime_population.epsilon_sq_boot_ci`. The third carried the comment
"required by acceptance criteria".

**Why it matters.** A percentile interval that excludes its own point estimate
is a diagnostic: it says the bootstrap distribution is biased. Clamping
distorts the nominal coverage and removes the signal. In a study whose thesis
is that instrument properties must be measured before use, bending an interval
to satisfy an acceptance criterion is the one move that cannot be defended.

**What the clamp was hiding.** Removing it immediately failed a test, which is
the finding. On the three-well-separated-group fixture,
`epsilon_sq_boot_ci` returns point `0.8888` against interval
`[0.8499, 0.8886]` — the point sits *above* the upper bound. That is genuine
and reproducible: epsilon-squared is bounded above by 1, the fixture sits near
that ceiling, and block resampling can essentially only move the statistic
down, so the percentile interval is biased low exactly where the statistic is
most informative.

**Impact.** No published number changes. `epsilon_sq_boot_ci` supports the
regime-population diagnostic, not a certificate gate.

**Resolution.** Clamping removed from all three. Each now emits a
`RuntimeWarning` naming the bound and the point estimate when the interval
excludes it. The test that asserted the clamped ordering now asserts the bias
directly, with a tolerance, so the bias is documented rather than hidden and a
change in its size will fail. A BCa interval would be the principled fix if
these intervals ever become load-bearing.

---

## D16 — The ETH replication carries the same superseded margin

**Executed.** `eth_replication_20260625_224506.json` records
`tost.epsilon = 0.02`, i.e. the same rounded constant corrected in D1, rather
than the preregistered MDE.

**Impact.** None. ETH's calendar gauge has the same non-overlapping window
count as BTC's (20126 at `W=120`), so the preregistered margin is the same
0.019748, and recomputing the full 12-comparison TOST and Holm family under it
returns an **identical certified set**: calendar-volume at q=2, q=5 and q=60,
and volume-event-time at q=2. No ETH verdict moves.

**Resolution.** Recorded here rather than silently corrected, since the
artifact's verdicts stand. `fig02_transport_forest` draws both panels against
the preregistered margin so the two assets are judged by one rule.

---

## What this document does not contain

No deviation listed here changes the direction of any conclusion. The
confirmatory result remains a uniform 0/200 across all 96 cells; the size gate
remains conservative; the value gate remains cost-dominated under the frozen
convention. D1 and D10 change published numbers, and D1 changes one gate
verdict from "certified" to "not certified".

The audit that produced this list also checked what is *not* here. All 109
artifacts carrying a freeze reference were examined for ordering violations:
there are none, and every `code_commit` also post-dates the freeze its artifact
cites. The three freeze commits each touch exactly one file and were never
amended afterwards. The nine detector modules are byte-identical between the
development and public repositories. No `P_det` changed across any of the three
regenerations of the grid.
