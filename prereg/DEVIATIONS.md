# Deviations from the Frozen Protocol

This document lists every point at which the executed analysis departs from the
pre-registered protocol, and every point at which the manuscript attributed a
choice to the freeze that the freeze does not actually fix.

It exists because the difference between a declared deviation and one a
reviewer finds unaided is the difference between a rigorous study and a
suspicious one. Each item below is individually defensible. None of them would
be, discovered without warning.

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
