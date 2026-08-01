---
status: draft
plan: 11-01
authored: 2026-06-11
frozen_before_results: true
ratified: true
---

# v4.0 Pre-Registration: Calibrated Detector and Exclusion Bound

> **INTEGRITY CONTROL (D-09/D-12/D-13/D-15).** This document freezes the
> scientific parameters for v4.0 before any Phase 11-15 code touches real
> 2021-2025 BTCUSDT data. It covers signal injection, time-gauge invariance,
> thermodynamic profit bounds, persistence-null testing, paper structure, and
> provenance requirements. It contains no computed variance-ratio value, no
> `z_m2` statistic, no `cascade_fired` rate, no `P_det`, no `delta_star_90`,
> and no exclusion-gap measurement from real 2021-2025 data. The only numeric
> values in this file are frozen design constants, formulas, and thresholds.
> The git commit hash of this file is the no-HARKing anchor for the entire
> v4.0 milestone. Downstream Phase 11-15 reports must record that hash as
> `prereg_commit`, and real-data computations must fail closed unless the hash
> exists and its commit timestamp predates the run timestamp.

---

## 1. Data Span

*Carried forward from the earlier pre-registrations. The span is restated here
so this document is self-contained.*

- **In-scope development/run span:** 2021-01-01 through 2025-12-31 inclusive.
- **Data source:** available 1-minute OHLCV BTCUSDT bars under
  `data/binance_futures/symbol=BTCUSDT/year=2021` through `year=2025`.
- **Permitted fields:** timestamp, open, high, low, close, and volume.
- **Forbidden fields:** order book, individual trades, taker-buy flow, funding,
  open interest, liquidation flow, news, social data, and exchange metadata not
  already present in the OHLCV store.
- **2026 holdout:** the `year=2026` partition is untouched during Phases 11-14.
- **2026 holdout use:** loaded exactly once in Phase 15, after the paper draft is
  written, for the confirmatory holdout run.
- **Development rule:** code and unit tests may use synthetic data after this
  freeze commit; real 2021-2025 data may be used only after the freeze commit
  exists and the D-09 guard verifies it.
- **No data loaded in this document:** this file was authored as a specification.
  It does not load, inspect, summarize, or compute over the 2021-2025 data.

---

## 2. Pinned (W, q) Grid and Primary Cell

*The variance-ratio grid is carried forward unchanged and is the common grid for
the calibrated-detector result.*

```
WQ_GRID = {(60, 5), (120, 5), (240, 15)}
PRIMARY_WQ = (120, 5)
```

| Setting | W (bars) | q (bars) | Role | W/q |
|---------|----------|----------|------|-----|
| (60, 5) | 60 | 5 | Robustness fast cell | 12.0 |
| (120, 5) | 120 | 5 | PRIMARY headline cell | 24.0 |
| (240, 15) | 240 | 15 | Robustness slow cell | 16.0 |

- **Units:** W and q are measured in bars of the active time gauge.
- **Clock-time gauge:** one bar is one 1-minute OHLCV bar.
- **Volume-time gauge:** one bar is one aggregated equal-volume bar.
- **Intrinsic-time gauge:** one bar is one aggregated equal-realized-variance bar.
- **Stability criterion:** each W/q pair is at least 10, satisfying the
  Lo-MacKinlay non-overlapping-block stability rule.
- **Primary reporting:** `(W=120, q=5)` supplies the headline number.
- **Surface reporting:** all three WQ cells are reported in the exclusion surface.

---

## 3. Signal Injection Specification

### 3.1 Estimand

The signal-injection estimand is the full protocol's detection probability:

```
P_det(delta_target, q_inject, W, q_vr)
    = Pr(full frozen cascade fires | injected AR(1) signal of target size delta_target)
```

The result is an instrument-characterization curve, not a trading strategy and
not a claim about realized predictability.

### 3.2 Injection Parameterization

- **Injected signal family:** positive AR(1) mean predictability only.
- **AR(1) coefficient:** `phi > 0`.
- **Negative phi:** excluded from Phase 11; not tested, not reported.
- **Innovation normalization:** innovations are orthogonalized using
  `sqrt(1 - phi^2)` so the innovation variance remains stable as phi changes.
- **Volatility scaling:** injected signal is scaled by a deterministic
  GARCH(1,1) conditional-volatility path, `sigma_t`.
- **GARCH source:** `arch` library version 7.2.0.
- **GARCH mean specification:** constant mean.
- **GARCH volatility specification:** GARCH(1,1), p=1, q=1.
- **GARCH innovation distribution:** normal.
- **GARCH estimation sample:** full 2021-2025 in-scope log-return series.
- **GARCH estimation timing:** performed only after this freeze commit exists.
- **Sigma path reuse:** one `sigma_t` path is fitted once and reused for all
  Monte Carlo draws in a run.
- **GARCH failure guard:** if maximum-likelihood estimation fails to converge or
  if `alpha + beta >= 0.999`, the volatility path switches algorithmically to
  EWMA.
- **EWMA fallback:** `lambda = 0.94`.
- **Fallback disclosure:** the result JSON and paper must record whether GARCH
  or EWMA supplied `sigma_t`.

### 3.3 Delta Grid

```
DELTA_GRID = {0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10}
```

- `delta_target` is the target median `|VR(q) - 1|` departure produced by the
  injected AR(1) signal.
- The grid has exactly 8 target effect sizes.
- The smallest target is `0.0005`.
- The largest target is `0.10`.
- The grid cannot be altered after this freeze commit.

### 3.4 Injection-Horizon Grid

```
INJECTION_Q_GRID = {2, 5, 15, 60}
```

- `q=1` is excluded as degenerate.
- Each injection horizon is crossed with every `delta_target` and every WQ cell.
- Total grid size is `8 * 4 * 3 = 96` grid points.

### 3.5 Monte Carlo Size and Seeds

```
N_MC = 200
MASTER_SEED = 43
```

- Exactly 200 Monte Carlo draws are run per grid point.
- Total planned cascade calls are `96 * 200 = 19200`.
- All random number generation uses `np.random.default_rng(seed)`.
- Legacy global seeding via `np.random.seed()` is forbidden.
- Per-draw seed is:

```
draw_seed = hash_combine(43, mc_index, grid_point_hash)
```

- `mc_index` is an integer in `[0, 199]`.
- `grid_point_hash` is derived deterministically from
  `(delta_target, injection_q, W, q_vr)`.
- Collection order is not a scientific parameter; `P_det` is a count fraction.

### 3.6 Phi-to-Delta Mapping

- **Primary calibration:** synthetic i.i.d. returns.
- **Validation calibration:** GARCH-matched synthetic returns.
- **Mapping output:** a lookup from `phi` to measured median `delta` for each
  injection horizon q.
- **Inversion:** `invert_phi_delta_mapping(delta_target, q)` returns the
  positive phi that produces the requested target by interpolation.
- **Interpolation direction:** invert the monotone `(delta, phi)` curve.
- **Validation tolerance:** absolute delta-deviation tolerance is `0.00025`.
- **Validation rule:** check the absolute deviation at every crossed
  `(delta_target, q)` grid point.
- **Fallback trigger:** if any validation point exceeds `0.00025`, switch the
  entire mapping to the GARCH-matched calibration.
- **No per-point discretion:** the fallback is all-or-nothing, not a per-grid
  hand edit.
- **No post-result tuning:** tolerance and fallback rule cannot change after
  any real-data injection run has begun.

### 3.7 Full Frozen Cascade

The cascade is the existing v3.0 Pre-Check B machinery, extracted into a pure
function for repeated calls:

1. Fit `RollingQuantileDetector` labels on the supplied close series.
2. Compute VR and `z_m2` over the frozen WQ grid.
3. Draw non-overlapping samples.
4. Compute M2 p-values.
5. Apply Holm correction within the q-family.
6. Produce closure table, horizon profile, noise floor, MDE, and label
   provenance.

The cascade function contains no file I/O, no gate guard, and no provenance
stamping. The caller owns those controls.

### 3.8 Cascade-Fires Rule

```
CASCADE_FIRES = any(Holm-adjusted p < 0.05) AND correct_sign
correct_sign = (VR > 1) for positive phi
ALPHA = 0.05
HOLM_FAMILY_SIZE = 4
```

- The detection test is one-sided in sign because only positive phi is injected.
- A p-value without the correct VR sign is not a detection.
- A correct sign without Holm-adjusted p-value below 0.05 is not a detection.
- Failed numerical draws count as non-detections.
- Degenerate draws count as non-detections.
- Missing output for a draw counts as a non-detection unless re-run and
  completed before aggregation.

---

## 4. Time-Gauge Invariance Specification

### 4.1 Estimand

The gauge-invariance estimand asks whether the efficiency-null conclusion is
stable under three clocks:

1. Clock time.
2. Volume time.
3. Intrinsic time.

The allowed verdicts are:

- `gauge-invariant within margin`
- `gauge violation detected`

Either verdict is valid if the frozen test is run correctly.

### 4.2 Clock-Time Gauge

- Source bars are the original 1-minute OHLCV bars.
- W and q in `WQ_GRID` are interpreted as counts of 1-minute bars.
- This is the reference gauge for comparison.

### 4.3 Volume-Time Gauge

```
VOLUME_THRESHOLDS_BTC = {500, 1000, 2000}
PRIMARY_VOLUME_THRESHOLD_BTC = 1000
```

- Volume bars are built from 1-minute OHLCV volume accumulation.
- Open is the first constituent bar's open.
- High is the maximum constituent high.
- Low is the minimum constituent low.
- Close is the last constituent close.
- Volume is the summed constituent volume.
- The final incomplete bar is discarded.
- If a single 1-minute bar exceeds the threshold by itself, it becomes one
  volume bar and the overshoot is recorded.
- The volume unit is BTC, not USD.
- This is a deliberate OHLCV approximation, not a tick-data volume bar.
- Threshold `500` is a stress case.
- Threshold `1000` is primary.
- Threshold `2000` is a coarser robustness case.

### 4.4 Intrinsic-Time Gauge

```
RV_THRESHOLDS = {0.0001, 0.0005, 0.001}
PRIMARY_RV_THRESHOLD = 0.0005
```

- Intrinsic bars are built by accumulating realized variance from 1-minute
  returns until the active threshold is reached.
- The same OHLC aggregation rules as volume bars apply.
- The final incomplete bar is discarded.
- Threshold `0.0005` is primary.
- Thresholds `0.0001` and `0.001` are robustness cases.

### 4.5 Fixed-Bar and Fixed-Calendar Views

Both views are required:

1. **Fixed-bar-count view:** use the same W and q bar counts across all gauges.
2. **Fixed-calendar-span view:** adjust W and q to approximate the same calendar
   span across gauges.

The fixed-bar view is the primary gauge definition. The fixed-calendar view is
the robustness check that prevents a misleading bar-count-only comparison.

### 4.6 Equivalence Test

- **Test family:** TOST, two one-sided tests.
- **Why TOST:** CI overlap is not an equivalence test and is not allowed as the
  primary gauge-invariance claim.
- **Equivalence margin:** `epsilon = MDE(q, W, n_eff, alpha=0.05, power=0.80)`.
- **MDE source:** same MDE convention used by the Pre-Check B cascade.
- **Per-comparison alpha:** `0.05`.
- **Per-horizon reporting:** all q horizons are reported.
- **Family correction:** Holm correction within the gauge-comparison family.

### 4.7 Gauge Figures

Required figures for Phase 12:

1. Three-panel horizon profile by gauge.
2. Gauge-overlap matrix by q.
3. Volume-time attenuation plot for the short-horizon bid-ask-bounce signature.

---

## 5. Thermodynamic Profit-Bound Specification

### 5.1 Estimand

The thermodynamic bound translates any residual VR departure into an upper bound
on extractable growth, then compares that bound with trading costs.

### 5.2 VR-to-Correlation Approximation

For the Gaussian headline bound:

```
rho_max ~= |VR(q) - 1| / (2 * (1 - 1/q))
I_gaussian = -0.5 * log(1 - rho_max^2)
```

- `I_gaussian` is measured in nats.
- The Gaussian bound is the headline bound because it isolates mean
  predictability by construction.
- The bound is an upper bound, not a realized strategy return.

### 5.3 KSG Estimator

```
KSG_K = 5
KSG_BOOTSTRAP_RESAMPLES = 500
```

- KSG is estimated with `scipy.spatial.cKDTree`.
- KSG is reported for `sign(r_t)` versus `sign(r_{t-q})` as the directional
  nonparametric check.
- KSG on raw GARCH returns may be reported as an upper bound because volatility
  clustering can create raw-return dependence without directional edge.
- KSG on raw returns is not the headline bound.

### 5.4 Kelly Growth Bound

```
G_MAX_NATS_PER_TRADE = I_nats
G_MAX_BPS_PER_TRADE = I_nats * 10000
```

- The Kelly conversion is an information-theoretic upper bound.
- No leverage rule, strategy implementation, or execution model is inferred.
- The paper reports the bound against costs, not an optimized trading strategy.

### 5.5 Cost Schedule

```
TAKER_FEE_BPS_PER_FILL = 4.5
TAKER_ROUND_TRIP_BPS_RANGE = [9.0, 10.0]
MAKER_FEE_BPS_PER_FILL = 1.8
MAKER_ROUND_TRIP_BPS_RANGE = [3.6, 5.6]
```

- Venue reference is Binance BTCUSDT perpetual VIP0 with BNB-discounted taker
  fee for the taker headline.
- Maker costs are a scenario, not the headline.
- If `G_MAX_BPS_PER_TRADE < round_trip_cost_bps`, the verdict is
  `demon runs at a loss`.
- If the bound exceeds costs, the verdict is `bound exceeds costs`.
- Either verdict is publishable and phase-complete if the frozen calculation is
  executed correctly.

### 5.6 Required Figure

The thermodynamic figure is a bound-vs-cost bar chart:

- x-axis: q horizon.
- y-axis: bps per trade.
- bars: Gaussian headline bound and nonparametric directional check.
- reference bands: taker and maker round-trip cost ranges.

---

## 6. Persistence Null Specification

### 6.1 Estimand

The persistence null is a second estimand: first-passage survival under a
martingale-like null.

```
MAX_HORIZON = 120
MARTINGALE_THETA = 0.5
PERSISTENCE_ALPHA = 0.05
```

### 6.2 Survival Function

- For each non-overlapping window, record whether price has crossed its starting
  level by horizon t.
- Estimate survival probability `S(t)` for t from 1 to 120.
- Fit exponent theta in `S(t) proportional to t^(-theta)`.
- Under the martingale null, theta is `0.5`.
- Per-window returns are demeaned before survival computation.

### 6.3 Arcsin-Law Reference

The theoretical reference survival curve is:

```
S0(t) = (2 / pi) * arcsin(sqrt(t0 / t))
```

- `t0 = 1`.
- The continuous-path reference is used as the null shape.
- This is a shape test, not a return forecast.

### 6.4 KS Test

```
PERSISTENCE_BOOTSTRAP_RESAMPLES = 500
PERSISTENCE_BLOCK_RULE = circular block bootstrap
```

- Kolmogorov-Smirnov statistic compares empirical `S(t)` with `S0(t)`.
- Textbook KS critical values are not used.
- The null distribution is built with block-bootstrap simulation under the
  martingale null because per-horizon survival estimates are serially dependent.
- Reject at alpha `0.05` only if the bootstrap p-value is below `0.05`.

### 6.5 Required Figure

The persistence figure overlays:

- empirical survival curve,
- martingale reference curve,
- bootstrap confidence band,
- fitted theta annotation.

---

## 7. Detection-Efficiency and Exclusion-Region Specification

### 7.1 Detection Probability

For every grid point:

```
P_det = number_of_detected_draws / 200
```

- Detected draws are draws where `CASCADE_FIRES` is true.
- Failed draws are included in the denominator and excluded from the numerator.
- Numerical failures must be counted and reported.

### 7.2 Confidence Intervals

```
PDET_CI_ALPHA = 0.05
PDET_CI_METHOD = Clopper-Pearson exact binomial
```

For x detections out of n draws:

```
ci_lo = BetaInv(alpha / 2, x, n - x + 1)
ci_hi = BetaInv(1 - alpha / 2, x + 1, n - x)
```

Boundary cases:

- If `x = 0`, `ci_lo = 0.0`.
- If `x = n`, `ci_hi = 1.0`.

### 7.3 Detection Threshold

```
DETECTION_THRESHOLD = 0.90
delta_star_90(q, W) = min(delta_target where P_det >= 0.90)
```

- If no grid point reaches `0.90`, `delta_star_90` is reported as above grid.
- If multiple grid points reach `0.90`, the smallest such delta is used.
- Monotone PCHIP interpolation may be used for visualization.
- The table value is determined by the grid crossing rule, not by hand-tuned
  interpolation.

### 7.4 Calibration Completeness

Calibration is complete when:

1. All 96 grid-point JSON files exist.
2. Each JSON contains exactly 200 draw records.
3. Every draw is classified as detected, non-detected, or failed-counted-as-
   non-detected.
4. `P_det` is reported for every grid point.
5. Clopper-Pearson 95 percent confidence intervals are reported for every grid
   point.
6. `delta_star_90` is finite or explicitly above-grid for every `(q, W)` cell.
7. The exclusion plot is generated from the JSON files, not from ad hoc arrays.

Violations are documented as limitations. They are not silently repaired by
changing thresholds or adding unregistered grid points.

### 7.5 Exclusion Gap

The exclusion gap is the distance between the detector's calibrated
`delta_star_90` surface and the measured data-region used by the later paper.
It is the result, not a failure condition.

---

## 8. Multiple Testing and Correction Rules

### 8.1 Signal-Injection Cascade

```
CASCADE_ALPHA = 0.05
CASCADE_HOLM_FAMILY_SIZE = 4
```

- Holm correction is applied within the cascade q-family.
- Both raw and Holm-adjusted p-values are recorded.
- The detection rule uses the Holm-adjusted p-values.

### 8.2 Time-Gauge Tests

- Gauge comparisons are Holm-corrected within their comparison family.
- TOST is the primary equivalence test.
- CI overlap can be shown descriptively but cannot determine the verdict.

### 8.3 Persistence Tests

- Persistence uses bootstrap-calibrated KS p-values.
- The alpha level is `0.05`.

### 8.4 Thermodynamic Bound

- The thermodynamic bound is an inequality calculation and cost comparison.
- It is not a discovery p-value family.

---

## 9. Disposition Rules

### 9.1 Signal Injection

- If `P_det` reaches 0.90 at a finite delta for each required cell, report the
  finite `delta_star_90` surface.
- If a cell does not reach 0.90 on the frozen grid, report `above grid` for that
  cell and treat it as a limitation.
- If P_det is non-monotone across adjacent deltas, report the non-monotonicity
  and use the grid-crossing rule for the table.
- No grid point may be added post hoc to repair a non-monotone or above-grid
  result.

### 9.2 Time Gauge

- `gauge-invariant within margin` and `gauge violation detected` are both valid.
- The paper reports the result of the frozen TOST procedure.
- A gauge violation is not a failed phase.

### 9.3 Thermodynamic Bound

- `demon runs at a loss` and `bound exceeds costs` are both valid.
- The phase succeeds if the chain from VR to MI to Kelly bound to costs is run
  with the frozen assumptions.

### 9.4 Persistence Null

- `consistent with martingale persistence` and `persistence deviation detected`
  are both valid.
- The phase succeeds if the survival and bootstrap-KS procedures are executed
  as frozen.

### 9.5 Paper Structure

- Primary target: Journal of Empirical Finance full article, about 6000 words.
- Derived short form: Finance Research Letters letter, about 2500 words.
- Main figures: 5.
- Required paper elements:
  1. calibrated-detector protocol box,
  2. detection-efficiency/exclusion plot,
  3. gauge-invariance horizon profile,
  4. thermodynamic bound-vs-cost chart,
  5. persistence survival figure.
- The bounded-null voice is mandatory: do not claim that BTC is efficient in
  general; claim only the calibrated limits of this instrument on this data.

---

## 10. Integrity Controls and Downstream Wiring

### 10.1 Freeze-First Ordering

- This file must be committed before any Phase 11-15 real-data computation.
- The file commit is the v4.0 freeze commit.
- Downstream outputs must record the freeze commit hash.
- Any post-freeze edit to this file voids the original freeze unless recorded as
  a forward-only amendment before any affected result is read.

### 10.2 D-09 Gate Guard

The extended gate guard must verify both:

1. `.planning/phases/07-pre-registration-freeze/07-PREREGISTRATION.md`
2. `.planning/phases/11-signal-injection-the-ligo-calibration/09-PREREGISTRATION.md`

The guard fails closed if any path is:

- missing,
- uncommitted,
- committed after the run timestamp,
- not available in git history,
- modified in the working tree relative to the recorded commit.

The guard returns a mapping:

```
{
  prereg_path: commit_hash
}
```

### 10.3 Provenance Stamp

Every real-data JSON artifact from Phases 11-15 carries:

- `prereg_commit`,
- `code_commit`,
- `run_ts_iso`,
- `run_date`,
- `library_versions`,
- `master_seed`,
- `data_span`,
- `holdout_status`,
- `injected` where applicable,
- D-15 provenance stamp.

### 10.4 Output Isolation

- Injection-run JSONs are written under `data/injection_runs/`.
- Injection-run outputs are separate from verdict outputs under
  `backtest_results/precheck/`.
- Synthetic-test outputs must not be mixed with real-data outputs.
- Real-data injection files must identify `synthetic: false`.
- Synthetic test files must identify `synthetic: true`.

---

## 11. NO-PEEKING Clause

1. This file is authored before any Phase 11-15 code runs on real 2021-2025
   BTCUSDT data.
2. No computed VR value from real 2021-2025 data appears in this file.
3. No computed `z_m2` statistic from real 2021-2025 data appears in this file.
4. No cascade-fired count or detection rate from real 2021-2025 data appears in
   this file.
5. No `P_det` value from real 2021-2025 injection runs appears in this file.
6. No `delta_star_90` value from real 2021-2025 injection runs appears in this
   file.
7. No exclusion-gap measurement appears in this file.
8. No gauge-invariance test statistic appears in this file.
9. No thermodynamic-bound value computed from real data appears in this file.
10. No persistence KS statistic or theta estimate from real data appears in this
    file.
11. All thresholds and formulas are frozen before results.
12. 2026 data remains untouched during Phases 11-14.

**Current no-peeking status:** HELD at authoring.

---

## 12. What This File Is Not

This file is a specification. It is not:

- a result report,
- an injection-run report,
- a trading-strategy design,
- a detector bake-off,
- a post-hoc robustness menu,
- a manuscript draft,
- a code implementation plan beyond scientific constants,
- evidence that BTC is or is not efficient.

This file contains no:

- real-data VR result,
- real-data p-value,
- real-data `z_m2`,
- real-data detection probability,
- real-data exclusion threshold,
- real-data gauge statistic,
- real-data mutual information estimate,
- real-data Kelly bound,
- real-data persistence statistic.

The absence of computed real-data values is part of the integrity control.

---

## 13. Ratification Record

**Ratification status:** ratified.

**Human decision:** delegated to AI assistant (Antigravity), approved by user.

**Ratification date:** 2026-06-11.

**Freeze commit:** pending.

**No-peeking attestation:** confirmed.

**Ratification checklist:**

- [x] All required sections are present.
- [x] Delta grid is enumerated.
- [x] Injection q-grid is enumerated.
- [x] WQ grid and primary cell are enumerated.
- [x] `N_MC = 200` is frozen.
- [x] `MASTER_SEED = 43` is frozen.
- [x] GARCH spec is frozen.
- [x] EWMA fallback is frozen.
- [x] Phi-to-delta validation tolerance is frozen.
- [x] Volume thresholds are frozen.
- [x] Intrinsic-time RV thresholds are frozen.
- [x] TOST margin rule is frozen.
- [x] KSG k is frozen.
- [x] Cost schedule is frozen.
- [x] Persistence horizon and bootstrap rule are frozen.
- [x] JSON schema is frozen.
- [x] No computed real-data results appear.

---

## Appendix A: Reference Formulas

### A.1 AR(1) Injection

```
signal_t = phi * signal_{t-1} + sqrt(1 - phi^2) * innovation_t
injected_return_t = return_t + sigma_t * scale(phi, delta_target, q) * signal_t
```

- `innovation_t` is drawn from `default_rng(seed)`.
- `phi` is positive.
- `sigma_t` comes from the frozen GARCH/EWMA rule.
- `scale(...)` is determined by the frozen phi-to-delta mapping.

### A.2 Variance Ratio

```
VR(q) = var_q / var_1
var_1 = (1 / (T - 1)) * sum((r_t - mu)^2)
var_q = (1 / m) * sum((R_t^q - q * mu)^2)
m = q * (T - q + 1) * (1 - q / T)
R_t^q = r_t + r_{t-1} + ... + r_{t-q+1}
```

### A.3 M2 Heteroskedasticity-Robust Statistic

```
z_m2 = sqrt(nq) * (VR - 1) / sqrt(phi_m2)
phi_m2 = sum_{k=1}^{q-1} 4 * (1 - k/q)^2 * delta_k
```

Here `phi_m2` is the Lo-MacKinlay M2 robust asymptotic-variance term, not the
AR(1) injection coefficient.

### A.4 Holm-Bonferroni

```
ordered p-values: p_(1) <= ... <= p_(k)
reject H_(i) while p_(i) <= alpha / (k - i + 1)
```

For the cascade q-family, `k = 4` and `alpha = 0.05`.

### A.5 Clopper-Pearson Exact Interval

```
lo = BetaInv(alpha / 2, x, n - x + 1)
hi = BetaInv(1 - alpha / 2, x + 1, n - x)
```

Boundary cases are frozen in Section 7.

### A.6 Gaussian VR-to-MI Bound

```
rho_max ~= |VR(q) - 1| / (2 * (1 - 1/q))
I = -0.5 * log(1 - rho_max^2)
G_bps ~= I * 10000
```

### A.7 Hash Combine

```
hash_input = f"{master_seed}:{mc_index}:{grid_point_hash}"
draw_seed = first_32_bits(sha256(hash_input))
```

The exact implementation must be deterministic and platform-stable.

---

## Appendix B: Injection-Run JSON Schema

Each grid-point JSON file must contain one object with the following required
fields.

### B.1 Top-Level Fields

| Field | Type | Required | Rule |
|-------|------|----------|------|
| `schema_version` | string | yes | exactly `"v4.0-injection-gridpoint-1"` |
| `injected` | boolean | yes | exactly `true` |
| `synthetic` | boolean | yes | `false` for real-data grid runs |
| `delta_target` | number | yes | member of `DELTA_GRID` |
| `injection_q` | integer | yes | member of `INJECTION_Q_GRID` |
| `W` | integer | yes | W from `WQ_GRID` |
| `q_vr` | integer | yes | q from `WQ_GRID` |
| `phi` | number | yes | positive finite value |
| `N_mc` | integer | yes | exactly `200` |
| `master_seed` | integer | yes | exactly `43` |
| `grid_point_hash` | string | yes | deterministic hash of grid metadata |
| `prereg_commit` | string | yes | commit hash of this document |
| `code_commit` | string | yes | code commit hash for the run |
| `run_ts_iso` | string | yes | ISO timestamp |
| `data_span` | object | yes | start and end dates |
| `holdout_status` | string | yes | must state 2026 untouched |
| `sigma_source` | string | yes | `"garch"` or `"ewma"` |
| `p_det` | number | yes | detections / 200 |
| `ci_lo` | number | yes | Clopper-Pearson lower bound |
| `ci_hi` | number | yes | Clopper-Pearson upper bound |
| `n_detected` | integer | yes | detected draw count |
| `n_failed` | integer | yes | failed draw count |
| `draws` | array | yes | exactly 200 draw objects |

### B.2 Draw Fields

Each draw object must contain:

| Field | Type | Required | Rule |
|-------|------|----------|------|
| `mc_index` | integer | yes | 0 through 199 |
| `seed` | integer | yes | from `hash_combine` |
| `cascade_fired` | boolean | yes | frozen detection rule |
| `correct_sign` | boolean | yes | `VR > 1` for positive phi |
| `holm_min_p` | number or null | yes | null only if draw failed |
| `status` | string | yes | `"ok"` or `"failed"` |
| `failure_reason` | string or null | yes | null when status is ok |
| `per_wq` | object | yes | cascade per-WQ output or empty on failure |
| `holm_correction` | object | yes | cascade Holm output or empty on failure |

### B.3 Validation Rules

- `len(draws) == 200`.
- `n_detected` equals count of draws with `cascade_fired = true`.
- `n_failed` equals count of draws with `status = "failed"`.
- `p_det == n_detected / 200`.
- Failed draws remain in the denominator.
- Every file validates without reading any other grid-point file.
- JSON values are self-contained enough for a referee to recompute the
  aggregate counts.

---

## Appendix C: Meta-Verification Checklist

This appendix is intentionally redundant with Section 13 so automated and human
checks can both inspect the freeze.

1. Data span is 2021-01-01 through 2025-12-31 inclusive.
2. 2026 holdout rule is explicit.
3. `WQ_GRID = {(60, 5), (120, 5), (240, 15)}` appears.
4. `PRIMARY_WQ = (120, 5)` appears.
5. `DELTA_GRID` has 8 values.
6. `INJECTION_Q_GRID` has 4 values.
7. Grid size is stated as 96.
8. `N_MC = 200` appears.
9. Planned cascade calls are stated as 19200.
10. `MASTER_SEED = 43` appears.
11. Positive phi only is stated.
12. GARCH(1,1) with constant mean and normal innovations is stated.
13. `arch` version 7.2.0 is stated.
14. `alpha + beta >= 0.999` fallback guard is stated.
15. `lambda = 0.94` EWMA fallback is stated.
16. Phi-to-delta validation tolerance `0.00025` is stated.
17. Failed draws count as non-detections.
18. Cascade fires requires Holm p below 0.05 and correct sign.
19. Volume thresholds `{500, 1000, 2000}` BTC are stated.
20. Primary volume threshold `1000` BTC is stated.
21. RV thresholds `{0.0001, 0.0005, 0.001}` are stated.
22. Primary RV threshold `0.0005` is stated.
23. TOST replaces CI-overlap as primary gauge-equivalence test.
24. `KSG_K = 5` appears.
25. Cost schedule appears.
26. Persistence max horizon `120` appears.
27. Persistence bootstrap resamples `500` appears.
28. JSON schema is specified.
29. No computed real-data value appears.
30. Ratification is pending until the human approves.
