# Claims Guardrail

This file keeps the paper **"Certifying Regime Detectors Before Use"** from drifting back
into weaker or riskier framings. The method is a pre-use **certification protocol**
(detector admissibility certification). "GCDE" / "gauge-calibrated detector exclusion" is
the legacy repository name and survives only as provenance.

## Primary Claim

Allowed:

> We introduce a pre-use certification protocol for regime-detector outputs that, for a
> single frozen claim, measures power, empirical size, cross-clock transport,
> detector-output information, and net information value, and returns an auditable
> certificate with a disposition and a gate-level failure profile.

Allowed:

> Certification asks whether a detector output is admissible as evidence for a declared
> claim; it does not compare detectors or rank performance.

Not allowed:

> We discovered Bitcoin inefficiency.

Not allowed:

> We built a profitable volatility-regime strategy.

Not allowed:

> We show the VR cascade is the best detector.

## Sensitivity and cMDE Claims

Allowed:

> The preregistered 96-cell injection grid excludes detection of variance-ratio departures
> `delta <= 0.10` at every window-horizon cell (0 of 200 fires per cell).

Allowed:

> Disclosed post-freeze positive controls fire at `q=2, delta=0.15` and `q=5, delta=0.30`
> and remain silent at `q=5, delta=0.15` and `q=5, delta=0.20`; these eight cells are
> exploratory, not confirmatory.

Allowed:

> The certified minimum detectable effect (cMDE, `delta_90`) is the smallest
> declared-family amplitude the frozen detector recovers with probability at least 0.90 on
> the executed grid, at the declared asset, frequency, and horizon.

Not allowed:

> The detector cannot detect variance-ratio structure.

Not allowed:

> Zero detections globally prove absence of regimes.

Not allowed:

> A `cMDE` / `delta_90` value beyond the executed amplitude grid, or for horizons, clocks,
> or signal families not run.

## Transport (Cross-Clock) Claims

Allowed:

> Output-statistic equivalence is certified at `q=2` across calendar, volume, and
> event-time bars, and is not certified at `q=5` for the same three scheme pairs.

Allowed:

> Detector claims calibrated in one market clock are not certified portable to another
> without a fresh transport check. ("clock" is the artifact field name; use
> "calendar/volume/event-time" in prose.)

Not allowed:

> Clock (gauge) invariance.

Not allowed:

> Calendar labels are false because volume labels differ.

Not allowed:

> Transport disagreement proves detector failure.

## E-value / Anytime-Valid Claims

Allowed:

> Casting each testing gate as an e-value gives static admission control by the
> intersection-union test with no multiplicity penalty across gates, and anytime-valid
> admission by Ville's inequality under optional stopping and unbounded monitoring.

Allowed:

> The information gate's permutation p-values calibrate to e-values `E=2.17` at `q=2` and
> `E=15.8` at `q=5`; both fall short of the `1/alpha = 20` admission threshold at
> `alpha=0.05`.

Not allowed:

> The e-value bound proves profitability or market inefficiency.

Not allowed:

> Anytime-valid admission is an unconditional market guarantee. (It holds under the
> declared empirical sequential null.)

## Information and Cost Claims

Allowed:

> The raw sign-pair artifact is a market-information scale diagnostic, not
> detector-contingent information.

Allowed:

> Economic admissibility requires `I(D_t; Y_{t+q})`, not merely `I(sign r_t; sign r_{t+q})`.

Allowed:

> The frozen VR trigger has small but resolvable detector-contingent MI at `q=2` and `q=5`;
> the net available-work bound is negative per-epoch under the 10 bps convention and its
> sign flips under per-trigger attribution — a convention dependence the certificate
> exposes rather than averages away.

Allowed:

> Saturated Monte Carlo trigger cells with `0/200` or `200/200` fires have zero within-cell
> path-level trigger entropy, so within-cell trigger MI is zero.

Allowed:

> If detector-contingent available work is non-positive after costs under the declared
> convention, detector-only use is economically excluded under that model.

Allowed:

> EUR/USD robustness requires bid/ask data (HistData Generic ASCII ticks); daily FRED or
> ECB reference rates are not cost-gate evidence. The current EUR/USD appendix is a
> BTC/ETH-matched-span cost-gate sample, not a full second-asset admissibility certificate.

Not allowed:

> Positive mutual information proves profitability.

Not allowed:

> Raw sign-pair information proves detector value.

Not allowed:

> A cost bound is realized PnL.

## Repair and Admissible Certificate

Allowed:

> Recentering the decision statistic on its empirical null — by exact offline
> re-thresholding of the frozen per-draw statistics, with no new simulation — lowers the
> cMDE roughly tenfold (`0.15 -> 0.02` at `q=2`, `0.30 -> 0.10` at `q=5`) and restores
> size control.

Allowed:

> The repaired detector still abstains on real BTC (the recentered observed statistic is
> negative), converting an underpowered withhold into a powered exclusion, and yields the
> protocol's first `admissible` disposition for a bounded, scoped `q=2` scientific claim.

Not allowed:

> The recentering constants are confirmed. (They are post-freeze / exploratory; a
> preregistered confirmatory rerun with the constants frozen is required.)

Not allowed:

> `admissible` means profitable, generally valid, or an economic certificate. (It is a
> bounded scientific admission for one claim tuple.)

## Benchmark Claims

Allowed:

> Detectors should be compared only after validity gates, common-target eligibility, and
> timestamp alignment; the three-family exhibit reports three different dispositions for
> three different reasons, not a ranking.

Allowed:

> A detector can be excluded from a common volatility-state benchmark because it measures
> serial dependence rather than volatility state (target-mismatched). Instrument failure is
> a valid benchmark outcome.

Allowed:

> A native-frequency benchmark may truncate the time domain, but it must not reinterpret
> `q=5` by downsampling bars.

Not allowed:

> Low agreement means one detector is wrong.

Not allowed:

> The benchmark is a trading horse race.

Not allowed:

> ML/HMM/GARCH/quantile combinations are the novelty.

## Novelty Wording

Strong wording:

> The contribution is a detector-admissibility certification operator: a frozen claim
> tuple, a characterization vector of measured operating characteristics, and a disposition
> generated by an explicit precedence rule.

Strong wording:

> Certification turns detector labels into admissible or inadmissible measurements for a
> declared target, horizon, sampling scheme, and cost model.

Weak wording:

> We propose a protocol combining several validation checks.

Weak wording:

> The strength of the paper is its honesty.

Replacement:

> The strength of the paper is that it defines and estimates the operating limits of
> detector outputs before those outputs are admitted as evidence.
