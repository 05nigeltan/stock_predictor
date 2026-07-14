# Spec v2.1 — Finalization Amendments (2026-07-13)

Deltas against the v2 one-pager. Everything not listed here carries over
from v2 unchanged. These were finalized interactively before any Phase 2
code was written; the rationale for each is logged below the table.

| # | Item | v2.1 (final) |
|---|------|--------------|
| 1 | **Annualization** | `IR = mean(α_d) / std(α_d) · √252`, computed on **daily calendar-time active returns** (see #3). v2's `√(52/5)` was a bug: it under-annualized by √5, which would have silently broken every comparison against literature IRs. |
| 2 | **Position rule** | **Long/flat.** beat→+1, flat→+1, lag→0. No shorting in the primary. |
| 3 | **Holding construction** | **5 overlapping tranches**, 1/5 capital each, one tranche opens per day and holds 5 trading days (Jegadeesh–Titman calendar-time). Daily active return: `α_t = (mean of the 5 live tranche signals − 1) · r_t − cost_t`. |
| 4 | **Costs** | One-way 2.5bps (= 5bps round trip). Charged per tranche on every change in that tranche's position: `cost_t = (2.5bps / 5) · Σ_j |Δs_t^j|`. The always-long benchmark pays zero. |
| 5 | **Inference** | Moving-block bootstrap (block = 10 trading days) is primary; Newey–West (lag ≥ 5) reported alongside. Handles both the tranche-induced overlap and the non-normality of α. |
| 6 | **PT variant** | Generalized **m-class Pesaran–Timmermann** on the full 3×3 prediction-vs-outcome table, all test periods (n ≈ 503). Not the binary version, not fired-subset. |
| 7 | **Activity floor** | A strategy must have `s_t = 0` on ≥10% of tranche-days to be evaluable (guards the degenerate near-zero-std IR). `power.py` reports `IR_min` at 10/20/30% activity; the floor is revisited only if power.py shows 10% is unpriceable. |
| 8 | **Null model in power.py** | Zero-skill signals with persistent runs (Markov, mean run length 5 days) and the drift drag included — E[α | no skill] < 0 because any deviation from long forfeits positive drift. IID null reported as a sensitivity. |
| 9 | **Joint power** | The 80%-power / α = 0.10 two-sided requirement applies to the **conjunction** (IR ≥ IR_min ∧ ≥8/10 folds ∧ PT p < 0.10), Monte-Carloed jointly — not to each gate separately. |
| 10 | **Sliding-window variant** | Fixed train length = the first expanding fold's span (2005→2011, ~7y). Reported alongside expanding, never primary. |

---

## Rationale log

### A9 · Position mapping: literal ternary → long/flat

**Superseded:** beat→+1, flat→0, lag→−1 (v2's literal reading).

**Why:** the label lives in *excess-vs-drift* space; the position pays *raw
return*. Two mismatches follow. (a) flat→0 is economically irrational: a
flat prediction says "return ≈ drift," and drift is positive — the model
predicts a normal up-week and the rule responds by exiting, paying the
drift drag on 34% of periods with no hypothesis behind it. (b) lag→−1
shorts on a signal that only says "below drift": with drift ≈ +30bps and
τ ≈ 70bps, a marginal lag call has expected raw return ≈ −40bps — barely
clears costs, and milder lag predictions are outright positive. Shorting
bets on a different event than the one the model was trained to predict.

**Consequence, stated honestly:** deviations now come only from lag calls,
so the effective sample is ~150 independent active periods, not v2's
estimated ~250. The MDE will be correspondingly larger. The long/short
variant (beat/flat→+1, lag→−1) may be *reported* as a secondary line, but
the contract is long/flat.

### A10 · Non-overlapping periods → calendar-time tranches

**Superseded:** "pooled across non-overlapping 5-day periods" (v2), which
leaves the phase offset unstated. §2.2 already documented a 1.5pp baseline
swing from that exact arbitrariness. The tranche construction uses every
day, removes the phase choice from the primary metric entirely, and its
induced autocorrelation is handled by the block bootstrap (#5), which v2
already mandated for the overlap problem anyway.

### A11 · Annualization bug

`√(52/5) ≈ 3.22` treats "52/5" as periods per year. A 5-trading-day period
recurs ~50.4 times per year (252/5), so the factor is `√50.4 ≈ 7.10`; on
daily tranche returns it is `√252`. Caught at spec review, before any
number was computed against it. Cost of the error: zero. Cost if uncaught:
every reported IR understated 2.2×, and `IR_min` mispriced against the
literature in the direction that makes an undetectable criterion look
achievable.

### A12 · Fold gate recalibrated 8/10 → 6/10  (2026-07-14)

**Superseded:** "mean net active return > 0 in ≥8/10 folds" (v2, via A8).

**Why it died:** the 6→8 raise in A8 was calibrated against a coin-flip
null — P(≥6/10 | p=0.5) = 37.7% — but power.py showed the coin flip is the
wrong null. Under the true null (zero skill **with the drift drag**), the
per-fold probability of positive α ranges from 6% (2017) to 49% (2018):
deviating from long in a relentless bull year is nearly unwinnable without
skill. Against that null, ≥8/10 has a 0.0% false-pass rate and
single-handedly pushed the joint criterion's MDE to IR ≈ 0.96 — detectable
but unachievable, v1's failure in a different costume. ≥6/10 under the
true null: 2.1% false-pass alone, ≤0.2% jointly — still far inside the
α = 0.10 budget.

**Replaced by:** ≥6/10, restoring the gate to its designed false-positive
budget under the correct null. A calibration fix, not a lowered bar: the
joint false-pass rate remains ~0.1%.

**Cost of the error:** one diagnostic run. Same lesson as A4: a threshold
is only as meaningful as the null it was calibrated against.

---

## power.py results — FROZEN (seed 20260713, gate ≥6/10, run 2026-07-14)

Full pre-registered record in `power_results.md`. The numbers that bind:

| activity | IR_min (luck bar, null 95th pct) | MDE (true IR @ 80% power) | lag-call hit @ MDE | verdict |
|---|---|---|---|---|
| 10% | **0.06** | 0.46 | 56.9% (vs ~30% base) | marginal |
| 20% | −0.11 | 0.46 | 54.4% | marginal |
| 30% | −0.24 | 0.47 | 53.4% | marginal |

- `IR_min` is evaluated at the model's **realized** activity (nearest MC
  level above). Joint false-pass under the null: ≤0.2% at every level.
- MDE ≈ 0.46 at every activity level — the criterion detects only effects
  at the optimistic edge of the published literature (IR ~0.4). A true IR
  of 0.3 would be caught well under 80% of the time. **This is the honest
  best this dataset supports** — accepted with eyes open, per the
  build-order decision point, rather than papered over.
- The kill criterion (≤25 experiments / 8 weeks) is unchanged and now has
  a defensible statistical floor under it.

---

## Build order (frozen)

1. `splitter.py` — date-based purged walk-forward. 10 expanding folds,
   1-year test windows ending 2021-12-31, 5-day purge computed against the
   panel calendar (label span must not intersect the test window), 2-day
   embargo. Date-based, never positional: rows may be absent from the
   modelling matrix (VIX drops). Sliding variant per #10.
2. `power.py` — on the splitter's real folds: zero-skill Monte Carlo of
   the joint criterion (#8, #9), MDE table for each candidate metric
   (v1 Δ, m-class PT, mean-α t-stat, active-return IR, DM on P&L), and the
   `IR_min` curve vs activity (#7). Output pre-registered into the spec
   before any feature or model exists.
3. **Decision point:** if no metric's MDE lands below a plausible effect
   size (IR ~0.4 is the optimistic edge of the literature), the
   pre-registered response is the negative-result write-up or a change of
   problem (longer sample / cross-sectional pooling) — not a lowered bar.
   **→ RESOLVED 2026-07-14:** after the A12 recalibration, MDE ≈ 0.46 —
   marginal, at the plausible edge. Proceeding, eyes open.
4. Only then: remaining Phase 2 (validate.py, FRED VIX ingest,
   point-in-time recompute test) and Phase 3 features.

**Phase 3 pre-registration (2026-07-14):** all feature diagnostics, drift
plots, and feature-selection decisions use data ≤ 2021-12-31 only. The
handoff's "plot every feature over the full history" would otherwise
include the locked holdout — choosing features by their behaviour over
2022+ is selection on holdout data, and no mechanical check catches it.

**Phase 3 drift review (2026-07-15):** 21 features built and registered
(`make_features.FEATURES`); all pass the PIT harness. Drift screen
(`reports/feature_drift.{md,png}`, train halves, flag at |shift| > 0.25
pooled sd) flagged 6: `vol_10d`, `vol_21d`, `vix_lag1` (regime-persistent
BY DESIGN — they are the conditioning variables; the "drift" is
volatility clustering, i.e. the hypothesis), and `ret_63d`,
`sma20_to_sma200`, `rsi_14` (half-sample means differ because the halves'
regime composition differs — GFC in the first, low-vol bull in the
second; each is relative/bounded with no monotone march). All 21
retained. No feature shows mechanical non-stationarity.
