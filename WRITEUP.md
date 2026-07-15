# Short-Term Reversal on SPY Is Real, Linear, and Unpayable

**A pre-registered negative result.** SPY, 5-day horizon, 2005–2026.
Nigel Tan, July 2026.

---

## Abstract

We tested whether 5-day short-term reversal on SPY, conditioned on
volatility regime, can beat buy-and-hold net of costs. It cannot — and
the interesting part is *how* it fails. The directional signal is
**real**: it passes the Pesaran–Timmermann test out-of-sample on every
model tried (p ≤ 0.02, 2012–2021), and the primary model's PT skill
persisted on a locked 2022–2026 holdout (p = 0.04) evaluated exactly
once. It is **linear**: logistic regression, random forest, and
LightGBM converge to the same 39–40% ternary accuracy; boosting's early
stopping quits after ~100 stumps of a 2,000-tree budget. It is
**unconditioned**: the volatility-interaction that was supposed to make
it tradeable — the core of the hypothesis — does not exist (per-σ SHAP
response of the reversal signal is flat across vol regimes), and the
models' confidence tails carry no extra precision (top-decile lag calls:
33.5% precise vs. the 51.5% the power analysis requires). Every
pre-registered route to monetization was tested and priced; the best
configuration loses ~0.6%/yr net against buy-and-hold on the holdout.
Four experiments of a 25-experiment budget were spent; the project was
stopped by **mechanism exhaustion, not budget exhaustion**. Three
success-criterion bugs and one simulator lookahead were caught by
pre-registration machinery *before* any model existed; the supersession
log (A1–A13) documenting them is, deliberately, half the contribution.

---

## 1. Hypothesis (pre-registered, Phase 1)

Weak-form efficiency treated as approximately true; the tested
exception: **5-day reversals are stronger in high-volatility regimes**
(overreaction + liquidity imbalances), so a tree model could learn
*when* reversal signals are reliable. Falsification pre-registered
(§1.5): if SHAP shows no importance for `ret_5d` or a positive
contribution from recent returns, the hypothesis dies. Kill criterion:
≤25 experiments or 8 weeks.

## 2. Methodology (the part that did the work)

- **Label:** ternary on excess-vs-drift return, vol-scaled threshold
  (τ = 0.4σ₅d) — chosen by train-side class-balance sweep, never by
  test performance.
- **Criterion (v2.1, after three logged failures):** annualized IR of
  net active return of a long/flat tranche book (5 overlapping
  tranches, 2.5bps one-way) vs. always-long, ≥6/10 walk-forward folds
  positive, m-class PT p < 0.10. The criterion is *label-blind*: it
  scores dollars, so no labeling choice can move the goalposts.
- **Power analysis before any model:** on the real fold structure,
  IR_min (95th-pct luck bar) = 0.06 at 10% activity; **MDE at 80%
  power ≈ IR 0.44**, requiring ~51.5% lag-call precision vs. a ~30%
  base — at the optimistic edge of published 5-day index-timing
  effects. The criterion could have detected a plausible effect; the
  zero-skill joint false-pass rate is ≤0.2%.
- **Leak controls, all mechanically tested:** date-based purged/
  embargoed walk-forward splitter (17 tests incl. gapped-index and
  hand-computed ground truth, mutation-tested); point-in-time recompute
  harness over the whole pipeline (~20 truncated-history rebuilds; a
  BACKWARD/FORWARD column registry every feature must join); NYSE-
  calendar validation; VIX lagged one session (0/5,412 sessions
  missing, measured not asserted).
- **Sample honesty:** 4,211 overlapping train windows ≈ **842
  independent** observations; pooled test 2012–2021 ≈ 503; holdout
  2022–2026 ≈ 225. All inference on independent units or block
  bootstrap (10-day blocks).

## 3. Results

### The bar, then the ladder (pooled walk-forward test folds, 2012–2021)

| configuration | acc | activity | IR [90% CI] | folds+ | PT p |
|---|---|---|---|---|---|
| random floor (Σp²) | 33.5% | — | — | — | — |
| B5 anti-persistence rule | 36.0% | 35% | −0.77 [−1.30,−0.29] | 2/10 | **0.01** |
| B6 depth-3 tree | 35.8% | 34% | −0.81 [−1.39,−0.30] | 3/10 | 0.02 |
| E1 logistic | **39.0%** | 26% | **−0.49** [−1.10, 0.03] | 2/10 | 0.02 |
| E2 random forest | 39.8% | 24% | −0.65 [−1.21,−0.09] | 5/10 | 0.00 |
| E3 LightGBM (frozen cfg) | 39.1% | 16% | −0.57 [−1.12,−0.06] | 3/10 | 0.02 |
| E4 LightGBM, top-decile tail | 40.0% | 20% | −0.57 [−1.04,−0.11] | 4/10 | 0.00 |

Three findings carry the story:

1. **Direction was never the problem.** Even a one-line rule (predict
   the opposite of last week's class) has significant directional skill
   out-of-sample. The Phase 1 "suggestive, not significant" reversal
   probe (+1.8 SE in-sample) was confirmed at p = 0.01.
2. **No nonlinear lift.** Linear, bagged, and boosted models converge
   on one accuracy. Early stopping wanted 10–251 trees of 2,000.
3. **The tail is empty (E4).** Fired on its top-decile-confidence lag
   days, the model was 33.5% precise vs. 30.0% base — one sixth of the
   +21.5pp the MDE demands. The model knows *that* reversal exists,
   not *when* it is strong.

### The falsification test (§1.5, out-of-sample SHAP, A1)

- **Direction: confirmed.** All five position-cluster features have
  negative correlation with their own BEAT-class SHAP (ret_5d −0.70):
  high recent returns uniformly reduce predicted outperformance.
- **Importance: weak.** `ret_5d` ranks 12/21; the top attributions are
  the conditioners themselves (`sma20_to_sma200`, `vol_21d`).
- **Vol strengthening: absent.** Per-σ response of the reversal signal
  is flat across vol regimes. Volatility matters as a level, not as an
  amplifier. *The half of the hypothesis that was supposed to pay is
  the half that is not there.*

### The locked holdout (2022-01 → 2026-07, evaluated once, protocol H1)

| strategy | acc | activity | IR [90% CI] | ann. α | yrs+ | PT p | lag prec |
|---|---|---|---|---|---|---|---|
| always-long (SPY +66.7%) | 36.2% | 0% | 0.00 | 0.00% | — | — | — |
| anti-persistence rule | 33.9% | 36% | −0.46 [−1.20, 0.27] | −3.05% | 2/5 | 0.73 | 30.3% |
| LightGBM (E3 cfg) | 40.7% | 18% | −0.15 [−0.93, 0.51] | −0.63% | 2/5 | **0.04** | 39.1% |

The negative result is **stable on unseen years**, with a telling
split: the naive rule's directional skill *vanished* (PT 0.01 → 0.73),
while the model's persisted (0.04) and its fired-day precision reached
its best value anywhere (39.1% vs ~30% base) — **and it still lost
money**. Even surviving, better-than-ever selectivity does not clear
drift-drag plus costs. 39.1% is well short of the 51.5% the power
analysis showed is needed; the holdout independently lands on the same
arithmetic.

## 4. Interpretation

The result is *consistent with*, not contrary to, the literature it
tested. Jegadeesh (1990) and Lehmann (1990) document short-term
reversal **cross-sectionally** — losers beating winners across many
stocks, harvested market-neutrally. At the single-index level the
effect largely diversifies away, and a long/flat expression must
additionally out-earn the equity risk premium it forfeits when flat
plus transaction costs. We measured exactly that gap: a real ranking
signal (PT), too shallow (linear, ~100 stumps), with no volatility
gearing (SHAP), whose best selectivity (39% precision) is structurally
below the required (51.5%).

## 5. The methodology is the other half of the result

Three success criteria and one simulator died *before* any model ran,
each caught by a pre-registered check and logged (full chain A1–A13):
v1's precision criterion was structurally undetectable (up-calls cancel
against the benchmark; effective n ≈ 47); v2's fold gate was calibrated
against a coin-flip null when the true drift-drag null makes per-fold
wins 6–49% likely; the annualization constant was off by √5; the
tranche simulator had a one-session lookahead invisible to every
aggregate check and caught by the first hand-computed ground-truth
test. Any one of these, undetected, would have made every experiment
uninterpretable. Total experiment budget consumed by the actual
science: **4 of 25**.

## 6. Limitations

Single asset (n is the binding constraint: MDE ≈ IR 0.44 is detectable
but only at the literature's optimistic edge); one label family
(vol-scaled ternary; triple-barrier untested); costs modeled as flat
5bps round trip; holdout spans one macro regime cluster (2022 bear,
2023–25 bull); calendar features flagged (`month` #3 by SHAP) but
unaudited — irrelevant to a failing strategy, a red flag had it passed.

## 7. What could change the answer (pre-priced, not vibes)

1. **Cross-sectional reversal on individual stocks** — where the
   anomaly is actually documented; ~30 low-ρ names triple the
   effective sample and remove the drift-drag structurally
   (market-neutral). Requires a point-in-time universe with delisted
   names — survivorship bias specifically *manufactures* fake reversal
   profits (dip-buying known survivors). This is a new project.
2. **Meta-labeling / triple-barrier** (AFML ch. 3): make the label the
   P&L event itself. The one principled label change; pairs with (1).
3. **SPY to 1993** doubles n and would sharpen this negative result;
   it cannot plausibly reverse it (the failures are point-estimate
   failures, 3–4 SE from the bar, not power failures).

Adding correlated index ETFs (QQQ: ρ ≈ 0.9, +5% effective n) and label
re-sweeps on the same features are pre-priced as non-answers.

---

*Full records: `experiment_log.md` (contract, B1–B6, E1–E4, A1, H1),
`context/spec_v2.1.md` (criterion + supersession log A1–A13),
`power_results.md`, `reports/` (per-run records, SHAP figure, holdout).
Repo: github.com/05nigeltan/stock_predictor. All results reproducible
from cached raw pulls at seed 20260713.*
