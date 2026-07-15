# Experiment log — append-only

## The contract (pre-registered 2026-07-15, before E1)

**An experiment is any evaluation of a candidate configuration on
walk-forward test folds** — any test-fold metric read with intent to
select, tune, or compare. One numbered E-entry per named configuration.
A hyperparameter search under a single pre-registered search spec is ONE
experiment, provided its internal validation uses train-side data only;
reading test folds per trial would make each trial an experiment.

Not experiments: B-entries (baselines — they are the bar, not attempts
to clear it); train-only diagnostics (feature distributions, drift
plots); the single locked-holdout evaluation at the end.

**Budget: ≤25 E-entries or 8 weeks from E1, whichever first** (spec §0
kill criterion). When the budget is spent with no configuration passing
the gates, the pre-registered response is: stop, run the holdout once,
write up the negative result.

Gates (spec v2.1, post-A13): `IR ≥ IR_min(activity)` ∧ `≥6/10 folds
positive` ∧ `m-class PT p < 0.10`. Luck bars: IR_min = 0.06 / −0.11 /
−0.25 at 10/20/30% activity. MDE ≈ 0.44.

---

## Baselines (2026-07-15, `phase4_baselines.py`, seed 20260713)

Pooled walk-forward test folds 2012–2021, 2517 days. Accuracy floor
(Σp²) = 33.5%. Full record: `reports/phase4_baselines.md`.

| # | baseline | acc | activity | IR [90% CI] | folds+ | PT p | verdict |
|---|----------|-----|----------|-------------|--------|------|---------|
| B1 | always-long | 34.9% | 0% | 0.00 (anchor) | — | — | the benchmark itself |
| B2 | dummy most_frequent | 34.9% | 0% | ≡ B1 | — | — | modal class is beat → collapses to always-long |
| B3 | dummy stratified | 32.9% | 35% | −0.89 [−1.41,−0.39] | 1/10 | 0.23 | fails all — matches power.py's null (E[IR]≈−0.7 at 30%): the luck model, live |
| B4 | persistence rule | 33.1% | 30% | −0.74 [−1.36,−0.26] | 2/10 | 0.52 | fails all — momentum still dead at 5d, now out-of-sample |
| B5 | **anti-persistence rule** | 36.0% | 35% | −0.77 [−1.30,−0.29] | 2/10 | **0.01** | fails IR+folds, **passes PT** |
| B6 | depth-3 tree | 35.8% | 34% | −0.81 [−1.39,−0.30] | 3/10 | **0.02** | fails IR+folds, passes PT |

### What the bar means for Phase 5

1. **The reversal signal is real and now significant out-of-sample.**
   Phase 1 called anti-persistence "suggestive, not significant"
   (+1.8 SE on train). On the pooled test folds the m-class PT rejects
   at p = 0.01. The hypothesis survived its first honest test.
2. **Directional skill is nowhere near sufficient.** Both B5 and B6
   have it and both lose ~5–7%/yr of active return: firing on ~35% of
   days pays more drift-drag and costs than the signal earns back.
3. **Therefore PT is not the discriminating gate — IR is.** A trivial
   one-line rule already passes PT. A model that "beats PT" has matched
   a baseline, not beaten it. The entire game is SELECTIVITY: keep the
   reversal signal, fire on far fewer days (near the 10% activity
   floor), in the volatility regimes where it pays. This is precisely
   the conditioning hypothesis (§1.4) — now with a quantified bar.
4. B6's printed tree independently rediscovered conditional reversal
   (buy weakness below trend; fade strength in high vol) — the right
   *shape* of rule, unprofitable at this selectivity.

---

## Experiments

*Budget clock started 2026-07-15 (E1). 24 remaining.*

### E1 — logistic regression, argmax (2026-07-15)

**Config (pre-registered):** multinomial LogisticRegression(C=1.0, L2,
lbfgs, max_iter=5000), 21 features standard-scaled inside the fold
pipeline (train-fit only), argmax, no thresholding. Seed 20260713.
Full record: `reports/e1_logistic.md`.

**Result:** acc **39.0%** | activity 25.9% | IR **−0.49** [−1.10, 0.03]
| ann.α −2.79% | folds 2/10 | PT p = 0.02 → **fails IR + folds gates.**

**Reading:**
1. Best accuracy so far by 3pp (B5 36.0, B6 35.8) and the least-bad IR
   of any active strategy (B3–B6 sat at −0.74 to −0.89). The linear
   share of the signal is substantial — the bar for trees is now
   acc 39.0 / IR −0.49, not the baselines.
2. Improvement came largely through LOWER activity (26% vs ~35%): less
   indiscriminate firing, less drag. Consistent with the selectivity
   thesis; the 90% CI's upper edge already touches +0.03.
3. Final-fold coefficients: reversal-shaped — high `rsi_14` /
   `px_to_sma20` push toward lag (overbought ⇒ fade). Negative partials
   on `ret_3d`/`ret_10d` inside the same ≥0.8-correlated position
   cluster are collinearity shadows, per the correlation map — do not
   read individual signs within a cluster. §1.5's falsification test
   stays with SHAP on the trees.

### E2 — random forest, argmax (2026-07-15)

**Config (pre-registered):** RandomForestClassifier(n_estimators=500,
min_samples_leaf=50, max_features='sqrt', seed 20260713), argmax, no
scaling, no tuning. Full record: `reports/e2_random_forest.md`.

**Result:** acc **39.8%** | activity 24.1% | IR **−0.65** [−1.21, −0.09]
| ann.α −2.54% | folds **5/10** | PT p = 0.00 → **fails IR + folds.**

**Reading:**
1. **The nonlinear lift over linear is ~nil.** +0.8pp accuracy over E1
   is inside noise (SE ≈ 2.2pp at n_indep ≈ 503); the primary metric is
   WORSE (IR −0.65 vs −0.49, CI now fully below zero). Directionally
   consistent with the honest prior: on 21 hand-encoded features and
   ~850 effective samples, there may be little nonlinear structure left
   for trees to find.
2. Fold profile inverted vs E1: 5/10 positive folds (best yet) but a
   worse pooled IR — RF wins small in half the years and loses big in
   the others; its alpha is more volatile per unit of mean.
3. Importances (record only, cluster-smeared): both clusters carry —
   trend/position (`sma20_to_sma200`, `ret_10d`, `px_to_sma20`) and vol
   regime (`vol_21d`, `vix_lag1`). Consistent with conditioning story.
4. Bar for E3 (LightGBM) unchanged and now two-sided: beat E1's IR
   −0.49 to justify boosting; beat 39.8% acc to claim any nonlinear
   signal at all.

### E3 — LightGBM, argmax, frozen regularize-hard config (2026-07-15)

**Config (pre-registered):** multiclass, lr=0.01, num_leaves=7,
max_depth=3, min_data_in_leaf=50, feature/bagging fraction 0.7,
lambda_l2=10, ceiling 2000 trees, early_stopping_rounds=100 on the last
15% of each fold's train (chronological, 5-session inner purge, never
test). Argmax, no tuning. Full record: `reports/e3_lightgbm.md`.

**Result:** acc **39.1%** | activity **16.2%** | IR **−0.57**
[−1.12, −0.06] | ann.α −2.13% | folds 3/10 | PT p = 0.02 → **fails
IR + folds.**

**The ladder verdict (E1→E2→E3, the pre-registered model progression):**
1. **Accuracy converged: 39.0 / 39.8 / 39.1.** Three model families,
   one number, all within noise. There is no nonlinear lift on these 21
   features — the linear model already extracts what the trees extract.
   This is the plan doc's honestly-expected outcome, now measured.
2. **Early stopping wanted almost nothing: 10–251 trees (mean 97) of a
   2000 ceiling.** The boosted signal is shallow; after ~100 small
   trees there is nothing left to fit that survives regularization.
3. Activity fell down the ladder (26% → 24% → 16%) but the IR did not
   follow (−0.49 / −0.65 / −0.57): even LightGBM's more selective
   argmax lag calls don't clear cost + drag. Selectivity by argmax
   sharpening is not enough — if it exists anywhere, it is in the
   probability TAIL, which no experiment has yet read.
4. Every rung passes PT (p ≤ 0.02). Direction was never the problem.

**Remaining pre-registered mechanism, untested:** probability
thresholding — fire only when P(lag) clears a bar chosen on TRAIN-SIDE
validation, targeting the ~10% activity floor where power.py says
~51.5% lag precision meets the MDE. That is E4. If the tail carries no
extra precision, the negative result is effectively complete.

### E4 — LightGBM tail, fixed top-decile rule (2026-07-15)

**Config (decisions locked before the run):** E3's model verbatim;
quantile rule (ranking only — calibration irrelevant by design);
threshold = 90th pct of P(lag) on each fold's inner-validation days,
applied to test as it arrives; flat iff fired; PT preds LAG-when-fired
else argmax{beat,flat}; Q = 0.90 fixed, never tuned. Full record:
`reports/e4_lgbm_tail.md`.

**Result:** acc 40.0% | activity 19.5% | IR **−0.57** [−1.04, −0.11] |
folds 4/10 | PT p = 0.00 | **lag precision on fired days: 33.5%** vs
30.0% base and **51.5% required** → **fails IR + folds.**

**Reading:**
1. **The tail is empty.** The model's most-confident lag calls are
   +3.5pp better than the base rate, a sixth of the +21.5pp the MDE
   demands. Confidence ranking carries almost no extra precision:
   the model knows THAT reversal exists, not WHEN it is strong.
2. IR identical to E3's argmax (−0.57) — thresholding changed which
   days fire, not how good firing is.
3. Implementation honesty note: the valid-set quantile transferred
   imperfectly (realized activity 19.5%, not 10% — test-time P(lag)
   runs hotter than inner-valid). Doesn't rescue the verdict: precision
   at the achieved selectivity is nowhere near the bar, and deeper
   selectivity with an uninformative ranking cannot help.

### A1 — SHAP falsification test, §1.5 (2026-07-15; analysis, not an E-entry)

**Method:** E3's ten walk-forward models, each explaining its own test
window (out-of-sample TreeSHAP via LightGBM `pred_contrib`), pooled
2012–2021. Full record: `reports/shap_falsification.{md,png}`.

**Findings against §1.5's three predictions:**
1. **Direction — CONFIRMED, on the beat side, coherently.** §1.5
   predicted "a negative contribution from recent returns." On the
   BEAT class every member of the position cluster has negative
   corr with its own SHAP (ret_5d −0.70, bb_pctb −0.65, rsi_14 −0.61,
   px_to_sma20 −0.43, ret_10d −0.40): high recent returns uniformly
   reduce predicted beat. The LAG mirror is only partial (ret_5d
   +0.58, rsi_14 +0.30, others ~0 or negative) — the model fades
   strength by stepping toward FLAT more than by predicting lag.
2. **Importance — weak on the letter.** ret_5d ranks 12/21 at ~12% of
   the top feature's attribution. The top features are the
   conditioners themselves: sma20_to_sma200, vol_21d.
3. **Vol strengthening — NOT confirmed.** Per raw unit of ret_5d the
   high-vol slope is SHALLOWER (0.174 vs 0.400, ratio 0.4×); because
   high-vol ret_5d spans ~3× the range, the per-sigma response is
   roughly FLAT across regimes. The predicted amplification is absent
   — vol enters as a level feature in its own right (#2 by |SHAP|),
   not as a multiplier on the reversal signal. The interaction §1.4
   bet on does not appear.

**Verdict:** the reversal hypothesis is *not dead* on direction — the
sign §1.5 demanded is present, out-of-sample, consistent with PT ≤
0.02 everywhere. But it is small (rank 12), expressed as "reduce beat
/ step aside" rather than "predict lag", and the volatility
conditioning — the part of the hypothesis that was supposed to make it
MONETIZABLE — is not there. This is the attribution-level portrait of
E1–E4's verdict: real, shallow, linear, unconditioned.

**Curiosity flagged, not chased:** `month` ranks #3 by |SHAP| —
calendar seasonality that prominent would have been a red flag worth
auditing had any configuration passed the gates.

### H1 — HOLDOUT PROTOCOL (pre-registered 2026-07-15, BEFORE the run)

The locked holdout (2022-01-01 → data end) is evaluated **exactly
once**, by one execution of `holdout.py`, which computes three
strategies and nothing else:
1. **always-long** — the anchor.
2. **anti-persistence rule** — the hypothesis in executable form,
   pre-registered in Phase 1 (B5's rule, zero fitted parameters).
3. **LightGBM, frozen E3 config, argmax** — and only this model,
   because spec §0 designated LightGBM primary in v2, before any
   results existed. E1's better test-fold IR does NOT earn a slot:
   selecting the holdout visitor by test performance is the exact
   contamination the lock prevents.

Mechanics: expanding annual retrains (train through each year-end,
predict the next year — the project's operating assumption), 5-day
purge + 2-day embargo via the splitter, inner-valid early stopping as
in E3, evaluation via evaluate.py (IR + 90% MBB CI, per-year means,
m-class PT, activity, fired-day lag precision). Purely descriptive: no
configuration passed the gates on 2012–2021, so the holdout tests the
NEGATIVE result's stability, not a success claim. After this run the
holdout is spent.

### Status after E1–E4 (budget: 21 remaining)

Every mechanism pre-registered in the plan has now been tested and
priced: linear (E1), bagging (E2), boosting (E3), tail selectivity
(E4). Uniform verdict: **directional skill is real (PT ≤ 0.02
everywhere), linear, shallow, and cannot be monetized against
always-long at any tested selectivity.** No configuration approaches
the gates. The pre-registered decision point is live: negative-result
write-up + single holdout run, or a NEW pre-registered hypothesis
(spec amendment required — e.g. meta-labeling), not further variations
on the tested mechanisms.
