Price Prediction Project — Plan v2 (Trees-First)
Revision of the original LSTM-first plan. The model family changes; the methodology (stationarity, leakage, walk-forward, baselines, labelling) does not. Everything you learned in Phase 1 still applies.

---
What changes, and what doesn't
This content is only supported in a Lark Docs
The single biggest shift: the model is no longer the interesting part. The features are. With an LSTM you hand it a window of raw returns and hope it discovers structure. With trees, you encode the structure. That means Phase 3 becomes the heart of the project, not Phase 5.

---
Prerequisite: the ML ladder (2 weeks, do this in parallel with Phase 2)
You said you're a beginner — so climb the tree family in order. Each rung explains the next.
1. Decision tree. One tree, splitting on thresholds to reduce impurity (Gini/entropy). Understand: what a split is, what depth does, why a deep tree overfits perfectly.
2. Bagging. Train many trees on bootstrap samples, average them. Understand: variance reduction.
3. Random forest. Bagging + random feature subsets at each split. Understand: decorrelating the trees is what makes the average work.
4. Boosting. Trees trained sequentially, each one fitting the errors of the ensemble so far. Understand: this reduces bias, not variance — the opposite of bagging. This is the conceptual jump.
5. Gradient boosting. Boosting where "fit the errors" is formalized as gradient descent in function space. Understand: the learning rate, and why many shallow trees beat few deep ones.
6. LightGBM/XGBoost. Engineering optimizations on gradient boosting (histogram binning, leaf-wise growth). Understand: the key hyperparameters and what each one trades off.
Resources:
- StatQuest (Josh Starmer) on YouTube — the Decision Trees, Random Forests, AdaBoost, Gradient Boost, and XGBoost series. This is the single best beginner explanation of the tree family anywhere. Watch them in order. Free.
- Hastie, Tibshirani & Friedman, Elements of Statistical Learning (free PDF, Stanford) — chapters 9, 10, 15. Rigorous. Read after StatQuest, not before.
- scikit-learn user guide, sections on Decision Trees and Ensemble Methods — the practical bridge.
- LightGBM docs: "Parameters Tuning" — read this before you touch a single hyperparameter.
- Grinsztajn et al., "Why do tree-based models still outperform deep learning on tabular data?" (NeurIPS 2022) — read once you've built something; it will make sense retroactively and explain why your results look the way they do.

---
Phase 1 — Framing (unchanged, mostly done)
Your one-pager stands as written. Only two edits:
- Replace "LSTM sweep" with "LightGBM" as the primary model in your success criterion.
- Add a line: primary metric is directional accuracy vs. the constant-up baseline; secondary is precision on the traded class.
Everything else — instrument, label, baselines, walk-forward params, cost assumption, kill criterion, EMH paragraph — carries over untouched.

---
Phase 2 — Data pipeline (unchanged)
SPY, adjusted close, Tiingo, parquet. Same as before. Trees don't change anything here.
One simplification: you no longer need to build 3D sliding-window tensors. Your data structure is a flat DataFrame: one row per date, one column per feature, one column for the label. Much simpler code.

---
Phase 3 — Feature engineering (NOW THE CORE — 2–3 weeks)
This is where the project lives. The LSTM would have tried to learn temporal structure from raw windows; you now encode it explicitly. Every feature must be stationary and relative.
Lagged returns (the model's "memory" — this replaces the LSTM's recurrence):
for lag in [1, 2, 3, 5, 10, 21, 63]:
    df[f'ret_{lag}d'] = np.log(df.close / df.close.shift(lag))
That single loop is doing the job the LSTM's hidden state was supposed to do — and it's explicit, inspectable, and 7 parameters instead of 50,000.
Volatility (the most predictable thing in finance — don't skip this):
df['vol_10d'] = df.log_ret.rolling(10).std()
df['vol_21d'] = df.log_ret.rolling(21).std()
df['vol_ratio'] = df.vol_10d / df.vol_21d          # vol regime: expanding or contracting?
Relative position (overlaid indicators, made stationary — from your notes):
df['px_to_sma20'] = df.close / df.close.rolling(20).mean() - 1
df['px_to_sma50'] = df.close / df.close.rolling(50).mean() - 1
df['sma20_to_sma200'] = df.close.rolling(20).mean() / df.close.rolling(200).mean() - 1
Bounded oscillators (scale by known bounds, no fitting):
df['rsi_14'] = ta.rsi(df.close, 14) / 100.0
df['bb_pctb'] = ...   # Bollinger %B, already normalized
Volume (unbounded — must be transformed):
df['vol_zscore'] = (np.log(df.volume) - np.log(df.volume).rolling(63).mean()) / np.log(df.volume).rolling(63).std()
Context / regime:
df['vix_level'] = vix.close                        # already stationary-ish
df['vix_change'] = vix.close.pct_change(5)
# Your existing HMM regime label plugs in here as a categorical feature.
Calendar: day-of-week, month, days-to-month-end. Trees handle these natively as integers.
Discipline for every single feature, no exceptions:
1. Is it stationary? (Plot it across 20 years. Does the mean drift? If yes, fix it.)
2. Does it use only information available at time t? (Every rolling() and shift() must look backward. shift(-1) anywhere outside the label column is a bug.)
3. Does it have a stable meaning across regimes? (Would this feature mean the same thing in 2008 and 2021?)
Deliverable: a make_features.py producing a flat DataFrame, plus a notebook with a plot of every feature over the full history so you can see which ones drift.

---
Phase 4 — Baselines (unchanged, 1 week)
DummyClassifier (most_frequent, stratified), persistence, class balance. Numbers written down before any model is trained.
Add one: a single decision tree, depth 3. It's fast, interpretable (you can literally print the tree), and gives you a "simplest real model" rung between the dummy and the boosted ensemble. If LightGBM doesn't beat a depth-3 tree by much, that's diagnostic.

---
Phase 5 — Modelling (was 2–3 weeks of LSTM, now ~2 weeks of trees)
Model progression — build in this order, evaluate each on walk-forward:
1. Logistic regression on your features. A linear baseline. If trees don't beat this, your features have no nonlinear structure.
2. Random forest. Low-effort, hard to overfit badly, good sanity check.
3. LightGBM. Your workhorse.
LightGBM starting configuration for a small, noisy dataset like yours — this is the opposite of the defaults, which assume big clean data. Regularize hard:
params = {
    'objective': 'binary',            # or 'multiclass' for ternary labels
    'learning_rate': 0.01,            # low — many small steps
    'num_leaves': 7,                  # SMALL. Default 31 will overfit 1000 samples.
    'max_depth': 3,                   # shallow trees
    'min_data_in_leaf': 50,           # forces each leaf to be supported by real data
    'feature_fraction': 0.7,          # column subsampling
    'bagging_fraction': 0.7,          # row subsampling
    'bagging_freq': 1,
    'lambda_l2': 10.0,                # L2 regularization
    'n_estimators': 2000,             # high ceiling...
    # ...with early stopping deciding the real number:
}
# early_stopping_rounds=100 on a validation split — same concept as the LSTM's
# early stopping. You did NOT escape this by switching models.
Class imbalance (from your notes — remember, don't downsample):
'scale_pos_weight': (y==0).sum() / (y==1).sum()    # binary
# or 'class_weight': 'balanced'  in the sklearn API
Hyperparameter tuning: use Optuna with your walk-forward splitter as the CV object. Do not grid-search naively — with ~1,000 samples, aggressive tuning IS overfitting. Tune a handful of parameters, few trials, and keep a final holdout you never touch.
Interpretation — this is the payoff trees give you that LSTMs never will:
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
This tells you which features actually carry signal, and in which direction. On a project where the honest expected outcome is "little to no edge," this is likely to be your most valuable finding — "volatility features carry signal, price-momentum features carry none" is a real, defensible result.
Probability calibration: boosted trees output probabilities, but they're often miscalibrated. Since you'll threshold them ("only trade if P(up) > 0.55"), calibration matters. Look at sklearn.calibration.CalibratedClassifierCV and reliability diagrams. This is a genuinely useful thing to learn.

---
Phase 6 — Evaluation & backtest (unchanged)
Walk-forward with purging and embargo. Directional accuracy vs. baseline, mean ± std across folds, fold-by-fold win count. Diebold-Mariano test. Backtest with 5bps costs. Same as before.

---
Phase 7 — Now the LSTM comes back (optional, 1–2 weeks)
Build it last, as a comparison, not a foundation. You now have a leak-free pipeline, real baselines, and a strong tree benchmark — which means you can evaluate the LSTM honestly, which the original paper could not.
Expected outcome: it loses to LightGBM. Writing that up rigorously is a better project than an LSTM that "works."
Stretch beyond that: meta-labeling (primary rule decides side, LightGBM decides whether to act). This is the genuinely interesting frontier, and it's tree-native.

---
Revised timeline
This content is only supported in a Lark Docs

---
The one-sentence summary of the pivot
You are trading "let the model discover temporal structure" for "encode temporal structure yourself and let the model find interactions among it" — which, given ~1,000 noisy samples, is the right trade, and which happens to teach you more transferable ML than an LSTM would.

Phase 1 document
Phase 1 — Problem Framing & Protocol
Price Prediction Project · SPY · 5-day horizon · tree-based models
Status: Phase 1 CLOSED. Proceed to Phase 2 (data pipeline + purged walk-forward splitter).

§0 · The Spec (one-pager)
Everything below this table is justification, working, or reference. This table is the contract.




Instrument
SPY, adjusted daily close. yfinance (primary), Tiingo (secondary). 2005–present.
Horizon
5 trading days
Label
Ternary on excess return: excess_5d = fwd_ret_5d − drift_5d, where drift_5d = 252d trailing mean of backward 5-day returns. Threshold τ = k · σ_5d, σ_5d = σ_daily(EWMA-20) · √5. k = 0.4
Class balance
35.4% beat / 34.0% flat / 30.6% lag
Features
Reversal signals (lagged returns) + volatility/regime conditioning variables
Model
LightGBM primary. Logistic regression and random forest as intermediate rungs. LSTM deferred to Phase 7 as comparison.
Validation
Expanding-window walk-forward, 10 folds, 1-year test windows, 5-day purge + 2-day embargo
Cost assumption
5bps round trip
Holdout
2022–present. Untouched until exactly one final evaluation. If evaluated more than once, the number is void.
Success
Precision-on-traded exceeds the always-long rate on the same fired periods by ≥3pp, at coverage ≥20%, in ≥6/10 folds, net of 5bps, with Pesaran–Timmermann p < 0.10
Kill
After ≤25 experiments or 8 weeks (whichever first), if no configuration meets the above: stop, run the locked holdout once, write it up as a negative result.

Definitions
Coverage — fraction of periods where the model fires (predicts up or down, not flat)
Precision-on-traded — of the periods it fired, what fraction were correct in direction
Scoring convention — fired predictions are scored against the sign of the actual excess return, ignoring the flat band. Fires landing in the true flat band are tracked separately as a transaction-cost drag.
# §0 · The Spec (one-pager) — v2

Supersedes v1. The instrument, label, validation protocol, cost assumption
and holdout are unchanged. **Only the success criterion is rewritten**, for
the reason logged in Appendix A8.

Everything below this table is justification, working, or reference.
This table is the contract.

---

| | |
|---|---|
| **Instrument** | SPY, adjusted daily close. Tiingo `adjClose` (primary), yfinance `auto_adjust` (reconciliation only). 2005–present. |
| **Horizon** | 5 trading days |
| **Label** | Ternary on excess return: `excess_5d = fwd_ret_5d − drift_5d`, where `drift_5d` = 252d trailing mean of *backward* 5-day returns. Threshold `τ = k · σ_5d`, `σ_5d = σ_daily(EWMA-20) · √5`. `k = 0.4` |
| **Class balance** | 35.4% beat / 34.0% flat / 30.6% lag |
| **Features** | Reversal signals (lagged returns) + volatility/regime conditioning variables |
| **Model** | LightGBM primary. Logistic regression and random forest as intermediate rungs. LSTM deferred to Phase 7 as comparison. |
| **Validation** | Expanding-window walk-forward, 10 folds, 1-year test windows, 5-day purge + 2-day embargo. Sliding window run in parallel and reported. |
| **Trading rule** | Position `s_t ∈ {+1, 0, −1}` from the model's ternary prediction, held 5 days. Benchmark is always-long (`s_t = +1`). **Active return** `α_t = (s_t − 1) · r_t` — i.e. the model only differs from buy-and-hold when it goes flat or short. |
| **Cost assumption** | 5bps round trip, charged on every change in `s_t` |
| **Holdout** | 2022–present. Untouched until exactly one final evaluation. If evaluated more than once, the number is void. |
| **PRIMARY metric** | **Annualised information ratio of the net active return**, pooled across non-overlapping 5-day periods in the 10 walk-forward test folds. `IR = mean(α) / std(α) · √(52/5)` |
| **Success** | `IR ≥ IR_min` (see below), **AND** mean net active return > 0 in ≥8/10 folds, **AND** Pesaran–Timmermann p < 0.10 on directional skill |
| **`IR_min`** | ⚠️ **TO BE SET BY `power.py`, NOT CHOSEN.** Defined as the minimum detectable effect at 80% power, two-sided, α = 0.10, on the actual pooled test-fold sample size. Pre-register before any model is trained. |
| **Kill** | After ≤25 experiments or 8 weeks (whichever first), if no configuration meets the above: stop, run the locked holdout once, write it up as a negative result. |

---

## Why the primary metric changed

The v1 criterion — *precision-on-fired minus always-long precision on the
same fired periods, by ≥3pp* — is **structurally undetectable**. Proof:

```
precision_model  = [ n_up·a_up + n_down·a_down       ] / n_fired
precision_always = [ n_up·a_up + n_down·(1 − a_down) ] / n_fired
                    ^^^^^^^^^^ identical, cancels exactly

Δ = (n_down / n_fired) · (2·a_down − 1)
```

Every up-call makes the same call as the benchmark, so it is right or wrong
in lockstep with it and contributes **exactly zero** to Δ. The effective
sample size of the entire criterion is therefore not the ~503 independent
test periods — it is the ~47 independent **down-calls** available at the
minimum allowed 20% coverage.

Measured consequences (verified numerically, 2026-07-13):

| coverage | n_fired | n_down | SE(Δ) | 3pp bar, in σ |
|---|---|---|---|---|
| 20% | 101 | 50 | 7.05pp | 0.43 |
| 50% | 252 | 126 | 4.46pp | 0.67 |
| 100% | 503 | 252 | 3.15pp | 0.95 |

The bar is undetectable **across the entire allowed coverage range**, not
just at the corner. A zero-skill model clears it by luck alone ~13.5% of
the time. Widening coverage does not rescue it: the SE falls as √n while
the bar stays fixed.

**The root cause is information loss, not sample size.** Δ collapses each
5-day period to a binary hit/miss, discarding the magnitude of every
observation. A +0.05% week and a +3% week score identically, and the flat
class — 34% of all periods — contributes nothing at all.

## Why the replacement fixes it

The active-return IR is benchmarked against always-long, so up-calls still
contribute zero — **that cancellation is inherent to benchmarking, not a
flaw in the statistic**. But two things change:

1. **The flat class now counts.** Going flat when the benchmark is long
   produces `α_t = −r_t`, a real contribution. 34% of the sample rejoins
   the estimate.
2. **Magnitudes enter.** Being right on a −3% week and being right on a
   −0.05% week are no longer the same event. This is where most of the
   information in a return series lives.

Effective sample rises from ~47 to roughly the count of periods where
`s_t ≠ +1` — on the order of 250 — *and* each carries magnitude
information rather than one bit.

It also **reconciles the statistical and economic criteria**, which v1 left
inconsistent. v1's economic bar said "the backtest must beat buy-and-hold
in dollars," but Δ does not measure dollars: a model that calls small
down-weeks right and large ones wrong can post an excellent Δ while losing
money. The IR *is* the dollar criterion, expressed per unit of risk.

## What `power.py` must produce before `IR_min` can be filled in

Do not pick `IR_min`. Compute it. The script must output, for each
candidate metric, the **minimum detectable effect** at 80% power on the
real pooled test-fold sample produced by `splitter.py`:

| metric | effective n | SE under null | MDE @ 80% power | detectable? |
|---|---|---|---|---|
| Δ (v1 precision difference) | ~47 | 7.05pp | — | **NO** |
| Pesaran–Timmermann | 503 | — | — | ? |
| Mean net active return (t-stat) | ~250 | — | — | ? |
| **Active-return IR** ← candidate primary | ~250 | — | — | ? |
| Diebold–Mariano on P&L | 503 | — | — | ? |

Requirements:
- Compute on the **actual** fold structure, not an analytic approximation.
- Account for the 5-day label overlap (use non-overlapping periods, or a
  Newey–West / block-bootstrap correction — state which).
- Include a **zero-skill Monte Carlo** for each metric: what fraction of
  the time does pure noise clear the proposed bar? If >10%, the bar is not
  a bar.
- Report the MDE **in units a reader can price against the literature**.
  An IR of 0.4 on 5-day SPY timing is at the optimistic edge of what
  published work claims; an MDE of 1.2 would mean the criterion is
  detectable but unachievable — the same failure as v1 in a different
  costume.

## The outcome this rewrite must not paper over

There is a live possibility that **no** metric's MDE lands below any
plausible effect size. Seventeen years of daily SPY at a 5-day horizon is
~850 independent observations, ~500 of them in test. That is a small
dataset, and no choice of statistic conjures information that isn't there.

If `power.py` shows that, the correct response is **not** to lower the bar
until something passes. It is to state the result:

> *This dataset cannot support detection of any edge small enough to be
> plausible on 5-day SPY. The pre-registration caught this before any
> model was trained.*

That is a publishable negative result and a better project than a 56%
accuracy number nobody can interpret. The alternatives — if the finding is
to be avoided rather than reported — are to change the problem, not the
statistic:

- **Longer sample.** SPY back to 1993 adds ~12 years (~1,200 independent
  periods). Cheap. Regime-heterogeneous, which is its own problem.
- **Cross-sectional pooling.** Not SPY + QQQ + IWM — those are ~0.9
  correlated and add almost no independent information (503 → ~545). It
  requires ~30 genuinely low-correlation names, which converts this into a
  stock-picking project, not market timing. That is a different project.
- **Shorter horizon, more observations.** 1-day labels give 5× the sample
  but a worse signal-to-noise ratio. Probably a wash; worth pricing.

---

## Appendix A8 · Precision-difference criterion → Return-based criterion

**Superseded:** *"Precision-on-traded exceeds the always-long rate on the
same fired periods by ≥3pp, at coverage ≥20%, in ≥6/10 folds, net of 5bps,
with Pesaran–Timmermann p < 0.10."*

**Why it died:** The up-call term cancels identically against the
always-long benchmark, reducing the criterion to
`Δ = (n_down/n_fired)·(2·a_down − 1)`. The effective sample collapses from
~503 independent test periods to ~47 down-calls. The 3pp bar sits 0.43σ
above zero at minimum coverage and 0.95σ at full coverage; a zero-skill
model clears it 13.5% of the time. Undetectable at every allowed coverage.

Separately, the fold-consistency screen was inert: under a coin flip,
P(≥6 of 10 folds positive) = **37.7%**. Raised to ≥8/10 (P = 5.5%).

**Replaced by:** annualised IR of net active return, with `IR_min` set from
a pre-registered power analysis rather than chosen. Retains PT as a
directional-skill gate, since PT tests all 503 samples rather than the 47
down-calls.

**Cost of the error:** one afternoon of power analysis, before any feature
was engineered. The pre-registration in §2.3 explicitly anticipated this
contingency and it triggered exactly as designed. **This is the
methodology working, not failing** — the alternative was discovering it
after eight weeks of modelling, at which point every experiment run against
the old bar would have been uninterpretable.


§1 · Hypothesis
1.1 Position on market efficiency
Weak-form market efficiency provides the starting point for this project. It argues that historical price information is already incorporated into current prices, implying that trading strategies based solely on past prices should not consistently generate abnormal returns after accounting for risk and transaction costs. Empirical evidence broadly supports this view, which is why I expect any predictive edge derived from historical data to be small, conditional, and difficult to detect rather than large or persistent.
However, weak-form efficiency is best viewed as an approximation rather than an absolute rule. Financial markets exhibit several well-documented anomalies, including short-term reversal, which has been attributed to temporary overreaction, liquidity imbalances, and market microstructure effects. Rather than attempting to reject weak-form efficiency outright, this project tests whether short-term reversal represents one such limited exception, and whether a tree-based model can identify the market conditions under which this anomaly is most likely to occur.
1.2 The hypothesis
I do not expect a tree-based model to accurately predict the conditional mean of daily stock returns, as decades of empirical evidence suggest that this is close to unpredictable. Instead, I hypothesize that short-term price reversals over a five-day horizon are stronger during periods of high market volatility, when temporary liquidity imbalances and investor overreaction are more pronounced. My hypothesis is that a tree-based model can learn the nonlinear relationship between recent price movements and volatility regimes, allowing it to identify when reversal signals are more likely to be reliable than noise.
If the model fails to demonstrate consistent out-of-sample improvement over the reversal and naïve benchmarks defined in §0, the hypothesis will be rejected.
1.3 Why short-term reversal
Short-term reversal is one of the most well-documented market anomalies at short investment horizons. Unlike many technical indicators whose predictive power is inconsistent, it has been observed across different markets and is supported by both behavioural finance and market microstructure theories. Large price movements over a few days are often driven by temporary overreaction, panic trading, profit-taking, or liquidity imbalances rather than by changes in fundamental value. As these temporary pressures dissipate, prices tend to partially reverse toward their previous levels. This makes short-term reversal a suitable hypothesis for a five-day prediction horizon, where behavioural and liquidity effects are more likely to dominate than longer-term momentum.
1.4 Why volatility (as a conditioning variable, not a predictor)
Volatility clustering is one of the most consistently observed empirical properties of financial markets. Unlike many technical indicators whose effectiveness is debated, it has been repeatedly documented across markets, asset classes, and time periods: large price movements tend to be followed by further large movements, and quiet periods tend to persist. This makes volatility a useful proxy for identifying market regimes.
Tree-based models are particularly suited to learning nonlinear interactions. Rather than assuming reversal always works, the model can learn conditions such as "reversal is informative during high-volatility, stressed markets where liquidity imbalances dominate, but less reliable during calm, sustained trends." Volatility is not being used to predict returns directly; it is being used to determine when a reversal signal is likely to be trustworthy.
Many candidate predictors exist — macroeconomic variables, company fundamentals, news sentiment, other technical indicators. But these either require external data, are released at lower frequencies than daily returns, or have weaker empirical support. Short-term reversal and volatility clustering are among the most extensively replicated stylized facts in financial economics, and both are observable directly from historical price data.
1.5 Falsification test
If SHAP shows no importance for ret_5d, or shows it with a positive contribution, the reversal hypothesis is dead. The hypothesis predicts a negative contribution from recent returns, strengthening in high-volatility regimes.

§2 · Phase 1 Results
All numbers computed on the training period (2005-04-06 → 2021-12-31) before any model exists.
Metric
Value
Meaning
Train rows (overlapping)
4,216


Effective independent samples
~843
Overlapping 5d labels → n/5
Economic bar (always-long, raw returns)
60.1%
The backtest must beat buy-and-hold in dollars
Statistical bar (majority class, excess label)
35.4%
The model must beat this on accuracy
Precision bar (always-long, sign of excess)
54.1%
Precision-on-fired must beat this by ≥3pp
Random floor (Σpᵢ²)
33.5%
The true floor — not 33%
Persistence
31.9%
Predict same class as previous 5d window
Anti-persistence
36.4%
Predict the opposite — the reversal probe
SE at n=843
±1.6pp


Detection threshold (~2 SE)
~3.3pp


Median τ
0.70% of price (14× cost assumption)
Label is economically meaningful
Mean excess return
+0.013%
≈0 — de-meaning succeeded

2.1 Finding: no 5-day momentum; weak evidence of reversal
Persistence (31.9%) falls below the random floor (33.5%). Predicting "same as last week" does worse than guessing. Anti-persistence (36.4%) beats it by +2.9pp — but that is only +1.8 SE, which does not clear the pre-registered ~2 SE bar.
Verdict: suggestive, not significant. The ordering (anti-persistence > majority > random > persistence) is exactly what the short-term-reversal literature predicts, which is a coherent story rather than noise scattered at random. But the magnitude is inside the error bars. This is a hypothesis with a plausible prior and weak supporting evidence — which is what a hypothesis is supposed to be before it is tested properly. It earns ret_5d a place in the feature set with a predicted sign, and nothing more.
2.2 Finding: measurement noise is visible in my own numbers


v1 script
v2 script
Non-overlapping always-long
60.2%
61.7%

Same ticker, same end date. The only change was the start date moving ~3 months (drift warm-up), which shifted which days landed in the non-overlapping sample.
A 1.5pp swing from an arbitrary sampling offset, against a printed SE of 1.7pp. Measurement noise, observed live, in a number that looked like a constant.
This is why the success bar is expressed relative to a baseline, never as an absolute percentage, and why results are reported as distributions across folds rather than point estimates.
2.3 The statistical power problem (unresolved)
The success bar is 3pp. The detection threshold at n=843 is ~3.3pp. The bar sits at or below the measurement noise.
The 3pp figure is therefore a necessary but not sufficient screen; the Pesaran–Timmermann p-value does the actual significance work. If PT and the 3pp screen disagree, PT wins.
⚠️ BLOCKING TODO (Phase 2): Compute the SE on the pooled walk-forward test folds once the splitter exists. The ±1.6pp above is on the training set; the test-fold SE will be larger (likely ~2.5pp). If it comes back ≥3pp, the criterion must be reconciled before any modelling begins — by raising the bar, leaning entirely on PT, or pooling across assets to grow the sample. Discovering the bar was undetectable after eight weeks of experiments is the expensive way to learn this.

§3 · Concepts (learning reference)
3.1 Efficient Market Hypothesis
Form
Claim
Implication
Weak
Past price and volume data is fully reflected
Technical analysis is useless
Semi-strong
Prices adjust quickly to all public information
Fundamental analysis is useless
Strong
All information, including private, is reflected
No investor can win long-term

Where it breaks: Strong form clearly doesn't hold. Semi-strong is contested (value investing by P/E has outperformed long-run). Weak form is approximately true — see §1.1 for my position and why reversal is the specific exception being tested.
3.2 Fundamental vs Technical analysis


Fundamental
Technical
Goal
Find fair value; is it under/overvalued?
Identify short-term trends, entry/exit points
Tools
Financial statements, earnings, macro data
Price charts, candlesticks, volume, indicators
Horizon
Months to decades
Intraday to weeks
Philosophy
Market eventually corrects to true worth
Fundamentals are already priced in; patterns repeat

3.3 Log vs simple returns
Log returns are preferred because they are time-additive — total return over a period is the sum of individual log returns, not a product. This makes multi-period aggregation trivial.
Disadvantage: a log return is not a linear function of the asset's price change, which is the strength of simple returns.
Reference: https://www.pfolio.io/academy/log-vs-simple-returns
3.4 Stationarity and differencing
Stationary = the series' statistical properties don't depend on when you observe it. Trends and seasonality break stationarity. White noise is stationary. A stationary series looks roughly horizontal with constant variance.
Differencing = take differences between consecutive observations. This is how you remove a trend. Unit root tests (ADF) objectively determine whether differencing is required.
⚠️ The critical insight: scaling ≠ stationarizing
Min-max and standardization are affine transforms:
x' = (x − min) / (max − min)
x' = (x − μ) / σ
Both are "subtract a constant, divide by a constant." An affine transform of a trending series is still a trending series — you squash the y-axis, but the mean still marches upward. This is why the CS230 paper's scaling still leaves a non-stationary dataset.
Differencing / log returns is a structural transform, not affine. That is what removes the trend.
What scaling is actually for
It solves an optimization problem, not a statistical one:
Neural nets initialize assuming inputs are O(1). Raw volume (millions) alongside RSI (0–100) makes volume's gradients dominate.
Distance-based methods (kNN, SVM, PCA) are scale-dependent by construction.
Trees don't care at all — they split on thresholds, which are scale-invariant. Diagnostic: if scaling changes a tree model's results, there's a bug.
Correct pipeline, in order: (1) stationarize → (2) scale (only if the model needs it).
Trees are scale-invariant but NOT extrapolation-capable. A tree splits on thresholds seen in training; a test value beyond the training range saturates flat. Stationarity is still mandatory for trees.
📚 Hyndman & Athanasopoulos, Forecasting: Principles and Practice (otexts.com/fpp3) — stationarity & differencing chapter. scikit-learn user guide, "Preprocessing data."
3.5 Bounded vs unbounded indicators — why the scaler matters
Problem A: the scaler must be fitted, and the fit doesn't transfer.
Fit on the whole dataset → the min/max encode information from the test period. Lookahead leakage. (This is what the CS230 paper does.)
Fit on the training window only (correct) → but for an unbounded trending series, test values exceed the training range. AMZN training max ~$800; test values hit $2,000 → min-max maps that to 2.5, not [0,1]. The network has never seen an input outside [0,1] and extrapolates badly.
Problem B: the mapping isn't stable over time. "RSI = 70" always means RSI = 70. But "price = $500," min-max scaled, means something different in 2016 than in 2020. The feature has no consistent semantics.
Why bounded indicators escape both. RSI is confined to [0,100] by construction. So scale by the known theoretical bound:
df['rsi_scaled'] = df['rsi'] / 100.0   # no fitting, no leakage, stable forever

No parameters estimated from data → nothing to leak, nothing to break out of range. That's the asymmetry.
Handling unbounded features: transform to something stationary/bounded first.
Prices → log returns
Volume → log volume, or rolling trailing z-score
MACD → divide by price or by rolling volatility
Anything → percentile-rank within a trailing window (bounds to [0,1] by construction)
3.6 Overlaid indicators vs oscillators


Overlaid
Oscillators (sub-plot)
Where drawn
On the price chart, same y-axis
Separate pane below
Scale
Same units as price (dollars)
Own scale, often bounded
Examples
SMA, EMA, Bollinger Bands, VWAP, Ichimoku
RSI, MACD, stochastic, ATR, OBV

A 50-day MA is overlaid because it's literally an average of prices — denominated in dollars, living in the same numeric range. Which means it inherits price's non-stationarity.
Fix: convert to ratios.
df['px_to_sma20'] = df.close / df.close.rolling(20).mean() - 1

This preserves the relationship and achieves stationarity. (The Daniel paper says "scale overlaid indicators together with price to preserve the relationship" — that preserves the relationship but stays non-stationary. I'm using ratios instead.)
3.7 Epochs
Sample — one training example
Batch — group processed before one weight update (batch size 32 = 32 samples/update)
Iteration — one weight update
Epoch — one full pass through all training samples
770 samples, batch 32 → ⌈770/32⌉ = 25 iterations/epoch. 100 epochs = 2,500 updates.
Why fixed epochs is wrong: the right number depends on convergence speed, which varies with architecture, learning rate, and data. Too few → underfit. Too many → overfit (training loss falls while validation loss rises).
Fix: early stopping. Monitor validation loss; stop after N epochs without improvement ("patience"); restore best weights. Set max_epochs high and let early stopping decide.
This concept survives the switch to trees. Gradient boosting adds trees sequentially, so "how many trees" is the identical overfitting dial as "how many epochs" — and it's solved the identical way: early_stopping_rounds on a validation set.
3.8 Confusion matrix


Predicted: Up
Predicted: Down
Actual: Up
True Positive
False Negative
Actual: Down
False Positive
True Negative

Accuracy = (TP+TN)/total
Precision = TP/(TP+FP) — of the days I said "up," how often was I right?
Recall = TP/(TP+FN) — of the days that went up, how many did I catch?
F1 = harmonic mean of the two
Why accuracy alone lies: markets rise ~54% of days (and ~60% of 5-day windows). A model that always predicts "up" scores 54%+ and has learned nothing. The confusion matrix exposes it instantly — the entire "Predicted: Down" column is empty.
Precision matters more than recall here. If I only trade when the model fires, precision is my win rate. A model with 60% precision and 20% recall is a good model — it rarely fires, but it's right when it does. Recall doesn't cost money; wrong predictions do.
3.9 Labelling schemes (in ascending sophistication)


Scheme
Problem
(a)
Regression on next-day return
Daily returns are ~95% noise; MSE dominated by unpredictable outliers
(b)
Binary direction
A +0.01% day and a +3% day get the same label
(c)
Fixed-threshold ternary
Lets the model say "no signal" — but 0.5% is a big move in a calm market and noise in a volatile one
(d)
Volatility-scaled threshold ← CHOSEN
τ_t = k · σ_t. "Big move" now means "big relative to current conditions"
(e)
Triple-barrier (López de Prado)
Most principled — labels the data the way you'd actually trade it (profit target, stop loss, time limit; whichever hits first)

Plus the de-meaning step (my addition, forced by the data): label on excess return, not raw. Without it, no value of k balanced the classes — the up/down ratio sat stubbornly at ~1.6:1, because a symmetric ±τ band around zero can't remove a positive drift. De-meaning changes the question from "will it go up?" (answer: usually — and buy-and-hold already captures that) to "will it beat its own recent drift?" This dropped the precision baseline from 62.3% → 54.1%, turning an unwinnable fight into a fair one.
Labels are computed, not annotated. shift(-HORIZON) is the labelling step. No manual annotation, no Kaggle pre-labelled datasets (which hide their design decisions and often leak). Pandas is the annotator.
Class imbalance — what NOT to do
The drift is signal, not bias. Downsampling the up-labels deletes real information and miscalibrates the model's probabilities. Preference order:
Change the label so classes balance naturally ← what I did (excess returns)
Keep the imbalance, fix the metric (report vs baseline, use precision/recall)
Class weights in the loss function
Downsampling — last resort only
3.10 Walk-forward validation
Train on the past, test on the immediate future, roll forward. Simulates real-world retraining and prevents lookahead bias.
Expanding window — start date fixed, training period grows
Sliding window — fixed-length training window; oldest data drops off as new data arrives
How I chose my parameters:
Parameter
Value
Reasoning
Purge
5 days
= the label horizon. Not a judgment call. Training samples whose label windows overlap the test set must be deleted.
Embargo
2 days
~1% of dataset (LdP). Guards against serial correlation bleeding across the boundary.
Test window / step
1 year
Matches a realistic "retrain annually" operating assumption — validation as a literal simulation of operations.
Folds
10
Enough for a fold-level distribution. (5 folds → std of 5 numbers is itself wildly noisy.)
Expanding vs sliding
Run both, report both
A genuine empirical question. Expanding assumes the relationship is stable forever; sliding assumes it drifts. If results flip, that itself is a finding.

sklearn.TimeSeriesSplit does NOT purge. With 5-day overlapping labels, a training sample adjacent to the test boundary shares up to 4 of its 5 label days with the test period. This must be hand-built.
3.11 Statistical tests
Test
Question it answers
Pesaran–Timmermann ← PRIMARY
Does my directional forecast have any skill, correcting for the base rate? Cannot be fooled by a model riding the drift.
Diebold–Mariano
Are two models' forecast errors significantly different? (Model-vs-model, continuous forecasts.)
McNemar
Paired comparison of two classifiers on the same fold.

📚 Pesaran & Timmermann (1992), JBES, "A Simple Nonparametric Test of Predictive Performance."

§4 · Reading list
Core (do these)
Hyndman & Athanasopoulos, Forecasting: Principles and Practice — free, otexts.com/fpp3. §5.2 (simple forecasting methods / naive baselines), §5.8 (evaluating accuracy), stationarity chapter.
López de Prado, Advances in Financial Machine Learning — ch. 3 (labelling, triple-barrier, meta-labelling), ch. 7 (purged CV, embargo), ch. 11–12 (backtest overfitting).
Fabrice Daniel, "Financial Time Series Data Processing for ML" — arxiv.org/pdf/1907.03010 ✅ read
StatQuest (YouTube) — Decision Trees → Random Forests → AdaBoost → Gradient Boost → XGBoost, in order.
Why trees over LSTMs
Grinsztajn, Oyallon & Varoquaux (2022), "Why do tree-based models still outperform deep learning on tabular data?" (NeurIPS) — the paper. Explains the three mechanisms.
Shwartz-Ziv & Armon (2022), "Tabular Data: Deep Learning is Not All You Need"
Zeng et al. (2023), "Are Transformers Effective for Time Series Forecasting?" (DLinear)
M5 competition results — every top finisher used LightGBM.
The anomalies I'm betting on
Jegadeesh (1990) / Lehmann (1990) — short-term reversal
Any GARCH primer — volatility clustering
Reference
Larry Harris, Trading and Exchanges — where transaction costs actually come from
scikit-learn user guide: Preprocessing, Ensemble Methods, Classification Metrics, TimeSeriesSplit, DummyClassifier

§5 · Open TODOs
[ ] BLOCKING: compute SE on pooled walk-forward test folds (see §2.3)
[ ] Build the purged walk-forward splitter (TimeSeriesSplit doesn't purge)
[ ] Cross-check the 60.1% always-long rate on 1993–2004 — if it lands nearer 56%, the baseline is regime-dependent, which affects per-fold comparison
[ ] Read up on Diebold–Mariano properly
[ ] Decide expanding vs sliding empirically (run both)

APPENDIX A · Supersession log
Preserved deliberately. The process of being wrong and correcting is the most valuable part of this document.

A1. Momentum hypothesis → Reversal hypothesis
Superseded: hypothesis that 5-day price momentum persists — that stocks rising over the previous five days are more likely to keep rising over the next five.
Original text (dead):
"I expect the exploitable structure to live in (a) time-series momentum, which is well-documented and survives out-of-sample, and (b) volatility clustering... My hypothesis is that a model can learn when momentum signals are reliable versus when they're noise."
Why it died: The empirical results did not support it. Persistence (31.9%) came in below the random floor (33.5%) — 5-day price movements exhibit no momentum whatsoever. The time horizon was mismatched to the phenomenon: momentum is documented at 3–12 month lookbacks (Jegadeesh & Titman; Moskowitz, Ooi & Pedersen), whereas at weekly horizons the documented anomaly is the opposite — short-term reversal (Jegadeesh 1990; Lehmann 1990), driven by temporary overreaction and liquidity effects.
Replaced by: §1.2. Same structure (a directional signal, conditioned on volatility regime), opposite sign.
Note the sign flip that came with it: the volatility-conditioning logic inverted too. Momentum works in calm sustained trends; reversal works in high-volatility stressed markets. Same reasoning, opposite prediction. Getting this backwards in the doc would mean misreading the SHAP results later.
Cost of the error: one 30-line script, run in week one. This is the argument for computing baselines before models.

A2. Raw-return labels → Excess-return labels
Superseded: thresholding on the raw 5-day forward return.
Why it died: No value of k could balance the classes — up/down sat at ~1.6:1 across the entire sweep, because a symmetric ±τ band around zero cannot remove a positive drift. Also meant the model had to beat a 62.3% precision baseline, i.e. out-predict the equity risk premium, which buy-and-hold delivers for free.
Replaced by: excess_5d = fwd_ret_5d − drift_5d, drift from a 252d trailing mean of backward returns. Classes now 35/34/31. Precision baseline fell to 54.1% — a fair fight.
Bug caught in passing: the first draft computed the drift as fwd_ret_5d.rolling(252).mean().shift(1). That leaks — fwd_ret_5d at time t already contains information from t+4, and a single .shift(1) doesn't remove it. The drift must be built from backward returns.

A3. τ = 0.5 × σ_daily → τ = 0.4 × σ_5d (units bug)
Superseded: τ = 0.5 × σ_20d where σ_20d is the std of daily returns.
Why it died: Units mismatch. A daily volatility applied as a threshold on a 5-day return. Returns scale with √t, so σ_5d ≈ σ_daily × √5 ≈ 2.24 × σ_daily. The threshold was therefore sitting at 0.5/2.24 ≈ 0.22 standard deviations of the distribution actually being labelled — capturing only ~18% as "flat" when a balanced split was intended.
Replaced by: σ_5d = σ_daily · √5, then τ = 0.4 × σ_5d. Sanity-checked: median τ = 0.70% of price = 14× the cost assumption.
Lesson: the number looked principled. Print the class balance and check; don't assume the math worked.

A4. Absolute performance bars → Baseline-relative bars
Superseded: "If no model achieves at least 55% directional accuracy... I will discontinue development."
Why it died: The always-long rate on 5-day SPY windows is 60.1%. A 55% bar is below the naive baseline — a constant would fail it while a model passing it would still be worse than buy-and-hold. The project would have survived the kill check while producing something strictly inferior to doing nothing.
Replaced by: every performance bar is now phrased relative to a baseline. In a domain with a 60% base rate, no absolute number can express what "good" means.
Rule adopted: No naked accuracy figures. Every rate is reported against a baseline, with a standard error.

A5. Downsampling the majority class → Better labels
Superseded: "To combat long-term trend bias, downsample the up-labels in the training set to get an even class balance."
Why it died: The upward drift is signal, not bias — it's the most reliable feature of equity markets. Downsampling deletes real information and miscalibrates the model's predicted probabilities. It also treats a symptom rather than the cause.
Replaced by: change the label so the classes balance naturally (A2). Class weights as fallback. Downsampling is a last resort.

A6. LSTM-first → Trees-first
Superseded: replicating the CS230 paper's LSTM architecture as the primary model.
Why it died:
Sample size. ~843 independent observations vs. an LSTM's ~50,000 parameters, at a signal-to-noise ratio near 1:20. That's not a modelling problem, it's a memorization guarantee.
The sequence advantage is illusory. Financial daily data has short, weak memory. The useful structure (recent returns, volatility, distance from a moving average) can be hand-engineered in ten lines of pandas — which is exactly the LSTM's only advantage, removed.
The evidence is one-sided. Grinsztajn et al. (NeurIPS 2022); M5 competition.
Replaced by: LightGBM primary. LSTM deferred to Phase 7 as a comparison — building it after a leak-free pipeline exists means it can finally be evaluated honestly, which the original paper could not do.
What did NOT change: stationarity, leakage discipline, walk-forward, baselines, labelling design, early stopping. All methodology, all transferable. Only the scaling requirement relaxed — and even then, only partially (trees are scale-invariant but cannot extrapolate; stationarity is still mandatory).

A7. Diebold–Mariano → Pesaran–Timmermann (primary test)
Superseded: using DM as the significance test against the baseline.
Why it died: DM compares the forecast errors of two models on continuous forecasts. That's not the question. The question is: "does my directional forecast have any skill at all, given the base rate?"
Replaced by: Pesaran–Timmermann — the standard market-timing test, which corrects for the base rate and therefore cannot be fooled by a model that is merely riding the drift. That is precisely the failure mode at risk here. DM retained for model-vs-model comparisons.

APPENDIX B · Superseded raw text
Kept verbatim for the record. Do not build from these.
B1 · Original EMH paragraph (superseded by §1.1)
One limitation of the Efficient Market Hypothesis is its assumption that market prices rapidly and accurately incorporate all publicly available information. While EMH does not require every investor to be rational, it relies on the idea that irrational trades are largely offset by rational investors and arbitrage, leaving prices close to their fair value. In practice, however, behavioural finance has documented that investors are often influenced by emotions such as fear, greed, and overconfidence. These behavioural biases can lead to phenomena such as panic selling, speculative bubbles, and momentum trading, causing prices to deviate from fundamental values for extended periods. As a result, markets may not always be perfectly efficient, creating opportunities for predictive models to exploit temporary inefficiencies.
EMH also argues that historical price information should not consistently provide an investor with abnormal returns, as any predictable patterns would already have been exploited. However, empirical studies have identified short-term market anomalies such as momentum and mean reversion, suggesting that past price movements can sometimes contain information about future price behaviour. While these patterns are neither permanent nor guaranteed, they indicate that historical data may still have predictive value over certain time horizons.
Why superseded: argues a stronger anti-EMH position than I actually hold, and re-raises momentum (dead per A1). §1.1 states the position properly: weak-form efficiency is approximately true, which is why any edge should be expected to be small and conditional — and reversal is the specific documented exception being tested.
"Markets aren't perfectly rational" is true, universally known, and gets you to the starting line — where several thousand better-resourced people are already standing. The useful question is not "are markets inefficient?" but "why is THIS specific inefficiency still available to me?"
B2 · Original momentum hypothesis (superseded by §1.2)
I do not expect a tree-based model to accurately predict the conditional mean of daily stock returns, as decades of empirical evidence suggest that this is close to unpredictable. Instead, I hypothesize that the model can improve decision-making by exploiting two persistent market characteristics: time-series momentum and volatility clustering. Time-series momentum provides a potential directional signal, while volatility clustering provides information about the current market regime. My hypothesis is that the predictive value of momentum is regime-dependent — that is, momentum signals are more reliable during some volatility regimes than others — and that a tree-based model can learn these nonlinear interactions better than simple rule-based approaches. If no consistent improvement is observed over appropriate baselines across out-of-sample validation, this hypothesis will be rejected.
Why superseded: see A1. Persistence at 31.9% vs 33.5% random killed it. Note the structure survived — a directional signal conditioned on volatility regime — only the signal's identity and sign changed.


Extra research papers? Might have some overlap, but it’s fine.
Tabular Data & Trees vs. Deep Learning
Grinsztajn, Oyallon & Varoquaux (2022)
"Why do tree-based models still outperform deep learning on tabular data?" (NeurIPS)
Read on ResearchGate
Shwartz-Ziv & Armon (2022)
"Tabular Data: Deep Learning is Not All You Need"
Read on OpenReview
Time Series Forecasting Benchmarks
Elsayed et al.
"Do We Really Need Deep Learning Models for Time Series Forecasting?"
Read on Semantic Scholar
Zeng et al. (2023) (The DLinear paper)
"Are Transformers Effective for Time Series Forecasting?"
Read on AAAI / Open Journal
Medium-Term Momentum & Time Series Momentum
Jegadeesh & Titman (1993)
"Returns to Buying Winners and Selling Losers: Implications for Stock Market Efficiency"
Read on the University of Houston / Bauer Portal
Moskowitz, Ooi & Pedersen (2012)
"Time Series Momentum"
Read on the Copenhagen Business School Research Portal
Short-Term Reversals (1-Week & 1-Month Horizons)
Jegadeesh (1990)
"Evidence of Predictable Behavior of Security Returns"
Read on ResearchGate
Lehmann (1990)
"Fads, Martingales, and Market Efficiency"
Read on the National University of Singapore Repository


Phase 2 considerations: 
I would use Tiingo's adjusted close for modelling:
This project uses Tiingo's adjusted closing prices rather than reconstructing them manually from raw prices, dividends, and stock splits. Although adjusted prices are not point-in-time—historical prices are retroactively rescaled whenever new corporate actions occur—this does not materially affect the features used in this study. Since all model inputs are expressed as returns or rolling ratios, any retroactive adjustment is applied equally to consecutive observations and therefore cancels when returns are calculated. Self-reconstruction would improve reproducibility by creating a fully deterministic data pipeline, but it offers little statistical advantage for the return-based features considered in this project.
The only potential impact would arise if level-based features or dollar-denominated thresholds were introduced in future work, in which case a self-reconstructed adjustment pipeline would be preferable.
The VIX series is lagged by one trading day to eliminate potential look-ahead bias arising from differences in market closing and publication times. After alignment, the number of missing observations is assessed. As the remaining missing values are expected to be negligible, these rows are removed uniformly across all models. This ensures that Logistic Regression, Random Forest, and LightGBM are trained and evaluated on an identical feature matrix, enabling fair comparison while avoiding unsupported assumptions through imputation. VIX serves as a supplementary market-wide volatility indicator, whereas the primary regime features are realised volatility measures derived directly from SPY returns, which share the same trading calendar and require no temporal adjustment.


Possible data quality issue:
2015-08-21, 08-24, 08-26, 08-27, 09-01. That's the August 2015 flash-crash week, when SPY gapped down ~5% at the open, hundreds of ETFs hit limit-down halts, and the official closing print was genuinely ambiguous across venues. Vendors resolved it differently.

Is it material? 17bps against a median τ of 70bps is ~24% of your label threshold — enough to flip a borderline label on those specific days. But it's 9 rows in 4,211 (0.2%), and every aggregate reproduced anyway. So: not ideal, but no choice anyways.


# Price Prediction Project — Session Handoff (Phase 1 → Phase 2)

Exported from a claude.ai chat on 2026-07-13, for continuation in Claude Code.
This is a working log, not a polished doc — decisions, reasoning, and open
threads are kept in the order they happened so context isn't lost.

---

## Where things stand

**Phase 1: CLOSED.** Baselines recomputed on Tiingo and reproduce the
original (yfinance-based) numbers to within noise. See "Phase 1 recompute
results" below.

**Phase 2: IN PROGRESS.** Ingest + raw cache + reconciliation are built and
run successfully (script below). Still to build: FRED VIX ingest, validation
module, purged walk-forward splitter, point-in-time recompute test, pooled
test-fold SE (this last one is a **gate** — see "The blocking question").

---

## Project spec (from the locked Phase 1 one-pager)

| | |
|---|---|
| Instrument | SPY, adjusted daily close. Tiingo primary (was yfinance). 2005–present. |
| Horizon | 5 trading days |
| Label | Ternary on excess return: `excess_5d = fwd_ret_5d − drift_5d`, `drift_5d` = 252d trailing mean of *backward* 5-day returns. Threshold `τ = k · σ_5d`, `σ_5d = σ_daily(EWMA-20) · √5`. `k = 0.4` |
| Class balance | 35.4% beat / 34.0% flat / 30.6% lag |
| Features | Reversal signals (lagged returns) + volatility/regime conditioning variables (Phase 3, not yet built) |
| Model | LightGBM primary. Logistic regression + random forest as intermediate rungs. LSTM deferred to Phase 7 as comparison. |
| Validation | Expanding-window walk-forward, 10 folds, 1-year test windows, 5-day purge + 2-day embargo |
| Cost assumption | 5bps round trip |
| Holdout | 2022–present. Untouched until exactly one final evaluation. |
| Success | Precision-on-traded exceeds always-long rate by ≥3pp, at coverage ≥20%, in ≥6/10 folds, net of 5bps, Pesaran–Timmermann p < 0.10 |
| Kill | ≤25 experiments or 8 weeks, whichever first, if no config meets the bar → stop, run holdout once, write up as negative result |

**Model family:** trees-first (LightGBM), not LSTM-first. LSTM comes back in
Phase 7 as a comparison once a leak-free pipeline + real baselines + a tree
benchmark exist. Full rationale for the pivot is in Appendix A6 of the
Phase 1 doc (not reproduced here — ask if you need it restated).

**The blocking question (§2.3 of Phase 1 doc, still open):** the success bar
(3pp) sits at or below the training-set detection threshold (~3.3pp at
n≈843). The training-set SE is ±1.6pp; the pooled walk-forward *test-fold*
SE will likely be larger (~2.5pp estimated). **Once the splitter exists,
compute this SE before building any feature or model.** If it comes back
≥3pp, the success criterion is undetectable as specified and must be
reconciled — raise the bar, lean entirely on the Pesaran–Timmermann test, or
pool across assets to grow the sample. This gates Phase 3.

---

## Phase 2 decisions made this session

### 1. Data source: Tiingo, adjusted close, not self-reconstructed

> This project uses Tiingo's adjusted closing prices rather than
> reconstructing them manually from raw prices, dividends, and stock
> splits. Although adjusted prices are not point-in-time — historical
> prices are retroactively rescaled whenever new corporate actions occur —
> this does not materially affect the features used in this study, since
> all model inputs are expressed as returns or rolling ratios and the
> adjusted series is the total-return path. Self-reconstruction would
> improve reproducibility (a fully deterministic pipeline) but offers
> little statistical advantage for return-based features. Would revisit if
> level-based or dollar-denominated features are introduced later.

**Caveat surfaced during review:** the "cancels" framing is only exactly
true for *consecutive-day* returns with no event between them. Rolling
ratios (`px_to_sma20` etc.) that straddle an ex-dividend date carry a small
residual (~0.3%, one quarterly SPY dividend). Small, but "cancels" was
technically imprecise — the accurate claim is "the adjusted series is the
total-return path, and every feature computed on it is the total-return
analogue of the feature intended."

**Empirically confirmed** (see recompute run below): Tiingo adjClose and
yfinance `auto_adjust=True` — both total-return series — reproduce
Phase 1's baselines to within noise. Max daily-return divergence 17bps,
concentrated in the Aug 2015 flash-crash week (known vendor
close-price-resolution landmine on 2015-08-21/24/26/27, 09-01 — not a
pipeline bug, logged as a data-quality note, not chased further since
aggregates are unaffected).

### 2. VIX alignment

> The VIX series is lagged by one trading day to eliminate potential
> look-ahead bias arising from differences in market closing and
> publication times (SPY closes 4:00pm ET; Cboe calculates VIX to 4:15pm ET
> since SPX options trade 15 min longer). After alignment, the number of
> missing observations is assessed. As the remaining missing values are
> expected to be negligible, these rows are removed uniformly across all
> models. This ensures Logistic Regression, Random Forest, and LightGBM are
> trained and evaluated on an identical feature matrix, enabling fair
> comparison while avoiding unsupported assumptions through imputation.
> VIX serves as a supplementary market-wide volatility indicator, whereas
> the primary regime features are realised volatility measures derived
> directly from SPY returns, which share the same trading calendar and
> require no temporal adjustment.

**Open implementation note (not yet resolved in code):** the drop must
happen *after* the full panel (features + labels) is built on the complete
SPY trading calendar, not at alignment time — otherwise `shift(-5)` on a
row-dropped index silently reaches further than 5 trading days on affected
rows, corrupting labels rather than just removing them. Sequence must be:
(1) build curated panel on complete calendar, VIX NaNs allowed → (2)
compute all features/labels on that complete panel → (3) drop VIX-NaN rows
only when materializing the final modelling matrix.

**Also unresolved:** "expected to be negligible" was asserted, not
measured. Count actual VIX-missing rows against the SPY calendar and
pre-register a threshold (e.g. >0.5% of days) above which dropping stops
being free.

**Consequence for the walk-forward splitter:** once rows can be absent
from the modelling matrix, purge/embargo logic must be date-based (compute
each label's date span against the panel calendar and exclude any training
row whose span intersects the test window), not positional
(`iloc`-based) — needed for the fold boundaries regardless of the VIX
issue.

---

## Phase 1 recompute results (Tiingo, ran successfully by user)

```
[cache] wrote data/raw/tiingo_spy_2005-01-01.parquet  (5413 rows, 2005-01-03 -> 2026-07-10)

SOURCE RECONCILIATION — Tiingo adjClose vs yfinance auto_adjust
  Overlapping dates          : 5413
  Dates in Tiingo only       : 0
  Max |daily return diff|    : 0.1683%
  Days with diff > 5bps      : 9  (all Aug 2015 flash-crash week)

Train period: 2005-04-06 → 2021-12-23
Train rows (overlapping)      : 4211
Purged at holdout boundary    : 5
Effective independent samples : ~842

ECONOMIC BAR — always-long
  Overlapping (daily) : 60.1%  (SE ±1.7%)   [v3 yfinance: 60.1% — MATCH]
  Non-overlapping     : 61.8%  (n=843)

CLASS BALANCE at k=0.4: 35.4% beat / 34.0% flat / 30.6% lag
  [v3 yfinance: 35.4 / 34.0 / 30.6 — MATCH]

STATISTICAL BAR (k=0.4)
  Random-guess floor       : 33.5%
  Majority-class           : 35.4%
  Persistence               : 32.0%   [v3: 31.9% — within noise, boundary purge]
  Anti-persistence          : 36.4%   [v3: 36.4% — MATCH]
  SE: ±1.6% (n_indep=842); detection threshold ~3.3%

PRECISION BAR
  Always-long precision (sign of excess ret): 54.1%  [v3: 54.1% — MATCH]

SANITY CHECKS
  Median tau: 0.70% of price (14x cost assumption)
  Mean excess return: +0.0145% (~0, de-meaning OK)
  Max index gap: 5 calendar days (no missing rows)

REVERSAL VERDICT: anti-persistence beats random by +2.9% (+1.8 SE)
  → SUGGESTIVE, not significant. Same verdict as v3.
```

**Conclusion:** the adjusted-close decision is empirically vindicated, not
just argued — two independently-sourced total-return series with different
adjustment implementations agree to the decimal on every baseline.

---

## Current script: `phase1_baselines_v4.py`

This is the working, tested (by user, successfully run) version. Ported
from a v3 yfinance-only script. Changes from v3:
- Data source: Tiingo EOD (`adjClose`), yfinance kept only as a
  reconciliation check, not an input
- Boundary purge fix: v3 let training-period rows read `fwd_ret_5d` values
  reaching into the 2022+ holdout; this version drops the last `HORIZON`
  rows before `TRAIN_END`
- Raw pull cached to parquet (`data/raw/tiingo_spy_2005-01-01.parquet`) —
  reruns don't silently pick up revised vendor data
- Column-flatten-before-rename bug fixed (yfinance MultiIndex columns)
- Unexplained-jump check: flags `|log_ret| > 20%` with no matching
  `splitFactor` change

```python
"""
Phase 1 — Baselines and Label Calibration  (v4)
================================================
Changes from v3:
  - DATA SOURCE: Tiingo EOD (adjClose) replaces yfinance as primary.
    yfinance retained ONLY as a reconciliation check, not as an input.
    Both are total-return (split + dividend adjusted) series, so the
    baselines should reproduce. Confirm, don't assume.
  - BOUNDARY PURGE: v3 kept rows whose fwd_ret_5d reached into 2022,
    meaning training baselines were computed partly from HOLDOUT prices.
    Fixed: the last HORIZON rows before TRAIN_END are dropped.
  - RAW CACHE: the Tiingo pull is written to parquet and reused. Rerunning
    this script cannot silently pick up revised vendor data.
  - Column handling fixed (flatten MultiIndex BEFORE renaming).

Install:  pip install pandas numpy pyarrow requests
          pip install yfinance          # only for the reconciliation check
Set:      export TIINGO_TOKEN=...
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------
TICKER      = "SPY"
START       = "2005-01-01"
TRAIN_END   = "2021-12-31"   # everything after this is the LOCKED HOLDOUT
HORIZON     = 5              # label horizon, in trading days
VOL_SPAN    = 20             # EWMA span for volatility estimate
DRIFT_WIN   = 252            # trailing window for drift estimate
K           = 0.4            # CHOSEN from the v2 sweep: gives ~35/34/31 balance

RAW_DIR     = Path("data/raw")
RECONCILE   = True           # cross-check Tiingo vs yfinance on adjusted close
TOKEN       = os.environ["TIINGO_TOKEN"]


# ---------------------------------------------------------------
# 1. LOAD — Tiingo EOD, cached to an immutable raw zone
# ---------------------------------------------------------------
def fetch_tiingo(ticker: str, start: str, token: str) -> pd.DataFrame:
    """Raw pull. Returns exactly what Tiingo sends, no transformation."""
    url = f"https://api.tiingo.com/tiingo/daily/{ticker.lower()}/prices"
    r = requests.get(
        url,
        params={"startDate": start, "format": "json"},
        headers={"Authorization": f"Token {token}"},
        timeout=60,
    )
    r.raise_for_status()
    raw = pd.DataFrame(r.json())
    raw["date"] = pd.to_datetime(raw["date"], utc=True).dt.tz_localize(None).dt.normalize()
    return raw.set_index("date").sort_index()


def load_raw(ticker: str, start: str, token: str) -> pd.DataFrame:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"tiingo_{ticker.lower()}_{start}.parquet"
    if path.exists():
        print(f"[cache] reusing {path}")
        return pd.read_parquet(path)
    raw = fetch_tiingo(ticker, start, token)
    raw.to_parquet(path)
    print(f"[cache] wrote {path}  ({len(raw)} rows, "
          f"{raw.index.min().date()} -> {raw.index.max().date()})")
    return raw


raw = load_raw(TICKER, START, TOKEN)

# Basic integrity: no dupes, monotonic, no missing prices.
assert raw.index.is_unique,               "duplicate dates in raw pull"
assert raw.index.is_monotonic_increasing, "raw pull is not sorted"
assert raw["adjClose"].notna().all(),     "NaN in adjClose"
assert (raw["adjClose"] > 0).all(),       "non-positive adjClose"

# Unexplained-jump check: any |log return| > 20% must coincide with a split.
_lr = np.log(raw.adjClose / raw.adjClose.shift(1))
_suspect = raw.loc[(_lr.abs() > 0.20) & (raw.splitFactor == 1.0)]
if len(_suspect):
    print(f"\n!! {len(_suspect)} large moves with no split factor — inspect:")
    print(_suspect[["close", "adjClose", "divCash", "splitFactor"]])
    print("   (2008-10-13 and 2020-03-16 are real. Anything else is a bug.)\n")

df = raw[["adjClose", "adjVolume"]].rename(
    columns={"adjClose": "close", "adjVolume": "volume"}
).copy()


# ---------------------------------------------------------------
# 1b. RECONCILIATION — Tiingo vs yfinance (auto_adjust=True)
#     Both are total-return series. They should agree to a few bps.
# ---------------------------------------------------------------
if RECONCILE:
    import yfinance as yf

    yfd = yf.download(TICKER, start=START, auto_adjust=True, progress=False)
    if isinstance(yfd.columns, pd.MultiIndex):          # flatten BEFORE renaming
        yfd.columns = yfd.columns.get_level_values(0)
    yfd = yfd[["Close"]].rename(columns={"Close": "yf_close"})

    # Compare RETURNS, not levels. The two vendors anchor their adjustment
    # factors differently, so levels can differ by a constant scale while the
    # returns — the only thing this project consumes — agree exactly.
    cmp = df[["close"]].join(yfd, how="inner")
    r_tg = np.log(cmp.close / cmp.close.shift(1))
    r_yf = np.log(cmp.yf_close / cmp.yf_close.shift(1))
    diff = (r_tg - r_yf).dropna()

    print("=" * 60)
    print("SOURCE RECONCILIATION — Tiingo adjClose vs yfinance auto_adjust")
    print("=" * 60)
    print(f"  Overlapping dates          : {len(cmp)}")
    print(f"  Dates in Tiingo only       : {len(df) - len(cmp)}")
    print(f"  Max |daily return diff|    : {diff.abs().max():.4%}")
    print(f"  Days with diff > 5bps      : {(diff.abs() > 0.0005).sum()}")
    if (diff.abs() > 0.0005).sum():
        print("  Worst offenders:")
        print(diff.reindex(diff.abs().sort_values(ascending=False).index)
                  .head(5).to_string())
    print("  >>> If the two disagree materially, STOP and find out why before")
    print("      trusting any number below.\n")


# ---------------------------------------------------------------
# 2. RETURNS, DRIFT, AND THE LABEL TARGET
# ---------------------------------------------------------------
df["log_ret"] = np.log(df.close / df.close.shift(1))

# Forward window. The ONLY legitimate forward-looking column in the codebase.
df["fwd_ret_5d"] = np.log(df.close.shift(-HORIZON) / df.close)

# Trailing drift from BACKWARD returns only — every term is history at time t.
past_ret_5d    = np.log(df.close / df.close.shift(HORIZON))
df["drift_5d"] = past_ret_5d.rolling(DRIFT_WIN, min_periods=60).mean()

# Label target: did the next 5 days BEAT THE RECENT DRIFT?
df["excess_5d"] = df.fwd_ret_5d - df.drift_5d


# ---------------------------------------------------------------
# 3. VOLATILITY-SCALED THRESHOLD
# ---------------------------------------------------------------
sigma_daily    = df.log_ret.ewm(span=VOL_SPAN).std()
df["sigma_5d"] = sigma_daily * np.sqrt(HORIZON)


def make_labels(frame, k):
    """Ternary label on EXCESS returns: +1 beat drift, -1 lagged drift, 0 flat."""
    tau = k * frame.sigma_5d
    return np.select(
        [frame.excess_5d > tau, frame.excess_5d < -tau],
        [1, -1],
        default=0,
    )


def se_of_acc(acc, n):
    """Standard error of an accuracy estimate. Print this next to EVERY rate."""
    return np.sqrt(acc * (1 - acc) / n)


# ---------------------------------------------------------------
# 4. SPLIT — train vs LOCKED holdout, WITH A BOUNDARY PURGE
# ---------------------------------------------------------------
# v3 bug: rows in the last HORIZON trading days of the training period have a
# fwd_ret_5d that reads prices from January 2022 — i.e. from the holdout. Small
# (5 rows), but the holdout is supposed to be untouched, and the same purge
# logic will be needed at every walk-forward fold boundary in Phase 2. Do it
# here so the rule is stated once and obeyed everywhere.
train_mask = df.index <= pd.Timestamp(TRAIN_END)
train_all  = df.loc[train_mask]
train      = train_all.iloc[:-HORIZON].dropna(
    subset=["fwd_ret_5d", "sigma_5d", "excess_5d", "drift_5d"]
).copy()

purged = len(train_all.dropna(subset=["fwd_ret_5d", "sigma_5d",
                                      "excess_5d", "drift_5d"])) - len(train)

n_indep = len(train) // HORIZON

print(f"Data source : Tiingo adjClose (total return)")
print(f"Train period: {train.index.min().date()} → {train.index.max().date()}")
print(f"Train rows (overlapping)      : {len(train)}")
print(f"Purged at holdout boundary    : {purged}")
print(f"Effective independent samples : ~{n_indep}\n")


# ---------------------------------------------------------------
# 5. THE ECONOMIC BAR — always-long
# ---------------------------------------------------------------
always_long = (train.fwd_ret_5d > 0).mean()
nonoverlap  = train.iloc[::HORIZON]
al_nonovlp  = (nonoverlap.fwd_ret_5d > 0).mean()

print("=" * 60)
print("ECONOMIC BAR — always-long")
print("=" * 60)
print(f"  Overlapping (daily) : {always_long:.1%}  "
      f"(SE ±{se_of_acc(always_long, n_indep):.1%})")
print(f"  Non-overlapping     : {al_nonovlp:.1%}  (n={len(nonoverlap)})")
print(f"  v3 (yfinance) said  : 60.1%  — does this reproduce?")
print("  >>> Your BACKTEST must beat buy-and-hold in DOLLARS, net of costs.\n")


# ---------------------------------------------------------------
# 6. CLASS BALANCE SWEEP (kept for the record)
# ---------------------------------------------------------------
print("=" * 60)
print("CLASS BALANCE vs k   (excess returns, training period only)")
print("=" * 60)
print(f"{'k':>6} {'beat':>8} {'flat':>8} {'lag':>8} {'coverage':>10}")
print("-" * 60)
for k in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]:
    y = pd.Series(make_labels(train, k), index=train.index)
    b = y.value_counts(normalize=True)
    marker = "  <-- CHOSEN" if abs(k - K) < 1e-9 else ""
    print(f"{k:>6.1f} {b.get(1,0):>8.1%} {b.get(0,0):>8.1%} "
          f"{b.get(-1,0):>8.1%} {b.get(1,0)+b.get(-1,0):>10.1%}{marker}")
print(f"\n  v3 (yfinance) at k=0.4: 35.4 / 34.0 / 30.6\n")


# ---------------------------------------------------------------
# 7. THE STATISTICAL BAR — baselines at K
# ---------------------------------------------------------------
train["y"] = make_labels(train, K)

p          = train.y.value_counts(normalize=True)
acc_random = (p ** 2).sum()          # random guess weighted by class freq — NOT 33%
majority   = train.y.value_counts().idxmax()
acc_major  = (train.y == majority).mean()

# --- Persistence: predict the previous non-overlapping window's class ---
# Label at t-5 covers (t-5 -> t), fully observed at t. Leak-free.
train["y_persist"] = train.y.shift(HORIZON)
m_p         = train.y_persist.notna()
acc_persist = (train.y[m_p] == train.y_persist[m_p]).mean()

# --- Anti-persistence: predict the OPPOSITE of last window ---
# THIS IS THE REVERSAL PROBE. If reversal is real, this beats random.
train["y_anti"] = -train.y.shift(HORIZON)
m_a          = train.y_anti.notna()
acc_anti     = (train.y[m_a] == train.y_anti[m_a]).mean()

se_rand = se_of_acc(acc_random, n_indep)

print("=" * 60)
print(f"STATISTICAL BAR — excess-return label (k = {K})")
print("=" * 60)
print("  Class balance:")
print(train.y.value_counts(normalize=True).sort_index().to_string())
print()
print(f"  Random-guess accuracy (sum p_i^2)        : {acc_random:.1%}"
      f"   <-- the true floor, NOT 33%")
print(f"  Majority-class accuracy (predicts {majority:+d})    : {acc_major:.1%}")
print(f"  Persistence accuracy                     : {acc_persist:.1%}")
print(f"  Anti-persistence accuracy                : {acc_anti:.1%}   <-- REVERSAL PROBE")
print(f"\n  SE on any of these: ±{se_rand:.1%}  (n_indep = {n_indep})")
print(f"  A result must clear ~{2*se_rand:.1%} above random to be worth believing.")
print(f"\n  v3 (yfinance): random 33.5 | majority 35.4 | persist 31.9 | anti 36.4\n")


# ---------------------------------------------------------------
# 8. THE PRECISION BAR
# ---------------------------------------------------------------
# SCORING CONVENTION (option b): a fired prediction is scored against the SIGN
# of the actual excess return, ignoring the flat band.
true_sign    = np.sign(train.excess_5d)
al_precision = (true_sign == 1).mean()

print("=" * 60)
print("PRECISION BAR — sign-based, no peeking")
print("=" * 60)
print(f"  Always-long precision (sign of excess ret): {al_precision:.1%}"
      f"  (SE ±{se_of_acc(al_precision, n_indep):.1%})")
print(f"  v3 (yfinance) said: 54.1%")
print("  >>> At eval time, compare your model's precision-on-fired against")
print("      always-long ON THE SAME FIRED PERIODS — not against this global")
print("      number. This is just the reference point.\n")


# ---------------------------------------------------------------
# 9. SANITY CHECKS
# ---------------------------------------------------------------
print("=" * 60)
print("SANITY CHECKS")
print("=" * 60)

median_tau = (K * train.sigma_5d).median()
print(f"  Median tau               : {median_tau:.2%} of price")
print(f"  Cost assumption          : 0.05% round trip")
print(f"  Ratio (tau / cost)       : {median_tau / 0.0005:.0f}x   (near 1x = broken)")
print(f"  Mean excess return       : {train.excess_5d.mean():+.4%}   (should be ~0)")
print(f"  Max index gap (cal days) : {train.index.to_series().diff().max().days}"
      f"   (>5 outside holidays = missing rows)\n")

print("  REVERSAL VERDICT:")
edge = acc_anti - acc_random
print(f"    Anti-persistence beats random by {edge:+.1%} "
      f"({edge/se_rand:+.1f} standard errors)")
if edge > 2 * se_rand:
    print("    >>> SIGNIFICANT. Real evidence for short-term reversal.")
    print("        Expect ret_5d to carry NEGATIVE SHAP contribution in Phase 5.")
elif edge > 0:
    print("    >>> SUGGESTIVE, not significant. Directionally consistent with the")
    print("        reversal literature but within noise. Treat as a hint that")
    print("        points your feature engineering — not as a finding.")
else:
    print("    >>> NO reversal signal. Both momentum and reversal look dead at 5d.")
    print("        Any edge must come purely from CONDITIONING on vol/regime.")

print("\n  Compare every number above against its v3 counterpart. If any of them")
print("  moved by more than its SE, that is a finding — log it in Appendix A.")
```

---

## Phase 2 checklist — remaining work

1. **FRED VIX ingest.** Tiingo has no index coverage (`^VIX` isn't a
   security). Use FRED's `VIXCLS`. Lag one trading day before joining to
   SPY (see decision #2 above). Sequence matters: build panel → compute
   features/labels → drop VIX-NaN rows last, not at join time.
2. **Validation module** (`data/validate.py`) — pull the inline asserts and
   jump-check out of the script above into reusable functions:
   `no_duplicate_dates`, `no_gaps_vs_exchange_calendar` (use
   `pandas_market_calendars` against the real NYSE calendar, not just a
   raw day-gap heuristic), `prices_positive_and_nonnull`,
   `no_unexplained_jumps`, `volume_sane`.
3. **Purged walk-forward splitter.** Date-based purge/embargo (not
   `iloc`-based), because rows can be missing from the panel. Parameters
   already decided: 5-day purge (= label horizon), 2-day embargo, 1-year
   test windows, 10 folds, expanding window (also run sliding, compare —
   open empirical question per the Phase 1 doc).
4. **Point-in-time recompute test.** Pick ~20 random dates, truncate raw
   data to each, rerun the full feature+label pipeline, assert identical
   output to the full-history run. Catches any centered rolling window or
   globally-fit transform.
5. **Pooled test-fold SE — the gate.** Once the splitter exists, compute
   this before touching Phase 3. If ≥3pp, renegotiate the success
   criterion before building features.
6. **VIX-missing count** — measure against a pre-registered threshold
   rather than asserting "negligible."
7. Cross-check the 60.1% always-long rate on 1993–2004 (open TODO from the
   original Phase 1 doc, unresolved).
8. Read up on Diebold–Mariano properly (open TODO, unresolved).

## Reading list surfaced this session

- CRSP calculations doc (`crsp.com/products/documentation/crsp-calculations`)
  — primary source on the adjustment methodology Tiingo follows.
- QuantConnect `DataNormalizationMode` docs — clearest practical treatment
  of Raw / Adjusted / SplitAdjusted / TotalReturn tradeoffs.
- López de Prado, *AFML* ch. 2 — adjusted-series backtest pitfalls, the
  "ETF trick," and aligning series sampled on different clocks.
- FRED `VIXCLS` series page — notes on non-publication-day handling.
- Cboe VIX Methodology PDF — calculation/dissemination timing (VIX runs to
  4:15pm ET / 3:15pm CT vs SPY's 4:00pm ET close); also documents that on
  days VIX can't be calculated, Cboe republishes the last value (i.e. an
  upstream forward-fill already baked into the raw series on rare days).
- LightGBM "Advanced Topics" — native missing-value handling (not used in
  the end, since the decision was to drop rows uniformly for a fair
  three-model comparison, but relevant if that decision is revisited).
- Hyndman & Athanasopoulos, *fpp3* — missing values in time series,
  general framing.
