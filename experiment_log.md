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

*(none yet — E1 opens Phase 5)*
