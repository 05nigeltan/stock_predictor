"""
experiments.py — E-entries against the 25-experiment budget
===========================================================
One function per named configuration; each run appends its row to
reports/ and is logged by hand in experiment_log.md (append-only).
The harness is imported from phase4_baselines so every experiment is
measured by literally the same code path as the bar it must clear.

Contract reminder (experiment_log.md): the configuration is named in
full BEFORE test folds are read. No threshold sweeps, no post-hoc
variants inside one E-number.
"""

import pandas as pd

from evaluate import ALPHA_TEST
from phase4_baselines import (SEED, evaluate_strategy, load_everything,
                              walkforward_classifier)


def report(entry, x, extra_lines=()):
    lines = [
        "=" * 78,
        f"{entry} — pooled walk-forward test folds 2012-2021",
        "=" * 78,
        f"  {'config':<22} {'acc':>6} {'activ':>6} "
        f"{'IR [90% CI]':>21} {'ann.a':>7} {'folds+':>6} {'PT p':>6}  verdict",
        "  " + "-" * 92,
        f"  {x['name']:<22} {x['acc']:>6.1%} {x['activity']:>6.1%} "
        f"{x['ir']:>6.2f} [{x['lo']:>5.2f},{x['hi']:>5.2f}] "
        f"{x['ann_mu']:>7.2%} {x['wins']:>5d}/10 {x['pt']:>6.2f}  "
        f"{x['verdict']}",
        *extra_lines,
    ]
    print("\n".join(lines))
    path = f"reports/{entry.lower().replace(' ', '_')}.md"
    with open(path, "w") as fh:
        fh.write(f"# {entry} — pre-registered (seed {SEED}, "
                 f"{pd.Timestamp.now().date()})\n\n```\n"
                 + "\n".join(lines) + "\n```\n")
    print(f"\n[written] {path}")
    return x


def e1():
    """E1 — multinomial logistic regression, the linear rung.

    Pre-registered config: LogisticRegression(C=1.0, L2, lbfgs,
    max_iter=5000), features standard-scaled INSIDE the fold pipeline
    (scaler fit on each fold's train only), argmax predictions, no
    probability thresholding. Purpose: price the LINEAR share of the
    signal. If trees don't beat this, the features have no nonlinear
    structure worth modelling.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    df, matrix, folds, test_dates, fold_id, y_true, r = load_everything()
    codes, last_model = walkforward_classifier(
        lambda: make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=5000)),
        matrix, folds)
    x = evaluate_strategy("E1 logistic (argmax)", codes, r, y_true, fold_id)

    # Coefficients of the final fold's model, for the record: which
    # features carry the linear signal, and with what sign.
    import numpy as np
    lr = last_model.named_steps["logisticregression"]
    feats = matrix.columns[:-1]
    lag_row = list(lr.classes_).index(-1.0)
    coef = pd.Series(lr.coef_[lag_row], index=feats).sort_values(key=abs,
                                                                 ascending=False)
    extra = ["", "  Final-fold coefficients, LAG class (top 8 by |coef|,",
             "  standardized units — sign read: positive pushes toward lag):"]
    extra += [f"    {n:>16}: {v:+.3f}" for n, v in coef.head(8).items()]
    return report("E1 logistic", x, extra)


def e2():
    """E2 — random forest, the bagging rung.

    Pre-registered config: RandomForestClassifier(n_estimators=500,
    min_samples_leaf=50, max_features='sqrt', random_state=SEED),
    argmax, no scaling (trees are scale-invariant), no tuning.
    min_samples_leaf=50 mirrors the frozen LightGBM philosophy: with
    ~843 effective samples, every leaf must be supported by real data.
    Purpose: the hard-to-overfit-badly sanity rung between the linear
    model and boosting. Must beat E1 (acc 39.0 / IR -0.49) to show the
    features carry nonlinear structure.
    """
    from sklearn.ensemble import RandomForestClassifier

    df, matrix, folds, test_dates, fold_id, y_true, r = load_everything()
    codes, last_model = walkforward_classifier(
        lambda: RandomForestClassifier(
            n_estimators=500, min_samples_leaf=50, max_features="sqrt",
            random_state=SEED, n_jobs=-1),
        matrix, folds)
    x = evaluate_strategy("E2 rand forest (argmax)", codes, r, y_true,
                          fold_id)

    # Final-fold impurity importances, for the record only — biased
    # toward high-cardinality features and smeared by the correlated
    # clusters; the real attribution verdict waits for SHAP in Phase 5.
    feats = matrix.columns[:-1]
    imp = pd.Series(last_model.feature_importances_, index=feats
                    ).sort_values(ascending=False)
    extra = ["", "  Final-fold impurity importances (top 8; record only —",
             "  cluster-smeared, SHAP is the verdict):"]
    extra += [f"    {n:>16}: {v:.3f}" for n, v in imp.head(8).items()]
    return report("E2 random forest", x, extra)


def e3():
    """E3 — LightGBM, the workhorse, frozen regularize-hard config.

    Pre-registered config (unchanged from the plan doc): multiclass,
    learning_rate=0.01, num_leaves=7, max_depth=3, min_data_in_leaf=50,
    feature_fraction=0.7, bagging_fraction=0.7 (freq 1), lambda_l2=10.0,
    n_estimators=2000 ceiling with early_stopping_rounds=100 on
    multiclass logloss. Validation for early stopping: the LAST 15% of
    each fold's train window (chronological), separated from the inner
    train by a 5-session purge — same purge rule as every other
    boundary; never test data. Argmax, no tuning. Seed 20260713.

    Two-sided question: beat E1's IR (-0.49) to justify boosting; beat
    E2's acc (39.8%) to claim nonlinear signal exists at all.
    """
    import lightgbm as lgb
    import numpy as np

    from make_features import FEATURES, HORIZON

    df, matrix, folds, test_dates, fold_id, y_true, r = load_everything()
    X, y = matrix[list(FEATURES)], matrix["y"]

    params = dict(objective="multiclass", num_class=3, learning_rate=0.01,
                  num_leaves=7, max_depth=3, min_data_in_leaf=50,
                  feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1,
                  lambda_l2=10.0, n_estimators=2000, random_state=SEED,
                  verbosity=-1)

    out, best_iters = [], []
    for f in folds:
        assert f.train_dates.max() < f.test_dates.min()
        n_valid = int(0.15 * len(f.train_dates))
        inner = f.train_dates[: -(n_valid + HORIZON)]   # 5-session purge
        valid = f.train_dates[-n_valid:]

        model = lgb.LGBMClassifier(**params)
        model.fit(X.loc[inner], y.loc[inner],
                  eval_set=[(X.loc[valid], y.loc[valid])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
        best_iters.append(model.best_iteration_)
        out.append(model.predict(X.loc[f.test_dates]))

    codes = (np.concatenate(out) + 1).astype(np.int64)
    x = evaluate_strategy("E3 lightgbm (argmax)", codes, r, y_true, fold_id)

    extra = ["", f"  Early stopping picked {min(best_iters)}-"
                 f"{max(best_iters)} trees per fold "
                 f"(mean {np.mean(best_iters):.0f} of 2000 ceiling)"]
    return report("E3 lightgbm", x, extra)


def e4():
    """E4 — the probability tail: E3's LightGBM, fixed top-decile rule.

    Pre-registered config (decisions locked 2026-07-15, before any run):
      - Base model: E3's frozen config verbatim, same inner-valid early
        stopping, seed 20260713. Nothing about the model changes.
      - Rule: quantile, not absolute — only the RANKING of P(lag) is
        used, so LightGBM's miscalibration is irrelevant by design.
      - Threshold: per fold, the 90th percentile of P(lag) on that
        fold's INNER-VALIDATION days (out-of-fit, the honest analogue
        of the test-time score distribution; the test window's own
        quantile would need the future). Fixed at 0.90. Never tuned.
      - Position: flat iff P(lag) >= threshold, else long.
      - PT predictions: LAG when fired, else argmax of {beat, flat}.
    The question: are the model's MOST CONFIDENT lag calls better than
    its average lag calls? power.py's bar at ~10% activity: ~51.5%
    lag precision, IR_min = 0.06.
    """
    import lightgbm as lgb
    import numpy as np

    from make_features import FEATURES, HORIZON

    Q = 0.90                                   # fixed, pre-registered

    df, matrix, folds, test_dates, fold_id, y_true, r = load_everything()
    X, y = matrix[list(FEATURES)], matrix["y"]

    params = dict(objective="multiclass", num_class=3, learning_rate=0.01,
                  num_leaves=7, max_depth=3, min_data_in_leaf=50,
                  feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1,
                  lambda_l2=10.0, n_estimators=2000, random_state=SEED,
                  verbosity=-1)

    out, thresholds = [], []
    for f in folds:
        assert f.train_dates.max() < f.test_dates.min()
        n_valid = int(0.15 * len(f.train_dates))
        inner = f.train_dates[: -(n_valid + HORIZON)]
        valid = f.train_dates[-n_valid:]

        model = lgb.LGBMClassifier(**params)
        model.fit(X.loc[inner], y.loc[inner],
                  eval_set=[(X.loc[valid], y.loc[valid])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])

        cls = list(model.classes_)
        i_lag, i_flat, i_beat = (cls.index(c) for c in (-1.0, 0.0, 1.0))

        thr = np.quantile(model.predict_proba(X.loc[valid])[:, i_lag], Q)
        thresholds.append(thr)

        p = model.predict_proba(X.loc[f.test_dates])
        fired = p[:, i_lag] >= thr
        rest = np.where(p[:, i_beat] >= p[:, i_flat], 1.0, 0.0)
        out.append(np.where(fired, -1.0, rest))

    codes = (np.concatenate(out) + 1).astype(np.int64)
    x = evaluate_strategy(f"E4 lgbm tail (q={Q})", codes, r, y_true, fold_id)

    fired = codes == 0
    lag_hit = (y_true[fired] == 0).mean() if fired.any() else float("nan")
    base = (y_true == 0).mean()
    extra = ["",
             f"  Fired on {fired.sum()} of {len(codes)} days; per-fold "
             f"thresholds {min(thresholds):.3f}-{max(thresholds):.3f}",
             f"  Lag precision on fired days: {lag_hit:.1%}   "
             f"(base rate {base:.1%}; power.py's bar at ~10% activity: "
             f"~51.5%)"]
    return report("E4 lgbm tail", x, extra)


if __name__ == "__main__":
    import sys
    globals()[sys.argv[1] if len(sys.argv) > 1 else "e1"]()
