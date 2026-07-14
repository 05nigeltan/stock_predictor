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


if __name__ == "__main__":
    import sys
    globals()[sys.argv[1] if len(sys.argv) > 1 else "e1"]()
