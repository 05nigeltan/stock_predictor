"""
phase4_baselines.py — the bar, written down before any model exists
===================================================================
Every number here is a REFERENCE POINT, logged as B-entries in
experiment_log.md. None counts against the 25-experiment budget; the
budget starts at E1, the first candidate configuration evaluated on
test folds with intent to beat these.

Six baselines, all pushed through the same chain the criterion lives on
(evaluate.py: long/flat tranche book, costs, IR + 90% block-bootstrap
CI, >=6/10 fold gate, m-class PT):

  B1  always-long          the benchmark itself: IR = 0 by definition.
                           The anchor, not a gated strategy.
  B2  dummy most_frequent  predicts the modal train class, walk-forward.
  B3  dummy stratified     random draws at train class frequencies.
  B4  persistence rule     predict the PREVIOUS non-overlapping window's
                           class (label at t-5, fully observed at t).
  B5  anti-persistence     the OPPOSITE — the reversal hypothesis as an
                           executable strategy. THE baseline to beat.
  B6  depth-3 tree         simplest real model, walk-forward on the 21
                           features. If LightGBM barely beats this,
                           that is diagnostic.

Classification accuracy is reported next to the strategy metrics, but
the CRITERION is the strategy side — spec v2.1.

IR_min lookup: frozen luck bars from power_results.md (seed 20260713,
post-A13), keyed by realized activity, nearest level, conservative side.
Activity < 10% -> below the evaluability floor (spec #7).
"""

import numpy as np
import pandas as pd

from evaluate import (ALPHA_TEST, LAG, active_returns, annualized_ir,
                      moving_block_bootstrap_ir, pt_pvalue)
from ingest import build_panel
from make_features import FEATURES, HORIZON, build_dataset, materialize
from splitter import PurgedWalkForward

# Frozen luck bars (power_results.md, post-A13). Keyed by activity level.
IR_MIN = {0.10: 0.06, 0.20: -0.11, 0.30: -0.25}
ACTIVITY_FLOOR = 0.10
SEED = 20260713


def ir_min_for(activity):
    if activity < ACTIVITY_FLOOR:
        return None                                  # below evaluability floor
    level = min((a for a in IR_MIN if a >= activity), default=0.30)
    return IR_MIN[level]


def load_everything():
    panel = build_panel(write=False)
    df = build_dataset(panel)                        # complete calendar
    matrix = materialize(panel)
    folds = PurgedWalkForward(panel.index).split(matrix.index)

    test_dates = folds[0].test_dates.append([f.test_dates for f in folds[1:]])
    fold_id = np.concatenate(
        [np.full(len(f.test_dates), f.fold) for f in folds])

    y_true = (df["y"].loc[test_dates].to_numpy() + 1).astype(np.int64)
    r = df["log_ret"].loc[test_dates].to_numpy()
    return df, matrix, folds, test_dates, fold_id, y_true, r


def evaluate_strategy(name, codes, r, y_true, fold_id):
    """codes: 1-D int predictions in {LAG,FLAT,BEAT} on the pooled test."""
    alpha = active_returns(codes[None, :], r)[0]
    ir, lo, hi = moving_block_bootstrap_ir(alpha, seed=SEED)
    _, ann_mu = annualized_ir(alpha[None, :])
    wins = sum(alpha[fold_id == k].mean() > 0 for k in range(10))
    pt = pt_pvalue(codes[None, :], y_true)[0]
    activity = (codes == LAG).mean()
    acc = (codes == y_true).mean()

    bar = ir_min_for(activity)
    if name == "always-long":
        verdict = "anchor (IR=0 by definition)"
    elif bar is None:
        verdict = f"below {ACTIVITY_FLOOR:.0%} activity floor"
    else:
        passed = (ir >= bar) and (wins >= 6) and (pt < ALPHA_TEST)
        verdict = ("PASSES ALL GATES" if passed else
                   f"fails ({'IR' if ir < bar else ''}"
                   f"{' folds' if wins < 6 else ''}"
                   f"{' PT' if pt >= ALPHA_TEST else ''} )".replace("( ", "(")
                   .replace(" )", ")"))
    return dict(name=name, acc=acc, activity=activity, ir=ir, lo=lo, hi=hi,
                ann_mu=ann_mu[0], wins=wins, pt=pt, verdict=verdict)


def walkforward_classifier(make_model, matrix, folds):
    """Train per expanding fold, predict its test window. The splitter
    guarantees purge/embargo; asserted anyway because it is cheap."""
    out = []
    for f in folds:
        assert f.train_dates.max() < f.test_dates.min()
        model = make_model()
        model.fit(matrix.loc[f.train_dates, list(FEATURES)],
                  matrix.loc[f.train_dates, "y"])
        out.append(model.predict(matrix.loc[f.test_dates, list(FEATURES)]))
    return (np.concatenate(out) + 1).astype(np.int64), model  # codes, last model


def main():
    from sklearn.dummy import DummyClassifier
    from sklearn.tree import DecisionTreeClassifier, export_text

    df, matrix, folds, test_dates, fold_id, y_true, r = load_everything()
    results = []

    # B1 — always-long
    results.append(evaluate_strategy(
        "always-long", np.full(len(y_true), 2, dtype=np.int64),
        r, y_true, fold_id))

    # B2/B3 — dummies, walk-forward
    for name, factory in [
        ("dummy most_frequent",
         lambda: DummyClassifier(strategy="most_frequent")),
        ("dummy stratified",
         lambda: DummyClassifier(strategy="stratified", random_state=SEED)),
    ]:
        codes, _ = walkforward_classifier(factory, matrix, folds)
        results.append(evaluate_strategy(name, codes, r, y_true, fold_id))

    # B4/B5 — persistence rules. The previous NON-OVERLAPPING window's
    # label: y at t-5 covers (t-5 -> t], fully observed at t. Computed on
    # the complete calendar (shift = 5 sessions), then read at test dates.
    y_prev = df["y"].shift(HORIZON).loc[test_dates].to_numpy()
    for name, sign in [("persistence rule", 1), ("anti-persistence rule", -1)]:
        codes = (sign * y_prev + 1).astype(np.int64)
        results.append(evaluate_strategy(name, codes, r, y_true, fold_id))

    # B6 — depth-3 tree, walk-forward
    codes, last_tree = walkforward_classifier(
        lambda: DecisionTreeClassifier(max_depth=3, random_state=SEED),
        matrix, folds)
    results.append(evaluate_strategy("depth-3 tree", codes, r, y_true, fold_id))

    # ---- report ------------------------------------------------------
    lines = []
    def emit(s=""):
        print(s); lines.append(s)

    p = pd.Series(y_true).value_counts(normalize=True)
    floor = (p ** 2).sum()
    emit("=" * 78)
    emit("PHASE 4 BASELINES — pooled walk-forward test folds 2012-2021")
    emit("=" * 78)
    emit(f"  Test days: {len(y_true)}   random-guess accuracy floor "
         f"(sum p^2): {floor:.1%}")
    emit(f"  Gates: IR >= IR_min(activity)  AND  >=6/10 folds  AND  "
         f"PT p < {ALPHA_TEST}")
    emit()
    emit(f"  {'baseline':<22} {'acc':>6} {'activ':>6} "
         f"{'IR [90% CI]':>21} {'ann.a':>7} {'folds+':>6} {'PT p':>6}  verdict")
    emit("  " + "-" * 92)
    for x in results:
        emit(f"  {x['name']:<22} {x['acc']:>6.1%} {x['activity']:>6.1%} "
             f"{x['ir']:>6.2f} [{x['lo']:>5.2f},{x['hi']:>5.2f}] "
             f"{x['ann_mu']:>7.2%} {x['wins']:>5d}/10 {x['pt']:>6.2f}  "
             f"{x['verdict']}")

    emit()
    emit("  Depth-3 tree, final fold (trained 2005-10 -> 2020-12) — the")
    emit("  'simplest real model', printed because it can be:")
    for ln in export_text(last_tree, feature_names=list(FEATURES)).splitlines():
        emit(f"    {ln}")

    with open("reports/phase4_baselines.md", "w") as fh:
        fh.write("# Phase 4 baselines — pre-registered "
                 f"(seed {SEED}, {pd.Timestamp.now().date()})\n\n```\n")
        fh.write("\n".join(lines))
        fh.write("\n```\n")
    print("\n[written] reports/phase4_baselines.md")


if __name__ == "__main__":
    main()
