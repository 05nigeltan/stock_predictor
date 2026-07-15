"""
holdout.py — THE single evaluation of the locked holdout (2022 -> end)
======================================================================
Protocol pre-registered in experiment_log.md (H1) before this ran.
Spec §0: "Untouched until exactly one final evaluation. If evaluated
more than once, the number is void." This script's one execution,
recorded in git, is that evaluation. Purely descriptive — nothing
passed the gates on 2012-2021, so this measures the stability of the
NEGATIVE result on unseen years, not a success claim.

Three strategies, nothing else (see H1 for why E1 is excluded):
  always-long | anti-persistence rule | LightGBM (frozen E3 config)

Operating simulation: expanding annual retrains through the holdout —
train through each year-end (purged, embargoed), predict the next
year. By 2024 the model legitimately knows 2022-23; that is how the
system would actually have been run.
"""

import numpy as np
import pandas as pd

from evaluate import (LAG, active_returns, annualized_ir,
                      moving_block_bootstrap_ir, pt_pvalue)
from ingest import build_panel
from make_features import FEATURES, HORIZON, build_dataset, materialize
from splitter import PurgedWalkForward

SEED = 20260713


def main():
    import lightgbm as lgb

    panel = build_panel(write=False)
    df = build_dataset(panel)
    matrix = materialize(panel)

    # Annual test windows 2022..2026(partial), expanding train from 2005.
    wf = PurgedWalkForward(panel.index, n_folds=5,
                           train_end=pd.Timestamp("2026-12-31"))
    folds = wf.split(matrix.index)
    wf.assert_leak_free(folds)
    assert folds[0].train_dates.max() < pd.Timestamp("2022-01-01"), \
        "fold 0 must train strictly inside the old locked-train period"

    test_dates = folds[0].test_dates.append([f.test_dates for f in folds[1:]])
    fold_id = np.concatenate(
        [np.full(len(f.test_dates), f.fold) for f in folds])
    y_true = (df["y"].loc[test_dates].to_numpy() + 1).astype(np.int64)
    r = df["log_ret"].loc[test_dates].to_numpy()
    years = test_dates.year.to_numpy()

    X, y = matrix[list(FEATURES)], matrix["y"]
    params = dict(objective="multiclass", num_class=3, learning_rate=0.01,
                  num_leaves=7, max_depth=3, min_data_in_leaf=50,
                  feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1,
                  lambda_l2=10.0, n_estimators=2000, random_state=SEED,
                  verbosity=-1)

    strategies = {}
    strategies["always-long"] = np.full(len(y_true), 2, dtype=np.int64)
    strategies["anti-persistence"] = (
        -df["y"].shift(HORIZON).loc[test_dates].to_numpy() + 1
    ).astype(np.int64)

    preds = []
    for f in folds:
        n_valid = int(0.15 * len(f.train_dates))
        inner = f.train_dates[: -(n_valid + HORIZON)]
        valid = f.train_dates[-n_valid:]
        model = lgb.LGBMClassifier(**params)
        model.fit(X.loc[inner], y.loc[inner],
                  eval_set=[(X.loc[valid], y.loc[valid])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
        preds.append(model.predict(X.loc[f.test_dates]))
    strategies["lightgbm (E3 cfg)"] = (
        np.concatenate(preds) + 1).astype(np.int64)

    # ---- report -------------------------------------------------------
    lines = []
    def emit(s=""):
        print(s); lines.append(s)

    emit("=" * 78)
    emit("LOCKED HOLDOUT — THE single evaluation  (protocol H1)")
    emit("=" * 78)
    emit(f"  Window : {test_dates.min().date()} -> {test_dates.max().date()}"
         f"   ({len(test_dates)} days, ~{len(test_dates) // HORIZON} "
         f"independent 5d samples)")
    emit(f"  Train boundary honored: fold-0 train ends "
         f"{folds[0].train_dates.max().date()} (purged+embargoed)")
    emit(f"  SPY buy-and-hold over the window: "
         f"{np.exp(r.sum()) - 1:+.1%} total")
    emit()
    emit(f"  {'strategy':<20} {'acc':>6} {'activ':>6} "
         f"{'IR [90% CI]':>21} {'ann.a':>7} {'yrs+':>5} {'PT p':>6} "
         f"{'lag prec':>9}")
    emit("  " + "-" * 88)

    per_year = {}
    for name, codes in strategies.items():
        alpha = active_returns(codes[None, :], r)[0]
        ir, lo, hi = moving_block_bootstrap_ir(alpha, seed=SEED)
        _, mu = annualized_ir(alpha[None, :])
        yr_means = {yr: alpha[years == yr].mean() * 252
                    for yr in np.unique(years)}
        per_year[name] = yr_means
        wins = sum(v > 0 for v in yr_means.values())
        pt = pt_pvalue(codes[None, :], y_true)[0]
        fired = codes == LAG
        lag_prec = (y_true[fired] == LAG).mean() if fired.any() else np.nan
        acc = (codes == y_true).mean()
        emit(f"  {name:<20} {acc:>6.1%} {fired.mean():>6.1%} "
             f"{ir:>6.2f} [{lo:>5.2f},{hi:>5.2f}] {mu[0]:>7.2%} "
             f"{wins:>3d}/{len(yr_means)} {pt:>6.2f} "
             f"{lag_prec:>9.1%}" if fired.any() else
             f"  {name:<20} {acc:>6.1%} {fired.mean():>6.1%} "
             f"{ir:>6.2f} [{lo:>5.2f},{hi:>5.2f}] {mu[0]:>7.2%} "
             f"{wins:>3d}/{len(yr_means)} {pt:>6.2f} {'--':>9}")

    emit()
    emit("  Net active return by calendar year (annualized):")
    yrs = sorted(per_year["always-long"])
    emit("  " + f"{'strategy':<20}" + "".join(f"{y:>9}" for y in yrs))
    for name, ym in per_year.items():
        emit("  " + f"{name:<20}"
             + "".join(f"{ym[y]:>+9.1%}" for y in yrs))

    emit()
    emit("  Comparison, pooled 2012-2021 test folds (for the record):")
    emit("    anti-persistence: IR -0.77   lightgbm: IR -0.57")
    emit()
    emit("  THE HOLDOUT IS NOW SPENT. Per spec §0, any further evaluation")
    emit("  of 2022+ data by any configuration is void.")

    with open("reports/holdout_final.md", "w") as fh:
        fh.write(f"# Locked holdout — the single evaluation (H1; seed "
                 f"{SEED}, run {pd.Timestamp.now().date()})\n\n```\n")
        fh.write("\n".join(lines))
        fh.write("\n```\n")
    print("\n[written] reports/holdout_final.md")


if __name__ == "__main__":
    main()
