"""
shap_falsification.py — the §1.5 pre-registered falsification test
==================================================================
Phase 1 doc, §1.5, verbatim:

  "If SHAP shows no importance for ret_5d, or shows it with a positive
   contribution, the reversal hypothesis is dead. The hypothesis
   predicts a negative contribution from recent returns, strengthening
   in high-volatility regimes."

Sign translation for the ternary model: "negative contribution from
recent returns" = high recent returns push AWAY from beat and TOWARD
lag. In LAG-class SHAP terms the hypothesis predicts
corr(ret_5d, SHAP_lag(ret_5d)) > 0, and the BEAT-class mirror < 0.

Method: E3's exact walk-forward models (same config, same seed —
deterministic refit), each explaining ITS OWN test window via
LightGBM's native exact TreeSHAP (pred_contrib=True; no new
dependency), pooled over 2012-2021. Out-of-sample SHAP: attribution is
measured on the same days the evaluation was.

This is ANALYSIS of the already-run E3 configuration — no test-fold
selection decision is made here — logged as A1, not an E-entry.

Interpretive guard (correlation map, reports/feature_corr.png):
ret_5d sits in a >=0.8-correlated position cluster (rsi_14, bb_pctb,
px_to_sma20, ret_10d ...). Trees split on whichever cluster member is
locally convenient, smearing credit. The verdict below therefore also
reports the CLUSTER's aggregate reversal direction — §1.5's substance —
alongside the named-feature reading, and says which is which.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from make_features import FEATURES, HORIZON
from phase4_baselines import SEED, load_everything

# The position cluster around ret_5d (|rho| >= ~0.8 on train).
CLUSTER = ("ret_5d", "ret_10d", "px_to_sma20", "rsi_14", "bb_pctb")
BLUE, ORANGE = "#3465c0", "#d9711e"          # CVD-safe pair, fixed order


def fit_and_explain():
    import lightgbm as lgb

    df, matrix, folds, test_dates, fold_id, y_true, r = load_everything()
    X = matrix[list(FEATURES)]
    y = matrix["y"]
    params = dict(objective="multiclass", num_class=3, learning_rate=0.01,
                  num_leaves=7, max_depth=3, min_data_in_leaf=50,
                  feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1,
                  lambda_l2=10.0, n_estimators=2000, random_state=SEED,
                  verbosity=-1)

    F = len(FEATURES)
    shap_lag, shap_beat, X_test = [], [], []
    for f in folds:
        n_valid = int(0.15 * len(f.train_dates))
        inner = f.train_dates[: -(n_valid + HORIZON)]
        valid = f.train_dates[-n_valid:]
        model = lgb.LGBMClassifier(**params)
        model.fit(X.loc[inner], y.loc[inner],
                  eval_set=[(X.loc[valid], y.loc[valid])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
        assert list(model.classes_) == [-1.0, 0.0, 1.0]

        contrib = model.booster_.predict(X.loc[f.test_dates],
                                         pred_contrib=True)
        assert contrib.shape[1] == 3 * (F + 1)
        shap_lag.append(contrib[:, 0 * (F + 1): 1 * (F + 1) - 1])   # class -1
        shap_beat.append(contrib[:, 2 * (F + 1): 3 * (F + 1) - 1])  # class +1
        X_test.append(X.loc[f.test_dates])

    Xt = pd.concat(X_test)
    return (pd.DataFrame(np.vstack(shap_lag), columns=FEATURES, index=Xt.index),
            pd.DataFrame(np.vstack(shap_beat), columns=FEATURES, index=Xt.index),
            Xt)


def main():
    shap_lag, shap_beat, Xt = fit_and_explain()
    lines = []
    def emit(s=""):
        print(s); lines.append(s)

    imp = shap_lag.abs().mean().sort_values(ascending=False)
    rank = int(np.where(imp.index == "ret_5d")[0][0]) + 1

    emit("=" * 72)
    emit("SHAP FALSIFICATION TEST (§1.5) — E3 models, out-of-sample, pooled")
    emit("=" * 72)
    emit(f"  Days explained: {len(Xt)} (2012-2021 test folds)")
    emit(f"\n  LAG-class mean |SHAP| ranking (top 10 of {len(imp)}):")
    for i, (n, v) in enumerate(imp.head(10).items(), 1):
        mark = "  <-- ret_5d" if n == "ret_5d" else ""
        emit(f"    {i:>2}. {n:>16}  {v:.4f}{mark}")
    if rank > 10:
        emit(f"    ... ret_5d ranks {rank} ({imp['ret_5d']:.4f})")

    # --- Sign: the named feature, then the cluster ------------------
    emit(f"\n  Direction (hypothesis: high recent return -> toward LAG):")
    emit(f"  {'feature':>16} {'corr w/ own LAG-SHAP':>21} "
         f"{'corr w/ own BEAT-SHAP':>22}")
    for n in CLUSTER:
        cl = np.corrcoef(Xt[n], shap_lag[n])[0, 1]
        cb = np.corrcoef(Xt[n], shap_beat[n])[0, 1]
        emit(f"  {n:>16} {cl:>21.2f} {cb:>22.2f}")

    q = Xt["ret_5d"].quantile([0.25, 0.75])
    top = shap_lag["ret_5d"][Xt["ret_5d"] >= q[0.75]].mean()
    bot = shap_lag["ret_5d"][Xt["ret_5d"] <= q[0.25]].mean()
    emit(f"\n  ret_5d LAG-SHAP, top vs bottom quartile of ret_5d: "
         f"{top:+.4f} vs {bot:+.4f}")

    # Cluster aggregate: summed LAG-SHAP of the cluster vs a composite
    # "extension" score (mean of z-scored members) — the substance test.
    z = (Xt[list(CLUSTER)] - Xt[list(CLUSTER)].mean()) / Xt[list(CLUSTER)].std()
    ext = z.mean(axis=1).to_numpy()
    csum = shap_lag[list(CLUSTER)].sum(axis=1).to_numpy()
    c_cluster = np.corrcoef(ext, csum)[0, 1]
    emit(f"  CLUSTER (position/extension) summed LAG-SHAP vs extension "
         f"score: corr {c_cluster:+.2f}")

    # --- Vol conditioning -------------------------------------------
    hi = Xt["vol_21d"] >= Xt["vol_21d"].median()
    s_hi = np.polyfit(Xt.loc[hi, "ret_5d"], shap_lag.loc[hi, "ret_5d"], 1)[0]
    s_lo = np.polyfit(Xt.loc[~hi, "ret_5d"], shap_lag.loc[~hi, "ret_5d"], 1)[0]
    emit(f"\n  Vol conditioning (slope of ret_5d LAG-SHAP on ret_5d):")
    emit(f"    high-vol half: {s_hi:+.3f}    low-vol half: {s_lo:+.3f}"
         f"    ratio {s_hi / s_lo if s_lo != 0 else float('inf'):.1f}x")

    # --- Verdict per §1.5 -------------------------------------------
    named_alive = imp["ret_5d"] > 0.25 * imp.iloc[0] and top > bot
    emit("\n  VERDICT (§1.5):")
    emit(f"    Named feature (ret_5d): importance rank {rank}, "
         f"direction {'REVERSAL (high ret -> lag)' if top > bot else 'WRONG SIGN'}"
         f" -> {'alive' if named_alive else 'weak/dead on the letter'}")
    emit(f"    Cluster substance: corr {c_cluster:+.2f} "
         f"-> {'REVERSAL-shaped: extension pushes toward lag' if c_cluster > 0 else 'NOT reversal-shaped'}")
    emit(f"    Vol strengthening: {'CONFIRMED' if s_hi > s_lo > 0 or (s_hi > s_lo and s_hi > 0) else 'NOT confirmed'}"
         f" (predicted: steeper in high vol)")
    emit("\n    Context from E1-E4: whatever survives here is the DIRECTIONAL")
    emit("    signal (PT <= 0.02 everywhere) that could not be monetized.")
    emit("    This test adjudicates the HYPOTHESIS's shape, not the P&L.")

    # --- Figure -------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    top10 = imp.head(10)[::-1]
    ax1.barh(range(len(top10)), top10.values, height=0.55,
             color=[ORANGE if n == "ret_5d" else BLUE for n in top10.index])
    ax1.set_yticks(range(len(top10)), top10.index, fontsize=8)
    ax1.set_xlabel("mean |SHAP|, LAG class", fontsize=9)
    ax1.set_title("Attribution magnitude (ret_5d highlighted)",
                  fontsize=10, loc="left")
    ax1.grid(axis="x", alpha=0.25); ax1.set_axisbelow(True)
    for s in ("top", "right"):
        ax1.spines[s].set_visible(False)

    for mask, color, label in [(~hi, BLUE, "low-vol half"),
                               (hi, ORANGE, "high-vol half")]:
        xs = Xt.loc[mask, "ret_5d"]
        ys = shap_lag.loc[mask, "ret_5d"]
        ax2.scatter(xs, ys, s=6, alpha=0.35, color=color, label=label,
                    edgecolors="none")
        k, b = np.polyfit(xs, ys, 1)
        xr = np.linspace(xs.min(), xs.max(), 2)
        ax2.plot(xr, k * xr + b, color=color, lw=2)
    ax2.axhline(0, color="gray", lw=0.8, alpha=0.6)
    ax2.set_xlabel("ret_5d (trailing 5-day return)", fontsize=9)
    ax2.set_ylabel("SHAP toward LAG", fontsize=9)
    ax2.set_title("§1.5: reversal sign, by vol regime", fontsize=10,
                  loc="left")
    ax2.legend(frameon=False, fontsize=9)
    ax2.grid(alpha=0.25); ax2.set_axisbelow(True)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)
    fig.suptitle("SHAP falsification test — E3 walk-forward models, "
                 "out-of-sample, 2012-2021", fontsize=11)
    fig.tight_layout()
    fig.savefig("reports/shap_falsification.png", dpi=110)
    print("\n[written] reports/shap_falsification.png")

    with open("reports/shap_falsification.md", "w") as fh:
        fh.write(f"# SHAP falsification test (§1.5) — A1 "
                 f"(seed {SEED}, {pd.Timestamp.now().date()})\n\n```\n")
        fh.write("\n".join(lines))
        fh.write("\n```\n")
    print("[written] reports/shap_falsification.md")


if __name__ == "__main__":
    main()
