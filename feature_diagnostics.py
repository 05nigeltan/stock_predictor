"""
feature_diagnostics.py — the drift check  (Phase 3 deliverable, part 2)
=======================================================================
"Plot every feature over its history and see which ones drift"
(feature discipline rules 1 and 3: stationary; stable meaning across
regimes). Two outputs, both computed on the TRAINING PERIOD ONLY per the
pre-registered holdout-blind rule (spec v2.1, Phase 3 pre-registration —
feature selection must never see 2022+):

  reports/feature_drift.png   one panel per feature: daily values plus a
                              252d rolling mean, train period
  stdout + reports/feature_drift.md
                              drift table: each feature's mean in the
                              first vs second half of train, the shift in
                              units of pooled std, and a DRIFT flag

The flag is a screen, not a verdict: a feature it flags needs a reason to
stay (vol levels and VIX are regime-persistent BY DESIGN — they are the
conditioning variables). A feature it flags with no story gets fixed or
dropped, and that decision is logged in the spec.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ingest import assemble_panel
from make_features import FEATURES, build_dataset

TRAIN_END  = pd.Timestamp("2021-12-31")
DRIFT_FLAG = 0.25          # |mean shift| > this many pooled stds -> flag
OUT_DIR    = "reports"


def load_train_features():
    raw = pd.read_parquet("data/raw/tiingo_spy_2005-01-01.parquet")
    vix = pd.read_parquet("data/raw/fred_vixcls_2005-01-01.parquet")["vix"]
    df = build_dataset(assemble_panel(raw, vix))
    return df.loc[df.index <= TRAIN_END, list(FEATURES)]


def drift_table(feats: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in feats.columns:
        s = feats[col].dropna()
        a, b = s.iloc[: len(s) // 2], s.iloc[len(s) // 2:]
        pooled = s.std()
        shift = (b.mean() - a.mean()) / pooled if pooled > 0 else 0.0
        rows.append({
            "feature": col,
            "mean_1st_half": a.mean(), "mean_2nd_half": b.mean(),
            "shift_in_sd": shift,
            "flag": "DRIFT" if abs(shift) > DRIFT_FLAG else "",
        })
    return pd.DataFrame(rows).set_index("feature")


def plot_grid(feats: pd.DataFrame, table: pd.DataFrame, path: str):
    n = len(feats.columns)
    ncols, nrows = 3, int(np.ceil(n / 3))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 2.2 * nrows),
                             sharex=True)
    for ax, col in zip(axes.flat, feats.columns):
        s = feats[col].dropna()
        ax.plot(s.index, s.values, lw=0.3, alpha=0.5)
        ax.plot(s.index, s.rolling(252).mean(), lw=1.2)
        flagged = table.loc[col, "flag"] == "DRIFT"
        ax.set_title(col + ("   [DRIFT]" if flagged else ""),
                     fontsize=9, loc="left",
                     color="crimson" if flagged else "black")
        ax.tick_params(labelsize=7)
    for ax in axes.flat[n:]:
        ax.axis("off")
    fig.suptitle("Feature drift — TRAIN PERIOD ONLY (2005 – 2021-12-31); "
                 "thick line = 252d rolling mean", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print(f"[written] {path}")


def correlation_report(feats: pd.DataFrame, path: str, flag_at=0.8):
    """Train-only Spearman correlation. Not a selection tool — a map for
    reading SHAP in Phase 5: importance smears across correlated copies,
    so 'no single feature dominates' inside a cluster is not 'no signal'."""
    corr = feats.corr(method="spearman")

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(corr)), corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr)), corr.columns, fontsize=7)
    fig.colorbar(im, shrink=0.8)
    ax.set_title("Feature correlation (Spearman) — train period only",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print(f"[written] {path}")

    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    pairs = (upper.stack().loc[lambda s: s.abs() > flag_at]
                  .sort_values(key=abs, ascending=False))
    print(f"\nClusters (|rho| > {flag_at}) — expect SHAP to smear inside these:")
    for (a, b), rho in pairs.items():
        print(f"  {a:>16} ~ {b:<16} {rho:+.2f}")
    return pairs


if __name__ == "__main__":
    import os
    os.makedirs(OUT_DIR, exist_ok=True)

    feats = load_train_features()
    table = drift_table(feats)

    with pd.option_context("display.float_format", "{:.4f}".format):
        print(table.to_string())
    n_flagged = (table.flag == "DRIFT").sum()
    print(f"\n{n_flagged}/{len(table)} features flagged at "
          f"|shift| > {DRIFT_FLAG} pooled sd")

    with open(f"{OUT_DIR}/feature_drift.md", "w") as f:
        f.write("# Feature drift table — train period only "
                f"(through {TRAIN_END.date()})\n\n")
        f.write(table.round(4).to_markdown())
        f.write(f"\n\nFlag threshold: |mean shift| > {DRIFT_FLAG} pooled sd "
                "between train halves.\n")
    print(f"[written] {OUT_DIR}/feature_drift.md")

    plot_grid(feats, table, f"{OUT_DIR}/feature_drift.png")
    correlation_report(feats, f"{OUT_DIR}/feature_corr.png")
