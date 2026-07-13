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

Install:  pip install pandas numpy pyarrow requests python-dotenv
          pip install yfinance          # only for the reconciliation check
Set:      TIINGO_TOKEN=... in a .env file beside this script (or in the shell env)
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))

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
TOKEN       = os.environ.get("TIINGO_TOKEN")
if not TOKEN or TOKEN.startswith("paste_your"):
    raise SystemExit(
        "TIINGO_TOKEN is not set. Put your key in the .env file beside this "
        "script:\n\n    TIINGO_TOKEN=your_key_here\n"
    )


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