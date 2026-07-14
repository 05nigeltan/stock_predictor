"""
make_features.py — the flat modelling dataset  (Phase 3 deliverable,
seeded in Phase 2 with the label machinery so the point-in-time harness
has a pipeline to guard from day one)

THE REGISTRY — the discipline this module enforces:
Every column build_dataset() produces MUST be declared below, in exactly
one of two classes. The point-in-time harness (validate.point_in_time_
recompute) refuses unregistered columns and holds each class to a
different invariant:

  BACKWARD — uses only information available at time t.
             Invariant: recomputing as-of any historical date T yields
             bit-identical values for every t <= T. History never moves.
  FORWARD  — reads prices after t. The label machinery, nothing else.
             Invariant: identical where computable as-of T (t + HORIZON
             <= T), and NaN — never a value — where the future is out of
             reach. A number appearing there is fabricated data.

Phase 3 adds features by (a) computing them here from the complete panel
and (b) registering them in BACKWARD. A feature that can't pass the
harness doesn't ship. shift(-n) anywhere outside a FORWARD column is a
bug by definition (Phase 1 doc, feature discipline rule 2).

Label spec (frozen, Phase 1 / spec v2.1): ternary on excess return,
excess_5d = fwd_ret_5d - drift_5d, tau = K * sigma_5d, K = 0.4.
"""

import numpy as np
import pandas as pd

HORIZON   = 5
K         = 0.4
VOL_SPAN  = 20
DRIFT_WIN = 252

RET_LAGS = (1, 2, 3, 5, 10, 21, 63)

# Model inputs. Everything here is stationary and relative by
# construction (Phase 1 doc, feature discipline rule 1) and uses only
# information available at t (rule 2, enforced by the PIT harness).
FEATURES = (
    # Lagged returns — the model's "memory"; ret_5d is the reversal
    # signal with the pre-registered NEGATIVE predicted sign (§1.5).
    *(f"ret_{n}d" for n in RET_LAGS),
    # Volatility level and regime: is vol expanding or contracting?
    "vol_10d", "vol_21d", "vol_ratio",
    # Relative position — overlaid indicators made stationary as ratios.
    "px_to_sma20", "px_to_sma50", "sma20_to_sma200",
    # Bounded oscillators — scaled by known bounds, nothing fitted.
    "rsi_14", "bb_pctb",
    # Volume — unbounded, so transformed: trailing z-score of log volume.
    "volume_z",
    # Market-wide regime context (vix_lag1 itself is the level feature —
    # already lagged one session at ingest, see ingest.align_vix).
    "vix_lag1", "vix_chg_5d",
    # Calendar — trees consume these natively as integers.
    "dow", "month", "days_to_eom",
)

BACKWARD = (
    "close", "volume",                       # panel passthrough (plumbing)
    "log_ret", "drift_5d", "sigma_5d",       # label plumbing, reusable
) + FEATURES
FORWARD = ("fwd_ret_5d", "excess_5d", "y")


def build_dataset(panel: pd.DataFrame) -> pd.DataFrame:
    """Complete panel in (every SPY session, VIX NaNs allowed), flat
    dataset out — same calendar, one column per registered name. Row
    dropping happens at materialization, never here (sequencing rule)."""
    df = panel[["close", "volume", "vix_lag1"]].copy()

    df["log_ret"] = np.log(df.close / df.close.shift(1))

    # The ONLY legitimate forward-looking columns in the codebase.
    df["fwd_ret_5d"] = np.log(df.close.shift(-HORIZON) / df.close)

    past_ret_5d    = np.log(df.close / df.close.shift(HORIZON))
    df["drift_5d"] = past_ret_5d.rolling(DRIFT_WIN, min_periods=60).mean()
    df["excess_5d"] = df.fwd_ret_5d - df.drift_5d

    sigma_daily    = df.log_ret.ewm(span=VOL_SPAN).std()
    df["sigma_5d"] = sigma_daily * np.sqrt(HORIZON)

    tau = K * df.sigma_5d
    y = np.select([df.excess_5d > tau, df.excess_5d < -tau],
                  [1.0, -1.0], default=0.0)
    # Float with NaN where undefined: np.select's default would stamp a
    # silent 0 ("flat") on rows whose label cannot be computed.
    df["y"] = pd.Series(y, index=df.index).mask(
        df.excess_5d.isna() | df.sigma_5d.isna())

    # ---- FEATURES (Phase 3) — all trailing, all registered ----------
    for n in RET_LAGS:
        df[f"ret_{n}d"] = np.log(df.close / df.close.shift(n))

    df["vol_10d"]   = df.log_ret.rolling(10).std()
    df["vol_21d"]   = df.log_ret.rolling(21).std()
    df["vol_ratio"] = df.vol_10d / df.vol_21d

    sma20  = df.close.rolling(20).mean()
    sma50  = df.close.rolling(50).mean()
    sma200 = df.close.rolling(200).mean()
    df["px_to_sma20"]     = df.close / sma20 - 1
    df["px_to_sma50"]     = df.close / sma50 - 1
    df["sma20_to_sma200"] = sma20 / sma200 - 1

    # Wilder's RSI via its EWM identity: gain/(gain+loss) IS rsi/100,
    # already in [0, 1] — scaled by the known bound, nothing fitted.
    delta = df.close.diff()
    avg_gain = delta.clip(lower=0).ewm(
        alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = (-delta.clip(upper=0)).ewm(
        alpha=1 / 14, adjust=False, min_periods=14).mean()
    df["rsi_14"] = avg_gain / (avg_gain + avg_loss)

    # Bollinger %B with the classic population std (ddof=0). Bounded by
    # construction in calm markets; excursions past [0, 1] ARE the signal.
    sd20 = df.close.rolling(20).std(ddof=0)
    df["bb_pctb"] = (df.close - (sma20 - 2 * sd20)) / (4 * sd20)

    logv = np.log(df.volume)
    df["volume_z"] = ((logv - logv.rolling(63).mean())
                      / logv.rolling(63).std())

    # vix_lag1 (the level) came from the panel, already one session back.
    df["vix_chg_5d"] = df.vix_lag1.pct_change(5)

    # Calendar. days_to_eom counts CALENDAR days, not trading days:
    # trading-days-remaining would need future sessions from the panel
    # index, which a truncated history does not contain — the PIT harness
    # would (rightly) reject it. Calendar distance is a pure function of
    # the date itself.
    df["dow"]         = df.index.dayofweek.astype(float)
    df["month"]       = df.index.month.astype(float)
    df["days_to_eom"] = (df.index.daysinmonth - df.index.day).astype(float)

    unregistered = set(df.columns) - set(BACKWARD) - set(FORWARD)
    assert not unregistered, f"unregistered columns: {unregistered}"
    return df


if __name__ == "__main__":
    from ingest import build_panel

    df = build_dataset(build_panel(write=False))

    # Regression anchor: the frozen Phase 1 class balance at k=0.4.
    train = df.loc[df.index <= "2021-12-31"].iloc[:-HORIZON]
    bal = train.y.dropna().value_counts(normalize=True)
    print(f"\nClass balance (train, k={K}):  "
          f"beat {bal.get(1.0, 0):.1%}  flat {bal.get(0.0, 0):.1%}  "
          f"lag {bal.get(-1.0, 0):.1%}")
    print("Phase 1 frozen:                beat 35.4%  flat 34.0%  lag 30.6%")
