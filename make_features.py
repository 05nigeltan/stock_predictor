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

BACKWARD = (
    "close", "volume", "vix_lag1",           # panel passthrough
    "log_ret", "drift_5d", "sigma_5d",       # label plumbing, reusable
)
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
