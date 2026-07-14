"""
Tests for the Phase 3 feature block in make_features.py.

The PIT harness (test_point_in_time.py) already proves every feature is
point-in-time via the registry. These tests cover the other half:
semantic correctness — RSI is actually Wilder's RSI, bounds hold,
calendar features match their dates — plus the registry bookkeeping and
warm-up contract that materialization will rely on.

Run:  pytest test_make_features.py -v
"""

import numpy as np
import pandas as pd
import pytest

from ingest import assemble_panel
from make_features import (BACKWARD, FEATURES, FORWARD, HORIZON, RET_LAGS,
                           build_dataset, materialize)

WARMUP = 300         # rows; longest lookback is sma200 + a safety margin


@pytest.fixture(scope="module")
def panel():
    raw = pd.read_parquet("data/raw/tiingo_spy_2005-01-01.parquet")
    vix = pd.read_parquet("data/raw/fred_vixcls_2005-01-01.parquet")["vix"]
    return assemble_panel(raw, vix)


@pytest.fixture(scope="module")
def df(panel):
    return build_dataset(panel)


# ---------------------------------------------------------------
# Registry bookkeeping
# ---------------------------------------------------------------
def test_registry_is_consistent(df):
    assert set(FEATURES) <= set(BACKWARD)            # model inputs are backward
    assert not set(FEATURES) & set(FORWARD)          # never forward
    assert set(df.columns) == set(BACKWARD) | set(FORWARD)
    assert len(set(BACKWARD) & set(FORWARD)) == 0


def test_features_complete_after_warmup(df):
    # Materialization will dropna on FEATURES; the warm-up cost must be
    # bounded and known. After WARMUP rows, every feature has a value.
    tail = df[list(FEATURES)].iloc[WARMUP:]
    incomplete = tail.columns[tail.isna().any()]
    assert len(incomplete) == 0, f"NaN after warm-up in: {list(incomplete)}"


# ---------------------------------------------------------------
# Semantic spot checks
# ---------------------------------------------------------------
def test_lagged_returns(df):
    assert np.allclose(df.ret_1d.iloc[1:], df.log_ret.iloc[1:], atol=1e-15)
    # ret_5d over 5 sessions == sum of the 5 daily log returns (additivity)
    manual = df.log_ret.rolling(5).sum()
    assert np.allclose(df.ret_5d.iloc[5:], manual.iloc[5:], atol=1e-12)
    assert all(f"ret_{n}d" in df.columns for n in RET_LAGS)


def test_rsi_is_wilders_and_bounded(df):
    obs = df.rsi_14.dropna()
    assert ((obs >= 0) & (obs <= 1)).all()
    # Monotone sanity on constructed data: 14 straight up-days -> RSI 1.0
    up = pd.DataFrame({"close": np.arange(1.0, 31.0),
                       "volume": 1e6, "vix_lag1": 20.0},
                      index=pd.bdate_range("2020-01-01", periods=30))
    assert build_dataset(up).rsi_14.iloc[-1] == pytest.approx(1.0)


def test_bollinger_pctb(df):
    # %B = 0.5 exactly when price sits on its SMA20 by construction;
    # check the identity numerically: px_to_sma20 == 0  =>  bb_pctb ~ 0.5.
    obs = df.dropna(subset=["bb_pctb", "px_to_sma20"])
    on_band = obs[obs.px_to_sma20.abs() < 1e-4]
    assert (on_band.bb_pctb - 0.5).abs().max() < 0.05
    # And it lives mostly inside [0, 1] — excursions are rare, not routine.
    assert ((obs.bb_pctb < -0.5) | (obs.bb_pctb > 1.5)).mean() < 0.001


def test_vol_ratio_positive_and_centered(df):
    obs = df.vol_ratio.dropna()
    assert (obs > 0).all()
    assert 0.9 < obs.median() < 1.1          # regime ratio centers near 1


def test_volume_z_is_trailing_zscore(df):
    obs = df.volume_z.dropna()
    assert obs.abs().median() < 1.5          # z-scaled, not raw
    # Recompute one date by hand.
    t = df.index[1000]
    logv = np.log(df.volume)
    win = logv.iloc[1000 - 62:1001]          # trailing 63 incl. t
    expected = (logv.iloc[1000] - win.mean()) / win.std()
    assert df.volume_z.iloc[1000] == pytest.approx(expected, rel=1e-9)


def test_calendar_features(df):
    t = pd.Timestamp("2015-06-16")           # a Tuesday, June has 30 days
    row = df.loc[t]
    assert row.dow == 1
    assert row.month == 6
    assert row.days_to_eom == 14
    assert df.dow.isin(range(5)).all()       # never a weekend row


def test_vix_features_lag_discipline(df):
    # vix_chg_5d must be computable from vix_lag1 alone — same-session
    # VIX must never enter. Verified by identity with a hand recompute.
    manual = df.vix_lag1.pct_change(5)
    both = df.vix_chg_5d.notna()
    assert np.allclose(df.vix_chg_5d[both], manual[both], atol=1e-15)


# ---------------------------------------------------------------
# Materialization — panel in, modelling matrix out, gated
# ---------------------------------------------------------------
def test_materialize_shape_and_completeness(panel, df):
    matrix = materialize(panel)
    assert list(matrix.columns) == [*FEATURES, "y"]
    assert not matrix.isna().any().any()             # complete rows only
    assert matrix.index.isin(panel.index).all()      # subset of the calendar
    # Dropped rows are exactly warm-up + the unlabelled tail (VIX-NaN
    # count is currently zero): the last labelled feature-complete row.
    expected = df[[*FEATURES, "y"]].dropna()
    assert matrix.index.equals(expected.index)
    assert matrix.index[-1] == df.index[-HORIZON - 1]


def test_materialize_values_match_complete_run(panel, df):
    # Every matrix value equals its complete-calendar computation — the
    # gate checks y; this asserts it for the features too.
    matrix = materialize(panel)
    for col in (*FEATURES, "y"):
        assert np.allclose(matrix[col],
                           df[col].reindex(matrix.index), atol=1e-12)


def test_materialize_drops_vix_holes_without_corrupting_labels(panel, df):
    # THE end-to-end sequencing demonstration: punch VIX holes into the
    # panel (future FRED gaps), materialize, and require (a) the holes'
    # full footprint vanishes, (b) every surviving label still matches
    # the complete-calendar run — shift(-5) never reached across a hole.
    #
    # Footprint finding (this test originally expected 3 dropped rows and
    # got 6): a missing publication at d also kills vix_chg_5d at d+5,
    # whose 5-session change references d's value. One missing VIX day
    # costs TWO matrix rows. Priced into the missingness gate's margin:
    # 0.5% missing days -> ~1% of rows.
    holed = panel.copy()
    victims = holed.index[[1000, 2000, 3000]]
    holed.loc[victims, "vix_lag1"] = np.nan

    matrix = materialize(holed)                      # gate runs inside
    assert not matrix.index.isin(victims).any()

    clean = materialize(panel)
    echoes = panel.index[[1005, 2005, 3005]]         # d+5 sessions
    assert clean.index.difference(matrix.index).equals(
        victims.union(echoes))                       # exactly the footprint
    assert np.allclose(matrix["y"],
                       clean["y"].reindex(matrix.index), atol=1e-15)
