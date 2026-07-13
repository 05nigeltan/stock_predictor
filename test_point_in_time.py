"""
Tests for the point-in-time recompute harness (Phase 2, item 4), and the
regression anchor tying make_features to the frozen Phase 1 numbers.

The harness's job: truncate raw history to ~20 as-of dates, rerun the
whole pipeline (raw -> panel -> dataset), and prove no computed value
ever changes when the future is removed. Each sabotage below is one of
the classic ways that property dies; the harness must catch every one.

Run:  pytest test_point_in_time.py -v
"""

import numpy as np
import pandas as pd
import pytest

from ingest import assemble_panel
from make_features import BACKWARD, FORWARD, HORIZON, build_dataset
from validate import DataValidationError, point_in_time_recompute


@pytest.fixture(scope="module")
def raws():
    raw = pd.read_parquet("data/raw/tiingo_spy_2005-01-01.parquet")
    vix = pd.read_parquet("data/raw/fred_vixcls_2005-01-01.parquet")["vix"]
    return raw, vix


@pytest.fixture(scope="module")
def build(raws):
    raw, vix = raws
    def _build(T):
        r = raw if T is None else raw.loc[:T]
        v = vix if T is None else vix.loc[:T]
        return build_dataset(assemble_panel(r, v))
    return _build


# Fewer as-of dates than the real run: the sabotage tests below rebuild
# the pipeline per date, and 6 is plenty to hit every failure mode.
def run_harness(build, n_dates=6):
    point_in_time_recompute(build, BACKWARD, FORWARD, HORIZON,
                            n_dates=n_dates)


# ---------------------------------------------------------------
# The real pipeline is point-in-time
# ---------------------------------------------------------------
def test_real_pipeline_passes_full_harness(build):
    run_harness(build, n_dates=20)


# ---------------------------------------------------------------
# Sabotage — each classic PIT bug must be caught
# ---------------------------------------------------------------
def sabotaged(build, col, fn):
    """Wrap the real pipeline, add one bad column registered as BACKWARD
    (the lie under test: 'this feature only looks backward')."""
    def _build(T):
        df = build(T)
        df[col] = fn(df)
        return df
    return _build, BACKWARD + (col,)


def test_catches_centered_rolling_window(build):
    bad, backward = sabotaged(
        build, "vol_centered",
        lambda df: df.log_ret.rolling(20, center=True).std())
    with pytest.raises(DataValidationError, match="reads the future"):
        point_in_time_recompute(bad, backward, FORWARD, HORIZON, n_dates=6)


def test_catches_globally_fit_transform(build):
    # z-score with FULL-SAMPLE mean/std — the CS230 paper's scaling bug.
    # Every truncation refits the constants, so all history moves.
    bad, backward = sabotaged(
        build, "volume_z",
        lambda df: (np.log(df.volume) - np.log(df.volume).mean())
                   / np.log(df.volume).std())
    with pytest.raises(DataValidationError, match="reads the future"):
        point_in_time_recompute(bad, backward, FORWARD, HORIZON, n_dates=6)


def test_catches_forward_shift_mislabeled_backward(build):
    bad, backward = sabotaged(
        build, "ret_tomorrow", lambda df: df.log_ret.shift(-1))
    with pytest.raises(DataValidationError, match="reads the future"):
        point_in_time_recompute(bad, backward, FORWARD, HORIZON, n_dates=6)


def test_catches_fabricated_values_in_forward_tail(build):
    # A forward column with its tail NaNs filled: the values within
    # HORIZON rows of the data edge cannot exist yet.
    def bad(T):
        df = build(T)
        df["fwd_ret_5d"] = df["fwd_ret_5d"].fillna(0.0)
        return df
    with pytest.raises(DataValidationError, match="does not exist yet"):
        point_in_time_recompute(bad, BACKWARD, FORWARD, HORIZON, n_dates=6)


def test_refuses_unregistered_columns(build):
    def bad(T):
        df = build(T)
        df["mystery"] = 1.0
        return df
    with pytest.raises(DataValidationError, match="unregistered"):
        point_in_time_recompute(bad, BACKWARD, FORWARD, HORIZON, n_dates=6)


# ---------------------------------------------------------------
# Regression anchor — the new pipeline must reproduce Phase 1 exactly
# ---------------------------------------------------------------
def test_reproduces_frozen_phase1_class_balance(build):
    df = build(None)
    train = df.loc[df.index <= "2021-12-31"].iloc[:-HORIZON]
    bal = train.y.dropna().value_counts(normalize=True)
    assert bal[1.0]  == pytest.approx(0.354, abs=0.002)   # beat
    assert bal[0.0]  == pytest.approx(0.340, abs=0.002)   # flat
    assert bal[-1.0] == pytest.approx(0.306, abs=0.002)   # lag
