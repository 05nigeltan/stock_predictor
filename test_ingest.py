"""
Tests for ingest.py — alignment logic on toy data (no network), plus
spot checks against the cached FRED pull. The one that matters most is
the lookahead test: same-day VIX is information from 15 minutes after
the SPY close, and using it is the leak this module exists to prevent.

Run:  pytest test_ingest.py -v
"""

import numpy as np
import pandas as pd
import pytest

from ingest import align_vix, vix_missingness_gate, vix_sane
from validate import DataValidationError


# ---------------------------------------------------------------
# Toy fixtures: one trading week with a weekend in the middle
# ---------------------------------------------------------------
@pytest.fixture
def spy_cal():
    # Thu 5, Fri 6, Mon 9, Tue 10, Wed 11 (Jan 2023)
    return pd.DatetimeIndex(
        ["2023-01-05", "2023-01-06", "2023-01-09",
         "2023-01-10", "2023-01-11"])


@pytest.fixture
def vix(spy_cal):
    return pd.Series([20.0, 21.0, 22.0, 23.0, 24.0], index=spy_cal)


# ---------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------
def test_lag_is_one_session_not_one_calendar_day(spy_cal, vix):
    out = align_vix(vix, spy_cal)
    # Monday inherits FRIDAY's value — a Timedelta(days=1) implementation
    # would look for a Sunday publication and produce NaN.
    assert out.loc["2023-01-09"] == 21.0
    assert out.loc["2023-01-06"] == 20.0
    assert np.isnan(out.iloc[0])                 # structural warm-up NaN


def test_no_same_day_lookahead(spy_cal, vix):
    # The value at t must never be the value published at t. VIX prints
    # 4:15pm ET, 15 min after SPY's close — same-day use is the leak.
    out = align_vix(vix, spy_cal)
    same_day = vix.reindex(spy_cal)
    overlap = out.notna() & same_day.notna()
    assert (out[overlap] != same_day[overlap]).all()


def test_nonpublication_day_stays_nan(spy_cal, vix):
    gappy = vix.copy()
    gappy.loc["2023-01-06"] = np.nan             # Cboe holiday, NYSE open
    out = align_vix(gappy, spy_cal)
    assert np.isnan(out.loc["2023-01-09"])       # Monday inherits the NaN
    assert out.loc["2023-01-10"] == 22.0         # Tuesday inherits Monday


def test_vix_on_non_spy_days_is_ignored(spy_cal, vix):
    # A publication on a day SPY was closed must not shift the alignment.
    extra = pd.concat([vix, pd.Series([99.0],
                       index=pd.DatetimeIndex(["2023-01-07"]))]).sort_index()
    out = align_vix(extra, spy_cal)
    assert out.loc["2023-01-09"] == 21.0         # still Friday's, not 99


# ---------------------------------------------------------------
# Gates
# ---------------------------------------------------------------
def test_vix_sane_catches_garbage(vix):
    bad = vix.copy(); bad.iloc[2] = -1.0
    with pytest.raises(DataValidationError, match="outside"):
        vix_sane(bad)
    bad = vix.copy(); bad.iloc[2] = 999.0
    with pytest.raises(DataValidationError, match="outside"):
        vix_sane(bad)
    vix_sane(vix)                                # clean passes


def test_missingness_gate_measures_and_raises():
    cal = pd.bdate_range("2023-01-02", periods=200)
    panel = pd.DataFrame({"vix_lag1": 20.0}, index=cal)
    panel.iloc[0, 0] = np.nan                    # structural warm-up

    assert len(vix_missingness_gate(panel)) == 0  # warm-up NaN excluded

    panel.iloc[50, 0] = np.nan                   # 1/199 = 0.5025% > 0.5%
    with pytest.raises(DataValidationError, match="pre-registered"):
        vix_missingness_gate(panel)

    missing = vix_missingness_gate(panel, threshold=0.01)
    assert list(missing) == [cal[50]]            # reports exactly the gap


# ---------------------------------------------------------------
# Against the cached real pulls (no network — cache written by ingest.py)
# ---------------------------------------------------------------
@pytest.fixture(scope="module")
def real_panel():
    path = "data/curated/spy_vix_panel.parquet"
    return pd.read_parquet(path)


def test_real_panel_lag_spot_checks(real_panel):
    # The all-time VIX close, 82.69, was PUBLISHED Mon 2020-03-16 and must
    # appear in the panel on Tue 2020-03-17 — and not on the 16th.
    assert real_panel.loc["2020-03-17", "vix_lag1"] == pytest.approx(82.69)
    assert real_panel.loc["2020-03-16", "vix_lag1"] != pytest.approx(82.69)


def test_real_panel_is_complete_spy_calendar(real_panel):
    raw = pd.read_parquet("data/raw/tiingo_spy_2005-01-01.parquet")
    assert real_panel.index.equals(raw.index)    # no rows dropped at ingest
    assert real_panel["vix_lag1"].iloc[1:].notna().all()
