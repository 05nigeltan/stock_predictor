"""
Tests for validate.py — every check must (a) pass on the real data and
(b) catch a planted corruption. A validator that cannot fail is
decoration; each corruption here is the specific bug the check exists
to catch.

Run:  pytest test_validate.py -v
"""

import numpy as np
import pandas as pd
import pytest

from validate import (
    DataValidationError,
    forward_columns_match_complete_calendar,
    no_duplicate_dates,
    no_gaps_vs_exchange_calendar,
    no_unexplained_jumps,
    prices_positive_and_nonnull,
    validate_raw,
    volume_sane,
)

HORIZON = 5


@pytest.fixture(scope="module")
def raw():
    return pd.read_parquet("data/raw/tiingo_spy_2005-01-01.parquet")


# ---------------------------------------------------------------
# Clean data passes the whole battery
# ---------------------------------------------------------------
def test_real_data_passes_battery(raw):
    validate_raw(raw, name="real")


# ---------------------------------------------------------------
# Each check catches its planted corruption
# ---------------------------------------------------------------
def test_catches_duplicate_date(raw):
    dup = pd.concat([raw, raw.iloc[[100]]]).sort_index()
    with pytest.raises(DataValidationError, match="duplicate"):
        no_duplicate_dates(dup)


def test_catches_unsorted_index(raw):
    shuffled = raw.iloc[np.r_[0:50, 60:55:-1, 66:len(raw)]]
    with pytest.raises(DataValidationError, match="not sorted"):
        no_duplicate_dates(shuffled)


def test_catches_missing_trading_day(raw):
    # Delete an ordinary Tuesday — a raw >5-calendar-day heuristic would
    # never notice a single missing session; the calendar check must.
    victim = pd.Timestamp("2015-06-16")
    assert victim in raw.index
    with pytest.raises(DataValidationError, match="sessions missing"):
        no_gaps_vs_exchange_calendar(raw.drop(index=victim))


def test_catches_weekend_row(raw):
    # A Saturday row is a vendor bug the same heuristic would also miss.
    sat = raw.iloc[[200]].copy()
    sat.index = pd.DatetimeIndex([pd.Timestamp("2015-06-20")])
    with pytest.raises(DataValidationError, match="not NYSE sessions"):
        no_gaps_vs_exchange_calendar(pd.concat([raw, sat]).sort_index())


def test_special_closures_are_not_false_positives(raw):
    # Hurricane Sandy (2012-10-29/30) and the Bush funeral (2018-12-05):
    # the data correctly has no rows there, and the calendar knows why.
    window = raw.loc["2012-10-01":"2012-11-30"]
    assert pd.Timestamp("2012-10-29") not in window.index
    no_gaps_vs_exchange_calendar(window)                 # must NOT raise


def test_catches_nan_and_nonpositive_price(raw):
    bad = raw.copy()
    bad.iloc[500, bad.columns.get_loc("adjClose")] = np.nan
    with pytest.raises(DataValidationError, match="NaN in adjClose"):
        prices_positive_and_nonnull(bad)
    bad = raw.copy()
    bad.iloc[500, bad.columns.get_loc("adjClose")] = -1.0
    with pytest.raises(DataValidationError, match="non-positive"):
        prices_positive_and_nonnull(bad)


def test_jump_check_semantics(raw):
    # Fabricated 30% jump, splitFactor == 1 -> caught, with the date named.
    bad = raw.copy()
    i = bad.columns.get_loc("adjClose")
    bad.iloc[1000:, i] = bad.iloc[1000:, i] * 1.35
    date = str(bad.index[1000].date())
    with pytest.raises(DataValidationError, match=date):
        no_unexplained_jumps(bad)

    # Same jump WITH a matching split factor -> explained, must pass.
    explained = bad.copy()
    explained.iloc[1000, explained.columns.get_loc("splitFactor")] = 1.35
    no_unexplained_jumps(explained)

    # Same jump on the verified-real whitelist -> must pass.
    no_unexplained_jumps(bad, known_real=(bad.index[1000],))


def test_catches_zero_and_nan_volume(raw):
    bad = raw.copy()
    bad.iloc[300, bad.columns.get_loc("adjVolume")] = 0
    with pytest.raises(DataValidationError, match="zero-adjVolume"):
        volume_sane(bad)
    bad = raw.copy()
    bad.iloc[300, bad.columns.get_loc("adjVolume")] = np.nan
    with pytest.raises(DataValidationError, match="NaN"):
        volume_sane(bad)


def test_runner_runs_everything_before_raising(raw, capsys):
    # Two planted faults -> the report must show BOTH, then raise once.
    bad = raw.copy()
    bad.iloc[300, bad.columns.get_loc("adjVolume")] = 0
    bad.iloc[500, bad.columns.get_loc("adjClose")] = np.nan
    with pytest.raises(DataValidationError, match="2 check"):
        validate_raw(bad, name="two-faults")
    out = capsys.readouterr().out
    assert "FAIL  prices_positive_and_nonnull" in out
    assert "FAIL  volume_sane" in out


# ---------------------------------------------------------------
# THE SEQUENCING CONTRACT — the relocated leak from the splitter work.
# Build panel -> compute forward columns -> drop rows. This test plants
# the wrong order and the check must catch it.
# ---------------------------------------------------------------
@pytest.fixture(scope="module")
def complete_panel(raw):
    panel = raw[["adjClose"]].rename(columns={"adjClose": "close"}).copy()
    panel["fwd_ret_5d"] = np.log(panel.close.shift(-HORIZON) / panel.close)
    return panel


def test_correct_order_passes(complete_panel):
    # RIGHT: compute on the complete calendar, THEN drop VIX-NaN stand-ins.
    rng = np.random.default_rng(3)
    holes = rng.choice(len(complete_panel), 60, replace=False)
    matrix = complete_panel.drop(index=complete_panel.index[holes])
    forward_columns_match_complete_calendar(
        matrix, complete_panel, ["fwd_ret_5d"])


def test_wrong_order_is_caught(complete_panel):
    # WRONG: drop rows first, THEN shift(-5) on the gapped frame. Every
    # label within 5 rows upstream of a hole now reaches further than 5
    # trading days — spans the splitter never purges for.
    rng = np.random.default_rng(3)
    holes = rng.choice(len(complete_panel), 60, replace=False)
    gapped = complete_panel.drop(index=complete_panel.index[holes]).copy()
    gapped["fwd_ret_5d"] = np.log(
        gapped.close.shift(-HORIZON) / gapped.close)   # the bug
    with pytest.raises(DataValidationError,
                       match="computed AFTER dropping rows"):
        forward_columns_match_complete_calendar(
            gapped, complete_panel, ["fwd_ret_5d"])


def test_stray_dates_are_caught(complete_panel):
    stray = complete_panel.iloc[[10]].copy()
    stray.index = pd.DatetimeIndex([pd.Timestamp("2015-06-20")])  # Saturday
    matrix = pd.concat([complete_panel, stray]).sort_index()
    with pytest.raises(DataValidationError, match="absent from the complete"):
        forward_columns_match_complete_calendar(
            matrix, complete_panel, ["fwd_ret_5d"])
