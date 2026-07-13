"""
Tests for the purged walk-forward splitter (spec v2.1).

Ordered by what they protect against, not alphabetically. The first four
are the ones that matter: the label-span invariant, its gapped-index
version (with holes placed AT the fold boundaries, where a positional
implementation fails), the metamorphic purge-does-something check, and the
hand-computed ground truth. Everything after that is contract hygiene.

Run:  pytest test_splitter.py -v
"""

import numpy as np
import pandas as pd
import pytest

from splitter import HORIZON, EMBARGO, PurgedWalkForward

TRAIN_END = pd.Timestamp("2021-12-31")


# ---------------------------------------------------------------
# Fixtures — the real panel, and the modelling index with the same
# dropna the Phase 1 script applies.
# ---------------------------------------------------------------
@pytest.fixture(scope="module")
def calendar():
    return pd.read_parquet("data/raw/tiingo_spy_2005-01-01.parquet").index


@pytest.fixture(scope="module")
def model_index(calendar):
    close = pd.read_parquet("data/raw/tiingo_spy_2005-01-01.parquet")["adjClose"]
    drift = np.log(close / close.shift(HORIZON)).rolling(252, min_periods=60).mean()
    fwd   = np.log(close.shift(-HORIZON) / close)
    return calendar[drift.notna() & fwd.notna()]


def label_span_invariant(folds, calendar, horizon):
    """THE invariant: no training row's label span touches its test window.

    Deliberately NOT 'train dates < test dates' — that weaker claim passes
    with a broken purge. Label ends are recomputed here from the FULL
    calendar, independently of the splitter's own arithmetic.
    """
    for f in folds:
        test_start = f.test_dates.min()
        pos = calendar.get_indexer(f.train_dates)
        assert (pos >= 0).all(), f"fold {f.fold}: train date off calendar"
        label_end = calendar[pos + horizon]
        bad = f.train_dates[label_end >= test_start]
        assert len(bad) == 0, (
            f"fold {f.fold}: {len(bad)} training labels reach the test window "
            f"(first: {bad[0].date()} reaches "
            f"{calendar[calendar.get_loc(bad[0]) + horizon].date()})"
        )


# ===============================================================
# 1. THE INVARIANT
# ===============================================================
@pytest.mark.parametrize("scheme", ["expanding", "sliding"])
def test_no_training_label_touches_the_test_window(calendar, model_index, scheme):
    wf = PurgedWalkForward(calendar, scheme=scheme)
    label_span_invariant(wf.split(model_index), calendar, HORIZON)


# ===============================================================
# 2. THE INVARIANT, ON A GAPPED INDEX — the bug this splitter
#    most plausibly has. Holes are placed ADJACENT TO EVERY FOLD
#    BOUNDARY, not just at random: random holes mostly miss the
#    boundary and pass by luck.
# ===============================================================
def test_purge_survives_holes_at_every_fold_boundary(calendar, model_index):
    # For each fold boundary, delete the index rows sitting just before
    # test_lo. A positional purge (iloc[:-H]) now reaches 6-8 trading days
    # forward across the hole and leaks. A date-based purge must not.
    drop = set()
    for i in range(10):
        test_lo = TRAIN_END - (10 - i) * pd.DateOffset(years=1)
        c = int(np.searchsorted(calendar, test_lo, side="right")) - 1
        drop.update(calendar[[c, c - 1, c - 3]])     # holes hugging the boundary
    gapped = model_index[~model_index.isin(drop)]

    wf = PurgedWalkForward(calendar)
    label_span_invariant(wf.split(gapped), calendar, HORIZON)


def test_purge_survives_random_holes(calendar, model_index):
    rng = np.random.default_rng(0)
    holes = rng.choice(len(model_index), size=40, replace=False)
    gapped = model_index.delete(holes)
    wf = PurgedWalkForward(calendar)
    label_span_invariant(wf.split(gapped), calendar, HORIZON)


# ===============================================================
# 3. METAMORPHIC — a purge that removes zero rows passes every
#    invariant test above. Prove it removes something, and that it
#    removes EXACTLY the boundary rows.
# ===============================================================
def test_purge_removes_exactly_the_boundary_rows(calendar, model_index):
    wf = PurgedWalkForward(calendar)
    for f in wf.split(model_index):
        c = int(np.searchsorted(calendar, f.test_lo, side="right")) - 1
        pre = model_index[model_index <= calendar[c]]

        removed = pre.difference(f.train_dates)
        assert len(removed) > 0, f"fold {f.fold}: purge removed nothing (inert)"

        # Independent expectation: excluded iff the label's last price lands
        # after the embargo cutoff, computed from calendar positions here.
        pre_pos = calendar.get_indexer(pre)
        expected = pre[pre_pos + HORIZON > c - EMBARGO]
        assert set(removed) == set(expected), (
            f"fold {f.fold}: removed {len(removed)} rows, "
            f"expected {len(expected)} — purge is removing the wrong rows"
        )


def test_embargo_is_train_side_and_not_inert(calendar, model_index):
    # LdP's embargo removes training rows AFTER the test window — which
    # removes ZERO rows in a forward-chaining walk-forward, because no
    # training data follows the test set. This implementation embargoes the
    # train side of the boundary instead. This test makes that design
    # decision undeniable: the embargo must remove exactly the rows whose
    # label ends land within EMBARGO trading days of the boundary.
    no_emb   = PurgedWalkForward(calendar, embargo=0).split(model_index)
    with_emb = PurgedWalkForward(calendar, embargo=EMBARGO).split(model_index)
    for a, b in zip(no_emb, with_emb):
        assert a.test_dates.equals(b.test_dates)       # test sets unchanged
        removed = a.train_dates.difference(b.train_dates)
        assert len(removed) == EMBARGO > 0, (
            f"fold {a.fold}: embargo removed {len(removed)} rows, "
            f"expected {EMBARGO} — an inert embargo is a promise, not a control"
        )


def test_purge_scales_with_horizon(calendar, model_index):
    # Catches a hardcoded 5 anywhere in the purge path.
    p5  = sum(f.n_purged for f in
              PurgedWalkForward(calendar, horizon=5).split(model_index))
    p20 = sum(f.n_purged for f in
              PurgedWalkForward(calendar, horizon=20).split(model_index))
    assert p20 > p5, "purge width did not grow with horizon — hardcoded?"


# ===============================================================
# 4. GROUND TRUTH — 40 fake trading days, horizon=2, embargo=1,
#    2 folds, hand-computed expected sets, exact equality. The only
#    test in this file that cannot be fooled.
# ===============================================================
def test_ground_truth_toy():
    cal = pd.bdate_range("2021-01-01", periods=40)     # Jan 1 .. Feb 25
    wf = PurgedWalkForward(
        cal, horizon=2, embargo=1, n_folds=2,
        test_span=pd.DateOffset(days=7), train_end=pd.Timestamp("2021-02-25"),
    )
    f0, f1 = wf.split(cal)

    # Fold 0: test window (Feb 11, Feb 18]. Boundary c = Feb 11 (pos 29).
    #   trainable: pos + 2 <= 29 - 1  ->  pos <= 26  ->  Jan 1 .. Feb 8
    #   embargoed: label ends Feb 11            ->  Feb 9
    #   purged   : label ends Feb 12 / Feb 15   ->  Feb 10, Feb 11
    assert list(f0.test_dates) == list(pd.to_datetime(
        ["2021-02-12", "2021-02-15", "2021-02-16", "2021-02-17", "2021-02-18"]))
    assert f0.train_dates.equals(cal[:27])
    assert f0.train_dates[-1] == pd.Timestamp("2021-02-08")
    assert (f0.n_purged, f0.n_embargoed) == (2, 1)

    # Fold 1: test window (Feb 18, Feb 25]. Boundary c = Feb 18 (pos 34).
    #   trainable: pos <= 31  ->  Jan 1 .. Feb 15
    #   embargoed: Feb 16;  purged: Feb 17, Feb 18
    assert list(f1.test_dates) == list(pd.to_datetime(
        ["2021-02-19", "2021-02-22", "2021-02-23", "2021-02-24", "2021-02-25"]))
    assert f1.train_dates.equals(cal[:32])
    assert f1.train_dates[-1] == pd.Timestamp("2021-02-15")
    assert (f1.n_purged, f1.n_embargoed) == (2, 1)


# ===============================================================
# 5. STRUCTURAL / CONTRACT
# ===============================================================
def test_structure(calendar, model_index):
    wf = PurgedWalkForward(calendar)
    folds = wf.split(model_index)

    assert len(folds) == 10

    # Test windows strictly increasing, non-overlapping, and their union
    # covers (2011-12-31, 2021-12-31] exactly once — no gaps, no dupes.
    for a, b in zip(folds, folds[1:]):
        assert a.test_hi == b.test_lo                  # contiguous
        assert a.test_dates.max() < b.test_dates.min() # increasing, disjoint
    pooled = folds[0].test_dates.append([f.test_dates for f in folds[1:]])
    expected = model_index[(model_index > folds[0].test_lo)
                           & (model_index <= TRAIN_END)]
    assert pooled.is_unique and pooled.equals(expected)

    for f in folds:
        assert len(f.train_dates.intersection(f.test_dates)) == 0
        assert f.train_dates.isin(model_index).all()
        assert f.test_dates.isin(model_index).all()


def test_expanding_trains_are_nested_and_purged_rows_return(calendar, model_index):
    folds = PurgedWalkForward(calendar).split(model_index)
    for a, b in zip(folds, folds[1:]):
        assert set(a.train_dates) <= set(b.train_dates), \
            f"fold {b.fold}: expanding train set is not a superset of fold {a.fold}'s"
        # Rows purged at fold k's boundary are deep in the past by fold k+1
        # and must reappear — a purge that deletes rows permanently is wrong.
        purged_at_a = model_index[(model_index > a.train_dates.max())
                                  & (model_index <= a.test_lo)]
        assert purged_at_a.isin(b.train_dates).all(), \
            f"rows purged at fold {a.fold} never returned to training"


def test_sliding_constant_length_and_differs_from_expanding(calendar, model_index):
    sliding   = PurgedWalkForward(calendar, scheme="sliding").split(model_index)
    expanding = PurgedWalkForward(calendar).split(model_index)

    lengths = {len(f.train_dates) for f in sliding}
    assert len(lengths) == 1, f"sliding train length varies: {sorted(lengths)}"
    # The mode flag must actually branch.
    assert any(not s.train_dates.equals(e.train_dates)
               for s, e in zip(sliding, expanding))
    # Test sets identical across schemes — only training membership differs.
    for s, e in zip(sliding, expanding):
        assert s.test_dates.equals(e.test_dates)


def test_deterministic(calendar, model_index):
    a = PurgedWalkForward(calendar).split(model_index)
    b = PurgedWalkForward(calendar).split(model_index)
    for fa, fb in zip(a, b):
        assert fa.train_dates.equals(fb.train_dates)
        assert fa.test_dates.equals(fb.test_dates)


# ===============================================================
# 6. THE HOLDOUT GUARD — §0 says the holdout is void if touched.
#    A test, not a promise.
# ===============================================================
def test_no_fold_ever_touches_the_holdout(calendar, model_index):
    # Feed the splitter an index that INCLUDES 2022+ rows on purpose.
    with_holdout = calendar[calendar >= model_index.min()]
    assert with_holdout.max() > pd.Timestamp("2022-01-01")   # bait is real
    for f in PurgedWalkForward(calendar).split(with_holdout):
        assert f.train_dates.max() < pd.Timestamp("2022-01-01")
        assert f.test_dates.max()  < pd.Timestamp("2022-01-01")


# ===============================================================
# 7. EDGE CASES — must raise, never silently yield
# ===============================================================
def test_raises_when_test_window_runs_off_the_data():
    cal = pd.bdate_range("2021-01-01", periods=40)
    wf = PurgedWalkForward(cal, n_folds=2, test_span=pd.DateOffset(days=7),
                           horizon=2, embargo=1,
                           train_end=pd.Timestamp("2021-06-30"))  # beyond data
    with pytest.raises(ValueError, match="runs off the data"):
        wf.split(cal)


def test_raises_when_purge_consumes_entire_training_set():
    cal = pd.bdate_range("2021-01-01", periods=40)
    wf = PurgedWalkForward(cal, n_folds=2, test_span=pd.DateOffset(days=7),
                           horizon=30, embargo=1,
                           train_end=pd.Timestamp("2021-02-25"))
    with pytest.raises(ValueError, match="consumed the entire training set"):
        wf.split(cal)


def test_raises_on_dates_missing_from_calendar(calendar, model_index):
    off_calendar = model_index.union(pd.DatetimeIndex(["2015-08-22"]))  # a Saturday
    with pytest.raises(ValueError, match="not on the panel calendar"):
        PurgedWalkForward(calendar).split(off_calendar)


def test_raises_on_unknown_scheme(calendar):
    with pytest.raises(ValueError, match="unknown scheme"):
        PurgedWalkForward(calendar, scheme="rolling")
