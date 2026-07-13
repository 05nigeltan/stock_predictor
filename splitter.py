"""
Purged walk-forward splitter  (spec v2.1, build item 1)
=======================================================
Forward-chaining walk-forward with a purge and an embargo, computed in
DATES against the full panel calendar — never in row positions.

Why date-based is mandatory here (handoff decision #2, consequence note):
rows can be absent from the modelling matrix (VIX-NaN days get dropped at
materialization). On a row-dropped index, iloc arithmetic silently reaches
further than HORIZON trading days, so a positional purge under-purges
exactly at the fold boundaries it exists to protect. Every boundary below
is therefore computed on the COMPLETE exchange calendar (the raw SPY pull),
and the modelling index — holes and all — is filtered against it.

The three exclusion zones at each fold boundary, for a train row at date t
with label span [t, t+HORIZON trading days]:

  1. TEST      : t inside the test window            -> test row, not train
  2. PURGE     : label_end reaches past test_lo      -> label reads test prices
  3. EMBARGO   : label_end within EMBARGO trading days of test_lo
                 -> guards serial correlation bleeding across the boundary

Equivalently: a date t is trainable for a fold iff
    calpos(t) + HORIZON <= calpos(test_lo) - EMBARGO

Holdout discipline: dates after TRAIN_END are never yielded in any fold,
train or test. The splitter is structurally incapable of touching 2022+.

Schemes (both per spec; expanding primary, sliding reported alongside):
  expanding — train from the start of the supplied index
  sliding   — train limited to the most recent SPAN calendar positions,
              SPAN = the trading-day length of fold 0's expanding train set
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

HORIZON   = 5              # label horizon, trading days (= purge width)
EMBARGO   = 2              # trading days (~1% of dataset, LdP)
N_FOLDS   = 10
TEST_SPAN = pd.DateOffset(years=1)
TRAIN_END = pd.Timestamp("2021-12-31")


@dataclass(frozen=True)
class Fold:
    fold: int
    scheme: str
    test_lo: pd.Timestamp            # exclusive
    test_hi: pd.Timestamp            # inclusive
    train_dates: pd.DatetimeIndex = field(repr=False)
    test_dates: pd.DatetimeIndex = field(repr=False)
    n_purged: int = 0                # excluded: label span reaches past test_lo
    n_embargoed: int = 0             # excluded: label end inside the embargo
    n_windowed: int = 0              # excluded: sliding window only


class PurgedWalkForward:
    def __init__(self, calendar: pd.DatetimeIndex, *,
                 horizon: int = HORIZON, embargo: int = EMBARGO,
                 n_folds: int = N_FOLDS, test_span=TEST_SPAN,
                 train_end: pd.Timestamp = TRAIN_END,
                 scheme: str = "expanding"):
        if scheme not in ("expanding", "sliding"):
            raise ValueError(f"unknown scheme {scheme!r}")
        if not calendar.is_monotonic_increasing or not calendar.is_unique:
            raise ValueError("calendar must be sorted and duplicate-free")
        self.calendar  = calendar
        self.horizon   = horizon
        self.embargo   = embargo
        self.n_folds   = n_folds
        self.test_span = test_span
        self.train_end = pd.Timestamp(train_end)
        self.scheme    = scheme

    def _calpos(self, dates: pd.DatetimeIndex) -> np.ndarray:
        pos = self.calendar.get_indexer(dates)
        if (pos < 0).any():
            missing = dates[pos < 0][:3].tolist()
            raise ValueError(
                f"{(pos < 0).sum()} index dates are not on the panel calendar "
                f"(first few: {missing}). The modelling index must be a subset "
                f"of the calendar the labels were computed on."
            )
        return pos

    def split(self, index: pd.DatetimeIndex) -> list[Fold]:
        index = pd.DatetimeIndex(index)
        # Holdout guard: 2022+ never participates, in either role.
        index = index[index <= self.train_end]

        pos = self._calpos(index)                    # calendar position of each row
        label_end_pos = pos + self.horizon           # last price the label reads

        folds = []
        span = None                                  # sliding window, set at fold 0
        for i in range(self.n_folds):
            test_hi = self.train_end - (self.n_folds - 1 - i) * self.test_span
            test_lo = test_hi - self.test_span

            test_mask  = (index > test_lo) & (index <= test_hi)
            if not test_mask.any():
                raise ValueError(
                    f"fold {i}: test window ({test_lo.date()}, {test_hi.date()}] "
                    f"contains no index dates — window runs off the data"
                )

            # Boundary position: last calendar day at or before test_lo.
            c = int(np.searchsorted(self.calendar, test_lo, side="right")) - 1
            cutoff = c - self.embargo                # label_end must sit at/before this

            pre        = pos <= c                    # rows dated before the test window
            train_mask = pre & (label_end_pos <= cutoff)
            n_purged    = int((pre & (label_end_pos > c)).sum())
            n_embargoed = int((pre & (label_end_pos > cutoff)
                                   & (label_end_pos <= c)).sum())

            n_windowed = 0
            if self.scheme == "sliding":
                if span is None:                     # fold 0 defines the window
                    span = int(train_mask.sum())
                keep = np.zeros_like(train_mask)
                keep[np.flatnonzero(train_mask)[-span:]] = True
                n_windowed = int(train_mask.sum() - keep.sum())
                train_mask = keep
            elif span is None:
                span = int(train_mask.sum())         # recorded for symmetry

            if not train_mask.any():
                raise ValueError(
                    f"fold {i}: purge/embargo consumed the entire training set "
                    f"(horizon={self.horizon}, embargo={self.embargo}) — a model "
                    f"must never silently fit on zero rows"
                )

            folds.append(Fold(
                fold=i, scheme=self.scheme, test_lo=test_lo, test_hi=test_hi,
                train_dates=index[train_mask], test_dates=index[test_mask],
                n_purged=n_purged, n_embargoed=n_embargoed,
                n_windowed=n_windowed,
            ))
        return folds

    def assert_leak_free(self, folds: list[Fold]) -> None:
        """Independent re-check of every invariant the split claims.

        Deliberately re-derives label ends from scratch rather than trusting
        split()'s own arithmetic — a self-check that shares the bug it is
        checking for is decoration, not a check.
        """
        for f in folds:
            assert len(f.train_dates.intersection(f.test_dates)) == 0, \
                f"fold {f.fold}: train/test overlap"
            assert (f.test_dates > f.test_lo).all() and \
                   (f.test_dates <= f.test_hi).all(), \
                f"fold {f.fold}: test dates outside window"
            assert f.train_dates.max() < f.test_dates.min(), \
                f"fold {f.fold}: training reaches into the test window"

            tr_pos = self.calendar.get_indexer(f.train_dates)
            label_ends = self.calendar[tr_pos + self.horizon]
            first_test = f.test_dates.min()
            assert (label_ends < first_test).all(), \
                f"fold {f.fold}: a train label span touches the test window"

            # Embargo: at least EMBARGO calendar rows between every train
            # label end and the first test date.
            gap = (int(np.searchsorted(self.calendar, first_test))
                   - int(self.calendar.get_indexer([label_ends.max()])[0]))
            assert gap > self.embargo, \
                f"fold {f.fold}: embargo violated (gap {gap} <= {self.embargo})"

            assert f.train_dates.max() <= self.train_end
            assert f.test_dates.max() <= self.train_end, \
                f"fold {f.fold}: test window reaches into the locked holdout"


# ---------------------------------------------------------------
# Demo / verification on the real panel. Run:  python splitter.py
# ---------------------------------------------------------------
if __name__ == "__main__":
    raw = pd.read_parquet("data/raw/tiingo_spy_2005-01-01.parquet")
    calendar = raw.index                                  # complete SPY calendar

    # Modelling index = rows a label can be computed on (same dropna as the
    # Phase 1 script: drift warm-up gone, last HORIZON rows have no label).
    close = raw["adjClose"]
    drift = np.log(close / close.shift(HORIZON)).rolling(252, min_periods=60).mean()
    fwd   = np.log(close.shift(-HORIZON) / close)
    model_index = calendar[drift.notna() & fwd.notna()]

    for scheme in ("expanding", "sliding"):
        wf = PurgedWalkForward(calendar, scheme=scheme)
        folds = wf.split(model_index)
        wf.assert_leak_free(folds)

        print("=" * 78)
        print(f"SCHEME: {scheme}   (purge={HORIZON}d, embargo={EMBARGO}d, "
              f"leak check: PASSED)")
        print("=" * 78)
        print(f"{'fold':>4} {'train span':>25} {'n_train':>8} "
              f"{'purged':>7} {'embg':>5} {'wndw':>6} {'test year':>10} {'n_test':>7}")
        for f in folds:
            print(f"{f.fold:>4} "
                  f"{str(f.train_dates.min().date()):>12} -> {str(f.train_dates.max().date())} "
                  f"{len(f.train_dates):>8} {f.n_purged:>7} {f.n_embargoed:>5} "
                  f"{f.n_windowed:>6} {f.test_hi.year:>10} {len(f.test_dates):>7}")
        pooled = sum(len(f.test_dates) for f in folds)
        print(f"\n  Pooled test rows: {pooled}  "
              f"(independent 5d samples ~{pooled // HORIZON})\n")

    # --- The reason this splitter exists: survive holes in the index. ---
    # Punch 100 random holes (stand-ins for VIX-NaN drops) and re-verify.
    rng = np.random.default_rng(42)
    holed = model_index.delete(rng.choice(len(model_index), 100, replace=False))
    wf = PurgedWalkForward(calendar)
    folds = wf.split(holed)
    wf.assert_leak_free(folds)
    print("=" * 78)
    print("HOLE-TOLERANCE CHECK: 100 random rows removed from the modelling")
    print("index (simulated VIX-NaN drops). Leak check on the holed index:")
    print("PASSED — purge/embargo boundaries held because they are computed")
    print("against the panel calendar, not row positions.")
