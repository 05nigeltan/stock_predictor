"""
validate.py — reusable data-quality checks  (Phase 2, checklist item 2)
=======================================================================
Supersedes the inline asserts in phase1_baselines.py for all future
pipeline code (the Phase 1 script itself is left untouched — it is the
record of what was run, not a library).

Every check raises DataValidationError with the offending dates in the
message; nothing returns a boolean for the caller to forget to look at.
`validate_raw()` runs the standard battery on a raw Tiingo pull and
prints a pass table.

The one check that is NOT hygiene: `forward_columns_match_complete_calendar`.
The splitter's test suite proved a positional purge cannot under-purge on
a gapped index — which relocated the real leak risk upstream, to label
construction: shift(-H) on a frame that has already had rows dropped
(VIX-NaNs) silently reaches MORE than H trading days forward across each
hole, so the label spans exceed what the splitter purges for. The
sequencing rule (handoff decision #2) is: build every forward-looking
column on the COMPLETE calendar first, drop rows only at materialization.
That check is how the rule is enforced rather than remembered.

Location note: the handoff said data/validate.py, but data/ is the
gitignored raw-cache zone — code there would be untracked. It lives in
the project root with splitter.py and power.py.
"""

import numpy as np
import pandas as pd

# The two real (verified) large SPY moves with no split factor — see the
# Phase 1 jump check. Anything else the jump check flags is a data bug.
KNOWN_REAL_JUMPS = (pd.Timestamp("2008-10-13"), pd.Timestamp("2020-03-16"))


class DataValidationError(Exception):
    pass


def _fail(msg, dates=None, limit=5):
    if dates is not None and len(dates):
        shown = ", ".join(str(pd.Timestamp(d).date()) for d in dates[:limit])
        more = f" (+{len(dates) - limit} more)" if len(dates) > limit else ""
        msg += f": {shown}{more}"
    raise DataValidationError(msg)


# ---------------------------------------------------------------
# The standard battery
# ---------------------------------------------------------------
def no_duplicate_dates(df):
    if not df.index.is_unique:
        _fail("duplicate dates in index",
              df.index[df.index.duplicated()].unique())
    if not df.index.is_monotonic_increasing:
        _fail("index is not sorted ascending")


def no_gaps_vs_exchange_calendar(df, exchange="NYSE"):
    """Compare against the real exchange calendar, both directions.

    Not a raw day-gap heuristic: the calendar knows holidays and special
    closures (Hurricane Sandy 2012-10-29/30, national mourning days), so
    a >5-calendar-day gap around one of those is fine while a single
    missing ordinary Tuesday is not.
    """
    import pandas_market_calendars as mcal
    sched = mcal.get_calendar(exchange).schedule(
        df.index.min(), df.index.max())
    expected = sched.index

    missing = expected.difference(df.index)
    if len(missing):
        _fail(f"{len(missing)} {exchange} sessions missing from the data",
              missing)
    extra = df.index.difference(expected)
    if len(extra):
        _fail(f"{len(extra)} dates in the data are not {exchange} sessions "
              f"(weekend/holiday rows — vendor bug)", extra)


def prices_positive_and_nonnull(df, cols=("adjClose", "close")):
    for col in (c for c in cols if c in df.columns):
        bad = df.index[df[col].isna()]
        if len(bad):
            _fail(f"NaN in {col}", bad)
        bad = df.index[df[col] <= 0]
        if len(bad):
            _fail(f"non-positive values in {col}", bad)


def no_unexplained_jumps(df, price_col="adjClose", threshold=0.20,
                         known_real=KNOWN_REAL_JUMPS):
    """Any |daily log return| > threshold must coincide with a split
    (splitFactor != 1) or be on the verified-real whitelist."""
    lr = np.log(df[price_col] / df[price_col].shift(1))
    suspect = lr.abs() > threshold
    if "splitFactor" in df.columns:
        suspect &= df["splitFactor"] == 1.0
    unexplained = df.index[suspect].difference(pd.DatetimeIndex(known_real))
    if len(unexplained):
        _fail(f"|log return| > {threshold:.0%} with no split factor and not "
              f"on the verified-real whitelist", unexplained)


def volume_sane(df, col="adjVolume"):
    if col not in df.columns:
        return
    bad = df.index[df[col].isna()]
    if len(bad):
        _fail(f"NaN in {col}", bad)
    bad = df.index[df[col] < 0]
    if len(bad):
        _fail(f"negative {col}", bad)
    bad = df.index[df[col] == 0]
    if len(bad):
        _fail(f"zero-{col} sessions (SPY never has zero volume — vendor bug)",
              bad)


# ---------------------------------------------------------------
# The sequencing contract — NOT hygiene, see module docstring
# ---------------------------------------------------------------
def forward_columns_match_complete_calendar(matrix, complete, cols):
    """Every forward-looking column in the materialized modelling matrix
    must equal, row for row, the same column computed on the COMPLETE
    panel. If rows were dropped BEFORE the shift(-H), the values disagree
    exactly downstream of each hole — the leak the splitter cannot see.

    matrix   : the materialized modelling matrix (rows may be dropped)
    complete : the panel built on the full trading calendar
    cols     : the forward-looking columns to verify (e.g. ["fwd_ret_5d"])
    """
    stray = matrix.index.difference(complete.index)
    if len(stray):
        _fail("matrix contains dates absent from the complete panel", stray)
    for col in cols:
        a = matrix[col]
        b = complete[col].reindex(matrix.index)
        both = a.notna() & b.notna()
        mismatch = matrix.index[both][
            ~np.isclose(a[both], b[both], rtol=0, atol=1e-12)]
        nan_diff = matrix.index[a.notna() != b.notna()]
        bad = mismatch.union(nan_diff)
        if len(bad):
            _fail(
                f"'{col}' on the matrix disagrees with the complete-calendar "
                f"computation on {len(bad)} rows — forward-looking columns "
                f"were computed AFTER dropping rows (shift(-H) reached across "
                f"a hole). Build panel -> compute -> drop, in that order",
                bad)


# ---------------------------------------------------------------
# Point-in-time recompute harness (Phase 2, checklist item 4)
# ---------------------------------------------------------------
def _series_identical(a, b):
    """Equal values AND equal NaN positions."""
    na_a, na_b = a.isna().to_numpy(), b.isna().to_numpy()
    if (na_a != na_b).any():
        return a.index[na_a != na_b]
    ok = na_a | np.isclose(a.to_numpy(), b.to_numpy(), rtol=0, atol=1e-12,
                           equal_nan=True)
    return a.index[~ok]


def point_in_time_recompute(build, backward, forward, horizon,
                            n_dates=20, seed=20260713, min_history=300):
    """Truncate history to ~n_dates random as-of dates, rerun the whole
    pipeline on each truncation, and hold every column to its class
    invariant against the full-history run:

      backward : bit-identical for every t <= T. If recomputing with less
                 future changes a past value, the column read the future
                 (centered window, globally-fit transform, shift(-n)).
      forward  : identical wherever computable as-of T (t + horizon <= T),
                 and NaN — never a value — in the last `horizon` rows.

    `build(T)` must run the pipeline on raw data truncated to date T
    (T=None -> full history). Truncate the RAW inputs, not the outputs:
    the point is to exercise every computation under a shorter history.
    Every output column must be registered backward or forward.
    """
    full = build(None)

    unregistered = set(full.columns) - set(backward) - set(forward)
    if unregistered:
        _fail(f"unregistered columns (declare BACKWARD or FORWARD): "
              f"{sorted(unregistered)}")

    cal = full.index
    rng = np.random.default_rng(seed)
    positions = sorted(set(
        rng.integers(min_history, len(cal) - 1, size=n_dates).tolist()
        + [len(cal) - 1]))                       # always include the end

    for pos in positions:
        T = cal[pos]
        part = build(T)
        if not part.index.equals(cal[:pos + 1]):
            _fail(f"as-of {T.date()}: truncated run has a different "
                  f"calendar prefix than the full run")

        for col in backward:
            bad = _series_identical(part[col], full[col].iloc[:pos + 1])
            if len(bad):
                _fail(f"'{col}' is not point-in-time: recomputing as-of "
                      f"{T.date()} REWROTE {len(bad)} historical values "
                      f"(the column reads the future)", bad)

        for col in forward:
            known = part[col].iloc[:pos + 1 - horizon]
            bad = _series_identical(known, full[col].iloc[:pos + 1 - horizon])
            if len(bad):
                _fail(f"'{col}' as-of {T.date()} disagrees with the full "
                      f"run where it should be computable", bad)
            tail = part[col].iloc[pos + 1 - horizon:]
            fake = tail.index[tail.notna()]
            if len(fake):
                _fail(f"'{col}' as-of {T.date()} holds values within "
                      f"{horizon} rows of the data edge — the future "
                      f"needed to compute them does not exist yet", fake)


# ---------------------------------------------------------------
# Runner
# ---------------------------------------------------------------
RAW_BATTERY = [no_duplicate_dates, no_gaps_vs_exchange_calendar,
               prices_positive_and_nonnull, no_unexplained_jumps,
               volume_sane]


def validate_raw(df, name="raw", checks=RAW_BATTERY):
    """Run the battery, print a pass table, raise on the first failure
    AFTER running everything (so one bad check doesn't hide the others)."""
    failures = []
    print(f"validate: {name}  ({len(df)} rows, "
          f"{df.index.min().date()} -> {df.index.max().date()})")
    for check in checks:
        try:
            check(df)
            print(f"  PASS  {check.__name__}")
        except DataValidationError as e:
            print(f"  FAIL  {check.__name__}: {e}")
            failures.append((check.__name__, str(e)))
    if failures:
        raise DataValidationError(
            f"{name}: {len(failures)} check(s) failed: "
            + "; ".join(n for n, _ in failures))
    return df                                    # chainable: df = validate_raw(load(...))


if __name__ == "__main__":
    raw = pd.read_parquet("data/raw/tiingo_spy_2005-01-01.parquet")
    validate_raw(raw, name="tiingo_spy_2005-01-01.parquet")
