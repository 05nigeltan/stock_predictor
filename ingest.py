"""
ingest.py — data loads and the curated panel  (Phase 2, checklist item 1)
=========================================================================
Raw zone   : data/raw/      immutable vendor pulls, parquet, cached — a
                            rerun can never silently pick up revised data
Curated    : data/curated/  the panel: complete SPY calendar, VIX joined

VIX decisions (frozen in the handoff, decision #2):
  - Source: FRED `VIXCLS` (Tiingo has no index coverage; ^VIX is not a
    security). Public CSV endpoint, no API key.
  - Alignment: the VIX value usable at SPY date t is the value published
    on SPY session t-1. SPY closes 4:00pm ET; Cboe calculates VIX to
    4:15pm ET — same-day VIX is information from after the close being
    traded at the close. Implemented as reindex-to-SPY-calendar, THEN
    shift(1): both steps on the panel calendar, no positional reach.
  - Non-publication days ('.' in the CSV) stay NaN. No imputation — rows
    are dropped at materialization, uniformly for all models, so LogReg /
    RF / LightGBM see an identical matrix.
  - MEASURED missingness gate: lagged-VIX NaNs must be <= 0.5% of SPY
    sessions (pre-registered here, not asserted as "negligible"). Above
    that, ingestion fails and the drop-vs-impute decision is reopened.

Sequencing rule (enforced by validate.forward_columns_match_complete_
calendar at materialization): the panel keeps EVERY SPY session, VIX
NaNs included. Features and labels are computed on this complete panel;
VIX-NaN rows are dropped only when the modelling matrix is materialized
(Phase 3). Dropping first would let shift(-H) reach across the holes.
"""

import io
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

from validate import DataValidationError, validate_raw

load_dotenv(Path(__file__).with_name(".env"))

RAW_DIR      = Path("data/raw")
CURATED_DIR  = Path("data/curated")
START        = "2005-01-01"
VIX_MISS_MAX = 0.005                 # pre-registered: 0.5% of SPY sessions
FRED_CSV     = "https://fred.stlouisfed.org/graph/fredgraph.csv"


# ---------------------------------------------------------------
# Raw zone
# ---------------------------------------------------------------
def load_tiingo(ticker: str = "spy", start: str = START) -> pd.DataFrame:
    """Cached Tiingo EOD pull (same cache the Phase 1 script wrote)."""
    import os
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"tiingo_{ticker.lower()}_{start}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    r = requests.get(
        f"https://api.tiingo.com/tiingo/daily/{ticker.lower()}/prices",
        params={"startDate": start, "format": "json"},
        headers={"Authorization": f"Token {os.environ['TIINGO_TOKEN']}"},
        timeout=60,
    )
    r.raise_for_status()
    raw = pd.DataFrame(r.json())
    raw["date"] = (pd.to_datetime(raw["date"], utc=True)
                     .dt.tz_localize(None).dt.normalize())
    raw = raw.set_index("date").sort_index()
    raw.to_parquet(path)
    print(f"[cache] wrote {path}  ({len(raw)} rows)")
    return raw


def load_vix(start: str = START) -> pd.Series:
    """FRED VIXCLS, cached. Pulled from 10 days before `start` so the
    first SPY session has a t-1 value to inherit. '.' -> NaN, kept."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"fred_vixcls_{start}.parquet"
    if path.exists():
        return pd.read_parquet(path)["vix"]
    fetch_from = (pd.Timestamp(start) - pd.Timedelta(days=10)).date()
    r = requests.get(FRED_CSV, params={"id": "VIXCLS",
                                       "cosd": str(fetch_from)}, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), na_values=["."])
    df.columns = ["date", "vix"]                     # header names vary
    df["date"] = pd.to_datetime(df["date"])
    ser = df.set_index("date")["vix"].sort_index()
    ser.to_frame().to_parquet(path)
    print(f"[cache] wrote {path}  ({len(ser)} rows, "
          f"{ser.isna().sum()} non-publication days)")
    return ser


# ---------------------------------------------------------------
# Alignment and gates — pure functions, tested without network
# ---------------------------------------------------------------
def align_vix(vix: pd.Series, spy_calendar: pd.DatetimeIndex) -> pd.Series:
    """VIX value available at SPY date t = value published at SPY session
    t-1. Reindex to the SPY calendar first, then shift one SESSION (not
    one calendar day): over a weekend, Friday's VIX is Monday's value."""
    return vix.reindex(spy_calendar).shift(1).rename("vix_lag1")


def vix_sane(vix: pd.Series):
    """Level checks on the published series (pre-alignment)."""
    obs = vix.dropna()
    bad = obs.index[(obs <= 0) | (obs > 150)]
    if len(bad):
        raise DataValidationError(
            f"VIX outside (0, 150] on {len(bad)} days, first "
            f"{bad[0].date()} = {obs[bad[0]]}")


def vix_missingness_gate(panel: pd.DataFrame,
                         threshold: float = VIX_MISS_MAX) -> pd.DatetimeIndex:
    """The 'negligible' claim, measured. Returns the missing dates so the
    caller can report them; raises if they exceed the pre-registered
    threshold. The first session is excluded — its NaN is the structural
    lag warm-up, not a data gap."""
    scope = panel.index[1:]
    missing = scope[panel.loc[scope, "vix_lag1"].isna()]
    frac = len(missing) / len(scope)
    if frac > threshold:
        raise DataValidationError(
            f"lagged VIX missing on {len(missing)}/{len(scope)} sessions "
            f"({frac:.2%}) — exceeds the pre-registered {threshold:.1%}. "
            f"Dropping rows is no longer free; reopen the drop-vs-impute "
            f"decision before proceeding.")
    return missing


# ---------------------------------------------------------------
# Curated panel
# ---------------------------------------------------------------
def assemble_panel(raw: pd.DataFrame, vix: pd.Series) -> pd.DataFrame:
    """Pure assembly, no IO or gates — so the point-in-time harness can
    rebuild the panel from truncated raws, exercising the alignment step
    rather than truncating its output."""
    panel = pd.DataFrame(index=raw.index)
    panel["close"]    = raw["adjClose"]
    panel["volume"]   = raw["adjVolume"]
    panel["vix_lag1"] = align_vix(vix, raw.index)
    return panel


def build_panel(start: str = START, write: bool = True) -> pd.DataFrame:
    raw = validate_raw(load_tiingo("spy", start), name="tiingo spy")
    vix = load_vix(start)
    vix_sane(vix)

    # Complete SPY calendar. Every session keeps its row; VIX may be NaN.
    panel = assemble_panel(raw, vix)

    missing = vix_missingness_gate(panel)
    print(f"\nVIX missingness (measured, threshold {VIX_MISS_MAX:.1%}): "
          f"{len(missing)}/{len(panel) - 1} sessions "
          f"({len(missing) / (len(panel) - 1):.3%})")
    if len(missing):
        print("  missing dates: "
              + ", ".join(str(d.date()) for d in missing[:10])
              + (" ..." if len(missing) > 10 else ""))

    if write:
        CURATED_DIR.mkdir(parents=True, exist_ok=True)
        out = CURATED_DIR / "spy_vix_panel.parquet"
        panel.to_parquet(out)
        print(f"[curated] wrote {out}  ({len(panel)} rows, "
              f"{panel.index.min().date()} -> {panel.index.max().date()})")
    return panel


if __name__ == "__main__":
    panel = build_panel()
    obs = panel["vix_lag1"].dropna()
    print(f"\nVIX (lagged) sanity: min {obs.min():.2f}  max {obs.max():.2f}"
          f"  (2008 peak 80.86 and 2020 peak 82.69 should bound the max)")
    print(f"Highest lagged readings:")
    print(obs.nlargest(3).to_string())
