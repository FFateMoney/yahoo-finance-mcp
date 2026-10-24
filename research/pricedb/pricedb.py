"""Read the project's canonical daily price store (research/pricedb/data).

Prices are stored *raw* (as traded, not adjusted for later splits/dividends), one parquet file per
month, plus an append-only table of splits and dividends. Adjusted prices are rebuilt on read the
same way Yahoo does it, so `load()` matches `yf.download(auto_adjust=True)` without the store ever
having to rewrite old months when a stock splits or pays a dividend.

    import sys; sys.path.insert(0, "research/pricedb")
    import pricedb
    px = pricedb.load(start="2020-01-01")               # long: Date, Ticker, Open, High, Low, Close, Volume
    C = pricedb.wide("Close", start="2020-01-01")       # Date x Ticker
    cal = pricedb.calendar()                            # SPY trading days
    earn = pricedb.earnings()                           # ticker, date (Yahoo calendar, incl. upcoming)
"""
import glob
import os

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
PRICES = os.path.join(ROOT, "prices")
ACTIONS = os.path.join(ROOT, "actions.parquet")
TICKERS = os.path.join(ROOT, "tickers.csv")
EARN_YF = os.path.join(ROOT, "earnings", "yahoo.csv.gz")
EARN_SEC = os.path.join(ROOT, "earnings", "sec_8k.csv.gz")
MEMBERS = os.path.join(ROOT, "membership")
COLS = ["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]


def month_files(start=None, end=None):
    files = sorted(glob.glob(os.path.join(PRICES, "*", "*.parquet")))
    s = pd.Timestamp(start).strftime("%Y-%m") if start is not None else "0000-00"
    e = pd.Timestamp(end).strftime("%Y-%m") if end is not None else "9999-99"
    return [f for f in files if s <= os.path.basename(f)[:7] <= e]


def load_raw(start=None, end=None, tickers=None):
    """Raw (unadjusted) rows, sorted by Ticker, Date."""
    files = month_files(start, end)
    if not files:
        return pd.DataFrame(columns=COLS)
    flt = None
    if tickers is not None:
        flt = ds.field("Ticker").isin(list(tickers))
    px = ds.dataset(files, format="parquet").to_table(filter=flt).to_pandas()
    px["Date"] = pd.to_datetime(px["Date"])
    if start is not None:
        px = px[px["Date"] >= pd.Timestamp(start)]
    if end is not None:
        px = px[px["Date"] <= pd.Timestamp(end)]
    return px.sort_values(["Ticker", "Date"], ignore_index=True)


def actions(tickers=None):
    """Splits and dividends: Ticker, Date, Dividend (raw $), Split (ratio, 0 if none), PrevClose (raw)."""
    if not os.path.exists(ACTIONS):
        return pd.DataFrame(columns=["Ticker", "Date", "Dividend", "Split", "PrevClose"])
    a = pd.read_parquet(ACTIONS)
    a["Date"] = pd.to_datetime(a["Date"])
    if tickers is not None:
        a = a[a["Ticker"].isin(list(tickers))]
    return a


def factors(a):
    """Per event, the cumulative price factor that applies to every day *before* that event."""
    a = a.sort_values(["Ticker", "Date"]).copy()
    f = np.ones(len(a))
    split = a["Split"].to_numpy(float)
    f = np.where(split > 0, 1.0 / np.where(split > 0, split, 1.0), f)
    div = a["Dividend"].to_numpy(float)
    prev = a["PrevClose"].to_numpy(float)
    ok = (div > 0) & (prev > 0) & (div < prev)
    f = f * np.where(ok, 1.0 - div / np.where(ok, prev, 1.0), 1.0)
    a["f"] = f
    vf = np.where(split > 0, np.where(split > 0, split, 1.0), 1.0)  # volume scales the other way, splits only
    a["vf"] = vf
    # product of this and all later events, per ticker
    a["F"] = a.iloc[::-1].groupby("Ticker")["f"].cumprod().iloc[::-1]
    a["VF"] = a.iloc[::-1].groupby("Ticker")["vf"].cumprod().iloc[::-1]
    return a[["Ticker", "Date", "F", "VF"]]


def adjust(px, acts=None):
    """Raw -> split- and dividend-adjusted (as of the latest data), like yfinance auto_adjust=True."""
    if len(px) == 0:
        return px
    acts = actions(px["Ticker"].unique()) if acts is None else acts
    if len(acts) == 0:
        return px
    f = factors(acts).rename(columns={"Date": "EvDate"}).sort_values("EvDate")
    out = px.sort_values("Date").copy()
    out["_d"] = out["Date"]
    m = pd.merge_asof(out, f, left_on="_d", right_on="EvDate", by="Ticker",
                      direction="forward", allow_exact_matches=False)
    F = m["F"].fillna(1.0).to_numpy()
    VF = m["VF"].fillna(1.0).to_numpy()
    for c in ("Open", "High", "Low", "Close"):
        m[c] = m[c].to_numpy(float) * F
    m["Volume"] = m["Volume"].to_numpy(float) * VF
    return m[COLS].sort_values(["Ticker", "Date"], ignore_index=True)


def load(start=None, end=None, tickers=None, adjust_prices=True):
    """Daily OHLCV, long format. Adjusted (default) or raw."""
    px = load_raw(start, end, tickers)
    return adjust(px) if adjust_prices else px


def wide(field="Close", start=None, end=None, tickers=None, adjust_prices=True):
    px = load(start, end, tickers, adjust_prices)
    return px.pivot(index="Date", columns="Ticker", values=field).sort_index()


def calendar(start=None, end=None):
    """Trading days (SPY dates)."""
    return pd.DatetimeIndex(load_raw(start, end, ["SPY"])["Date"])


def last_date():
    files = month_files()
    if not files:
        return None
    return pd.read_parquet(files[-1], columns=["Date"])["Date"].max()


def tickers():
    """Ticker registry: Ticker, first, last, rows, status (ok / nodata), sources, tried."""
    return pd.read_csv(TICKERS)


def earnings(source="yahoo"):
    """Earnings announcement dates. yahoo: ticker, date (incl. upcoming); sec: 8-K item 2.02 filings."""
    if source == "sec":
        e = pd.read_csv(EARN_SEC, parse_dates=["filed"])
        return e.rename(columns={"filed": "date"})
    return pd.read_csv(EARN_YF, parse_dates=["date"])
