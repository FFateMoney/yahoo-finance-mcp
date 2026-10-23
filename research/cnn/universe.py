"""Point-in-time S&P 500 / S&P 400 membership on a daily calendar, 2000 on.

S&P 500: latest fja05680 daily history, then Wikipedia's change table for any later dates. Tickers with a
"-YYYYMM" suffix in fja are delisted/renamed; they count as members (for coverage) but have no prices.
S&P 400: research/factors/pit.py logic, applied only to tickers that are current S&P 400 members or
appear in the S&P 400 change table (reliable from 2012).
"""
import sys

import pandas as pd

sys.path.insert(0, "research/factors")
from pit import membership as pit400  # noqa: E402

F = "research/factors/data"


def _date(s):
    return pd.to_datetime(s.astype(str).str.replace(r"\[.*?\]", "", regex=True), errors="coerce")


def sp500_sets(cal):
    """Dict date -> set of member symbols (Yahoo style), for each date in cal."""
    fja = pd.read_csv("research/cnn/data/sp500_history.csv", parse_dates=["date"]).sort_values("date")
    snaps = [(d, {s.replace(".", "-") for s in t.split(",")}) for d, t in zip(fja["date"], fja["tickers"])]
    try:
        ch = pd.read_csv("research/cnn/data/sp500_changes.csv")
    except pd.errors.EmptyDataError:
        ch = pd.DataFrame()
    if ch.empty:
        ch = pd.DataFrame({"Date": [], "Added | Ticker": [], "Removed | Ticker": []})
    dcol = next(c for c in ch.columns if c.startswith("Date") or c.startswith("Effective"))
    add = next(c for c in ch.columns if "Added" in c and "Ticker" in c)
    rem = next(c for c in ch.columns if "Removed" in c and "Ticker" in c)
    ch["d"] = _date(ch[dcol])
    ch = ch[ch["d"] > snaps[-1][0]].sort_values("d")
    cur = set(snaps[-1][1])
    for d, g in ch.groupby("d"):
        cur = (cur - {str(s).replace(".", "-") for s in g[rem].dropna()}) | {str(s).replace(".", "-") for s in g[add].dropna()}
        snaps.append((d, set(cur)))
    out, k = {}, -1
    for d in cal:
        while k + 1 < len(snaps) and snaps[k + 1][0] <= d:
            k += 1
        out[d] = snaps[k][1] if k >= 0 else set()
    return out


def masks(C, dvol):
    """Boolean DataFrames (index=C.index, columns=C.columns): in S&P 500, in S&P 400; plus S&P 500 sets."""
    s5 = sp500_sets(C.index)
    m500 = pd.DataFrame({d: C.columns.isin(list(s)) for d, s in s5.items()}, index=C.columns).T
    t = pd.read_csv(f"{F}/membership/sp400_table1.csv")
    cur400 = set(pd.read_csv(f"{F}/constituents.csv").query("Universe == 'SP400'")["Symbol"])
    names = {str(s).replace(".", "-") for s in pd.concat([t["Added | Ticker"], t["Removed | Ticker"]]).dropna()}
    names |= {s.replace(".", "-") for s in cur400}
    cols = [c for c in C.columns if c in names]
    m400 = pd.DataFrame(False, index=C.index, columns=C.columns)
    m400[cols] = pit400(C[cols], dvol[cols]).values
    m400 &= ~m500
    return m500, m400, s5
