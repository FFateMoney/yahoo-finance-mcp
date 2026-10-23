"""Fallback for earnings dates when SEC is unreachable: Yahoo earnings calendar via yfinance (runs on GitHub Actions).
Output: research/fullflow/data/earnings_yf.csv.gz with columns ticker, filed (announcement date).
"""
import time

import pandas as pd
import yfinance as yf

ours = set(pd.read_csv("research/factors/data/constituents.csv")["yf"])
ours |= set(pd.read_csv("research/factors/data/prices_removed.csv.gz", usecols=["Ticker"])["Ticker"].unique())
ours -= {"SPY", "MDY"}
rows, miss = [], []
for k, t in enumerate(sorted(ours)):
    try:
        e = yf.Ticker(t).get_earnings_dates(limit=40)
        if e is None or len(e) == 0:
            miss.append(t)
        else:
            for ts in e.index:
                rows.append((t, pd.Timestamp(ts).tz_localize(None).normalize().date()))
    except Exception as ex:  # delisted names often have no calendar
        miss.append(t)
    if k % 100 == 0:
        print(k, t, len(rows), len(miss), flush=True)
    time.sleep(0.3)
pd.DataFrame(rows, columns=["ticker", "filed"]).drop_duplicates().to_csv("research/fullflow/data/earnings_yf.csv.gz", index=False)
print("done", len(rows), "dates;", len(ours) - len(miss), "tickers with dates;", len(miss), "missing")
