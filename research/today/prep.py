"""Build pick_today.py inputs from the canonical price store (research/pricedb):
research/factors/data/prices.csv.gz (current members + SPY/MDY, since 2024-03), constituents.csv,
and research/fullflow/data/earnings_yf.csv.gz (Yahoo calendar incl. upcoming, merged with the old file)."""
import sys

import pandas as pd

sys.path.insert(0, "research/pricedb")
import pricedb as P

cons = pd.read_csv(f"{P.MEMBERS}/constituents.csv")
cons.to_csv("research/factors/data/constituents.csv", index=False)
want = set(cons["yf"]) | {"SPY", "MDY"}
px = P.load(start="2024-03-01")
px = px[px["Ticker"].isin(want)][["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]]
px.to_csv("research/factors/data/prices.csv.gz", index=False, compression="gzip")
print("prices", px["Ticker"].nunique(), "tickers, last", px["Date"].max().date())
e = P.earnings().rename(columns={"date": "filed"})[["ticker", "filed"]]
old = pd.read_csv("research/fullflow/data/earnings_yf.csv.gz", parse_dates=["filed"])
out = pd.concat([old[~old["ticker"].isin(set(e["ticker"]))], e]).drop_duplicates()
out["filed"] = pd.to_datetime(out["filed"]).dt.strftime("%Y-%m-%d")
out.to_csv("research/fullflow/data/earnings_yf.csv.gz", index=False, compression="gzip")
print("earnings", len(out), "dates")
