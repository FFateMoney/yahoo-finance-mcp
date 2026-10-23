"""Download daily prices since 2000 for every ticker that was in the S&P 500 or S&P 400 (run in GitHub Actions).

Tickers: fja05680 S&P 500 history (suffix-free tickers only; suffixed ones are delisted/renamed and
not on Yahoo), Wikipedia S&P 500 change table, current S&P 500/400 members, and S&P 400 removals.
Writes research/cnn/work/prices.parquet (not committed; too large) and research/cnn/data/sp500_changes.csv.
"""
import io
import os
import urllib.request

import pandas as pd
import yfinance as yf

F = "research/factors/data"
OUT = "research/cnn"
UA = {"User-Agent": "Mozilla/5.0"}
os.makedirs(f"{OUT}/work", exist_ok=True)
os.makedirs(f"{OUT}/data", exist_ok=True)

html = urllib.request.urlopen(urllib.request.Request(
    "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", headers=UA)).read().decode()
for t in pd.read_html(io.StringIO(html)):
    if isinstance(t.columns, pd.MultiIndex):
        t.columns = [" | ".join(dict.fromkeys(map(str, c))) for c in t.columns]
    if any("Removed" in c for c in t.columns) and any("Added" in c for c in t.columns):
        t.to_csv(f"{OUT}/data/sp500_changes.csv", index=False)
        changes = t
        print("sp500 changes", t.shape, list(t.columns))
        break
else:
    raise RuntimeError("no S&P 500 change table")

fja = pd.read_csv(f"{F}/membership/sp500_history_fja05680.csv")
names = set()
for x in fja.loc[fja["date"] >= "2000-01-01", "tickers"]:
    names |= set(x.split(","))
clean = {s for s in names if "-" not in s}
print("fja tickers since 2000:", len(names), "suffix-free:", len(clean))

s400 = pd.read_csv(f"{F}/membership/sp400_table1.csv")
cols = [c for c in changes.columns if "Ticker" in c]
tickers = (clean | set(pd.read_csv(f"{F}/constituents.csv")["Symbol"])
           | set(s400["Added | Ticker"].dropna()) | set(s400["Removed | Ticker"].dropna())
           | set(changes[cols].stack().dropna().astype(str)))
tickers = sorted({s.strip().replace(".", "-") for s in tickers if s.strip()}) + ["SPY"]
print("tickers to download:", len(tickers))

frames = []
for i in range(0, len(tickers), 50):
    batch = tickers[i : i + 50]
    df = yf.download(batch, start="2000-01-01", interval="1d", auto_adjust=True,
                     group_by="ticker", threads=True, progress=False)
    for s in batch:
        if s not in df.columns.get_level_values(0):
            continue
        d = df[s].dropna(subset=["Close"]).reset_index()
        if len(d) == 0:
            continue
        d["Ticker"] = s
        frames.append(d[["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]])
out = pd.concat(frames)
out["Date"] = pd.to_datetime(out["Date"]).dt.tz_localize(None)
out.to_parquet(f"{OUT}/work/prices.parquet", index=False)
print(out["Ticker"].nunique(), "tickers with prices,", len(out), "rows,", out["Date"].min(), "->", out["Date"].max())
