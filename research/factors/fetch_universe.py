"""Download ~6y of daily prices for current S&P 500 + S&P 400 members (run in GitHub Actions)."""
import io
import urllib.request

import pandas as pd
import yfinance as yf

UA = {"User-Agent": "Mozilla/5.0"}


def wiki_table(url, sym_col, sector_col):
    html = urllib.request.urlopen(urllib.request.Request(url, headers=UA)).read().decode()
    for t in pd.read_html(io.StringIO(html)):
        if sym_col in t.columns and sector_col in t.columns:
            return t[[sym_col, sector_col]].set_axis(["Symbol", "Sector"], axis=1)
    raise RuntimeError(f"no table in {url}")


sp500 = wiki_table("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", "Symbol", "GICS Sector")
sp500["Universe"] = "SP500"
sp400 = wiki_table("https://en.wikipedia.org/wiki/List_of_S%26P_400_companies", "Symbol", "GICS Sector")
sp400["Universe"] = "SP400"
cons = pd.concat([sp500, sp400]).drop_duplicates("Symbol")
cons["yf"] = cons["Symbol"].str.replace(".", "-", regex=False)
cons.to_csv("research/factors/data/constituents.csv", index=False)
print(cons["Universe"].value_counts().to_dict())

tickers = cons["yf"].tolist() + ["SPY", "MDY"]
frames = []
for i in range(0, len(tickers), 50):
    batch = tickers[i : i + 50]
    df = yf.download(batch, period="6y", interval="1d", auto_adjust=True,
                     group_by="ticker", threads=True, progress=False)
    for t in batch:
        if t not in df.columns.get_level_values(0):
            continue
        d = df[t].dropna(subset=["Close"]).reset_index()
        d["Ticker"] = t
        frames.append(d[["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]])
out = pd.concat(frames)
out["Date"] = pd.to_datetime(out["Date"]).dt.strftime("%Y-%m-%d")
out = out.round({"Open": 4, "High": 4, "Low": 4, "Close": 4})
out.to_csv("research/factors/data/prices.csv.gz", index=False, compression="gzip")
print(out["Ticker"].nunique(), "tickers,", len(out), "rows,", out["Date"].min(), "->", out["Date"].max())
