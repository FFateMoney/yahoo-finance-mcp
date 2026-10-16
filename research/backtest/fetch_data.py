"""Download ~5y of daily prices for current S&P 500 members (run in GitHub Actions)."""
import io
import urllib.request

import pandas as pd
import yfinance as yf

URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
cons = pd.read_csv(io.StringIO(urllib.request.urlopen(URL).read().decode()))
cons["yf"] = cons["Symbol"].str.replace(".", "-", regex=False)
cons.to_csv("research/backtest/data/constituents.csv", index=False)

tickers = cons["yf"].tolist() + ["SPY"]
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
out.to_csv("research/backtest/data/prices.csv.gz", index=False, compression="gzip")
print(out["Ticker"].nunique(), "tickers,", len(out), "rows,", out["Date"].min(), "->", out["Date"].max())
