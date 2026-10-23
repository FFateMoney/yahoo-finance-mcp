"""Download prices for tickers removed from the S&P 400 since 2020-10 (to reduce survivorship bias; run in GitHub Actions)."""
import pandas as pd
import yfinance as yf

D = "research/factors/data"
t = pd.read_csv(f"{D}/membership/sp400_table1.csv")
t["d"] = pd.to_datetime(t["Date"].str.replace(r"\[.*?\]", "", regex=True), errors="coerce")
cur = set(pd.read_csv(f"{D}/constituents.csv")["Symbol"])
removed = sorted(set(t.loc[t["d"] >= "2020-10-01", "Removed | Ticker"].dropna()) - cur)
tickers = [s.replace(".", "-") for s in removed]
frames = []
for i in range(0, len(tickers), 50):
    batch = tickers[i : i + 50]
    df = yf.download(batch, period="6y", interval="1d", auto_adjust=True, group_by="ticker", threads=True, progress=False)
    for s in batch:
        if s not in df.columns.get_level_values(0):
            continue
        d = df[s].dropna(subset=["Close"]).reset_index()
        if len(d) == 0:
            continue
        d["Ticker"] = s
        frames.append(d[["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]])
out = pd.concat(frames)
out["Date"] = pd.to_datetime(out["Date"]).dt.strftime("%Y-%m-%d")
out = out.round({"Open": 4, "High": 4, "Low": 4, "Close": 4})
out.to_csv(f"{D}/prices_removed.csv.gz", index=False, compression="gzip")
print(len(removed), "removed tickers;", out["Ticker"].nunique(), "with prices")
