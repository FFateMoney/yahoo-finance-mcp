"""Fetch screening inputs for the short-term workflow (run in GitHub Actions).

Universe: current S&P 500, last close <= MAX_PRICE. Saves prices, yfinance info,
Finviz screener views and StockAnalysis statistics into research/screen/out/.
"""
import io
import json
import re
import time
import urllib.request

import pandas as pd
import requests
import yfinance as yf

MAX_PRICE = 62
OUT = "research/screen/out"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"}

URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
cons = pd.read_csv(io.StringIO(urllib.request.urlopen(URL).read().decode()))
cons["yf"] = cons["Symbol"].str.replace(".", "-", regex=False)
cons.to_csv(f"{OUT}/constituents.csv", index=False)

# 1. prices (2y) for everything, then universe by price
tickers = cons["yf"].tolist() + ["SPY"]
frames = []
for i in range(0, len(tickers), 50):
    batch = tickers[i:i + 50]
    df = yf.download(batch, period="2y", interval="1d", auto_adjust=False,
                     group_by="ticker", threads=True, progress=False)
    for t in batch:
        if t not in df.columns.get_level_values(0):
            continue
        d = df[t].dropna(subset=["Close"]).reset_index()
        d["Ticker"] = t
        frames.append(d[["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]])
px = pd.concat(frames)
px["Date"] = pd.to_datetime(px["Date"]).dt.strftime("%Y-%m-%d")
last = px.sort_values("Date").groupby("Ticker").tail(1).set_index("Ticker")["Close"]
uni = sorted(t for t, c in last.items() if c <= MAX_PRICE and t != "SPY")
px[px["Ticker"].isin(uni + ["SPY"])].round(4).to_csv(f"{OUT}/prices.csv.gz", index=False, compression="gzip")
print(len(uni), "tickers <=", MAX_PRICE, "last date", px["Date"].max())

# 2. yfinance info + earnings dates
KEYS = ["sector", "industry", "marketCap", "profitMargins", "freeCashflow", "operatingCashflow",
        "debtToEquity", "returnOnAssets", "returnOnEquity", "earningsQuarterlyGrowth", "earningsGrowth",
        "revenueGrowth", "currentRatio", "trailingPE", "forwardPE", "pegRatio", "priceToBook",
        "dividendYield", "exDividendDate", "earningsTimestamp", "earningsTimestampStart",
        "earningsTimestampEnd", "mostRecentQuarter", "beta", "totalDebt", "ebitda", "longName"]
rows = []
for t in uni:
    r = {"Ticker": t}
    try:
        tk = yf.Ticker(t)
        info = tk.info
        r.update({k: info.get(k) for k in KEYS})
        try:
            ed = tk.get_earnings_dates(limit=6)
            if ed is not None and len(ed):
                r["earnings_dates"] = ";".join(str(x.date()) for x in ed.index)
        except Exception as e:  # noqa: BLE001
            r["earnings_err"] = str(e)[:80]
    except Exception as e:  # noqa: BLE001
        r["err"] = str(e)[:120]
    rows.append(r)
pd.DataFrame(rows).to_csv(f"{OUT}/yf_info.csv", index=False)

# 3. Finviz screener views (valuation 121, financial 161, technical 171)
for v in (111, 121, 161, 171):
    parts = []
    for start in range(1, 520, 20):
        url = f"https://finviz.com/screener.ashx?v={v}&f=idx_sp500,sh_price_u{60 if MAX_PRICE <= 60 else 70}&r={start}"
        try:
            html = requests.get(url, headers=UA, timeout=20).text
            tabs = [x for x in pd.read_html(io.StringIO(html)) if "Ticker" in [str(c) for c in x.columns]]
        except Exception as e:  # noqa: BLE001
            print("finviz", v, start, e)
            break
        if not tabs:
            break
        tab = tabs[-1]
        if parts and tab["Ticker"].iloc[0] == parts[-1]["Ticker"].iloc[0]:
            break
        parts.append(tab)
        if len(tab) < 20:
            break
        time.sleep(1)
    if parts:
        pd.concat(parts).drop_duplicates("Ticker").to_csv(f"{OUT}/finviz_v{v}.csv", index=False)
        print("finviz", v, sum(len(p) for p in parts))

# 4. StockAnalysis statistics
FIELDS = ["Altman Z-Score", "Piotroski F-Score", "Interest Coverage", "Free Cash Flow",
          "Debt / Equity", "Profit Margin", "Return on Invested Capital (ROIC)", "Current Ratio"]
sa = {}
for t in uni:
    url = f"https://stockanalysis.com/stocks/{t.lower().replace('-', '.')}/statistics/"
    try:
        html = requests.get(url, headers=UA, timeout=20).text
        text = re.sub(r"<[^>]+>", "|", html)
        text = re.sub(r"\|+", "|", text)
        d = {}
        for f in FIELDS:
            m = re.search(re.escape(f) + r"\s*\|\s*([^|]{1,25})\|", text)
            if m:
                d[f] = m.group(1).strip()
        sa[t] = d
    except Exception as e:  # noqa: BLE001
        sa[t] = {"err": str(e)[:80]}
    time.sleep(0.7)
json.dump(sa, open(f"{OUT}/stockanalysis.json", "w"), indent=1, ensure_ascii=False)
print("stockanalysis", sum(1 for d in sa.values() if "Altman Z-Score" in d), "/", len(sa))
