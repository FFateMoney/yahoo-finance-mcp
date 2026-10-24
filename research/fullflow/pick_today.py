"""Today's R1 pick from the latest prices (process doc 资料/短线选股分析流程.md, third version, 2026-10-23 rules):
S&P 500/400 now, price > $5 and <= CAP, 21-day dollar volume > $5M, no earnings within ±1 month, score >= 8,
no IT / communication, above a rising 200-day average, below the 20-day average with RSI14 30-50.
Ranked by score, then past +3% touch rate (doc rule) — also shown: ties by lowest RSI.
Usage: python research/fullflow/pick_today.py [cap] [as-of date YYYY-MM-DD]
"""
import sys

import numpy as np
import pandas as pd

CAP = float(sys.argv[1]) if len(sys.argv) > 1 else 180.0
D = "research/factors/data"
cons = pd.read_csv(f"{D}/constituents.csv").set_index("yf")
px = pd.read_csv(f"{D}/prices.csv.gz", parse_dates=["Date"])
if len(sys.argv) > 2:
    px = px[px["Date"] <= sys.argv[2]]
P = {k: px.pivot(index="Date", columns="Ticker", values=k).sort_index() for k in ["High", "Close", "Volume"]}
Hi, C, V = (P[k].reindex(columns=cons.index) for k in ["High", "Close", "Volume"])
t = C.index[-1]
r1 = C.pct_change(fill_method=None)
up, dn = r1.clip(lower=0).rolling(14).mean(), (-r1.clip(upper=0)).rolling(14).mean()
ma200 = C.rolling(200).mean()
fwd_hi = Hi[::-1].rolling(10).max()[::-1].shift(-1)
touched = (fwd_hi >= C * 1.03).astype(float).where(fwd_hi.notna())
x = pd.DataFrame({
    "price": C.loc[t], "dvol": (C * V).rolling(21).mean().loc[t],
    "rsi": (100 - 100 / (1 + up / dn)).loc[t], "ma20": (C / C.rolling(20).mean() - 1).loc[t],
    "ma200": (C / ma200 - 1).loc[t], "ma200_up": (ma200 > ma200.shift(20)).loc[t],
    "touch": touched.shift(11).rolling(250, min_periods=120).mean().loc[t], "sector": cons["Sector"]})

e = pd.concat([pd.read_csv(f"research/fullflow/data/{f}", parse_dates=["filed"])[["ticker", "filed"]]
               for f in ("earnings_yf.csv.gz", "earnings.csv.gz")]).drop_duplicates()
past21 = C.index[-22]  # 21 trading days back
last_e = e[e["filed"] <= t].groupby("ticker")["filed"].max()
next_e = e[e["filed"] > t].groupby("ticker")["filed"].min()
x["last_earn"], x["next_earn"] = last_e, next_e
x["next_earn"] = x["next_earn"].fillna(x["last_earn"] + pd.Timedelta(days=91))  # not announced yet: estimate
x["no_earn"] = (x["last_earn"] < past21) & ((x["next_earn"] - t).dt.days > 28)

sc = pd.read_parquet("research/fullflow/data/scores.parquet")
sc = sc[sc["filed"] <= t].sort_values("filed").groupby("ticker").last()
x["score"] = sc["score"].where((t - sc["filed"]).dt.days <= 200)

m = ((x["price"] > 5) & (x["price"] <= CAP) & (x["dvol"] > 5e6) & x["no_earn"] & (x["score"] >= 8)
     & ~x["sector"].isin(["Information Technology", "Communication Services"]) & (x["ma200"] > 0) & x["ma200_up"]
     & (x["ma20"] < 0) & x["rsi"].between(30, 50))
q = x[m].copy()
q["doc_rank"] = (q["score"] * 10 + q["touch"]).rank(ascending=False)
q["rsi_rank"] = (q["score"] * 10 - q["rsi"]).rank(ascending=False)
cols = ["sector", "price", "score", "rsi", "ma20", "ma200", "touch", "last_earn", "next_earn", "doc_rank", "rsi_rank"]
print(f"数据日期 {t.date()}，股价上限 ${CAP:.0f}，通过全部条件 {len(q)} 只")
steps = {"价格/成交额": (x["price"] > 5) & (x["price"] <= CAP) & (x["dvol"] > 5e6), "+财报前后1个月": x["no_earn"],
         "+财务分>=8": x["score"] >= 8, "+排除科技通信": ~x["sector"].isin(["Information Technology", "Communication Services"]),
         "+200日线上方且上升": (x["ma200"] > 0) & x["ma200_up"], "+20日线下方且RSI 30-50": (x["ma20"] < 0) & x["rsi"].between(30, 50)}
acc = pd.Series(True, index=x.index)
for k, v in steps.items():
    acc &= v.fillna(False).astype(bool)
    print(f"{k}: {int(acc.sum())}")
print(q.sort_values("doc_rank")[cols].to_string() if len(q) else "今天没有符合全部条件的股票")
near = x[acc.index.isin(x.index) & steps["价格/成交额"] & x["no_earn"] & (x["score"] >= 8) & steps["+排除科技通信"]
         & steps["+200日线上方且上升"].fillna(False).astype(bool) & ~m]
print("\n只差“回调”条件的（等它们跌回 20 日线下方、RSI 30-50）：")
print(near[["sector", "price", "score", "rsi", "ma20", "touch", "next_earn"]].round(3).to_string())
