"""Jasper's full short-term process (短线选股分析流程, 2026-10-17 version), applied step by step, scored with the
limit-sell + keep-holding plan (+n%, 10 + 10 trading days, unfilled sold at the day-20 close).
Steps: 0 all stocks -> 1 no earnings release in the next 28 days -> 2 financial score >= 8 (point in time)
       -> 3 no Information Technology / Communication Services -> 4 above a rising 200-day average
       -> 5 trend (above 20 and 50-day averages, RSI 40-70) or pullback (below 20-day average, RSI 30-50).
Usage: python research/fullflow/flow.py   (needs research/limitfit/out/panel.parquet and research/fullflow/data/*)
"""
import numpy as np
import pandas as pd

D = "research/factors/data"
d = pd.read_parquet("research/limitfit/out/panel.parquet")
cons = pd.read_csv(f"{D}/constituents.csv").set_index("yf")
d["sector"] = d["ticker"].map(cons["Sector"]).fillna("Unknown")
d["sp500"] = d["ticker"].map(cons["Universe"]).eq("SP500")

# 200-day average rising: today's 200-day average above its value 20 trading days ago
px = pd.read_csv(f"{D}/prices.csv.gz", parse_dates=["Date"], usecols=["Date", "Ticker", "Close"])
rem = pd.read_csv(f"{D}/prices_removed.csv.gz", parse_dates=["Date"], usecols=["Date", "Ticker", "Close"])
px = pd.concat([px, rem[~rem["Ticker"].isin(cons.index)]])
C = px.pivot(index="Date", columns="Ticker", values="Close").sort_index().drop(columns=["SPY", "MDY"], errors="ignore")
ma200 = C.rolling(200).mean()
rising = (ma200 > ma200.shift(20)).stack().rename("ma200_up").reset_index().rename(columns={"Date": "date", "Ticker": "ticker"})
price = C.stack().rename("price").reset_index().rename(columns={"Date": "date", "Ticker": "ticker"})
d = d.merge(rising, on=["date", "ticker"], how="left").merge(price, on=["date", "ticker"], how="left")

# earnings: any announcement in the 28 calendar days after the signal (covers the whole holding period).
# Yahoo calendar (covers removed tickers too) plus SEC 8-K item 2.02 filing dates.
e = pd.concat([pd.read_csv(f"research/fullflow/data/{f}", parse_dates=["filed"])[["ticker", "filed"]]
               for f in ("earnings_yf.csv.gz", "earnings.csv.gz")]).drop_duplicates()
have_e = set(e["ticker"])
e = e.sort_values("filed")
nxt = []
for tk, g in d.groupby("ticker"):
    ed = e.loc[e["ticker"] == tk, "filed"].to_numpy()
    if len(ed) == 0:
        nxt.append(pd.Series(pd.NaT, index=g.index))
        continue
    pos = np.searchsorted(ed, g["date"].to_numpy(), side="right")
    nxt.append(pd.Series([ed[p] if p < len(ed) else pd.NaT for p in pos], index=g.index))
d["next_earn"] = pd.concat(nxt)
d["earn_data"] = d["ticker"].isin(have_e)
d["no_earn"] = d["earn_data"] & ((d["next_earn"] - d["date"]).dt.days > 28)

# point-in-time financial score: latest score whose filing date is on or before the signal date
sc = pd.read_parquet("research/fullflow/data/scores.parquet").sort_values("filed")
d = d.sort_values("date")
d = pd.merge_asof(d, sc.rename(columns={"filed": "score_filed"}), left_on="date", right_on="score_filed", by="ticker",
                  direction="backward")
d["score_age"] = (d["date"] - d["score_filed"]).dt.days
d.loc[d["score_age"] > 200, "score"] = np.nan  # stale (no filing for over 6 months)
mcap = d["price"] * d["shares"]
d["altman_z"] = 1.2 * d["wc_ta"] + 1.4 * d["re_ta"] + 3.3 * d["ebit_ta"] + 0.6 * mcap / d["tl"] + 1.0 * d["sales_ta"]

trend = (d["离20日线"] > 0) & (d["离50日线"] > 0) & d["RSI14"].between(40, 70)
pull = (d["离20日线"] < 0) & d["RSI14"].between(30, 50)
steps = [
    ("0 全部股票", pd.Series(True, index=d.index)),
    ("1 + 28天内不发财报", d["no_earn"]),
    ("2 + 财务打分≥8", d["score"] >= 8),
    ("3 + 排除科技和通信", ~d["sector"].isin(["Information Technology", "Communication Services"])),
    ("4 + 200日线上方且上升", (d["离200日线"] > 0) & d["ma200_up"].fillna(False).astype(bool)),
]
masks, m = [], pd.Series(True, index=d.index)
for name, s in steps:
    m = m & s.fillna(False).astype(bool)
    masks.append((name, m.copy()))
masks += [("5a + 顺势（20/50日线上方，RSI 40-70）", m & trend), ("5b + 回调（20日线下方，RSI 30-50）", m & pull),
          ("5 + 顺势或回调（完整流程）", m & (trend | pull))]
extra = [("对照：只避开财报", d["no_earn"]), ("对照：只看财务打分≥8", d["score"] >= 8),
         ("对照：完整流程 + 复核（利息覆盖≥3，Z≥1.81）", m & (trend | pull) & (d["int_cov"] >= 3) & (d["altman_z"] >= 1.81))]


def nw_t(x, lag=4):
    x = pd.Series(x).dropna()
    e_ = (x - x.mean()).to_numpy()
    T = len(e_)
    v = (e_ @ e_) / T
    for k in range(1, min(lag, T - 1) + 1):
        v += 2 * (1 - k / (lag + 1)) * (e_[k:] @ e_[:-k]) / T
    return x.mean() / np.sqrt(v / T)


last = d["date"].max()
W = {"全部（2021-11 起）": d["date"].min(), "2年": last - pd.DateOffset(years=2), "1年": last - pd.DateOffset(years=1)}
lines = [f"信号 {d['date'].min().date()} 至 {last.date()}，每 5 个交易日一次；收益扣交易成本",
         f"有财报日期数据的股票：{d.loc[d['earn_data'], 'ticker'].nunique()} / {d['ticker'].nunique()}；"
         f"有财务打分的行：{d['score'].notna().mean():.0%}"]
all_by_date = {n: d.groupby("date")[f"ret_{n}"].mean() for n in ["3%", "5%", "hold"]}
for w, t0 in W.items():
    lines.append(f"\n== {w} ==")
    for name, mk in masks + extra:
        s = d[mk & (d["date"] >= t0)]
        parts = [f"{name:<34} 平均每期 {s.groupby('date').size().reindex(d.loc[d['date'] >= t0, 'date'].unique()).fillna(0).mean():5.0f} 只"]
        for n in ["3%", "5%"]:
            per = s.groupby("date")[f"ret_{n}"].mean()
            rel = per - all_by_date[n].reindex(per.index)
            unf = s.loc[~s[f"hit_{n}"].astype(bool), f"ret_{n}"].mean()
            parts.append(f"n={n}: 每笔 {per.mean():+.2%} 比SPY {s.groupby('date')[f'ex_{n}'].mean().mean():+.2%} "
                         f"比全部 {rel.mean():+.2%}(t {nw_t(rel):+.1f}) 成交 {s[f'hit_{n}'].mean():.0%} 没成交 {unf:+.1%}")
        hold = s.groupby("date")["ret_hold"].mean()
        parts.append(f"直接持有20天 {hold.mean():+.2%}")
        lines.append(" | ".join(parts))
open("research/fullflow/out.txt", "w").write("\n".join(lines))
print("\n".join(lines))
