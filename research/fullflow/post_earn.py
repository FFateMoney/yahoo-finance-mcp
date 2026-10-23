"""R1 + skip stocks that reported earnings recently and fell after the report (CROX/YETI/SWK case, 2026-08).
Reports only the last 1 year / 3 months / 1 month (Jasper's preference from 2026-10-23).
Usage: python research/fullflow/post_earn.py   (needs flow_panel_daily.parquet)
"""
import numpy as np
import pandas as pd

D = "research/factors/data"
d = pd.read_parquet("research/fullflow/flow_panel_daily.parquet")
cons = pd.read_csv(f"{D}/constituents.csv").set_index("yf")
px = pd.read_csv(f"{D}/prices.csv.gz", parse_dates=["Date"], usecols=["Date", "Ticker", "Close"])
rem = pd.read_csv(f"{D}/prices_removed.csv.gz", parse_dates=["Date"], usecols=["Date", "Ticker", "Close"])
px = pd.concat([px, rem[~rem["Ticker"].isin(cons.index)]])
C = px.pivot(index="Date", columns="Ticker", values="Close").sort_index()
cal = C.index

e = pd.concat([pd.read_csv(f"research/fullflow/data/{f}.csv.gz", usecols=["ticker", "filed"]) for f in ("earnings", "earnings_yf")])
e["filed"] = pd.to_datetime(e["filed"])
e = e[e["filed"] <= cal[-1]].drop_duplicates()
e["k"] = cal.searchsorted(e["filed"])  # first trading day on/after the report
e = e[(e["k"] >= 1) & (e["k"] + 1 < len(cal))].drop_duplicates(["ticker", "k"])
rows = []
for r in e.itertuples():
    if r.ticker not in C:
        continue
    pre = C[r.ticker].iloc[r.k - 1]                     # close before the report
    post = C[r.ticker].iloc[r.k + 1]                    # close one day after (covers before-open and after-close reports)
    rows.append((r.ticker, cal[r.k], pre, post / pre - 1))
ev = pd.DataFrame(rows, columns=["ticker", "edate", "pre_close", "reaction"]).dropna().sort_values("edate")
d = d.sort_values("date")
d = pd.merge_asof(d, ev, left_on="date", right_on="edate", by="ticker", direction="backward")
pos = pd.Series(np.arange(len(cal)), index=cal)
d["since"] = pos.reindex(d["date"]).values - pos.reindex(d["edate"]).fillna(-10**6).values  # trading days since last report
d["below_pre"] = d["price"] < d["pre_close"]

d["base"] = (d["no_earn"] & (d["score"] >= 8) & ~d["sector"].isin(["Information Technology", "Communication Services"])
             & (d["离200日线"] > 0) & d["ma200_up"].fillna(False).astype(bool) & (d["离20日线"] < 0) & d["RSI14"].between(30, 50))
d["touch"] = d["过去触及率(+3%,10日)"]
d["RSI_neg"] = -d["RSI14"]
last = d["date"].max()
WIN = {"最近 1 年": last - pd.DateOffset(years=1), "最近 3 个月": last - pd.DateOffset(months=3), "最近 1 个月": last - pd.DateOffset(months=1)}
SKIP = {
    "R1（不加）": None,
    "跳过：20 天内发过财报": d["since"] <= 20,
    "跳过：20 天内发过财报且财报后下跌": (d["since"] <= 20) & (d["reaction"] < 0),
    "跳过：40 天内发过财报且财报后下跌": (d["since"] <= 40) & (d["reaction"] < 0),
    "跳过：40 天内发过财报且现价低于财报前": (d["since"] <= 40) & d["below_pre"],
    "跳过：财报后跌超 5%（60 天内）": (d["since"] <= 60) & (d["reaction"] < -0.05),
}


def picks(skip, cap=None, rank="touch"):
    m = d["base"].copy()
    if skip is not None:
        m &= ~skip
    if cap:
        m &= d["price"] <= cap
    q = d[m].copy()
    q["s"] = q["score"] * 10 + q[rank]
    return q.sort_values("s", ascending=False).groupby("date").head(1)


allavg = {w: d.loc[d["date"] >= t0].groupby("date")["ret_3%"].mean().mean() for w, t0 in WIN.items()}
lines = [f"信号 {d['date'].min().date()} 至 {last.date()}（每笔要等约 28 天出结果）。第 1 名每笔平均收益，n=3%，扣成本，没成交按第 28 天收盘价",
         "全部股票平均：" + "，".join(f"{w} {v:+.2%}" for w, v in allavg.items())]
for title, cap, rank in [("不限股价，原排序", None, "touch"), ("股价 ≤ $180，原排序", 180, "touch"), ("股价 ≤ $180，同分按 RSI 最低", 180, "RSI_neg")]:
    lines.append(f"\n== {title} ==")
    t = {}
    for name, sk in SKIP.items():
        tp = picks(sk, cap, rank)
        r = {}
        for w, t0 in WIN.items():
            x = tp.loc[tp["date"] >= t0, "ret_3%"]
            r[w] = f"{x.mean():+.2%}"
            r[w + " 亏>5%"] = f"{(x < -0.05).mean():.0%}"
        r["笔数(1年)"] = int((tp["date"] >= WIN["最近 1 年"]).sum())
        t[name] = r
    lines.append(pd.DataFrame(t).T.to_string())
case = d[d["ticker"].isin(["CROX", "YETI", "SWK"]) & d["date"].between("2026-08-10", "2026-09-05") & d["base"]]
lines.append("\nCROX/YETI/SWK 当时：\n" + case.groupby("ticker").agg(edate=("edate", "first"), reaction=("reaction", "first"),
                                                                  since=("since", "min"), below_pre=("below_pre", "mean")).to_string())
open("research/fullflow/post_earn.txt", "w").write("\n".join(lines))
print("\n".join(lines))
