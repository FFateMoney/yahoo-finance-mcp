"""Pick only the top-ranked stock (or top 5 / top 10) each day by one factor; real payoff under the limit plan.
Usage: python research/limitfit/top1.py   (run `explore.py 1 2024-09-01` first)
"""
import pandas as pd

d = pd.read_parquet("research/limitfit/out/panel_daily.parquet")
cons = pd.read_csv("research/factors/data/constituents.csv").set_index("yf")
cost = d["ticker"].map(cons["Universe"].map({"SP500": 0.0005, "SP400": 0.001})).fillna(0.001)
last = d["date"].max()
W = {"3个月": last - pd.DateOffset(months=3), "今年": pd.Timestamp("2026-01-01"),
     "1年": last - pd.DateOffset(years=1), "2年": last - pd.DateOffset(years=2)}
n = "5%"
hit = d[f"hit_{n}"]
d["real"] = (d[f"ret_{n}"] + 2 * cost).where(~hit, d[f"n_{n}"])
d["hold"] = d["ret_hold"] + 2 * cost
for f in ["60日波动率", "ATR占价格", "过去触及率(+3%,10日)"]:
    rk = d.groupby("date")[f].rank(ascending=False, method="first")
    print(f"\n== {f}，n={n} ==")
    for w, t0 in W.items():
        m = d["date"] >= t0
        out = [f"{w}:"]
        for k in [1, 5, 10, None]:
            s = d[m & (rk <= k)] if k else d[m]
            per = s.groupby("date")["real"].mean()
            out.append(f"{'第1名' if k == 1 else f'前{k}' if k else '全部'} 每笔 {per.mean():+.2%} 亏钱的天数 {(per < 0).mean():.0%} "
                       f"最差 {per.min():+.0%}")
        top1 = d[m & (rk == 1)]
        out.append(f"| 第1名成交率 {top1[f'hit_{n}'].mean():.0%}，不同股票 {top1['ticker'].nunique()} 只，"
                   f"最常出现 {', '.join(top1['ticker'].value_counts().head(3).index)}")
        print("  ".join(out))
