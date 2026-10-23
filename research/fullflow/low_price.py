"""Candidate rule plus a "price is low" condition: among stocks with financial score >= 8, no earnings in 28 days,
not tech/communication, above a rising 200-day average, require a low-price condition, then buy the one with the
highest (score, past +3% touch rate). Limit +3% / +5%, keep holding one more round, sell at day-20 close.
Usage: python research/fullflow/low_price.py   (run flow.py and flow.py --daily first)
"""
import numpy as np
import pandas as pd

LOW = {
    "不加（原候选：顺势或回调）": lambda d: d["full"],
    "回调（20日线下方，RSI 30-50）": lambda d: d["base"] & (d["离20日线"] < 0) & d["RSI14"].between(30, 50),
    "20日线下方": lambda d: d["base"] & (d["离20日线"] < 0),
    "20日线下方 3% 以上": lambda d: d["base"] & (d["离20日线"] < -0.03),
    "50日线下方": lambda d: d["base"] & (d["离50日线"] < 0),
    "20日区间位置 < 30%": lambda d: d["base"] & (d["20日区间位置"] < 0.3),
    "RSI < 40": lambda d: d["base"] & (d["RSI14"] < 40),
}


def prep(path):
    d = pd.read_parquet(path)
    d["base"] = (d["no_earn"] & (d["score"] >= 8) & ~d["sector"].isin(["Information Technology", "Communication Services"])
                 & (d["离200日线"] > 0) & d["ma200_up"].fillna(False).astype(bool))
    return d


def report(d, windows, lines):
    allm = d.groupby("date")["ret_3%"].mean()
    for w, t0 in windows.items():
        lines.append(f"\n== {w} ==  全部股票 n=3% 每笔 {allm[allm.index >= t0].mean():+.2%}")
        for name, f in LOW.items():
            q = d[f(d).fillna(False) & (d["date"] >= t0)].copy()
            if q.empty:
                continue
            q["s"] = q["score"] * 10 + q["过去触及率(+3%,10日)"]
            top = q.sort_values("s", ascending=False).groupby("date").head(1)
            band = [q.assign(r=np.random.default_rng(k).random(len(q))).sort_values("r").groupby("date").head(1)["ret_3%"].mean()
                    for k in range(100)]
            ndates = d.loc[d["date"] >= t0, "date"].nunique()
            lines.append(
                f"{name:<22} 有股票的期数 {top['date'].nunique() / ndates:4.0%}  每期 {q.groupby('date').size().mean():4.0f} 只 | "
                f"整组 n=3% {q.groupby('date')['ret_3%'].mean().mean():+.2%} | 第1名 n=3% {top['ret_3%'].mean():+.2%} "
                f"n=5% {top['ret_5%'].mean():+.2%} 成交 {top['hit_3%'].mean():.0%} 没成交 {top.loc[~top['hit_3%'].astype(bool), 'ret_3%'].mean():+.1%} "
                f"直接持有 {top['ret_hold'].mean():+.2%} | 随机1只范围 {np.percentile(band, 5):+.2%}~{np.percentile(band, 95):+.2%} | "
                f"{top['ticker'].nunique()} 只不同股票")


lines = []
d = prep("research/fullflow/flow_panel_daily.parquet")
last = d["date"].max()
lines.append(f"每个交易日入场，信号至 {last.date()}")
report(d, {"1年": last - pd.DateOffset(years=1), "3个月": last - pd.DateOffset(months=3),
           "1个月": last - pd.DateOffset(months=1)}, lines)
d5 = prep("research/fullflow/flow_panel.parquet")
lines.append("\n\n每 5 个交易日入场，按年份")

for y in [2022, 2023, 2024, 2025]:
    sub = d5[d5["date"].dt.year == y]
    report(sub, {f"{y} 年": sub["date"].min()}, lines)
open("research/fullflow/low_price.txt", "w").write("\n".join(lines))
print("\n".join(lines))
