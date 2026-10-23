"""Among the stocks that pass the full process each period, rank by one rule and buy only the top 1 (or top 3).
Usage: python research/fullflow/pick_one.py   (run flow.py first; it writes flow_panel.parquet)
"""
import numpy as np
import pandas as pd

import sys

DAILY = "--daily" in sys.argv
d = pd.read_parquet("research/fullflow/flow_panel_daily.parquet" if DAILY else "research/fullflow/flow_panel.parquet")
last = d["date"].max()
W = ({"1年": last - pd.DateOffset(years=1), "3个月": last - pd.DateOffset(months=3), "1个月": last - pd.DateOffset(months=1)}
     if DAILY else {"全部（2021-11 起）": d["date"].min(), "2年": last - pd.DateOffset(years=2), "1年": last - pd.DateOffset(years=1)})
p = d[d["full"]].copy()
rng = np.random.default_rng(0)
p["_rand"] = rng.random(len(p))
RULES = {  # higher = picked first
    "财务打分最高（同分看过去触及率）": p["score"] * 10 + p["过去触及率(+3%,10日)"],
    "过去 1 年碰到 +3% 最频繁": p["过去触及率(+3%,10日)"],
    "RSI 最低": -p["RSI14"],
    "20 日区间位置最低": -p["20日区间位置"],
    "离 52 周高点最近": p["离52周高点"],
    "动量 12-1 最强": p["动量12-1"],
    "1 个月涨幅最大": p["1月涨幅"],
    "成交额最大（最大的公司）": p["成交额(规模)"],
    "随机挑一只（对照）": p["_rand"],
}
lines = [f"每期通过完整流程的股票里，按规则只买第 1 名（或前 3 名）。信号 {d['date'].min().date()} 至 {last.date()}，{'每个交易日' if DAILY else '每 5 个交易日'}一次"]
allm = {n: d.groupby("date")[f"ret_{n}"].mean() for n in ["3%", "5%"]}
grp = {n: p.groupby("date")[f"ret_{n}"].mean() for n in ["3%", "5%"]}
for w, t0 in W.items():
    lines.append(f"\n== {w} ==   对照：全部股票 n=3% {allm['3%'][allm['3%'].index >= t0].mean():+.2%}，"
                 f"通过流程的平均 {grp['3%'][grp['3%'].index >= t0].mean():+.2%}；n=5% {allm['5%'][allm['5%'].index >= t0].mean():+.2%} / "
                 f"{grp['5%'][grp['5%'].index >= t0].mean():+.2%}")
    q0 = p[p["date"] >= t0]
    band = []
    for seed in range(200):
        r = np.random.default_rng(seed).random(len(q0))
        band.append(q0.assign(_r=r).sort_values("_r").groupby("date").head(1)["ret_3%"].mean())
    lines.append(f"随机挑 1 只的运气范围（n=3%，5%–95% 分位）：{np.percentile(band, 5):+.2%} 到 {np.percentile(band, 95):+.2%}")
    for name, sc in RULES.items():
        q = p.assign(_s=sc)[p["date"] >= t0]
        rk = q.groupby("date")["_s"].rank(ascending=False, method="first")
        out = [f"{name:<20}"]
        for n in ["3%", "5%"]:
            for k in (1, 3):
                per = q[rk <= k].groupby("date")[f"ret_{n}"].mean()
                out.append(f"n={n} 前{k} {per.mean():+.2%}")
            t1 = q[rk == 1]
            out.append(f"(第1名成交 {t1[f'hit_{n}'].mean():.0%}，没成交 {t1.loc[~t1[f'hit_{n}'].astype(bool), f'ret_{n}'].mean():+.1%}，"
                       f"亏钱的期数 {(t1[f'ret_{n}'] < 0).mean():.0%})")
        out.append(f"直接持有20天 第1名 {q[rk == 1]['ret_hold'].mean():+.2%}  第1名共 {q[rk == 1]['ticker'].nunique()} 只")
        lines.append("  ".join(out))
open("research/fullflow/pick_one_daily.txt" if DAILY else "research/fullflow/pick_one.txt", "w").write("\n".join(lines))
print("\n".join(lines))
