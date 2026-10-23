"""Jasper's payoff, unfilled positions counted as 0:
  filled within 10 trading days (~14 calendar days)  -> +n%
  filled on day 11-20 (the second, "rescue" round)    -> +n%
  not filled within 20 trading days (~28 days)        -> 0
Payoffs compared: within 14 days only, within 28 days, and per trading day held (payoff / days, unfilled = 20 days).
Correlation = within-date Spearman rank IC of each feature with the payoff, averaged over recent windows.
Real versions ("真实28天", "真实每天"): unfilled positions are valued at the day-20 close instead of 0
(no trading costs in either, so costs do not differ between S&P 500 and 400 names).
Usage: python research/limitfit/touch_ic.py   (run `explore.py 1 2024-09-01` first)
"""
import numpy as np
import pandas as pd

d = pd.read_parquet("research/limitfit/out/panel_daily.parquet")
NS = ["2%", "3%", "5%", "7%", "10%", "1σ"]
FEATS = [c for c in d.columns if not c.startswith(("ret_", "ex_", "hit_", "days_", "n_")) and c not in ("i", "tk", "date", "ticker")]
last = d["date"].max()
WIN = {"3个月": last - pd.DateOffset(months=3), "今年": pd.Timestamp("2026-01-01"),
       "1年": last - pd.DateOffset(years=1), "2年": last - pd.DateOffset(years=2)}
cons = pd.read_csv("research/factors/data/constituents.csv").set_index("yf")
cost = d["ticker"].map(cons["Universe"].map({"SP500": 0.0005, "SP400": 0.001})).fillna(0.001)
PAY = {"14天内": {}, "28天内": {}, "每天": {}, "真实28天": {}, "真实每天": {}}
for n in NS:
    hit, days, nn = d[f"hit_{n}"], d[f"days_{n}"], d[f"n_{n}"]
    PAY["14天内"][n] = (hit & (days <= 10)) * nn
    PAY["28天内"][n] = hit * nn
    PAY["每天"][n] = hit * nn / days
    real = (d[f"ret_{n}"] + 2 * cost).where(~hit, nn)  # filled: exactly n; unfilled: day-20 close vs entry
    PAY["真实28天"][n] = real
    PAY["真实每天"][n] = real / days


def nw_t(x, lag=20):  # daily signals with 20-day outcomes overlap
    x = pd.Series(x).dropna()
    e = (x - x.mean()).to_numpy()
    T = len(e)
    v = (e @ e) / T
    for k in range(1, min(lag, T - 1) + 1):
        v += 2 * (1 - k / (lag + 1)) * (e[k:] @ e[:-k]) / T
    return x.mean() / np.sqrt(v / T)


def daily_ic(x, y):
    """Per-date Spearman correlation of x and y."""
    t = pd.DataFrame({"x": x, "y": y, "date": d["date"]}).dropna()
    g = t.groupby("date")
    t["x"], t["y"] = g["x"].rank(), g["y"].rank()
    g = t.groupby("date")
    t["x"] -= g["x"].transform("mean")
    t["y"] -= g["y"].transform("mean")
    t["xy"], t["xx"], t["yy"] = t.x * t.y, t.x ** 2, t.y ** 2
    s = t.groupby("date")[["xy", "xx", "yy"]].sum()
    return s.xy / np.sqrt(s.xx * s.yy)


lines = [f"信号日期 {d['date'].min().date()} 至 {last.date()}，每个交易日一次。没成交的记 0（不是真实盈亏）。\n",
         "== 不挑股票的期望 =="]
for w, t0 in WIN.items():
    m = d["date"] >= t0
    lines.append(f"{w}（{d.loc[m, 'date'].nunique()} 天）")
    for n in NS:
        lines.append(f"  n={n:>3}: 14天内达到 {(d.loc[m, f'hit_{n}'] & (d.loc[m, f'days_{n}'] <= 10)).mean():.0%}，"
                     f"28天内达到 {d.loc[m, f'hit_{n}'].mean():.0%}；期望 14天 {PAY['14天内'][n][m].mean():.2%}，"
                     f"28天 {PAY['28天内'][n][m].mean():.2%}，每交易日 {PAY['每天'][n][m].mean():.3%}；"
                     f"真实 28天 {PAY['真实28天'][n][m].mean():+.2%}，真实每交易日 {PAY['真实每天'][n][m].mean():+.3%}，"
                     f"没成交平均 {PAY['真实28天'][n][m][~d.loc[m, f'hit_{n}']].mean():+.1%}，"
                     f"同期直接持有 {(d.loc[m, 'ret_hold'] + 2 * cost[m]).mean():+.2%}")

rows = []
for p, by_n in PAY.items():
    for n in NS:
        for f in FEATS:
            ic = daily_ic(d[f], by_n[n])
            ic_real = daily_ic(d[f], d[f"ret_{n}"]) if p == "28天内" else None
            for w, t0 in WIN.items():
                s = ic[ic.index >= t0]
                rows.append({"收益": p, "n": n, "因子": f, "区间": w, "IC": s.mean(), "t": nw_t(s), "IC为正天数": (s > 0).mean(),
                             "真实收益IC": None if ic_real is None else ic_real[ic_real.index >= t0].mean()})
r = pd.DataFrame(rows)
r.to_csv("research/limitfit/out/touch_ic.csv", index=False)
pd.set_option("display.width", 250, "display.max_rows", 2000)
for p in PAY:
    for n in NS:
        q = r[(r["收益"] == p) & (r["n"] == n)]
        ic = q.pivot(index="因子", columns="区间", values="IC")[list(WIN)]
        t = q.pivot(index="因子", columns="区间", values="t")[list(WIN)]
        tab = ic.map(lambda v: f"{v:+.3f}") + " (" + t.map(lambda v: f"{v:+.1f}") + ")"
        tab = tab.loc[ic["2年"].abs().sort_values(ascending=False).index]
        lines.append(f"\n== {p}，n={n}：IC（t），按 2 年 |IC| 排序 ==")
        lines.append(tab.to_string())
open("research/limitfit/out/touch_ic.txt", "w").write("\n".join(lines))
