"""Backtest the short-term screening rules and common price factors on S&P 500 daily data.

Usage: python research/backtest/backtest.py [prices.csv.gz]
"""
import sys

import numpy as np
import pandas as pd

H = 10  # holding period in trading days
path = sys.argv[1] if len(sys.argv) > 1 else "research/backtest/data/prices.csv.gz"
px = pd.read_csv(path, parse_dates=["Date"])
close = px.pivot(index="Date", columns="Ticker", values="Close").sort_index()
high = px.pivot(index="Date", columns="Ticker", values="High").reindex_like(close)
low = px.pivot(index="Date", columns="Ticker", values="Low").reindex_like(close)
spy = close.pop("SPY")
high = high.drop(columns="SPY")
low = low.drop(columns="SPY")


def rsi(c, n=14):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


ma20, ma50, ma200 = (close.rolling(n).mean() for n in (20, 50, 200))
r14 = rsi(close)
fwd = close.shift(-H) / close - 1
fwd_x = fwd.sub(spy.shift(-H) / spy - 1, axis=0)  # excess over SPY
fut_high = high[::-1].rolling(H).max()[::-1].shift(-1)
fut_low = low[::-1].rolling(H).min()[::-1].shift(-1)
hit_up3 = (fut_high >= close * 1.03).astype(float).where(fut_high.notna())
hit_dn3 = (fut_low <= close * 0.97).astype(float).where(fut_low.notna())
hit_dn5 = (fut_low <= close * 0.95).astype(float).where(fut_low.notna())

valid = ma200.notna() & fwd.notna()
start = valid.any(axis=1).idxmax()
dates = close.index[(close.index >= start) & fwd.notna().any(axis=1)]
reb = dates[::H]  # non-overlapping rebalance dates
print(f"Data: {close.shape[1]} stocks, {close.index[0].date()} -> {close.index[-1].date()}")
print(f"Test window: {reb[0].date()} -> {reb[-1].date()}, {len(reb)} non-overlapping {H}-day periods\n")

# ---------- Part A: the screening rules ----------
up200 = (close > ma200) & (ma200 > ma200.shift(20))
trend = up200 & (close > ma20) & (close > ma50) & (r14 >= 40) & (r14 <= 70)
pullback = up200 & (close < ma20) & (r14 >= 30) & (r14 <= 50)
below200 = close < ma200
conds = {
    "全部股票（基准）": valid,
    "200日均线上方且上升": up200 & valid,
    "规则A 顺势": trend & valid,
    "规则B 回调": pullback & valid,
    "200日均线下方": below200 & valid,
    "200下方 + RSI<30": below200 & (r14 < 30) & valid,
}


def per_period(mask, val):
    """Per rebalance date: mean of val over stocks where mask is true."""
    m = mask.loc[reb]
    v = val.loc[reb].where(m)
    return v.mean(axis=1), m.sum(axis=1)


def tstat(s):
    s = s.dropna()
    return s.mean() / s.std(ddof=1) * np.sqrt(len(s)) if len(s) > 2 else np.nan


rows = []
for name, m in conds.items():
    r, n = per_period(m, fwd)
    x, _ = per_period(m, fwd_x)
    u, _ = per_period(m, hit_up3)
    d3, _ = per_period(m, hit_dn3)
    d5, _ = per_period(m, hit_dn5)
    w, _ = per_period(m, (fwd > 0).astype(float).where(fwd.notna()))
    rows.append({
        "条件": name, "平均只数": n.mean(), "10日平均收益%": 100 * r.mean(),
        "超额SPY%": 100 * x.mean(), "超额t值": tstat(x), "上涨概率%": 100 * w.mean(),
        "触及+3%%": 100 * u.mean(), "触及-3%%": 100 * d3.mean(), "触及-5%%": 100 * d5.mean(),
    })
A = pd.DataFrame(rows).set_index("条件")
print("== Part A: 流程里的短线条件（10 个交易日后）==")
print(A.round(2).to_string(), "\n")

# relative to baseline, by year
base_r, _ = per_period(valid, fwd)
yr = pd.DataFrame(index=reb)
for name in ["规则A 顺势", "规则B 回调", "200下方 + RSI<30"]:
    r, _ = per_period(conds[name], fwd)
    yr[name] = 100 * (r - base_r)
print("== 各条件相对全部股票的超额（每年平均，%）==")
print(yr.groupby(yr.index.year).mean().round(2).to_string(), "\n")

# ---------- Part B: cross-sectional factors ----------
ret = lambda n: close / close.shift(n) - 1
lo20, hi20 = close.rolling(20).min(), close.rolling(20).max()
factors = {
    "动量 12-1个月": close.shift(21) / close.shift(252) - 1,
    "6个月涨幅": ret(126),
    "1个月涨幅": ret(21),
    "1周涨幅": ret(5),
    "RSI(14)": r14,
    "离200日线距离": close / ma200 - 1,
    "20日区间位置": (close - lo20) / (hi20 - lo20),
    "离52周高点": close / close.rolling(252).max() - 1,
    "60日波动率": close.pct_change().rolling(60).std(),
}
rows, ics = [], {}
for name, f in factors.items():
    qs, ic = [], []
    for d in reb:
        s = pd.DataFrame({"f": f.loc[d], "r": fwd.loc[d]}).dropna()
        if len(s) < 100:
            continue
        s["q"] = pd.qcut(s["f"].rank(method="first"), 5, labels=False) + 1
        qs.append(s.groupby("q")["r"].mean().rename(d))
        ic.append(s["f"].rank().corr(s["r"].rank()))
    Q = pd.DataFrame(qs) * 100
    spread = Q[5] - Q[1]
    ic = pd.Series(ic, index=Q.index)
    ics[name] = ic
    rows.append({"因子": name, **{f"Q{i}": Q[i].mean() for i in range(1, 6)},
                 "Q5-Q1%": spread.mean(), "t值": tstat(spread),
                 "Q5胜Q1占比%": 100 * (spread > 0).mean(), "IC均值": ic.mean(), "IC t值": tstat(ic)})
B = pd.DataFrame(rows).set_index("因子")
print("== Part B: 因子五分组（Q1=因子值最低，Q5=最高；10日收益%）==")
print(B.round(3).to_string(), "\n")
print("== 因子 IC 按年 ==")
print(pd.DataFrame(ics).groupby(lambda d: d.year).mean().round(3).to_string())

A.to_csv("research/backtest/result_rules.csv")
B.to_csv("research/backtest/result_factors.csv")
