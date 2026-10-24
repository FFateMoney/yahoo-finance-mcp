"""The full current method (process doc, 2026-10-24): R1 with ±1 month earnings rule, price <= $180, ties by lowest RSI
(and the doc's touch-rate ranking), +3% limit, -3% exit from day 11, day-20 close. Compared with all stocks under the same
exits and SPY held over the same days. Last 1 year / 3 months / 1 month; also a $180 one-at-a-time run per window.
Usage: python research/fullflow/full_set.py
"""
import numpy as np
import pandas as pd

exec(open("research/fullflow/exits.py", encoding="utf-8").read().split("def report")[0])  # outcome(), picks, d, WIN, cal
spyO, spyC = O["SPY"], Cl["SPY"]


def with_spy(rows, stop=-0.03):
    ret, days = outcome(rows, stop)
    i = cal.searchsorted(rows["date"].to_numpy())
    spy = spyC.to_numpy()[i + days] / spyO.to_numpy()[i + 1] - 1
    return rows.assign(ret=ret, days=days, spy=spy)


def seq(x, t0):
    """$180, one position at a time, whole shares, compounding; next pick from the exit day's close."""
    x = x.set_index("date").sort_index()
    cap, k, n, w = 180.0, cal.searchsorted(t0), 0, 0
    while k < len(cal) and cal[k] <= last:
        t = cal[k]
        if t in x.index:
            r = x.loc[t]
            sh = np.floor(cap / O.at[cal[k + 1], r["ticker"]])
            if sh >= 1:
                cap += sh * O.at[cal[k + 1], r["ticker"]] * r["ret"]
                n, w = n + 1, w + (r["ret"] > 0)
                k += int(r["days"])
                continue
        k += 1
    return cap / 180 - 1, n, w


lines = [f"信号截至 {last.date()}（结果要等约 28 天）。每笔平均，扣成本；SPY = 同样的买入日和卖出日拿 SPY"]
rows = {}
for name, rank in [("我的方法（同分按 RSI 最低）", "RSI_neg"), ("我的方法（文档原排序）", "touch")]:
    tp = with_spy(picks(SKIP["前后 1 个月内有财报的不买"], 180, rank).pipe(lambda q: q[q["date"] >= WIN["最近 1 年"]]))
    for w, t0 in WIN.items():
        x = tp[tp["date"] >= t0]
        s = seq(x, t0)
        rows.setdefault(name, {})[w] = f"{x['ret'].mean():+.2%}（SPY {x['spy'].mean():+.2%}）| $180 连续做 {s[0]:+.1%}，{s[1]} 笔赚 {s[2]}"
al = with_spy(d[(d["date"] >= WIN["最近 1 年"]) & d["ret_3%"].notna()][["date", "ticker"]])
for w, t0 in WIN.items():
    x = al[al["date"] >= t0]
    rows.setdefault("全部股票平均（同样卖法）", {})[w] = f"{x.groupby('date')['ret'].mean().mean():+.2%}（SPY {x.groupby('date')['spy'].mean().mean():+.2%}）"
    rows.setdefault("一直拿 SPY", {})[w] = f"{spyC[last + pd.Timedelta(days=28):].iloc[0] / spyO[spyO.index >= t0].iloc[0] - 1:+.1%}（{t0.date()} 到 {(last + pd.Timedelta(days=28)).date()} 左右）"
for k, v in rows.items():
    lines.append(f"\n{k}")
    lines += [f"  {w}: {s}" for w, s in v.items()]
open("research/fullflow/full_set.txt", "w").write("\n".join(lines))
print("\n".join(lines))
