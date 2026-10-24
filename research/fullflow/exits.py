"""R1 with a break-even exit in the second 14 days: +3% limit for all 20 trading days; from day 11 on, also sell as soon
as the price gets back to the buy price (0%) or to -3%. Unfilled at day 20 -> sell at the close.
Rules as of 2026-10-23 (no earnings within ±1 month). Reports last 1 year / 3 months / 1 month only.
Usage: python research/fullflow/exits.py
"""
import numpy as np
import pandas as pd

exec(open("research/fullflow/post_earn.py", encoding="utf-8").read().split("allavg =")[0])  # d, picks, SKIP, WIN, last

D = "research/factors/data"
px = pd.read_csv(f"{D}/prices.csv.gz", parse_dates=["Date"], usecols=["Date", "Ticker", "Open", "High", "Close"])
rem = pd.read_csv(f"{D}/prices_removed.csv.gz", parse_dates=["Date"], usecols=["Date", "Ticker", "Open", "High", "Close"])
px = pd.concat([px, rem[~rem["Ticker"].isin(cons.index)]])
O, H, Cl = (px.pivot(index="Date", columns="Ticker", values=k).sort_index() for k in ["Open", "High", "Close"])
Cl = Cl.ffill()
cal = O.index
L, LIM = 20, 0.03


def outcome(rows, stop):
    """Per-trade return for rows (date, ticker, cost) with the second-half exit at `stop` (None = original rule)."""
    i = cal.searchsorted(rows["date"].to_numpy())
    j = O.columns.get_indexer(rows["ticker"])
    k = np.arange(1, L + 1)
    op, hi = O.to_numpy()[i[:, None] + k, j[:, None]], H.to_numpy()[i[:, None] + k, j[:, None]]
    entry = op[:, 0]
    out = Cl.to_numpy()[i + L, j] / entry - 1
    days = np.full(len(i), L)
    done = np.zeros(len(i), bool)
    for t in range(L):
        lim = entry * (1 + LIM)
        hit = ~done & (hi[:, t] >= lim)
        out[hit] = (np.maximum(lim, op[:, t]) if t else lim)[hit] / entry[hit] - 1
        days[hit], done = t + 1, done | hit
        if stop is not None and t >= 10:
            lv = entry * (1 + stop)
            be = ~done & (hi[:, t] >= lv)
            out[be] = np.maximum(lv, op[:, t])[be] / entry[be] - 1  # already above the level at the open: sell at the open
            days[be], done = t + 1, done | be
    cost = np.where(rows["ticker"].map(cons["Universe"]).eq("SP500"), 0.0005, 0.001)
    return out - 2 * cost, days


def report(rows):
    r = {}
    for name, stop in [("原规则（第 28 天市价卖）", None), ("后 14 天回到 0% 就卖", 0.0), ("后 14 天回到 −3% 就卖", -0.03)]:
        ret, days = outcome(rows, stop)
        x = rows.assign(ret=ret, days=days)
        r[name] = {w: f"{x.loc[x['date'] >= t0, 'ret'].mean():+.2%} / 亏>5% {(x.loc[x['date'] >= t0, 'ret'] < -0.05).mean():.0%}"
                   f" / {x.loc[x['date'] >= t0, 'days'].mean():.1f}天" for w, t0 in WIN.items()}
    return pd.DataFrame(r).T.to_string()


lines = [f"信号截至 {last.date()}。格式：每笔平均收益 / 亏超 5% 的比例 / 平均持有交易日数（扣成本）"]
for title, cap, rank in [("股价 ≤ $180，同分按 RSI 最低", 180, "RSI_neg"), ("股价 ≤ $180，原排序", 180, "touch"), ("不限股价，原排序", None, "touch")]:
    tp = picks(SKIP["前后 1 个月内有财报的不买"], cap, rank)
    lines += [f"\n== {title} ==", report(tp[tp["date"] >= WIN["最近 1 年"]])]
allr = d[(d["date"] >= WIN["最近 1 年"]) & d["ret_3%"].notna()][["date", "ticker"]]
lines += ["\n== 全部股票（对照）==", report(allr)]
open("research/fullflow/exits.txt", "w").write("\n".join(lines))
print("\n".join(lines))
