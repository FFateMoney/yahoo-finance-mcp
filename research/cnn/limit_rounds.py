"""F136-F137: per-stock limit-sell target n, picked from the stock's own past, with "keep holding" rounds.

Usage: python research/cnn/limit_rounds.py   (needs research/cnn/work/prices.parquet)
Pre-registration: /mnt/project-files/回测/限价补救_预登记.md. Do not change the design after seeing results.

Plan: signal at close t, buy open t+1, limit sell at entry*(1+n). A round is 10 trading days (~14 calendar
days); if not filled, keep the order for another round, up to R rounds, then sell at the last close.
Main R=2 (Jasper: "再来14天"); R=6 is reported as a secondary trial.
Per stock and date, n is the value in NS whose plan had the highest mean return per trading day held over
the stock's own last 2 years of completed plans (no look-ahead). Score = that past value; the top quintile
is compared with all eligible stocks and with SPY over the same holding days.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "research/cnn")
from universe import masks  # noqa: E402

W = "research/cnn/work"
ROUND = 10
RS = [2, 6]
NS = np.array([0.02, 0.03, 0.04, 0.05, 0.07, 0.10])
LOOKBACK = 504  # trading days of the stock's own history used to pick n
MIN_PAST = 20
SPLIT = pd.Timestamp("2020-01-01")
M_TOTAL = 137
T_A = 3.5

px = pd.read_parquet(f"{W}/prices.parquet")
P = {k: px.pivot(index="Date", columns="Ticker", values=k).sort_index() for k in ["Open", "High", "Close", "Volume"]}
spy_c, spy_o = P["Close"].pop("SPY"), P["Open"].pop("SPY")
P["High"].pop("SPY")
P["Volume"].pop("SPY")
O, Hi, C, V = P["Open"], P["High"], P["Close"], P["Volume"]
dvol = C * V
m500, m400, _ = masks(C, dvol)
valid = ((m500 | m400) & (C > 5) & (dvol.rolling(21).mean() > 5e6)).to_numpy()
cost = np.where(m500, 0.05, 0.10) / 100
cal = C.index
Oa, Ha, Ca = O.to_numpy(), Hi.to_numpy(), C.ffill().to_numpy()
SO, SC = spy_o.to_numpy(), spy_c.ffill().to_numpy()

first = cal.searchsorted(pd.Timestamp("2009-01-01"))  # plans from 2009 feed the 2-year lookback for 2011
starts = list(range(first, len(cal) - 1, ROUND))


def simulate(i, R):
    """For signal row i: return/day-held arrays [len(NS), tickers], raw return, days, exit row, SPY return."""
    L = min(R * ROUND, len(cal) - 1 - i)
    entry = Oa[i + 1]
    hi, op = Ha[i + 1 : i + 1 + L], Oa[i + 1 : i + 1 + L]
    ret = np.full((len(NS), len(entry)), np.nan)
    days = np.full((len(NS), len(entry)), float(L))
    with np.errstate(invalid="ignore", divide="ignore"):
        for a, n in enumerate(NS):
            lim = entry * (1 + n)
            hit = np.zeros(len(entry), bool)
            fill = np.full(len(entry), np.nan)
            for k in range(L):
                new = ~hit & (hi[k] >= lim)
                fill[new] = lim[new] if k == 0 else np.maximum(lim[new], op[k][new])
                days[a, new] = k + 1
                hit |= new
            ret[a] = np.where(hit, fill, Ca[i + L]) / entry - 1
    ret[:, ~(entry > 0)] = np.nan
    exit_row = i + days.astype(int)  # last day held
    spy = SC[np.minimum(exit_row, len(cal) - 1)] / SO[i + 1] - 1
    return ret, days, exit_row, spy, L == R * ROUND


out = {}
for R in RS:
    sims = {i: simulate(i, R) for i in starts}
    rows = []
    for i in starts:
        if cal[i] < pd.Timestamp("2011-01-01"):
            continue
        ret, days, exit_row, spy, complete = sims[i]
        if not complete:
            continue
        # pick n per stock from completed past plans (exit before i) in the lookback window
        past = [j for j in starts if i - LOOKBACK <= j < i]
        num = np.zeros_like(ret)
        cnt = np.zeros_like(ret)
        for j in past:
            rj, dj, ej, _, _ = sims[j]
            done = (ej < i) & np.isfinite(rj)
            num += np.where(done, np.nan_to_num(rj) / dj, 0)
            cnt += done
        with np.errstate(invalid="ignore", divide="ignore"):
            perday = num / cnt
        perday[cnt < MIN_PAST] = np.nan
        ok = valid[i] & np.isfinite(perday).all(0) & np.isfinite(ret).all(0)
        if ok.sum() < 200:
            continue
        best = np.nanargmax(np.where(np.isfinite(perday), perday, -np.inf), 0)
        idx = np.arange(ret.shape[1])
        b = pd.DataFrame({
            "score": perday[best, idx], "n": NS[best], "r": ret[best, idx] * 100 - 2 * cost[i] * 100,
            "d": days[best, idx], "spy": spy[best, idx] * 100,
            "hit": (days[best, idx] < R * ROUND) | (ret[best, idx] >= NS[best] - 1e-9),
            "r3": ret[1] * 100 - 2 * cost[i] * 100, "d3": days[1], "spy3": spy[1] * 100,
        })[ok]
        b["x"] = b["r"] - b["spy"]
        b["xd"] = b["x"] / b["d"]
        b["x3"] = b["r3"] - b["spy3"]
        b["q"] = pd.qcut(b["score"].rank(method="first"), 5, labels=False) + 1
        b["date"] = cal[i]
        rows.append(b)
    S = pd.concat(rows)
    out[R] = S

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)


def nw_t(s, lag):
    s = pd.Series(s).dropna().to_numpy()
    e = s - s.mean()
    v = e @ e / len(s)
    for k in range(1, lag + 1):
        v += 2 * (1 - k / (lag + 1)) * (e[k:] @ e[:-k]) / len(s)
    return s.mean() / np.sqrt(v / len(s))


res = {}
for R, S in out.items():
    lag = R  # holdings overlap up to R rounds
    per = S.groupby("date")
    allx = per["x"].mean()
    top = S[S["q"] == 5]
    tp = top.groupby("date")
    ex = tp["x"].mean() - allx  # top quintile vs all eligible stocks, same date
    a, b = ex[ex.index < SPLIT], ex[ex.index >= SPLIT]
    qs = S.groupby("q")["x"].mean()
    mono = qs.rank().corr(pd.Series(range(1, 6), index=qs.index, dtype=float))
    yr = ex.groupby(ex.index.year).mean()
    o = {
        "期数": len(ex), "全部股票_成交率%": S["hit"].mean() * 100, "全部股票_平均持有天数": S["d"].mean(),
        "全部股票_扣成本收益%": S["r"].mean(), "全部股票_同期SPY%": S["spy"].mean(), "全部股票_超SPY%": S["x"].mean(),
        "全部_超SPY_t": nw_t(allx, lag), "固定n=3%_超SPY%": S["x3"].mean(),
        "Top组_成交率%": top["hit"].mean() * 100, "Top组_平均n%": top["n"].mean() * 100, "Top组_平均持有天数": top["d"].mean(),
        "Top组_扣成本收益%": top["r"].mean(), "Top组_超SPY%": top["x"].mean(), "Top组_每天超SPY%": top["xd"].mean(),
        "Top组没成交时平均%": top.loc[~top["hit"], "r"].mean(), "Top组最差5%分位%": top["r"].quantile(0.05),
        "Top组相对全部超额%": ex.mean(), "t(NW)": nw_t(ex, lag),
        "2011-19超额%": a.mean(), "2011-19_t": nw_t(a, lag), "2020-26超额%": b.mean(), "2020-26_t": nw_t(b, lag),
        "单调性": mono, "正年数": f"{(yr > 0).sum()}/{len(yr)}",
        **{f"Q{q}超SPY%": qs[q] for q in qs.index},
    }
    if o["t(NW)"] > T_A and o["2020-26_t"] > 1.5 and np.sign(o["2011-19_t"]) == np.sign(o["2020-26_t"]) and mono >= 0.8:
        g = "A 可用"
    elif o["t(NW)"] > 2 and o["2011-19_t"] > 0 and o["2020-26_t"] > 0:
        g = "B 观察"
    else:
        g = "C 无效"
    o["等级"] = g
    res[f"最多{R}轮({R * ROUND}天)"] = o
    print(f"R={R}: n chosen (share of stock-periods):", S["n"].value_counts(normalize=True).sort_index().round(3).to_dict())
    print(f"R={R}: top-group excess by year (%):", yr.round(3).to_dict(), "\n")

print(f"M={M_TOTAL}, A-grade t>{T_A}; returns are per plan, net of costs; 超SPY = minus SPY over the same holding days\n")
R_ = pd.DataFrame(res)
print(R_.to_string())
os.makedirs("research/cnn/out", exist_ok=True)
R_.T.to_csv("research/cnn/out/limit_rounds_result.csv")
