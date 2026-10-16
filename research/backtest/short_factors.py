"""Test short-horizon (default 14 trading days) price/volume factors on S&P 500 daily data.

Usage: python research/backtest/short_factors.py [holding_days]
Factors come from published anomalies (short-term reversal, MAX, idiosyncratic vol,
high-volume premium, overnight/intraday returns, frog-in-the-pan, etc.) and
WorldQuant-style 101 alphas, all computable from daily OHLCV.
"""
import sys

import numpy as np
import pandas as pd

H = int(sys.argv[1]) if len(sys.argv) > 1 else 14
D = "research/backtest/data"
px = pd.read_csv(f"{D}/prices.csv.gz", parse_dates=["Date"])
cons = pd.read_csv(f"{D}/constituents.csv")
sector = cons.set_index("yf")["GICS Sector"]
P = {k: px.pivot(index="Date", columns="Ticker", values=k).sort_index() for k in ["Open", "High", "Low", "Close", "Volume"]}
spy = P["Close"].pop("SPY")
for k in ["Open", "High", "Low", "Volume"]:
    P[k] = P[k].drop(columns="SPY")
O, Hi, L, C, V = P["Open"], P["High"], P["Low"], P["Close"], P["Volume"]
sector = sector.reindex(C.columns)

r1 = C.pct_change()
mkt = spy.pct_change()
overnight = O / C.shift(1) - 1
intraday = C / O - 1
dvol = C * V


def sector_demean(df):
    return df.T.groupby(sector).transform(lambda x: x - x.mean()).T


def roll_beta_resid_std(n):
    cov = r1.rolling(n).cov(mkt)
    beta = cov.div(mkt.rolling(n).var(), axis=0)
    resid = r1 - beta.mul(mkt, axis=0)
    return beta, resid.rolling(n).std()


beta60, _ = roll_beta_resid_std(60)
_, ivol21 = roll_beta_resid_std(21)

# volume-spike day in last 20 days (proxy for earnings/news day) and its return
vol_ratio = V / V.rolling(60).mean().shift(1)
spike = vol_ratio > 3
spike_ret = (C / C.shift(1) - 1).where(spike)
last_spike_ret = spike_ret.ffill(limit=20).fillna(0).where(C.notna())  # 0 = no spike day

FACTORS = {
    # reversal family
    "1周反转 (−5日涨幅)": -(C / C.shift(5) - 1),
    "1月反转 (−21日涨幅)": -(C / C.shift(21) - 1),
    "行业内1周反转": -sector_demean(C / C.shift(5) - 1),
    "行业内1月反转": -sector_demean(C / C.shift(21) - 1),
    # momentum family
    "行业1月动量": (C / C.shift(21) - 1).T.groupby(sector).transform("mean").T,
    "动量12-1": C.shift(21) / C.shift(252) - 1,
    "隔夜收益动量(21日)": overnight.rolling(21).sum(),
    "日内收益反转 (−21日日内)": -intraday.rolling(21).sum(),
    "稳步上涨 (21日上涨天数占比)": (r1 > 0).astype(float).rolling(21).mean().where(r1.notna()),
    # lottery / risk family
    "低MAX (−21日最大单日涨幅)": -r1.rolling(21).max(),
    "低特质波动 (−21日)": -ivol21,
    "低贝塔 (−60日)": -beta60,
    "负偏度 (−21日偏度)": -r1.rolling(21).skew(),
    # volume / liquidity family
    "放量 (5日/60日成交量)": V.rolling(5).mean() / V.rolling(60).mean(),
    "成交额下降 (−5日/60日)": -(dvol.rolling(5).mean() / dvol.rolling(60).mean()),
    "Amihud非流动性": (r1.abs() / dvol).rolling(21).mean(),
    "量价背离 (−corr(收盘,量,10))": -C.rolling(10).corr(V),
    # candle / 101-alpha style
    "收盘位置 (5日均值)": ((C - L) / (Hi - L)).replace([np.inf, -np.inf], np.nan).rolling(5).mean(),
    "Alpha101 (C−O)/(H−L)": ((C - O) / (Hi - L + 1e-3)),
    "Alpha12 sign(Δ量)×(−Δ价)": np.sign(V.diff()) * (-C.diff()),
    # event proxy (post-news drift)
    "放量日涨幅 (近20日成交>3倍那天)": last_spike_ret,
}

fwd = C.shift(-H) / C - 1
ma_ok = C.rolling(252).count() >= 252
dates = C.index[(C.index >= C.index[260]) & fwd.notna().sum(axis=1).gt(300)]
reb = dates[::H]
n = len(reb)
split = reb[int(n * 0.6)]
print(f"{C.shape[1]} stocks, holding {H} days, {n} non-overlapping periods {reb[0].date()} -> {reb[-1].date()}")
print(f"train = before {split.date()}, test = from {split.date()}\n")


def tstat(s):
    s = s.dropna()
    return s.mean() / s.std(ddof=1) * np.sqrt(len(s)) if len(s) > 2 else np.nan


def evaluate(f):
    qrets, ics, top_sets = [], [], []
    for d in reb:
        s = pd.DataFrame({"f": f.loc[d], "r": fwd.loc[d]}).dropna()
        if len(s) < 200:
            continue
        s["q"] = pd.qcut(s["f"].rank(method="first"), 5, labels=False) + 1
        g = s.groupby("q")["r"].mean()
        g["all"] = s["r"].mean()
        qrets.append(g.rename(d))
        ics.append(pd.Series({"ic": s["f"].rank().corr(s["r"].rank())}, name=d))
        top_sets.append(set(s.index[s["q"] == 5]))
    Q = pd.DataFrame(qrets) * 100
    ic = pd.DataFrame(ics)["ic"]
    ls = Q[5] - Q[1]
    long_x = Q[5] - Q["all"]
    turn = np.mean([1 - len(a & b) / max(len(a), 1) for a, b in zip(top_sets, top_sets[1:])])
    mono = pd.Series([Q[i].mean() for i in range(1, 6)]).rank().corr(pd.Series(range(1, 6), dtype=float))
    yr = ls.groupby(ls.index.year).mean()
    tr, te = ls[ls.index < split], ls[ls.index >= split]
    return {
        "Q1": Q[1].mean(), "Q5": Q[5].mean(), "Q5−Q1%": ls.mean(), "t值": tstat(ls),
        "IC": ic.mean(), "ICIR(年化)": ic.mean() / ic.std() * np.sqrt(252 / H),
        "单调性": mono, "Q5超额%": long_x.mean(), "Q5超额t": tstat(long_x),
        "正收益年数": f"{(yr > 0).sum()}/{len(yr)}",
        "训练段t": tstat(tr), "测试段t": tstat(te), "Q5换手": turn,
    }, ic, ls


rows, ic_all = {}, {}
for name, f in FACTORS.items():
    res, ic, _ = evaluate(f)
    rows[name] = res
    ic_all[name] = ic
R = pd.DataFrame(rows).T
R = R.sort_values("t值", key=lambda s: s.astype(float), ascending=False)
pd.set_option("display.width", 250)
print("== 单因子检验（Q5 = 因子值最高的 20%，已调整方向：因子值越高预期越好）==")
num = R.drop(columns=["正收益年数"]).astype(float).round(3)
num.insert(9, "正收益年数", R["正收益年数"])
print(num.to_string(), "\n")


def verdict(r):
    t, te, tr, mono = float(r["t值"]), float(r["测试段t"]), float(r["训练段t"]), float(r["单调性"])
    if t > 3 and tr > 1 and te > 1 and mono >= 0.8:
        return "严格有效"
    if t > 2 and tr > 0 and te > 0:
        return "勉强（t>2）"
    return "无效"


print("== 判定 ==")
print(R.apply(verdict, axis=1).to_string(), "\n")

# ---- combination: choose on train, test out of sample ----
train_ic = pd.DataFrame(ic_all)
train_ic = train_ic[train_ic.index < split]
icir_tr = train_ic.mean() / train_ic.std()
chosen = icir_tr[icir_tr > 0.15].sort_values(ascending=False).index.tolist()[:6]
print("训练段 ICIR（每期）:", icir_tr.round(3).sort_values(ascending=False).to_dict())
print("组合选中的因子（训练段 ICIR>0.15，最多 6 个）:", chosen)
if chosen:
    def z(df):
        rk = df.rank(axis=1, pct=True)
        return rk.sub(rk.mean(axis=1), axis=0).div(rk.std(axis=1), axis=0)
    combo = sum(z(FACTORS[k]) for k in chosen) / len(chosen)
    res, _, ls = evaluate(combo)
    te = ls[ls.index >= split]
    print("\n== 等权组合 ==")
    print(pd.Series(res).to_string())
    print(f"测试段 Q5−Q1 平均 {te.mean():.3f}%, t={tstat(te):.2f}, 期数 {te.count()}")
R.to_csv(f"research/backtest/result_short_factors_{H}d.csv")
