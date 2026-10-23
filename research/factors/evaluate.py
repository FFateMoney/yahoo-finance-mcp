"""Evaluate factors on S&P 500 + S&P 400 against the project's factor standard (因子评价标准.md).

Usage: python research/factors/evaluate.py [holding_days]
Signal at close of day t, buy at open t+1, sell at open t+1+H. Non-overlapping periods.
Every factor is registered below with its expected direction (higher = better) before testing.
"""
import sys

import numpy as np
import pandas as pd

H = int(sys.argv[1]) if len(sys.argv) > 1 else 5
D = "research/factors/data"
COST = {"SP500": 0.05, "SP400": 0.10}  # % per side
px = pd.read_csv(f"{D}/prices.csv.gz", parse_dates=["Date"])
cons = pd.read_csv(f"{D}/constituents.csv").set_index("yf")
P = {k: px.pivot(index="Date", columns="Ticker", values=k).sort_index() for k in ["Open", "High", "Low", "Close", "Volume"]}
spy = P["Close"]["SPY"]
for k in P:
    P[k] = P[k].drop(columns=["SPY", "MDY"], errors="ignore")
O, Hi, L, C, V = P["Open"], P["High"], P["Low"], P["Close"], P["Volume"]
cons = cons.reindex(C.columns)
sector, univ = cons["Sector"], cons["Universe"]
cost = univ.map(COST)

r1 = C.pct_change()
mkt = spy.pct_change()
overnight = O / C.shift(1) - 1
intraday = C / O - 1
dvol = C * V


def sector_demean(df):
    return df.T.groupby(sector).transform(lambda x: x - x.mean()).T


def beta_resid(n):
    beta = r1.rolling(n).cov(mkt).div(mkt.rolling(n).var(), axis=0)
    return beta, r1 - beta.mul(mkt, axis=0)


beta60, resid60 = beta_resid(60)
ivol21 = resid60.rolling(21).std()
vol_ratio = V / V.rolling(60).mean().shift(1)
spike = vol_ratio > 3
last_spike_ret = r1.where(spike).ffill(limit=20).fillna(0).where(C.notna())
gap = overnight.where(vol_ratio > 2)
last_gap = gap.ffill(limit=10).fillna(0).where(C.notna())
ret5 = C / C.shift(5) - 1
vol5_rel = (V.rolling(5).mean() / V.rolling(60).mean()).rank(axis=1, pct=True)

# Reference factors used for the "增量价值" check
REF = {
    "动量12-1": C.shift(21) / C.shift(252) - 1,
    "60日波动率": r1.rolling(60).std(),
    "离200日线": C / C.rolling(200).mean() - 1,
}

FACTORS = {
    # reversal
    "1周反转": -ret5,
    "1月反转": -(C / C.shift(21) - 1),
    "行业内1周反转": -sector_demean(ret5),
    "行业内1月反转": -sector_demean(C / C.shift(21) - 1),
    "残差反转(5日/波动)": -resid60.rolling(5).sum() / resid60.rolling(60).std(),  # Blitz et al. 2013
    "缩量反转": -ret5 * (1 - vol5_rel),  # Conrad, Hameed, Niden 1994
    # momentum / trend
    "动量12-1": REF["动量12-1"],
    "行业1月动量": (C / C.shift(21) - 1).T.groupby(sector).transform("mean").T,
    "52周高点接近度": C / C.rolling(252).max(),  # George & Hwang 2004
    "隔夜收益动量(21日)": overnight.rolling(21).sum(),  # Lou, Polk, Skouras 2019
    "日内收益反转(21日)": -intraday.rolling(21).sum(),
    "稳步上涨(21日上涨占比)": (r1 > 0).astype(float).rolling(21).mean().where(r1.notna()),
    # lottery / risk
    "低MAX(21日)": -r1.rolling(21).max(),  # Bali et al. 2011
    "低特质波动(21日)": -ivol21,
    "低贝塔(60日)": -beta60,
    "负偏度(21日)": -r1.rolling(21).skew(),
    # volume / liquidity
    "放量(5日/60日)": V.rolling(5).mean() / V.rolling(60).mean(),  # Gervais et al. 2001
    "Amihud非流动性": (r1.abs() / dvol).rolling(21).mean(),
    "量价背离(10日)": -C.rolling(10).corr(V),
    # candle / 101 alphas
    "收盘位置(5日)": ((C - L) / (Hi - L)).replace([np.inf, -np.inf], np.nan).rolling(5).mean(),
    "Alpha12": np.sign(V.diff()) * (-C.diff()),
    # event proxies
    "放量日涨幅(20日内)": last_spike_ret,  # PEAD proxy
    "放量跳空幅度(10日内)": last_gap,  # news momentum, Chan 2003
}
M = len(FACTORS)
T_A = 3.5 if M > 20 else 3.0

spy_fwd = (spy.shift(-(H + 1)) / spy.shift(-1) - 1) * 100
fwd = O.shift(-(H + 1)) / O.shift(-1) - 1
valid = (C > 5) & (dvol.rolling(21).mean() > 5e6) & fwd.notna()
dates = C.index[C.index >= C.index[260]]
dates = dates[valid.loc[dates].sum(axis=1) > 500]
reb = dates[::H]
split = reb[int(len(reb) * 0.6)]
print(f"{C.shape[1]} stocks, hold {H}d (open t+1 -> open t+1+{H}), {len(reb)} periods "
      f"{reb[0].date()} -> {reb[-1].date()}, train < {split.date()}, M={M}, A-grade t>{T_A}\n")


def tstat(s):
    s = pd.Series(s).dropna()
    return s.mean() / s.std(ddof=1) * np.sqrt(len(s)) if len(s) > 2 else np.nan


def ols_alpha_t(y, X):
    df = pd.concat([y.rename("y"), X], axis=1).dropna()
    A = np.column_stack([np.ones(len(df)), df.drop(columns="y").values])
    b, *_ = np.linalg.lstsq(A, df["y"].values, rcond=None)
    e = df["y"].values - A @ b
    cov = (e @ e) / (len(df) - A.shape[1]) * np.linalg.inv(A.T @ A)
    return b[0] / np.sqrt(cov[0, 0])


def sector_rank(f):
    return pd.concat([f[cols].rank(axis=1, pct=True) for _, cols in f.columns.groupby(sector).items()], axis=1)[f.columns]


def run(f, keep_series=False):
    fs = sector_rank(f)
    rows, tops = [], []
    for d in reb:
        s = pd.DataFrame({"f": f.loc[d], "fs": fs.loc[d], "r": fwd.loc[d] * 100,
                          "u": univ, "c": cost}).where(valid.loc[d], axis=0).dropna()
        if len(s) < 300:
            continue
        s["q"] = pd.qcut(s["f"].rank(method="first"), 5, labels=False) + 1
        s["qs"] = pd.qcut(s["fs"].rank(method="first"), 5, labels=False) + 1
        allm = s["r"].mean()
        top = s[s["q"] == 5]
        row = {f"Q{i}": s.loc[s["q"] == i, "r"].mean() for i in range(1, 6)}
        row.update({
            "all": allm, "x5": top["r"].mean() - allm,
            "x10": s.nlargest(10, "f")["r"].mean() - allm,
            "xs": s.loc[s["qs"] == 5, "r"].mean() - allm,
            "ic": s["f"].rank().corr(s["r"].rank()),
            "cost5": top["c"].mean(),
        })
        for u in COST:
            su = s[s["u"] == u]
            qu = pd.qcut(su["f"].rank(method="first"), 5, labels=False) + 1
            row[f"x_{u}"] = su.loc[qu == 5, "r"].mean() - su["r"].mean()
        rows.append(pd.Series(row, name=d))
        tops.append(set(top.index))
    R = pd.DataFrame(rows)
    R["turn"] = [np.nan] + [1 - len(a & b) / max(len(a), 1) for a, b in zip(tops, tops[1:])]
    return R


vol_ls = None  # high-vol long-short series, for exposure check
ref_corr = {}


def summarize(name, R, f):
    x5 = R["x5"]
    mono = pd.Series([R[f"Q{i}"].mean() for i in range(1, 6)]).rank().corr(pd.Series(range(1, 6), dtype=float))
    turn = R["turn"].mean()
    roundtrip = 2 * R["cost5"].mean()
    net = x5.mean() - turn * roundtrip
    breakeven = x5.mean() / (2 * turn) if turn > 0 else np.nan
    yr = x5.groupby(x5.index.year).mean()
    tr, te = x5[x5.index < split], x5[x5.index >= split]
    mkt_x = spy_fwd.reindex(R.index) - R["all"]
    alpha_t = ols_alpha_t(x5, pd.concat([mkt_x.rename("mkt"), vol_ls.reindex(R.index).rename("vol")], axis=1))
    corr = max(abs(np.nanmean([f.loc[d].rank().corr(g.loc[d].rank()) for d in reb[::4]])) for k, g in REF.items() if k != name)
    ic = R["ic"]
    out = {
        "RankIC": ic.mean(), "ICIR": ic.mean() / ic.std(), "IC胜率": (ic > 0).mean(), "IC_t": tstat(ic),
        "单调性": mono, "Top组超额%": x5.mean(), "t": tstat(x5), "跑赢比例": (x5 > 0).mean(),
        "Top10超额%": R["x10"].mean(), "Top10_t": tstat(R["x10"]),
        "换手率": turn, "净超额%": net, "盈亏平衡成本%": breakeven, "假设成本%": R["cost5"].mean(),
        "训练t": tstat(tr), "测试t": tstat(te), "正年数": f"{(yr > 0).sum()}/{len(yr)}",
        "行业中性t": tstat(R["xs"]), "剔除暴露alpha_t": alpha_t, "与已有因子最大相关": corr,
        "SP500超额t": tstat(R["x_SP500"]), "SP400超额t": tstat(R["x_SP400"]),
    }
    pos, n = (yr > 0).sum(), len(yr)
    if (out["t"] > T_A and out["测试t"] > 1.5 and np.sign(tstat(tr)) == np.sign(tstat(te)) and net > 0
            and breakeven >= 3 * R["cost5"].mean() and alpha_t > 2 and mono >= 0.8 and pos >= 2 * n / 3):
        g = "A 可用"
    elif out["t"] > 2 and tstat(tr) > 0 and tstat(te) > 0 and net > 0:
        g = "B 观察"
    else:
        g = "C 无效"
    out["等级"] = g
    return out


vol_R = run(REF["60日波动率"])
vol_ls = vol_R["Q5"] - vol_R["Q1"]
res = {}
for name, f in FACTORS.items():
    res[name] = summarize(name, run(f), f)
    print(f"{name}: t={res[name]['t']:.2f} {res[name]['等级']}", flush=True)
T = pd.DataFrame(res).T.sort_values("t", key=lambda s: s.astype(float), ascending=False)
pd.set_option("display.width", 400)
pd.set_option("display.max_columns", 40)
num = T.drop(columns=["正年数", "等级"]).astype(float).round(3)
num["正年数"], num["等级"] = T["正年数"], T["等级"]
print("\n", num.to_string())
T.to_csv(f"research/factors/result_{H}d.csv")
