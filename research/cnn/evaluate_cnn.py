"""Evaluate the CNN chart signal (F127) and two reference factors against the project's factor standard.

Usage: python research/cnn/evaluate_cnn.py   (needs research/cnn/work/prices.parquet and pred_*.parquet)
Universe: point-in-time S&P 500 + S&P 400, price > $5, 21-day dollar volume > $5M.
Trade: signal at close t, buy open t+1, sell open t+6 (last close if the stock stops trading first).
Sub-periods per pre-registration: 2011-2019 ("first") and 2020-2026 ("post-paper").
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "research/cnn")
from universe import masks  # noqa: E402

W = "research/cnn/work"
H = 5
COST = {"SP500": 0.05, "SP400": 0.10}  # % per side
SPLIT = pd.Timestamp(os.environ.get("SPLIT", "2020-01-01"))  # override only for local smoke tests
M_TOTAL = 129
T_A = 3.5

px = pd.read_parquet(f"{W}/prices.parquet")
P = {k: px.pivot(index="Date", columns="Ticker", values=k).sort_index() for k in ["Open", "Close", "Volume"]}
spy = P["Close"].pop("SPY")
P["Open"].pop("SPY")
P["Volume"].pop("SPY")
O, C, V = P["Open"], P["Close"], P["Volume"]
dvol = C * V
m500, m400, s500 = masks(C, dvol)

pred = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{W}/pred_*.parquet"))])
n_models = pred.groupby(["Date", "Ticker"]).size().max()
cnn = pred.groupby(["Date", "Ticker"])["p"].mean().unstack().reindex(columns=C.columns)
reb = cnn.index.intersection(C.index)

entry = O.shift(-1)
exit_ = O.shift(-(H + 1)).fillna(C.ffill().shift(-(H + 1)))  # delisted within the window: last close
fwd = (exit_ / entry - 1).where(entry.notna())
spy_fwd = (spy.shift(-(H + 1)) / spy.shift(-1) - 1) * 100
member = m500 | m400
valid = member & (C > 5) & (dvol.rolling(21).mean() > 5e6) & fwd.notna()
cost = pd.DataFrame(np.where(m500, COST["SP500"], COST["SP400"]), index=C.index, columns=C.columns)
r1 = C.pct_change()

FACTORS = {
    "CNN图像(I20/R5)": cnn,
    "1周反转": -(C / C.shift(5) - 1),
    "动量12-1": C.shift(21) / C.shift(252) - 1,
}
VOL = r1.rolling(60).std()


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


def run(f):
    rows, tops = [], []
    for d in reb:
        s = pd.DataFrame({"f": f.loc[d], "r": fwd.loc[d] * 100, "c": cost.loc[d], "u": m500.loc[d]})
        s = s[valid.loc[d] & s["f"].notna()]
        if len(s) < 200:
            continue
        s["q"] = pd.qcut(s["f"].rank(method="first"), 5, labels=False) + 1
        allm = s["r"].mean()
        top = s[s["q"] == 5]
        row = {f"Q{i}": s.loc[s["q"] == i, "r"].mean() for i in range(1, 6)}
        row.update({"all": allm, "x5": top["r"].mean() - allm, "x10": s.nlargest(10, "f")["r"].mean() - allm,
                    "ic": s["f"].rank().corr(s["r"].rank()), "cost5": top["c"].mean(), "n": len(s)})
        for name, sub in (("SP500", s[s["u"]]), ("SP400", s[~s["u"]])):
            if len(sub) >= 50:
                qu = pd.qcut(sub["f"].rank(method="first"), 5, labels=False) + 1
                row[f"x_{name}"] = sub.loc[qu == 5, "r"].mean() - sub["r"].mean()
        rows.append(pd.Series(row, name=d))
        tops.append(set(top.index))
    R = pd.DataFrame(rows)
    R["turn"] = [np.nan] + [1 - len(a & b) / max(len(a), 1) for a, b in zip(tops, tops[1:])]
    return R


vol_R = run(VOL)
vol_ls = vol_R["Q5"] - vol_R["Q1"]


def summarize(R, name):
    x5 = R["x5"]
    mono = pd.Series([R[f"Q{i}"].mean() for i in range(1, 6)]).rank().corr(pd.Series(range(1, 6), dtype=float))
    turn = R["turn"].mean()
    net = x5.mean() - turn * 2 * R["cost5"].mean()
    breakeven = x5.mean() / (2 * turn) if turn > 0 else np.nan
    yr = x5.groupby(x5.index.year).mean()
    a, b = x5[x5.index < SPLIT], x5[x5.index >= SPLIT]
    mkt_x = spy_fwd.reindex(R.index) - R["all"]
    alpha_t = ols_alpha_t(x5, pd.concat([mkt_x.rename("mkt"), vol_ls.reindex(R.index).rename("vol")], axis=1))
    out = {
        "期数": len(R), "平均股票数": R["n"].mean(), "RankIC": R["ic"].mean(), "ICIR": R["ic"].mean() / R["ic"].std(),
        "IC胜率": (R["ic"] > 0).mean(), "IC_t": tstat(R["ic"]), "单调性": mono,
        "Q1%": R["Q1"].mean(), "Q3%": R["Q3"].mean(), "Q5%": R["Q5"].mean(),
        "Top组超额%": x5.mean(), "t": tstat(x5), "跑赢比例": (x5 > 0).mean(),
        "Top10超额%": R["x10"].mean(), "Top10_t": tstat(R["x10"]), "换手率": turn, "净超额%": net,
        "盈亏平衡成本%": breakeven, "假设成本%": R["cost5"].mean(),
        "2011-19超额%": a.mean(), "2011-19_t": tstat(a), "2020-26超额%": b.mean(), "2020-26_t": tstat(b),
        "正年数": f"{(yr > 0).sum()}/{len(yr)}", "剔除暴露alpha_t": alpha_t,
        "SP500超额t": tstat(R.get("x_SP500")), "SP400超额t": tstat(R.get("x_SP400")),
    }
    pos, n = (yr > 0).sum(), len(yr)
    if (out["t"] > T_A and out["2020-26_t"] > 1.5 and np.sign(tstat(a)) == np.sign(tstat(b)) and net > 0
            and breakeven >= 3 * R["cost5"].mean() and alpha_t > 2 and mono >= 0.8 and pos >= 2 * n / 3):
        g = "A 可用"
    elif out["t"] > 2 and tstat(a) > 0 and tstat(b) > 0 and net > 0:
        g = "B 观察"
    else:
        g = "C 无效"
    out["等级"] = g
    return out, yr


print(f"CNN models averaged: {n_models}; rebalance dates {reb[0].date()} -> {reb[-1].date()} ({len(reb)}); "
      f"M={M_TOTAL}, A-grade t>{T_A}\n")

# coverage: share of point-in-time S&P 500 members with usable prices on rebalance dates
cov = pd.Series({d: len(s500[d] & set(C.columns[C.loc[d].notna()])) / max(len(s500[d]), 1) for d in reb})
print("S&P 500 point-in-time coverage by year (share of members with Yahoo prices):")
print(cov.groupby(cov.index.year).mean().round(3).to_string(), "\n")

res, years, Rs = {}, {}, {}
for name, f in FACTORS.items():
    Rs[name] = run(f)
    res[name], years[name] = summarize(Rs[name], name)
T = pd.DataFrame(res).T
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
print(T.T.to_string(), "\n")
print("Top-group excess by year (%):")
print(pd.DataFrame(years).round(3).to_string(), "\n")
print("Mean quintile returns, CNN (%):", Rs["CNN图像(I20/R5)"][[f"Q{i}" for i in range(1, 6)]].mean().round(3).to_dict())
corr = {k: np.nanmean([cnn.loc[d].rank().corr(FACTORS[k].loc[d].where(valid.loc[d]).rank()) for d in reb[::4]])
        for k in ["1周反转", "动量12-1"]}
print("Cross-sectional rank correlation of CNN with:", {k: round(v, 3) for k, v in corr.items()})
os.makedirs("research/cnn/out", exist_ok=True)
T.to_csv("research/cnn/out/result.csv")
Rs["CNN图像(I20/R5)"].to_csv("research/cnn/out/periods_cnn.csv")
pred.groupby(["Date", "Ticker"])["p"].mean().reset_index().to_parquet("research/cnn/out/cnn_signal.parquet", index=False)
