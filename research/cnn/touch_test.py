"""F130-F135: do simple features raise the "+x% limit sell within 10 days" outcome above same-volatility peers?

Usage: python research/cnn/touch_test.py   (needs research/cnn/work/prices.parquet; CNN uses research/cnn/out/cnn_signal.parquet)
Pre-registration: /mnt/project-files/回测/触及概率_预登记.md. Do not change the design after seeing results.

Plan per stock: signal at close t, buy open t+1, limit sell at entry*(1+x) during days t+1..t+10
(filled at max(limit, open) on the hit day), else sell close t+10. Costs 0.05% / 0.10% per side.
Baseline: within each date, stocks are split into 5 buckets by 20-day volatility; every number is
measured against the average of the stock's own bucket on that date.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "research/cnn")
from universe import masks  # noqa: E402

W = "research/cnn/work"
N = 10
TARGETS = [0.02, 0.03, 0.05]
MAIN = 0.03
SPLIT = pd.Timestamp("2020-01-01")
M_TOTAL = 135
T_A = 3.5

px = pd.read_parquet(f"{W}/prices.parquet")
P = {k: px.pivot(index="Date", columns="Ticker", values=k).sort_index() for k in ["Open", "High", "Close", "Volume"]}
for k in P:
    P[k].pop("SPY")
O, Hi, C, V = P["Open"], P["High"], P["Close"], P["Volume"]
dvol = C * V
m500, m400, _ = masks(C, dvol)
valid = (m500 | m400) & (C > 5) & (dvol.rolling(21).mean() > 5e6)
cost = pd.DataFrame(np.where(m500, 0.05, 0.10), index=C.index, columns=C.columns) / 100
r1 = C.pct_change()
vol20 = r1.rolling(20).std()

d = r1.clip(-1, 1)
up, dn = d.clip(lower=0).rolling(14).mean(), (-d.clip(upper=0)).rolling(14).mean()
FEATURES = {
    "1周反转": -(C / C.shift(5) - 1),
    "距20日高点回撤": -(C / C.rolling(20).max() - 1),
    "量比5/60": V.rolling(5).mean() / V.rolling(60).mean(),
    "动量12-1": C.shift(21) / C.shift(252) - 1,
    "RSI14低": -(100 - 100 / (1 + up / dn)),
}
if os.path.exists("research/cnn/out/cnn_signal.parquet"):
    FEATURES["CNN图像(I20/R5)"] = (pd.read_parquet("research/cnn/out/cnn_signal.parquet")
                                   .pivot(index="Date", columns="Ticker", values="p").reindex(columns=C.columns))

cal = C.index
start = cal.searchsorted(pd.Timestamp("2011-01-01"))
reb = list(range(start, len(cal) - N - 1, N))  # non-overlapping 10-day periods
Oa, Ha, Ca = O.to_numpy(), Hi.to_numpy(), C.to_numpy()


def plan(i, x):
    """Gross plan return and hit flag for every ticker, signal at row i."""
    entry = Oa[i + 1]
    lim = entry * (1 + x)
    hi, op = Ha[i + 1 : i + N + 1], Oa[i + 1 : i + N + 1]
    hit_any = np.zeros(len(entry), bool)
    fill = np.full(len(entry), np.nan)
    for k in range(N):
        new = ~hit_any & (hi[k] >= lim)
        fill[new] = lim[new] if k == 0 else np.maximum(lim[new], op[k][new])
        hit_any |= new
    last = pd.DataFrame(Ca[i + 1 : i + N + 1]).ffill().to_numpy()[-1]  # last close (stock may stop trading)
    exit_ = np.where(hit_any, fill, last)
    return exit_ / entry - 1, hit_any.astype(float), last / entry - 1


rows = []
for i in reb:
    dt = cal[i]
    ok = valid.iloc[i].to_numpy() & np.isfinite(Oa[i + 1]) & np.isfinite(vol20.iloc[i].to_numpy())
    if ok.sum() < 200:
        continue
    base = pd.DataFrame({"vol": vol20.iloc[i].to_numpy(), "c": cost.iloc[i].to_numpy()}, index=C.columns)
    for x in TARGETS:
        r, h, hold = plan(i, x)
        base[f"r{x}"], base[f"h{x}"] = r * 100, h
    base["hold"] = hold * 100
    for name, f in FEATURES.items():
        base[name] = f.loc[dt] if dt in f.index else np.nan
    b = base[ok & np.isfinite(base[f"r{MAIN}"])].copy()
    b["vb"] = pd.qcut(b["vol"].rank(method="first"), 5, labels=False)
    b["date"] = dt
    rows.append(b)
S = pd.concat(rows)
S["net"] = S[f"r{MAIN}"] - 2 * S["c"] * 100
print(f"periods {S['date'].nunique()} ({S['date'].min().date()} -> {S['date'].max().date()}), "
      f"stock-periods {len(S)}, M={M_TOTAL}, A-grade t>{T_A}\n")

print("Baseline by volatility bucket (all stocks):")
g = S.groupby("vb")
print(pd.DataFrame({"年化波动%": g["vol"].mean() * np.sqrt(252) * 100,
                    **{f"触及+{int(x*100)}%": g[f"h{x}"].mean() * 100 for x in TARGETS},
                    "限价+3%收益%": g[f"r{MAIN}"].mean(), "扣成本%": g["net"].mean(), "持有收益%": g["hold"].mean()}).round(3).to_string(), "\n")


def tstat(s):
    s = pd.Series(s).dropna()
    return s.mean() / s.std(ddof=1) * np.sqrt(len(s)) if len(s) > 2 else np.nan


res, years = {}, {}
for name in FEATURES:
    T = S.dropna(subset=[name]).copy()
    if T.empty:
        continue
    key = ["date", "vb"]
    for c in [f"r{MAIN}", f"h{MAIN}", "hold"] + [f"h{x}" for x in TARGETS]:
        T["x_" + c] = T[c] - T.groupby(key)[c].transform("mean")
    T["q"] = T.groupby(key)[name].transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False) + 1)
    top = T[T["q"] == 5]
    per = top.groupby("date").agg(ex=(f"x_r{MAIN}", "mean"), hx=(f"x_h{MAIN}", "mean"), holdx=("x_hold", "mean"))
    miss = top[top[f"h{MAIN}"] == 0]
    peer_miss = T[T[f"h{MAIN}"] == 0]
    a, b = per["ex"][per.index < SPLIT], per["ex"][per.index >= SPLIT]
    qs = T.groupby("q")[f"x_r{MAIN}"].mean()
    mono = qs.rank().corr(pd.Series(range(1, 6), index=qs.index, dtype=float))
    yr = per["ex"].groupby(per.index.year).mean()
    out = {
        "期数": len(per), "Top组触及+3%率%": top[f"h{MAIN}"].mean() * 100, "同档基准触及率%": (top[f"h{MAIN}"] - top[f"x_h{MAIN}"]).mean() * 100,
        "超额触及率(百分点)": per["hx"].mean() * 100, "超额触及率_t": tstat(per["hx"]),
        "限价+3%超额收益%": per["ex"].mean(), "t": tstat(per["ex"]),
        "2011-19超额%": a.mean(), "2011-19_t": tstat(a), "2020-26超额%": b.mean(), "2020-26_t": tstat(b),
        "未触及时平均%": miss[f"r{MAIN}"].mean(), "同档未触及时平均%": peer_miss[f"r{MAIN}"].mean(),
        "持有超额%": per["holdx"].mean(), "单调性": mono, "正年数": f"{(yr > 0).sum()}/{len(yr)}",
        **{f"超额触及+{int(x*100)}%": (top[f"x_h{x}"].mean() * 100) for x in TARGETS},
    }
    if out["t"] > T_A and tstat(b) > 1.5 and np.sign(tstat(a)) == np.sign(tstat(b)) and mono >= 0.8 \
            and out["未触及时平均%"] >= out["同档未触及时平均%"] - 0.3:
        g_ = "A 可用"
    elif out["t"] > 2 and tstat(a) > 0 and tstat(b) > 0:
        g_ = "B 观察"
    else:
        g_ = "C 无效"
    out["等级"] = g_
    res[name], years[name] = out, yr

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
R = pd.DataFrame(res)
print(R.to_string(), "\n")
print("Top-group excess plan return by year (%):")
print(pd.DataFrame(years).round(3).to_string())
os.makedirs("research/cnn/out", exist_ok=True)
R.T.to_csv("research/cnn/out/touch_result.csv")
