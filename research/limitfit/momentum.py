"""Which non-volatility features (momentum, trend, volume...) pick the stock most likely to rise most in 14 days.
Outcome: open t+1 to close t+10 (10 trading days ~ 14 calendar days), no costs.
Single features: within-date rank IC over 3 months / this year / 1 year / 2 years (daily signals, 2024-09 on).
Combination: LightGBM trained on 2021-11..2024-07 (every 5 days), scored on 2024-09 on.
Usage: python research/limitfit/momentum.py   (needs out/panel.parquet and out/panel_daily.parquet from explore.py)
"""
import lightgbm as lgb
import numpy as np
import pandas as pd

D = "research/factors/data"
FEATS = ["1周涨幅", "1月涨幅", "3月涨幅", "6月涨幅", "动量12-1", "离20日线", "离50日线", "离200日线", "离52周高点",
         "离52周低点", "RSI14", "5日/60日成交量", "成交额(规模)", "隔夜收益(21日)", "日内收益(21日)", "上涨天数占比(21日)",
         "行业1月涨幅", "相对行业1月", "布林位置", "20日区间位置", "60日偏度"]
H = 10

# same price matrix as explore.py, so panel row/column indices line up
px = pd.read_csv(f"{D}/prices.csv.gz", parse_dates=["Date"])
cons = pd.read_csv(f"{D}/constituents.csv").set_index("yf")
rem = pd.read_csv(f"{D}/prices_removed.csv.gz", parse_dates=["Date"])
px = pd.concat([px, rem[~rem["Ticker"].isin(cons.index)]])
O = px.pivot(index="Date", columns="Ticker", values="Open").sort_index()
C = px.pivot(index="Date", columns="Ticker", values="Close").sort_index()
spy_o, spy_c = O.pop("SPY"), C.pop("SPY").ffill()
O, C = O.drop(columns=["MDY"], errors="ignore"), C.drop(columns=["MDY"], errors="ignore").ffill()
Oa, Ca = O.to_numpy(), C.to_numpy()


def attach(d):
    i, tk = d["i"].to_numpy(), d["tk"].to_numpy()
    ok = i + H < len(C)
    f = np.full(len(d), np.nan)
    f[ok] = Ca[i[ok] + H, tk[ok]] / Oa[i[ok] + 1, tk[ok]] - 1
    d["fwd"] = f
    d["spy"] = np.where(ok, spy_c.to_numpy()[np.minimum(i + H, len(C) - 1)] / spy_o.to_numpy()[i + 1] - 1, np.nan)
    d["fwd_adj"] = d["fwd"] / d["60日波动率"]  # rise per unit of the stock's own volatility
    return d.dropna(subset=["fwd"])


old = attach(pd.read_parquet("research/limitfit/out/panel.parquet"))
new = attach(pd.read_parquet("research/limitfit/out/panel_daily.parquet"))
old = old[old["date"] < pd.Timestamp("2024-07-15")]  # 10-day outcomes end before the test period starts
last = new["date"].max()
W = {"3个月": last - pd.DateOffset(months=3), "今年": pd.Timestamp("2026-01-01"),
     "1年": last - pd.DateOffset(years=1), "2年": last - pd.DateOffset(years=2)}


def nw_t(x, lag=10):
    x = pd.Series(x).dropna()
    e = (x - x.mean()).to_numpy()
    T = len(e)
    v = (e @ e) / T
    for k in range(1, min(lag, T - 1) + 1):
        v += 2 * (1 - k / (lag + 1)) * (e[k:] @ e[:-k]) / T
    return x.mean() / np.sqrt(v / T)


def daily_ic(d, x, y):
    t = pd.DataFrame({"x": d[x], "y": d[y], "date": d["date"]}).dropna()
    g = t.groupby("date")
    t["x"], t["y"] = g["x"].rank(), g["y"].rank()
    g = t.groupby("date")
    t["x"] -= g["x"].transform("mean")
    t["y"] -= g["y"].transform("mean")
    s = t.assign(xy=t.x * t.y, xx=t.x ** 2, yy=t.y ** 2).groupby("date")[["xy", "xx", "yy"]].sum()
    return s.xy / np.sqrt(s.xx * s.yy)


def picks(d, score, w0):
    """Mean 14-day rise of the top 1 / top 5 / top fifth by score, and of all stocks, from w0 on."""
    m = d[d["date"] >= w0]
    rk = m.groupby("date")[score].rank(ascending=False, method="first")
    q = m.groupby("date")[score].rank(pct=True)
    res = {}
    for name, sel in [("第1名", rk == 1), ("前5", rk <= 5), ("前1/5", q > 0.8)]:
        per = m[sel].groupby("date")["fwd"].mean()
        res[name] = per.mean()
        allm = m.groupby("date")["fwd"].mean().reindex(per.index)
        res[name + "跑赢全部的天数"] = (per > allm).mean()
    res["全部"] = m.groupby("date")["fwd"].mean().mean()
    res["SPY"] = m.groupby("date")["spy"].first().mean()
    res["第1名不同股票数"] = m[rk == 1]["ticker"].nunique()
    return res


lines = [f"14 天（10 个交易日）涨幅，信号 {new['date'].min().date()} 至 {last.date()}；训练段 {old['date'].min().date()} 至 {old['date'].max().date()}\n"]
lines.append("== 单个因子：和 14 天涨幅的相关性 IC（t）；括号外第二列是扣掉波动影响后的 IC ==")
rows = []
for f in FEATS:
    ic, ica = daily_ic(new, f, "fwd"), daily_ic(new, f, "fwd_adj")
    ic_old = daily_ic(old, f, "fwd").mean()
    row = {"因子": f, "训练段IC(2021-24)": round(ic_old, 3)}
    for w, t0 in W.items():
        s, sa = ic[ic.index >= t0], ica[ica.index >= t0]
        row[w] = f"{s.mean():+.3f} ({nw_t(s):+.1f}) / {sa.mean():+.3f}"
    rows.append(row)
tab = pd.DataFrame(rows).set_index("因子")
pd.set_option("display.width", 250)
lines.append(tab.to_string())

# per-feature picks, direction fixed from the training period
lines.append("\n== 单个因子挑股：方向按 2021-24 训练段定，每天挑第 1 名 / 前 5 / 前 1/5 的 14 天平均涨幅 ==")
prow = []
for f in FEATS:
    sign = np.sign(daily_ic(old, f, "fwd").mean())
    new["_s"] = sign * new[f]
    for w, t0 in W.items():
        r = picks(new, "_s", t0)
        prow.append({"因子": f, "方向": "高" if sign > 0 else "低", "区间": w, **{k: r[k] for k in ["第1名", "前5", "前1/5", "全部", "SPY", "第1名不同股票数"]}})
pt = pd.DataFrame(prow)
pt.to_csv("research/limitfit/out/momentum_picks.csv", index=False)
show = pt.pivot(index=["因子", "方向"], columns="区间", values="前5")[list(W)].map(lambda v: f"{v:+.2%}")
lines.append("前 5 名平均：\n" + show.to_string())
lines.append("对照（全部股票平均 / SPY）：" + "，".join(
    f"{w} {new[new['date'] >= t0].groupby('date')['fwd'].mean().mean():+.2%} / {new[new['date'] >= t0].groupby('date')['spy'].first().mean():+.2%}"
    for w, t0 in W.items()))

# combination
X_old = old[FEATS].groupby(old["date"]).rank(pct=True)
X_new = new[FEATS].groupby(new["date"]).rank(pct=True)
X_old.columns = X_new.columns = [f"f{k}" for k in range(len(FEATS))]
y = old.groupby("date")["fwd"].rank(pct=True) - 0.5  # rank target: which stock rises most that day
params = dict(objective="regression", learning_rate=0.03, num_leaves=15, min_data_in_leaf=1000, feature_fraction=0.7,
              bagging_fraction=0.7, bagging_freq=1, lambda_l2=10, verbose=-1, seed=0)
m = lgb.train(params, lgb.Dataset(X_old, y), num_boost_round=300)
new["组合"] = m.predict(X_new)
ic = daily_ic(new, "组合", "fwd")
lines.append("\n== 组合模型（2021-24 学，2024-09 之后检验）==")
for w, t0 in W.items():
    s = ic[ic.index >= t0]
    r = picks(new, "组合", t0)
    lines.append(f"{w}: IC {s.mean():+.3f} (t {nw_t(s):+.1f})  第1名 {r['第1名']:+.2%}（跑赢全部的天数 {r['第1名跑赢全部的天数']:.0%}，"
                 f"{r['第1名不同股票数']} 只不同股票） 前5 {r['前5']:+.2%}  前1/5 {r['前1/5']:+.2%}  全部 {r['全部']:+.2%}  SPY {r['SPY']:+.2%}")
imp = pd.Series(m.feature_importance("gain"), index=FEATS)
lines.append("模型最看重：" + "，".join(f"{k} {v:.0%}" for k, v in (imp / imp.sum()).sort_values(ascending=False).head(6).items()))
today = new[new["date"] == last]
lines.append(f"\n（最后一个有 14 天结果的信号日 {last.date()} 的第 1 名：{today.loc[today['组合'].idxmax(), 'ticker']}）")
open("research/limitfit/out/momentum.txt", "w").write("\n".join(lines))
print("\n".join(lines))
