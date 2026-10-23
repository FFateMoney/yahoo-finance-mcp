"""Exploratory feature combinations: LightGBM trained on one half of the dates, scored on the other half."""
import numpy as np
import pandas as pd
import lightgbm as lgb

d = pd.read_parquet("research/limitfit/out/panel.parquet")
NS = ["2%", "3%", "5%", "7%", "10%", "1σ"]
FEATS = [c for c in d.columns if not c.startswith(("ret_", "ex_", "hit_", "days_")) and c not in ("i", "tk", "date", "ticker")]
dates = np.sort(d["date"].unique())
MID = dates[len(dates) // 2]
GAP = pd.Timedelta(days=35)  # holding period is up to ~28 calendar days
X = d[FEATS].groupby(d["date"]).rank(pct=True)  # within-date ranks
X.columns = [f"f{k}" for k in range(len(FEATS))]  # LightGBM rejects some characters in names
for n in NS:
    d[f"dm_{n}"] = d[f"ex_{n}"] - d.groupby("date")[f"ex_{n}"].transform("mean")


def nw_t(x, lag=4):
    x = pd.Series(x).dropna()
    e = x - x.mean()
    T = len(e)
    v = (e @ e) / T
    for k in range(1, lag + 1):
        v += 2 * (1 - k / (lag + 1)) * (e[k:].to_numpy() @ e[:-k].to_numpy()) / T
    return x.mean() / np.sqrt(v / T)


halves = {"前半训练→后半": (d["date"] < MID - GAP, d["date"] >= MID),
          "后半训练→前半": (d["date"] >= MID + GAP, d["date"] < MID)}
params = dict(objective="regression", learning_rate=0.03, num_leaves=15, min_data_in_leaf=2000,
              feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1, lambda_l2=10, verbose=-1, seed=0)
imp_all = {}
for name, (tr, te) in halves.items():
    preds = {}
    for n in NS:
        y = d.loc[tr, f"dm_{n}"].clip(-0.3, 0.3)
        m = lgb.train(params, lgb.Dataset(X[tr], y), num_boost_round=300)
        preds[n] = pd.Series(m.predict(X[te]), index=d.index[te])
        imp_all.setdefault(n, []).append(pd.Series(m.feature_importance("gain"), index=FEATS))
        t = d[te].assign(p=preds[n])
        t["q"] = t.groupby("date")["p"].transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False))
        top = t[t["q"] == 4]
        s = top.groupby("date")[f"dm_{n}"].mean()
        s10 = t[t.groupby("date")["p"].rank(ascending=False) <= 10].groupby("date")[f"dm_{n}"].mean()
        q = t.groupby(["date", "q"])[f"dm_{n}"].mean().unstack().mean()
        print(f"{name} n={n:>3}: 前1/5 相对全部 {s.mean():+.2%} (t {nw_t(s):+.1f}), 前10只 {s10.mean():+.2%} (t {nw_t(s10):+.1f}), "
              f"比SPY {top.groupby('date')[f'ex_{n}'].mean().mean():+.2%}, 成交率 {top[f'hit_{n}'].mean():.0%}, "
              f"没成交 {top.loc[~top[f'hit_{n}'], f'ret_{n}'].mean():+.1%}, 同组直接持有20天比SPY {top.groupby('date')['ex_hold'].mean().mean():+.2%}, 五组 {' '.join(f'{x:+.2%}' for x in q)}")
    # per-stock n: choose the n with the highest predicted value
    P = pd.DataFrame(preds)
    best_n = P.idxmax(axis=1)
    t = d[te].assign(p=P.max(axis=1), n=best_n)
    t["dm"] = [t.at[k, f"dm_{n}"] for k, n in zip(t.index, t["n"])]
    t["ex"] = [t.at[k, f"ex_{n}"] for k, n in zip(t.index, t["n"])]
    t["q"] = t.groupby("date")["p"].transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False))
    s = t[t["q"] == 4].groupby("date")["dm"].mean()
    print(f"{name} 每只股票自选n: 前1/5 相对全部 {s.mean():+.2%} (t {nw_t(s):+.1f}), 比SPY {t[t['q']==4].groupby('date')['ex'].mean().mean():+.2%}, "
          f"选中的n分布 {t[t['q']==4]['n'].value_counts(normalize=True).round(2).to_dict()}\n")
print("== 特征重要性（两次训练平均，n=3%、5%、1σ）==")
imp = pd.concat([pd.concat(imp_all[n], axis=1).mean(axis=1) for n in ["3%", "5%", "1σ"]], axis=1, keys=["3%", "5%", "1σ"])
print((imp / imp.sum()).round(3).sort_values("5%", ascending=False).head(12).to_string())
