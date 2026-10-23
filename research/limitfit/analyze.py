"""Summaries of research/limitfit/out/panel.parquet (run explore.py first)."""
import numpy as np
import pandas as pd

d = pd.read_parquet("research/limitfit/out/panel.parquet")
NS = ["2%", "3%", "5%", "7%", "10%", "1σ"]
FEATS = [c for c in d.columns if not c.startswith(("ret_", "ex_", "hit_", "days_")) and c not in ("i", "tk", "date", "ticker")]
MID = d["date"].sort_values().unique()[len(d["date"].unique()) // 2]


def nw_t(x, lag=4):
    x = pd.Series(x).dropna()
    e = x - x.mean()
    T = len(e)
    v = (e @ e) / T
    for k in range(1, lag + 1):
        v += 2 * (1 - k / (lag + 1)) * (e[k:].to_numpy() @ e[:-k].to_numpy()) / T
    return x.mean() / np.sqrt(v / T)


print("== 不挑股票的基准（每笔，扣成本）==")
for n in NS:
    g = d.groupby("date")
    print(f"n={n:>3}: 成交率 {d[f'hit_{n}'].mean():.0%}  每笔收益 {g[f'ret_{n}'].mean().mean():+.2%}  "
          f"比SPY {g[f'ex_{n}'].mean().mean():+.2%} (t {nw_t(g[f'ex_{n}'].mean()):+.1f})  "
          f"没成交平均 {d.loc[~d[f'hit_{n}'], f'ret_{n}'].mean():+.1%}  平均持有 {d[f'days_{n}'].mean():.1f} 天")

d["q"] = 0
res = []
for f in FEATS:
    q = d.groupby("date")[f].transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False) if s.notna().sum() > 50 else np.nan)
    for n in NS:
        ex = d[f"ex_{n}"]
        dm = ex - ex.groupby(d["date"]).transform("mean")
        tab = pd.DataFrame({"date": d["date"], "q": q, "dm": dm, "hit": d[f"hit_{n}"], "ret": d[f"ret_{n}"]}).dropna(subset=["q"])
        by = tab.groupby(["date", "q"])["dm"].mean().unstack()
        for side, col in (("高", 4), ("低", 0)):
            s = by[col]
            sub = tab[tab["q"] == col]
            res.append({
                "特征": f, "n": n, "组": side, "相对全部/笔": s.mean(), "t": nw_t(s),
                "前半t": nw_t(s[s.index < MID]), "后半t": nw_t(s[s.index >= MID]),
                "正的日期占比": (s > 0).mean(), "成交率": sub["hit"].mean(),
                "没成交平均": sub.loc[~sub["hit"].astype(bool), "ret"].mean(),
                "单调": pd.Series(by.mean().to_numpy()).corr(pd.Series(range(5)), method="spearman"),
            })
r = pd.DataFrame(res)
r.to_csv("research/limitfit/out/single.csv", index=False)
pd.set_option("display.width", 250, "display.max_rows", 400)
fmt = r.copy()
for c in ["相对全部/笔", "成交率", "没成交平均", "正的日期占比"]:
    fmt[c] = fmt[c].map(lambda x: f"{x:+.2%}" if c != "成交率" and c != "正的日期占比" else f"{x:.0%}")
for c in ["t", "前半t", "后半t", "单调"]:
    fmt[c] = fmt[c].round(2)
print(f"\n== 单个特征：前 1/5 或后 1/5 组相对同日全部股票（分界 {pd.Timestamp(MID).date()}）==")
good = r[(r["t"].abs() > 2) & (np.sign(r["前半t"]) == np.sign(r["后半t"]))].sort_values("t", key=abs, ascending=False)
print(fmt.loc[good.index].to_string(index=False))
print("\n== n=3% 全部特征 ==")
print(fmt[(r["n"] == "3%")].sort_values("t").to_string(index=False))
