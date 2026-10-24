"""Why 7c worked in 2024-26 but not 2023 (Jasper 2026-10-24): market regime per window.
Universe: all S&P 500+400 names in the panel (point-in-time members). Per signal day: 20-trading-day forward return
(next open -> close 20 days later). Reports: SPY, equal-weight universe, high-vol vs low-vol quintile (60-day vol),
the 7c pick vs five-average split by SPY up/down over the next 20 days, sector mix and beta of the pick.
Usage: AV_START=... AV_END=... python research/blindtest/regime.py
"""
import io
import os
from contextlib import redirect_stdout

os.environ["KEEP_TECH"] = "1"
with redirect_stdout(io.StringIO()):
    exec(open("research/blindtest/trap3.py", encoding="utf-8").read().split("\ncs = Cl")[0])
a, b = f["date"].min(), f["date"].max()
P = pd.read_parquet("research/fullflow/flow_panel_daily.parquet", columns=["date", "ticker", "60日波动率", "贝塔", "sector"])
P["date"] = pd.to_datetime(P["date"])
P = P[(P["date"] >= a) & (P["date"] <= b)]
ii, jj = cal.searchsorted(P["date"].to_numpy()), O.columns.get_indexer(P["ticker"])
ok = (jj >= 0) & (ii + 21 < len(cal))
P = P[ok]; ii, jj = ii[ok], jj[ok]
P["fwd"] = Cl.to_numpy()[ii + 21, jj] / O.to_numpy()[ii + 1, jj] - 1
s_i = O.columns.get_loc("SPY")
spyf = pd.Series(Cl.to_numpy()[ii + 21, s_i] / O.to_numpy()[ii + 1, s_i] - 1, index=P.index)
P["spy"] = spyf
P["q"] = P.groupby("date")["60日波动率"].transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False))
d = P.groupby("date").agg(spy=("spy", "first"), ew=("fwd", "mean"))
d["hi"] = P[P.q == 4].groupby("date")["fwd"].mean(); d["lo"] = P[P.q == 0].groupby("date")["fwd"].mean()
d["ic"] = P.groupby("date").apply(lambda g: g["60日波动率"].rank().corr(g["fwd"].rank()))
d["beat"] = P.assign(b_=P["fwd"] > P["spy"]).groupby("date")["b_"].mean()
spy = Cl["SPY"][a:b]
dd = (spy / spy.cummax() - 1).min()
out = [f"窗口 {a.date()} ~ {b.date()}：SPY 区间涨幅 {spy.iloc[-1] / spy.iloc[0] - 1:+.1%}，最大回撤 {dd:.1%}，SPY 日波动 {spy.pct_change().std() * np.sqrt(252):.0%}（年化）",
       f"  每个信号日买入、拿 20 个交易日的平均：SPY {d.spy.mean():+.2%}，全部股票等权 {d.ew.mean():+.2%}，跑赢 SPY 的股票比例 {d.beat.mean():.0%}",
       f"  高波动 1/5 {d.hi.mean():+.2%}，低波动 1/5 {d.lo.mean():+.2%}，差 {(d.hi - d.lo).mean():+.2%}；波动率与之后 20 天收益的排名相关 {d.ic.mean():+.3f}",
       f"  之后 20 天 SPY 下跌的信号日占 {(d.spy < 0).mean():.0%}；SPY 跌超 3% 的占 {(d.spy < -0.03).mean():.0%}"]
for nm, m in (("SPY 之后 20 天上涨", d.spy > 0), ("SPY 之后 20 天下跌", d.spy <= 0)):
    days = d.index[m]
    q = f[f["date"].isin(days)]
    p7, p5 = q[q["被排序"] == 5]["结果_阶梯"].mean(), q["结果_阶梯"].mean()
    out.append(f"  {nm}（{len(days)} 天）：7c {p7:+.2%}，五只 {p5:+.2%}，差 {p7 - p5:+.2%}；高波动 − 低波动 {(d.hi - d.lo)[m].mean():+.2%}")
pk = f[f["被排序"] == 5]
fi = f
out.append(f"  7c 选的：贝塔 {fi[fi['被排序'] == 5]['贝塔'].mean():.2f}（五只平均 {fi['贝塔'].mean():.2f}），60 日波动 {pk['60日波动率'].mean() * np.sqrt(252):.0%}（五只 {f['60日波动率'].mean() * np.sqrt(252):.0%}）")
out.append("  7c 板块：" + ", ".join(f"{k} {v:.0%}" for k, v in pk["sector"].value_counts(normalize=True).head(5).items()))
out.append("  五只板块：" + ", ".join(f"{k} {v:.0%}" for k, v in f["sector"].value_counts(normalize=True).head(5).items()))
q = pk.copy(); q["月"] = q["date"].dt.to_period("M")
mm = pd.DataFrame({"7c": q.groupby("月")["结果_阶梯"].mean(), "五只": f.assign(月=f["date"].dt.to_period("M")).groupby("月")["结果_阶梯"].mean()})
mm["SPY20"] = d.assign(月=d.index.to_period("M")).groupby("月")["spy"].mean()
mm["高-低波动"] = (d.hi - d.lo).groupby(d.index.to_period("M")).mean()
out.append("  按月：\n" + (mm * 100).round(1).to_string())
txt = "\n".join(out)
open(f"research/blindtest/regime_{a.date()}.txt", "w").write(txt)
print(txt)
