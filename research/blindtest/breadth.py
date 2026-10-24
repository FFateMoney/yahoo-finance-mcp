"""Is the process useless in narrow bull markets? (Jasper 2026-10-24). Version in use (top 5, no sector rule, stepped exit),
2022-01 to 2026-08. Per calendar month and year: five-average return per 20 holding days (sum ret / sum days x 20) vs
buying every stock in the universe for 20 trading days (equal weight) vs SPY 20 days; breadth = equal weight - SPY.
Usage: python research/blindtest/breadth.py
"""
import io
import os
from contextlib import redirect_stdout

os.environ.update(KEEP_TECH="1", AV_START="2021-12-31", AV_END="2026-08-14")
with redirect_stdout(io.StringIO()):
    exec(open("research/blindtest/trap3.py", encoding="utf-8").read().split("\ncs = Cl")[0])
P = pd.read_parquet("research/fullflow/flow_panel_daily.parquet", columns=["date", "ticker"])
P["date"] = pd.to_datetime(P["date"])
P = P[(P["date"] >= f["date"].min()) & (P["date"] <= f["date"].max())]
ii, jj = cal.searchsorted(P["date"].to_numpy()), O.columns.get_indexer(P["ticker"])
ok = (jj >= 0) & (ii + 21 < len(cal)); P = P[ok]; ii, jj = ii[ok], jj[ok]
P["fwd"] = Cl.to_numpy()[ii + 21, jj] / O.to_numpy()[ii + 1, jj] - 1
s = O.columns.get_loc("SPY")
P["spy"] = Cl.to_numpy()[ii + 21, s] / O.to_numpy()[ii + 1, s] - 1
P["beat"] = P["fwd"] > P["spy"]
d = P.groupby("date").agg(ew=("fwd", "mean"), spy=("spy", "first"), beat=("beat", "mean"))
x = f.assign(r=f["结果_阶梯"], dd=f["持有天数_阶梯"], r7=np.where(f["被排序"] == 5, f["结果_阶梯"], 0.0), d7=np.where(f["被排序"] == 5, f["持有天数_阶梯"], 0))
out = []
for per, key in (("年", "Y"), ("月", "M")):
    g = x.groupby(x["date"].dt.to_period(key)).agg(r=("r", "sum"), dd=("dd", "sum"), r7=("r7", "sum"), d7=("d7", "sum"))
    t = pd.DataFrame({"五只": g.r / g.dd * 20, "7c": g.r7 / g.d7 * 20})
    dm = d.groupby(d.index.to_period(key)).mean()
    t["等权全部"], t["SPY"], t["宽度(等权-SPY)"], t["跑赢SPY比例"] = dm.ew, dm.spy, dm.ew - dm.spy, dm.beat
    t["五只-等权"], t["五只-SPY"] = t["五只"] - t["等权全部"], t["五只"] - t["SPY"]
    if key == "Y":
        out.append("== 按年（每 20 个交易日）==\n" + (t * 100).round(2).to_string())
    else:
        tm = t.dropna()
        c1 = tm["五只-等权"].corr(tm["宽度(等权-SPY)"]); c2 = tm["五只-SPY"].corr(tm["宽度(等权-SPY)"])
        out.append(f"\n== 按月（{len(tm)} 个月）==\n五只−等权 与 宽度 的相关 {c1:+.2f}；五只−SPY 与 宽度 的相关 {c2:+.2f}")
        tm = tm.assign(组=pd.qcut(tm["宽度(等权-SPY)"], 3, labels=["窄（等权落后最多）", "中", "宽"]))
        out.append((tm.groupby("组", observed=True)[["宽度(等权-SPY)", "五只", "等权全部", "SPY", "五只-等权", "五只-SPY"]].mean() * 100).round(2).to_string())
        up = tm[tm["SPY"] > 0]
        out.append("\n只看 SPY 上涨的月份（牛市里窄 vs 宽）：\n" + (up.groupby("组", observed=True)[["宽度(等权-SPY)", "五只", "等权全部", "SPY", "五只-等权", "五只-SPY"]].mean() * 100).round(2).to_string()
                   + "\n月数：" + up["组"].value_counts().sort_index().to_string().replace("\n", "，"))
txt = "\n".join(out)
open("research/blindtest/breadth.txt", "w").write(txt)
print(txt)
