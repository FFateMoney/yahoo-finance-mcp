"""Exit-rule grid (Jasper 2026-10-24: fit the exit on the train year, check on the test year). Parameters: target n,
day from which the limit drops to 0% (break-even), stop-loss level (active from day 1, on the low), max holding days.
Objective: return per 20 trading days (mean return / mean days * 20). Rows = daily top-5, two years.
Usage: python research/fullflow/exit_grid.py
"""
import itertools

exec(open("research/fullflow/exit_bs.py", encoding="utf-8").read().split("\nrows = feat")[0])  # sim, feat, MID, cal

rows = feat[["date", "ticker"]].assign(date=pd.to_datetime(feat["date"]))
tr, te = rows[rows["date"] <= MID], rows[rows["date"] > MID]
spy = Cl["SPY"]


def spy20(x):
    s = spy[x["date"].min():x["date"].max()]
    return (s.iloc[-1] / s.iloc[0]) ** (20 / (len(s) - 1)) - 1


res = []
for n, be, stop, L in itertools.product([0.02, 0.03, 0.05, 0.07, 0.10, 9.9], [10, 20, None], [None, -0.05, -0.08, -0.12], [10, 20, 30, 48]):
    if be is not None and be >= L:
        continue
    steps = [(0, n)] + ([(be, 0.0)] if be is not None else [])
    kw = dict(lim_steps=steps, stop_from=0 if stop else None, stop=stop or -1, L=L)
    row = dict(目标=("不设" if n > 1 else f"+{n:.0%}"), 回本日=(be + 1 if be is not None else "不设"), 止损=(f"{stop:.0%}" if stop else "不设"), 最长=L)
    for lab, x in (("训练", tr), ("测试", te)):
        r, dd = sim(x, **kw)
        row[f"{lab}每20日"] = r.mean() / dd.mean() * 20
        row[f"{lab}每笔"] = r.mean()
    res.append(row)
g = pd.DataFrame(res).sort_values("训练每20日", ascending=False).reset_index(drop=True)
best = g.iloc[0]
lines = [f"{len(g)} 种组合。一直拿 SPY 折合每 20 交易日：训练年 {spy20(tr):+.2%}，测试年 {spy20(te):+.2%}",
         f"训练年最优：目标 {best['目标']}，第 {best['回本日']} 天起回本卖，止损 {best['止损']}，最长 {best['最长']} 天 → 训练 {best['训练每20日']:+.2%}，测试 {best['测试每20日']:+.2%}",
         f"训练年排名和测试年排名的相关：{g['训练每20日'].rank().corr(g['测试每20日'].rank()):+.2f}",
         f"测试年里训练前 10 名的平均 {g.head(10)['测试每20日'].mean():+.2%}，全部组合平均 {g['测试每20日'].mean():+.2%}，测试年最优 {g['测试每20日'].max():+.2%}"]
fmt = g.copy()
for c in ("训练每20日", "测试每20日", "训练每笔", "测试每笔"):
    fmt[c] = fmt[c].map(lambda v: f"{v:+.2%}")
lines += ["\n训练年前 15：", fmt.head(15).to_string(), "\n训练年后 5：", fmt.tail(5).to_string()]
lines.append("\n按单个参数平均（测试每20日）：")
for c in ("目标", "回本日", "止损", "最长"):
    lines.append(f"  {c}: " + "，".join(f"{k} 训练 {v['训练每20日']:+.2%} 测试 {v['测试每20日']:+.2%}" for k, v in g.groupby(c)[["训练每20日", "测试每20日"]].mean().iterrows()))
open("research/fullflow/exit_grid.txt", "w").write("\n".join(lines))
print("\n".join(lines))
