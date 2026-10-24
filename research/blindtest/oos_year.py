"""Version in use (top 5, no sector rule, 7c filter, stepped exit) on another window, e.g. calendar 2023 (Jasper 2026-10-24).
Usage: AV_START=2022-12-31 AV_END=2023-12-31 python research/blindtest/oos_year.py
"""
import io
import os
from contextlib import redirect_stdout

os.environ["KEEP_TECH"] = "1"
with redirect_stdout(io.StringIO()):
    exec(open("research/blindtest/trap3.py", encoding="utf-8").read().split("\ncs = Cl")[0])
x = f
i0 = cal.searchsorted(x["date"].to_numpy())
spy = Cl["SPY"]
a, b = x["date"].min(), x["date"].max()
sr = spy[a:b]
out = [f"窗口 {a.date()} ~ {b.date()}（{x['date'].nunique()} 个信号日）；同期一直拿 SPY {sr.iloc[-1] / sr.iloc[0] - 1:+.1%}，折合每 20 交易日 {(sr.iloc[-1] / sr.iloc[0]) ** (20 / (len(sr) - 1)) - 1:+.2%}"]
for ex in ("阶梯", "现行"):
    for who, w in (("7c 选的", x["被排序"] == 5), ("五只平均", x["被排序"] > 0), ("排除科技前的五只平均（参考）", ~x["sector"].isin(["Information Technology", "Communication Services"]))):
        q = x[w]
        r, d = q[f"结果_{ex}"], q[f"持有天数_{ex}"]
        t = q["被套"] == 1
        out.append(f"{ex} {who}（{len(q)} 笔）：每笔 {r.mean():+.2%}、{d.mean():.1f} 天 → 一年约 {(1 + r.mean()) ** (250 / d.mean()) - 1:+.0%}；"
                   f"被套 {t.mean():.0%}（没被套 {r[~t].mean():+.2%}/{d[~t].mean():.1f} 天，被套 {r[t].mean():+.2%}/{d[t].mean():.1f} 天）；亏超 10% {(r < -0.10).mean():.0%}；赚钱 {(r > 0).mean():.0%}")
    w = x["被排序"] == 5
    diff = x[w].set_index("date")[f"结果_{ex}"] - x.groupby("date")[f"结果_{ex}"].mean()
    blk = diff.groupby(np.arange(len(diff)) // 20).mean()
    out.append(f"  {ex} 7c − 五只平均：{diff.mean():+.2%}，分块 t {blk.mean() / blk.std() * np.sqrt(len(blk)):.2f}")
    q = x[w].copy(); q["月"] = q["date"].dt.to_period("M")
    out.append("  7c 按月每笔：" + ", ".join(f"{k.month}月 {v:+.1%}" for k, v in q.groupby("月")[f"结果_{ex}"].mean().items()))

# $180, one at a time, sequential: buy the 7c pick at next open, sell per exit, next buy = next signal day after the sale
for ex in ("阶梯", "现行"):
    q = x[x["被排序"] == 5].sort_values("date").reset_index(drop=True)
    cash, k, n, path = 180.0, 0, 0, []
    dates = q["date"].to_numpy()
    while k < len(q):
        r_, d_ = q.loc[k, f"结果_{ex}"], int(q.loc[k, f"持有天数_{ex}"])
        cash *= 1 + r_; n += 1
        exit_i = cal.searchsorted(q.loc[k, "date"]) + d_
        k = int(np.searchsorted(cal[cal.searchsorted(dates)].to_numpy() if False else cal.searchsorted(dates), exit_i, side="left"))
        k = int(np.searchsorted(cal.searchsorted(dates), exit_i))
    out.append(f"$180 一次一只连续做（{ex}）：{n} 笔，期末 ${cash:.0f}（{cash / 180 - 1:+.1%}）")
txt = "\n".join(out)
open(f"research/blindtest/oos_{os.environ.get('AV_END', 'x')[:4]}.txt", "w").write(txt)
print(txt)
