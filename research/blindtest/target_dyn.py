"""Dynamic take-profit target (Jasper 2026-10-24): fixed +2..+6% vs a per-stock target from its own volatility
(m x 20-day daily vol x sqrt(10), clipped to 1.5%..10%). Stepped exit otherwise: target to day 20, 0% from day 21,
-3% from day 35, day-48 close. Daily top 5 (no sector rule, version in use) and the 7c pick; train year vs last year.
Usage: python research/blindtest/target_dyn.py
"""
import io
import os
from contextlib import redirect_stdout

os.environ["KEEP_TECH"] = "1"
with redirect_stdout(io.StringIO()):
    exec(open("research/blindtest/trap3.py", encoding="utf-8").read().split("\ncs = Cl")[0])


def run(x, tgt, L=48):
    i = cal.searchsorted(x["date"].to_numpy()); j = O.columns.get_indexer(x["ticker"])
    k = np.arange(1, L + 1)
    op, hi = O.to_numpy()[i[:, None] + k, j[:, None]], H.to_numpy()[i[:, None] + k, j[:, None]]
    en = op[:, 0]
    r, d, done = Cl.to_numpy()[i + L, j] / en - 1, np.full(len(i), L), np.zeros(len(i), bool)
    for t in range(L):
        lv = en * (1 + (tgt if t < 20 else 0.0 if t < 34 else -0.03))
        h = ~done & (hi[:, t] >= lv)
        r[h] = (np.maximum(lv, op[:, t]) if t else lv)[h] / en[h] - 1
        d[h], done = t + 1, done | h
    return r - 0.002, d


sig = f["20日波动率"].to_numpy() * np.sqrt(10)
T = {f"固定 +{v:.0%}": np.full(len(f), v) for v in (0.02, 0.03, 0.04, 0.05, 0.06)}
for m_ in (0.5, 0.75, 1.0, 1.25):
    T[f"波动 ×{m_}"] = np.clip(m_ * np.nan_to_num(sig, nan=0.03 / m_), 0.015, 0.10)
lines = []
for half, m in (("训练年", f["train"].to_numpy()), ("最近一年", ~f["train"].to_numpy())):
    for who, w in (("7c 选的", m & (f["被排序"] == 5).to_numpy()), ("五只平均", m)):
        x = f[w]
        rows = []
        for nm, tg in T.items():
            r, d = run(x, tg[w])
            blk = pd.Series(r / d * 20).groupby(np.arange(len(r)) // 20).mean()
            rows.append(dict(目标=nm, 平均目标=f"{tg[w].mean():+.1%}", 每笔=f"{r.mean():+.2%}", 天数=round(d.mean(), 1),
                             达到目标=f"{(d <= 20).mean() if True else 0:.0%}", 一年约=f"{(1 + r.mean()) ** (250 / d.mean()) - 1:+.0%}",
                             亏超10=f"{(r < -0.10).mean():.0%}"))
        lines.append(f"\n== {half} {who}（{len(x)} 笔）==\n" + pd.DataFrame(rows).to_string(index=False))
open("research/blindtest/target_dyn.txt", "w").write("\n".join(lines))
print("\n".join(lines))

# paired check: vol x0.5 / x0.75 vs fixed +3%, return per 20 trading days held, 20-day blocks
lines.append("\n== 配对比较（总收益 / 总持有天数，折合每 20 个持有日，按 20 个信号日分块）==")
for half, m in (("训练年", f["train"].to_numpy()), ("最近一年", ~f["train"].to_numpy())):
    for who, w in (("7c 选的", m & (f["被排序"] == 5).to_numpy()), ("五只平均", m)):
        x = f[w]
        r0, d0 = run(x, T["固定 +3%"][w])
        for nm in ("波动 ×0.5", "波动 ×0.75"):
            r1_, d1 = run(x, T[nm][w])
            b = pd.factorize(x["date"])[0] // 20  # 20 signal days per block; per-day return = sum(ret) / sum(days) within block
            g = pd.DataFrame(dict(b=b, r0=r0, d0=d0, r1=r1_, d1=d1)).groupby("b").sum()
            blk = (g["r1"] / g["d1"] - g["r0"] / g["d0"]) * 20
            tot = (r1_.sum() / d1.sum() - r0.sum() / d0.sum()) * 20
            lines.append(f"  {half} {who} {nm} − 固定 +3%：每 20 个持有日 {tot:+.2%}，分块 t {blk.mean() / blk.std() * np.sqrt(len(blk)):.2f}")
open("research/blindtest/target_dyn.txt", "w").write("\n".join(lines))
print("\n".join(lines[-9:]))
