"""Expected return of the 7c pick split into 'not trapped' and 'trapped' (Jasper 2026-10-24):
E[r] = (1-p)*E[r|not trapped] + p*E[r|trapped], same for days; per-year figure = E[r]/E[days]*250 compounded.
Usage: [KEEP_TECH=1] python research/blindtest/trap4.py
"""
import io
import os
from contextlib import redirect_stdout

with redirect_stdout(io.StringIO()):
    exec(open("research/blindtest/trap3.py", encoding="utf-8").read().split("\ncs = Cl")[0])
out = ["不排除科技、通信" if os.environ.get("KEEP_TECH") else "排除科技、通信（现行）"]
for half, m in (("训练年", f["train"]), ("最近一年", ~f["train"])):
    for who, x in (("7c 选的", f[m & (f["被排序"] == 5)]), ("五只平均", f[m])):
        for ex in ("阶梯",):
            r, d = x[f"结果_{ex}"], x[f"持有天数_{ex}"]
            t = x["被套"] == 1
            p = t.mean()
            yr = (1 + r.mean()) ** (250 / d.mean()) - 1
            out.append(f"{half} {who}（{len(x)} 笔）：被套概率 {p:.0%}；没被套 每笔 {r[~t].mean():+.2%}、{d[~t].mean():.1f} 天；"
                       f"被套 每笔 {r[t].mean():+.2%}、{d[t].mean():.1f} 天（其中最后回本或赚钱 {(r[t] > 0).mean():.0%}，亏超 10% {(r[t] < -0.10).mean():.0%}）"
                       f" → 期望每笔 {r.mean():+.2%}、{d.mean():.1f} 天 → 一年约 {yr:+.0%}；被套占用时间 {d[t].sum() / d.sum():.0%}")
print("\n".join(out))
