"""Trapped stocks among the daily five (Jasper 2026-10-24).
Part 1: which features tell apart the stocks that get trapped (no +3% within 10 trading days and still below cost at day 10)
— AUROC of each feature, ranked within the day, train year vs test year (last year).
Part 2: once trapped at day 10, sell when back to x% (x = -6..+3) or at the close of day L; loss, days held, and loss
net of the time cost (days x the average daily return of a fresh pick).
Usage: python research/blindtest/trap.py
"""
import io
import sys
from contextlib import redirect_stdout

with redirect_stdout(io.StringIO()):
    sys.argv = ["x", "/tmp/exit_bs_tmp"]
    exec(open("research/fullflow/exit_bs.py", encoding="utf-8").read().split("\nrows = feat")[0])  # feat, O, H, Cl, cal, MID

f = feat.copy()
f["date"] = pd.to_datetime(f["date"])
i = cal.searchsorted(f["date"].to_numpy())
j = O.columns.get_indexer(f["ticker"])
k = np.arange(1, 11)
op, hi = O.to_numpy()[i[:, None] + k, j[:, None]], H.to_numpy()[i[:, None] + k, j[:, None]]
entry = op[:, 0]
hit10 = (hi >= entry[:, None] * 1.03).any(1)
c10 = Cl.to_numpy()[i + 10, j] / entry - 1
f["被套"] = (~hit10 & (c10 < 0)).astype(int)
f["train"] = f["date"] <= MID


def auc(score, y):
    m = score.notna()
    s, y = score[m].to_numpy(), y[m].to_numpy()
    r = pd.Series(s).rank().to_numpy()
    n1, n0 = y.sum(), (1 - y).sum()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan


skip = {"date", "ticker", "sector", "来源", "上次财报日", "score_filed", "period_end", "被套", "train", "sp500", "ma200_up"}
cols = [c for c in f.columns if c not in skip and not c.startswith(("结果_", "持有天数_", "当天最差", "亏钱_")) and f[c].dtype != object]
rows = []
for c in cols:
    rk = f.groupby("date")[c].rank(pct=True)  # within-day rank: we only compare the five against each other
    a_tr, a_te = auc(rk[f["train"]], f.loc[f["train"], "被套"]), auc(rk[~f["train"]], f.loc[~f["train"], "被套"])
    # block bootstrap over 20-day blocks for the test-year AUC spread
    rows.append(dict(特征=c, 训练年AUC=a_tr, 最近一年AUC=a_te))
A = pd.DataFrame(rows)
A["两年同向"] = np.sign(A["训练年AUC"] - 0.5) == np.sign(A["最近一年AUC"] - 0.5)
A["强度"] = (A["训练年AUC"] - 0.5).abs() + (A["最近一年AUC"] - 0.5).abs()
A = A.sort_values("强度", ascending=False)
lines = [f"被套 = 10 个交易日内没碰到 +3% 且第 10 天收盘低于成本。训练年 {f.loc[f.train, '被套'].mean():.0%}，最近一年 {f.loc[~f.train, '被套'].mean():.0%}",
         f"AUC > 0.5 = 数值越大越容易被套（按当天五只内部排名算）。两年 AUC 的相关：{A['训练年AUC'].corr(A['最近一年AUC']):+.2f}",
         A.drop(columns="强度").round(3).head(20).to_string(index=False)]

# rough noise level: shuffle labels within day
rng = np.random.default_rng(0)
null = []
for _ in range(200):
    y = f.groupby("date")["被套"].transform(lambda s: s.sample(frac=1, random_state=int(rng.integers(1 << 30))).to_numpy())
    null.append(auc(f.loc[~f.train, "60日波动率"], y[~f.train]))
lines.append(f"噪声参考：当天内打乱标签后 AUC 的 5%–95% 范围约 {np.percentile(null, 5):.3f} ~ {np.percentile(null, 95):.3f}（逐笔独立假设，偏窄；相邻日重叠会让真实范围更宽）")

# Part 2
tr_ = f[f["被套"] == 1]
mu = {}
for lab, m in (("训练年", f["train"]), ("最近一年", ~f["train"])):
    x = f[m]
    ii = cal.searchsorted(x["date"].to_numpy()); jj = O.columns.get_indexer(x["ticker"])
    mu[lab] = np.nanmean(Cl.to_numpy()[np.minimum(ii + 20, len(cal) - 1), jj] / O.to_numpy()[ii + 1, jj] - 1) / 20
lines.append(f"\n== 被套以后（从第 11 个交易日起）==  新选一只的平均每天收益：训练年 {mu['训练年']:+.3%}，最近一年 {mu['最近一年']:+.3%}")
for lab, m in (("训练年", tr_["train"]), ("最近一年", ~tr_["train"])):
    x = tr_[m]
    ii = cal.searchsorted(x["date"].to_numpy()); jj = O.columns.get_indexer(x["ticker"])
    ok = ii + 61 < len(cal)
    ii, jj = ii[ok], jj[ok]
    kk = np.arange(1, 61)
    OP, HI = O.to_numpy()[ii[:, None] + kk, jj[:, None]], H.to_numpy()[ii[:, None] + kk, jj[:, None]]
    en = OP[:, 0]
    out = []
    for lvl in (-0.06, -0.05, -0.04, -0.03, -0.02, -0.01, 0.0, 0.01, 0.03):
        for L in (20, 30, 48, 60):
            r = Cl.to_numpy()[ii + L, jj] / en - 1
            d = np.full(len(ii), L)
            done = np.zeros(len(ii), bool)
            for t in range(10, L):
                lv = en * (1 + (0.03 if t < 10 else max(lvl, -1)))
                hit = ~done & (HI[:, t] >= np.minimum(lv, en * 1.03))
                r[hit] = np.maximum(np.minimum(lv, en * 1.03), OP[:, t])[hit] / en[hit] - 1
                d[hit], done = t + 1, done | hit
            r = r - 0.002
            out.append(dict(回到=f"{lvl:+.0%}", 最长=L, 平均亏损=r.mean(), 平均天数=d.mean(), 亏超10=(r < -0.10).mean(),
                            扣时间成本=r.mean() - d.mean() * mu[lab]))
    o = pd.DataFrame(out).sort_values("扣时间成本", ascending=False)
    for c in ("平均亏损", "扣时间成本"):
        o[c] = o[c].map(lambda v: f"{v:+.2%}")
    o["亏超10"] = o["亏超10"].map(lambda v: f"{v:.0%}")
    o["平均天数"] = o["平均天数"].round(1)
    lines += [f"\n{lab}：被套的 {ok.sum()} 笔（按扣时间成本排序，越高越好）", o.head(8).to_string(index=False), "  ……最差 3 个：", o.tail(3).to_string(index=False)]
open("research/blindtest/trap.txt", "w").write("\n".join(lines))
print("\n".join(lines))
