"""Count of trap traits (Jasper 2026-10-24): the 7 features stable in both years (trap2.txt); a stock 'has' a trait when it is
among the 2 lowest of the day's five on that feature (all 7: lower = more often trapped). Exclude the K stocks with the most
traits (ties: lower average rank first), buy the rest equally. Note: the 7 were chosen with both years -> last year is in-sample.
Usage: python research/blindtest/trap3.py
"""
import io
from contextlib import redirect_stdout

with redirect_stdout(io.StringIO()):
    exec(open("research/blindtest/trap2.py", encoding="utf-8").read().split("\n# combined score")[0])

T7 = ["20日波动率", "60日波动率", "残差波动率", "上次财报次日反应", "200日线斜率", "离52周低点", "6月涨幅"]
lowrk = pd.DataFrame({c: f.groupby("date")[c].rank(method="min") for c in T7})
f["特征数"] = (lowrk <= 2).sum(1)
f["均排名"] = lowrk.mean(1)
f["被排序"] = f.sort_values(["特征数", "均排名"], ascending=[False, True]).groupby("date").cumcount().reindex(f.index) + 1
cs = Cl["SPY"]
out = [f"7 个特征：{'、'.join(T7)}；每只满足 = 当天五只里排最低的 2 只之一。注意：这 7 个是看两年结果挑的，最近一年不是样本外。"]
for half, m in (("训练年", f["train"]), ("最近一年", ~f["train"])):
    x = f[m]
    ii = cal.searchsorted(x["date"].to_numpy())
    spy_ = {lab: np.mean(cs.to_numpy()[np.minimum(ii + L, len(cal) - 1)] / O["SPY"].to_numpy()[ii + 1] - 1) for lab, L in (("阶梯", 20), ("现行", 20))}
    out.append(f"\n== {half}（{x['date'].nunique()} 天）被套比例 {x['被套'].mean():.0%}；满足特征数分布 " +
               ", ".join(f"{k}个:{v:.0%}" for k, v in x["特征数"].value_counts(normalize=True).sort_index().items()))
    out.append("  按满足个数分组的被套比例：" + ", ".join(f"{k}个 {v:.0%}" for k, v in x.groupby("特征数")["被套"].mean().items()))
    for ex in ("阶梯", "现行"):
        rr, hd = f"结果_{ex}", f"持有天数_{ex}"
        rows = []
        for K in (0, 1, 2, 3, 4):
            keep = x[x["被排序"] > K]
            g = keep.groupby("date")
            diff = g[rr].mean() - x.groupby("date")[rr].mean()
            blk = diff.groupby(np.arange(len(diff)) // 20).mean()
            rows.append(dict(排除=f"{K} 只", 每笔=f"{keep[rr].mean():+.2%}", 平均持有=round(keep[hd].mean(), 1),
                             折合每20日=f"{keep[rr].mean() / keep[hd].mean() * 20:+.2%}", 被套=f"{keep['被套'].mean():.0%}",
                             亏钱=f"{(keep[rr] <= 0).mean():.0%}", 比五只=f"{diff.mean():+.2%}",
                             分块t=round(blk.mean() / blk.std() * np.sqrt(len(blk)), 2) if K else np.nan))
        k1 = x[x["被排序"] > 4]
        tech = k1["sector"].isin(["Information Technology", "Communication Services"])
        out.append(f"  {ex}：7c 选的那只里科技/通信占 {tech.mean():.0%}，它们每笔 {k1.loc[tech, rr].mean():+.2%}，其他 {k1.loc[~tech, rr].mean():+.2%}；中位数 {k1[rr].median():+.2%}；去掉最好的 5 笔后平均 {k1[rr].sort_values().iloc[:-5].mean():+.2%}")
        out.append(f"  {ex}卖法（同期拿 SPY 20 个交易日每笔约 {spy_[ex]:+.2%}）\n" + pd.DataFrame(rows).to_string(index=False))
import os
out.insert(0, "不排除科技、通信" if os.environ.get("KEEP_TECH") else "排除科技、通信（现行）")
out.insert(1, "科技/通信在五只里占比：" + ", ".join(f"{h} {f.loc[m_, 'sector'].isin(['Information Technology', 'Communication Services']).mean():.0%}" for h, m_ in (("训练年", f["train"]), ("最近一年", ~f["train"]))))
open("research/blindtest/trap3" + ("_keep_tech" if os.environ.get("KEEP_TECH") else "") + ".txt", "w").write("\n".join(out))
print("\n".join(out))
