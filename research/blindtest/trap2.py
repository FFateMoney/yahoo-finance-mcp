"""More features for trapped stocks (Jasper 2026-10-24): 22 extra single features, 10 composites (e.g. below MA20 / above MA200),
and a combined score fit on the train year only. Pre-registration: 排雷/被套特征扩展_预登记.md.
Usage: python research/blindtest/trap2.py
"""
exec(open("research/blindtest/trap.py", encoding="utf-8").read().split("\nskip = ")[0])  # f (with 被套, train), auc, O, H, Cl, Lw, cal

Dd = "research/factors/data"
vo = pd.concat([pd.read_csv(f"{Dd}/prices.csv.gz", parse_dates=["Date"], usecols=["Date", "Ticker", "Volume"]),
                pd.read_csv(f"{Dd}/prices_removed.csv.gz", parse_dates=["Date"], usecols=["Date", "Ticker", "Volume"])])
V = vo.drop_duplicates(["Date", "Ticker"]).pivot(index="Date", columns="Ticker", values="Volume").reindex(index=O.index, columns=O.columns)
C, Hh, Ll, Oo = Cl, H, Lw, O
r1 = C.pct_change(fill_method=None)
ma = {n: C.rolling(n).mean() for n in (5, 10, 20, 50, 200)}
down = (r1 < 0).astype(float)
streak = down.groupby((down != down.shift()).cumsum(), axis=0).cumsum() * down if False else None
# consecutive down closes (vectorised)
s = np.zeros(C.shape)
dn = (r1 < 0).to_numpy()
for t in range(1, len(C)):
    s[t] = np.where(dn[t], s[t - 1] + 1, 0)
streak = pd.DataFrame(s, index=C.index, columns=C.columns)
g = C.diff()
up2, dn2 = g.clip(lower=0).ewm(alpha=1 / 2).mean(), (-g.clip(upper=0)).ewm(alpha=1 / 2).mean()
hi20 = C.rolling(20).max()
days_hi20 = C.rolling(20).apply(lambda a: 19 - np.argmax(a), raw=True)
days_hi252 = C.rolling(252, min_periods=200).apply(lambda a: len(a) - 1 - np.argmax(a), raw=True)
dd60 = C / C.rolling(60).max() - 1
v20, v60 = r1.rolling(20).std(), r1.rolling(60).std()
gapdn = ((Oo / C.shift() - 1) < -0.01).astype(float).rolling(21).sum()
dvol = (V * (r1 < 0)).rolling(20).sum() / V.rolling(20).sum()
spy = r1["SPY"]
corr = r1.rolling(60).corr(spy)
NEW = {
    "当天涨跌": r1, "3日涨跌": C / C.shift(3) - 1, "连跌天数": streak, "当天跳空": Oo / C.shift() - 1,
    "收盘在当日区间位置": (C - Ll) / (Hh - Ll).replace(0, np.nan), "当天下影线": (np.minimum(Oo, C) - Ll) / C,
    "RSI2": 100 - 100 / (1 + up2 / dn2.replace(0, np.nan)),
    "离20日最高收盘": C / hi20 - 1, "离20日最高过了几天": days_hi20, "离10日线": C / ma[10] - 1,
    "5日线斜率": ma[5] / ma[5].shift(3) - 1, "60日最大回撤": dd60, "离52周高点过了几天": days_hi252,
    "200日线斜率": ma[200] / ma[200].shift(20) - 1, "50日线斜率": ma[50] / ma[50].shift(10) - 1, "50日线/200日线": ma[50] / ma[200] - 1,
    "20日/60日波动率": v20 / v60, "21日跳空下跌次数": gapdn,
    "当天量/20日均量": V / V.rolling(20).mean(), "20日下跌日成交量占比": dvol,
    "与SPY相关性60日": corr, "5日相对SPY": (C / C.shift(5)).div(C["SPY"] / C["SPY"].shift(5), axis=0) - 1,
}
i = cal.searchsorted(f["date"].to_numpy())
j = O.columns.get_indexer(f["ticker"])
for k, M in NEW.items():
    f[k] = M.to_numpy()[i, j]
rk = lambda c: f.groupby("date")[c].rank(pct=True)
below20, below50 = f["离20日线"] < 0, f["离50日线"] < 0
COMP = {
    "C1 20日线下且50日线上": (below20 & ~below50).astype(float),
    "C2 20和50日线下、200日线上": (below20 & below50 & (f["离200日线"] > 0)).astype(float),
    "C3 多头排列20>50>200": ((ma[20] > ma[50]) & (ma[50] > ma[200])).to_numpy()[i, j].astype(float),
    "C4 回调深+长期趋势强": rk("离200日线") - rk("离20日线"),
    "C5 超卖+收盘强": rk("收盘在当日区间位置") - rk("RSI14"),
    "C6 连跌>=3且当天收阳": ((streak.shift().to_numpy()[i, j] >= 3) & (f["当天涨跌"] > 0)).astype(float),
    "C7 均线下放量": (below20 & (f["当天量/20日均量"] > 1.5)).astype(float),
    "C8 强动量里的回调": rk("动量12-1") - rk("1月涨幅"),
    "C9 安静+近高点": rk("离52周高点") - rk("60日波动率"),
    "C10 200日线陡+离得近": rk("200日线斜率") - rk("离200日线"),
}
for k, v in COMP.items():
    f[k] = v

skip = {"date", "ticker", "sector", "来源", "上次财报日", "score_filed", "period_end", "被套", "train", "sp500", "ma200_up"}
allc = [c for c in f.columns if c not in skip and not c.startswith(("结果_", "持有天数_", "当天最差", "亏钱_")) and f[c].dtype != object]
R = pd.DataFrame({c: rk(c) for c in allc})
tr, te = f["train"].to_numpy(), ~f["train"].to_numpy()
y = f["被套"]
A = pd.DataFrame([dict(特征=c, 组=("新" if c in NEW else "复合" if c in COMP else "旧"),
                       训练年AUC=auc(R.loc[tr, c], y[tr]), 最近一年AUC=auc(R.loc[te, c], y[te]),
                       训练年占比=f.loc[tr, c].mean() if c in COMP and f[c].isin([0, 1]).all() else np.nan) for c in allc])
A["稳定"] = ((A["训练年AUC"] - .5).abs() > .04) & ((A["最近一年AUC"] - .5).abs() > .04) & (np.sign(A["训练年AUC"] - .5) == np.sign(A["最近一年AUC"] - .5))
A["强度"] = (A["训练年AUC"] - .5).abs() + (A["最近一年AUC"] - .5).abs()
lines = [f"被套比例：训练年 {y[tr].mean():.0%}，最近一年 {y[te].mean():.0%}。AUC > 0.5 = 数值越大越容易被套（当天五只内部排名）",
         f"全部 {len(A)} 个特征两年 AUC 相关 {A['训练年AUC'].corr(A['最近一年AUC']):+.2f}；只看新+复合 {A[A.组 != '旧']['训练年AUC'].corr(A[A.组 != '旧']['最近一年AUC']):+.2f}",
         "\n== 新特征 + 复合特征（按两年合计强度排序）==", A[A.组 != "旧"].sort_values("强度", ascending=False).drop(columns="强度").round(3).to_string(index=False),
         "\n== 稳定的（两年同向且都偏离 0.5 超过 0.04）==", A[A["稳定"]].sort_values("强度", ascending=False).drop(columns=["强度", "训练年占比"]).round(3).to_string(index=False)]

# combined score, fit on train year only
sel = A[(A["训练年AUC"] - .5).abs() > .03]
sc = pd.concat([R[c] if a > .5 else 1 - R[c] for c, a in zip(sel["特征"], sel["训练年AUC"])], axis=1).mean(1)  # missing features skipped
X = R[allc].fillna(.5).to_numpy() - .5
w = np.zeros(X.shape[1]); b = 0.
yt = y.to_numpy().astype(float)
for _ in range(3000):  # L2 logistic regression, gradient descent, train year only
    p = 1 / (1 + np.exp(-(X[tr] @ w + b)))
    w -= 0.5 * (X[tr].T @ (p - yt[tr]) / tr.sum() + 0.01 * w); b -= 0.5 * (p - yt[tr]).mean()
lr = pd.Series(X @ w, index=f.index)
lines += [f"\n== 多特征合成（只用训练年定方向/权重）==",
          f"平均排名打分（{len(sel)} 个特征）：训练年 AUC {auc(sc[tr], y[tr]):.3f}，最近一年 AUC {auc(sc[te], y[te]):.3f}",
          f"逻辑回归（{len(allc)} 个特征）：训练年 AUC {auc(lr[tr], y[tr]):.3f}，最近一年 AUC {auc(lr[te], y[te]):.3f}"]
for nm, s_ in (("平均排名打分", sc), ("逻辑回归", lr)):
    for half, m in (("训练年", tr), ("最近一年", te)):
        x = f[m].assign(s=s_[m].to_numpy())
        x["r"] = x.groupby("date")["s"].rank(method="first", ascending=False)
        res = []
        for K in (1, 2):
            dd = x.groupby("date").apply(lambda q: pd.Series(dict(all=q["结果_阶梯"].mean(), keep=q.loc[q["r"] > K, "结果_阶梯"].mean(),
                                                                trap=q.loc[q["r"] <= K, "被套"].mean())))
            dd = dd.dropna()
            diff = dd["keep"] - dd["all"]
            blk = diff.groupby(np.arange(len(diff)) // 20).mean()
            res.append(f"排除最高 {K} 只：剩下 {dd['keep'].mean():+.2%} vs 五只 {dd['all'].mean():+.2%}（差 {diff.mean():+.2%}，分块 t {blk.mean() / blk.std() * np.sqrt(len(blk)):.2f}），被排除的里被套 {dd['trap'].mean():.0%}")
        lines.append(f"  {nm} {half}（阶梯卖法）：" + "；".join(res))
lines.append(f"  参考：随便排除一只，被套比例 = 当年平均 {y[tr].mean():.0%} / {y[te].mean():.0%}")
lines.append("\n逻辑回归权重最大的 8 个（正 = 越大越容易被套）：\n" + pd.Series(w, index=allc).sort_values(key=abs, ascending=False).head(8).round(2).to_string())
open("research/blindtest/trap2.txt", "w").write("\n".join(lines))
print("\n".join(lines))
