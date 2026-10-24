"""Pre-registered avoid rules (排雷/排雷规则_预登记.md): exclude the 1 or 2 riskiest of the five by one feature.
python research/blindtest/avoid_rules.py TRAIN_CSV            -> train-year table and the selected rule
python research/blindtest/avoid_rules.py FEATURES_CSV --emit RULE K OUT_CSV  -> exclusion list (date,ticker) for grading
"""
import sys

import numpy as np
import pandas as pd

# code: (column, +1 = exclude the highest, -1 = exclude the lowest)
RULES = {"X1": ("上次财报次日反应", -1), "X2": ("21日最大单日跌幅", -1), "X3": ("贝塔", 1), "X4": ("60日波动率", 1),
         "X5": ("残差波动率", 1), "X6": ("离200日线", -1), "X7": ("离50日线", -1), "X8": ("动量12-1", -1),
         "X9": ("离52周高点", -1), "X10": ("相对行业1月", -1), "X11": ("de", 1), "X12": ("altman_z", -1)}


def excluded(df, rule, k):
    col, sgn = RULES[rule]
    risk = df[col] * sgn
    rk = risk.groupby(df["date"]).rank(ascending=False, method="first")
    return (rk <= k) & risk.notna()


def block_t(diff):
    b = diff.groupby(np.arange(len(diff)) // 20).mean()
    return b.mean() / b.std(ddof=1) * np.sqrt(len(b)), len(b)


df = pd.read_csv(sys.argv[1], encoding="utf-8-sig")
if "--emit" in sys.argv:
    i = sys.argv.index("--emit")
    rule, k, out = sys.argv[i + 1], int(sys.argv[i + 2]), sys.argv[i + 3]
    df[excluded(df, rule, k)][["date", "ticker"]].to_csv(out, index=False)
    sys.exit()

dates = sorted(df["date"].unique())
half = dates[len(dates) // 2]
res = []
for lab in ("阶梯", "现行"):
    r = f"结果_{lab}"
    five = df.groupby("date")[r].mean()
    for rule, (col, sgn) in RULES.items():
        for k in (1, 2):
            ex = excluded(df, rule, k)
            rem = df[~ex].groupby("date")[r].mean()
            diff = (rem - five).reindex(five.index)
            t, nb = block_t(diff)
            h1, h2 = diff[diff.index < half].mean(), diff[diff.index >= half].mean()
            hit = df.loc[ex, f"当天最差2只_{lab}"].mean()
            res.append(dict(卖法=lab, 规则=rule, 特征=col, 排除几只=k, 剩下减五只=diff.mean(), t=t, 前半年=h1, 后半年=h2,
                            排除的是最差2只=hit, 排除的平均=df.loc[ex, r].mean()))
res = pd.DataFrame(res)
pd.set_option("display.width", 250)
for lab in ("阶梯", "现行"):
    x = res[res["卖法"] == lab].sort_values("t", ascending=False).copy()
    for c in ("剩下减五只", "前半年", "后半年", "排除的平均"):
        x[c] = x[c].map(lambda v: f"{v:+.2%}")
    x["排除的是最差2只"] = x["排除的是最差2只"].map(lambda v: f"{v:.0%}")
    x["t"] = x["t"].round(2)
    print(f"\n== 训练年，{lab}卖法（五只平均 {df[f'结果_{lab}'].mean():+.2%}；分块 {nb} 块）==")
    print(x.drop(columns="卖法").to_string(index=False))
m = res[(res["卖法"] == "阶梯") & (res["前半年"] > 0) & (res["后半年"] > 0)]
if len(m):
    b = m.sort_values("t", ascending=False).iloc[0]
    print(f"\n按预登记选出：{b['规则']}（{b['特征']}），排除 {b['排除几只']} 只，训练年 t {b['t']:.2f}")
else:
    print("\n没有一条在前后半年都为正：训练年找不到排雷规则")
