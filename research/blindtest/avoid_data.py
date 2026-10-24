"""Dataset for "find the worst 1-2 of the five" (Jasper 2026-10-24): every trading day of the last two years, the current
method's top 5 (score then lowest RSI, filled from the calm post-earnings pool), with every feature known on that day.
Outcomes under the stepped exit (+3% to day 20, 0% days 21-34, -3% days 35-48, day-48 close) and the current exit
(+3%, -3% from day 11, day-20 close). Train year: outcomes included. Test year: features only; outcomes are recomputed by
avoid_grade.py and never written to the shared folder.
Usage: python research/blindtest/avoid_data.py OUT_DIR [--answers PATH]   (--answers: also write test-year outcomes, for grading only)
"""
import sys
from pathlib import Path

exec(open("research/fullflow/ladder.py", encoding="utf-8").read().split("\nlast_ok =")[0])  # d, outcome_steps, STEPS, cal

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
recent = d["since"] <= 21


def topn(skip, n):
    m = d["base"] & ~skip & (d["price"] <= 180)
    q = d[m].copy()
    q["s"] = q["score"] * 10 + q["RSI_neg"]
    return q.sort_values("s", ascending=False).groupby("date").head(n)


p1 = topn(recent, 5).assign(来源="主规则")
fb = topn(~recent | ~(d["reaction"].abs() <= 0.05) | (d["since"] < 1), 5).assign(来源="补选")
fb = fb[~fb.set_index(["date", "ticker"]).index.isin(p1.set_index(["date", "ticker"]).index)]
allp = pd.concat([p1, fb])
allp["k"] = allp.groupby("date").cumcount() + (allp["来源"] == "补选") * 100
allp = allp.sort_values(["date", "k"]).groupby("date").head(5)
n5 = allp.groupby("date").size()
allp = allp[allp["date"].isin(n5[n5 == 5].index)]

END = cal[len(cal) - 1 - LMAX]
MID = END - pd.DateOffset(years=1)
START = END - pd.DateOffset(years=2)
allp = allp[(allp["date"] > START) & (allp["date"] <= END)].reset_index(drop=True)

drop = [c for c in allp.columns if c.startswith(("ret_", "ex_", "hit_", "days_", "n_"))] + ["i", "tk", "k", "s", "base", "full",
        "next_earn", "no_earn", "earn_data", "below_pre", "touch", "RSI_neg", "pre_close"]
feat = allp.drop(columns=[c for c in drop if c in allp.columns])
feat["date"] = feat["date"].dt.date
feat["edate"] = pd.to_datetime(feat["edate"]).dt.date
feat = feat.rename(columns={"edate": "上次财报日", "reaction": "上次财报次日反应", "since": "距上次财报交易日"})

rows = allp[["date", "ticker"]]
names = {"阶梯": [k for k in STEPS if k.startswith("阶梯")][0], "现行": [k for k in STEPS if k.startswith("现行")][0]}
for lab, nm in names.items():
    r, dd = outcome_steps(rows, STEPS[nm], LEN[nm])
    feat[f"结果_{lab}"] = r.round(5)
    feat[f"持有天数_{lab}"] = dd
    rk = feat.groupby("date")[f"结果_{lab}"].rank(method="first")
    feat[f"当天最差1只_{lab}"] = (rk == 1).astype(int)
    feat[f"当天最差2只_{lab}"] = (rk <= 2).astype(int)
    feat[f"亏钱_{lab}"] = (feat[f"结果_{lab}"] <= 0).astype(int)
is_train = pd.to_datetime(feat["date"]) <= MID
out_cols = [c for c in feat.columns if c.startswith(("结果_", "持有天数_", "当天最差", "亏钱_"))]
feat[is_train].to_csv(OUT / "训练年_带结果.csv", index=False, encoding="utf-8-sig")
feat[~is_train].drop(columns=out_cols).to_csv(OUT / "测试年_只有特征.csv", index=False, encoding="utf-8-sig")
if "--answers" in sys.argv:
    feat[~is_train][["date", "ticker", *out_cols]].to_csv(sys.argv[sys.argv.index("--answers") + 1], index=False, encoding="utf-8-sig")
tr, te = feat[is_train], feat[~is_train]
print(f"训练年 {tr['date'].min()} ~ {tr['date'].max()}：{tr['date'].nunique()} 天，{len(tr)} 行")
print(f"测试年 {te['date'].min()} ~ {te['date'].max()}：{te['date'].nunique()} 天，{len(te)} 行")
for lab in names:
    print(lab, f"训练年 每笔 {tr[f'结果_{lab}'].mean():+.2%}，亏钱比例 {tr[f'亏钱_{lab}'].mean():.0%}，独立五只组合数（按 20 交易日不重叠）约 {tr['date'].nunique() // 20}")
print("列：", list(feat.columns))
