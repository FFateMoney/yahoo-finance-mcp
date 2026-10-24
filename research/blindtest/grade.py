"""Grade the blind test: python research/blindtest/grade.py ANSWER_CSV [QUESTION_DIR]
For each question: return of the chosen stock (buy next open; +3% limit, -3% from trading day 11, day-20 close; costs),
the five-stock average, best / worst of the five, SPY over the same days. Summary: chosen minus five-average (paired t).
"""
import sys

exec(open("research/fullflow/top5.py", encoding="utf-8").read().split("\nalld = ")[0])  # allp, with_spy, cal

ans = pd.read_csv(sys.argv[1], encoding="utf-8-sig", dtype=str).fillna("")
rows = []
for r in ans.itertuples(index=False):
    t = pd.Timestamp(r[1])
    five = with_spy(allp[allp["date"] == t][["date", "ticker"]].reset_index(drop=True))
    pick = str(r[2]).strip().upper()
    if pick not in set(five["ticker"]):
        print(f"第{r[0]}题 {t.date()}: 选择 '{pick}' 不在五只里，跳过")
        continue
    x = five.set_index("ticker")
    rows.append({"题号": r[0], "日期": t.date(), "选择": pick, "信心": r[3], "选中": x.at[pick, "ret"], "五只平均": x["ret"].mean(),
                 "最好": x["ret"].max(), "最好是": x["ret"].idxmax(), "最差": x["ret"].min(), "SPY同日": x.at[pick, "spy"],
                 "名次(1=最好)": int(x["ret"].rank(ascending=False)[pick]),
                 "五只结果": " ".join(f"{k}{v:+.1%}" for k, v in x["ret"].sort_values(ascending=False).items())})
g = pd.DataFrame(rows)
pd.set_option("display.width", 250)
fmt = g.copy()
for k in ("选中", "五只平均", "最好", "最差", "SPY同日"):
    fmt[k] = fmt[k].map(lambda v: f"{v:+.2%}")
print(fmt.to_string(index=False))
diff = g["选中"] - g["五只平均"]
n = len(g)
print(f"\n共 {n} 题。选中每笔 {g['选中'].mean():+.2%}，五只平均 {g['五只平均'].mean():+.2%}，同日 SPY {g['SPY同日'].mean():+.2%}")
print(f"选中 − 五只平均：{diff.mean():+.2%}（t {diff.mean() / diff.std(ddof=1) * np.sqrt(n):+.2f}），{(diff > 0).sum()}/{n} 题好于平均")
print(f"名次分布（随便挑时每个名次各约 20%）：" + "，".join(f"第{k}名 {(g['名次(1=最好)'] == k).sum()} 次" for k in range(1, 6)))
print(f"20 笔连续复利：选中 {np.prod(1 + g['选中']) - 1:+.1%}，五只平均 {np.prod(1 + g['五只平均']) - 1:+.1%}，同日 SPY {np.prod(1 + g['SPY同日']) - 1:+.1%}")
