"""Blind test kit (Jasper 2026-10-24): on ~20 past dates, the current method's top 5 (±1 month earnings, <= $180, score
then lowest RSI; filled up to 5 from the 7b calm post-earnings pool). Each question folder holds only data known on that date.
No answers are written; grade.py computes them later from the same data.
Usage: python research/blindtest/make.py OUT_DIR
"""
import shutil
import sys
from pathlib import Path

exec(open("research/fullflow/top5.py", encoding="utf-8").read().split("\nalld = ")[0])  # allp, d, cal, O, H, Cl, ev, cons

OUT = Path(sys.argv[1])
N_Q, GAP, SEED = 20, 25, 7
full = allp.groupby("date").size()
elig = [t for t in full[full == 5].index if pd.Timestamp("2023-01-01") <= t <= last]
rng = np.random.default_rng(SEED)
chosen = []
for t in rng.permutation(elig):
    t = pd.Timestamp(t)
    if all(abs(cal.searchsorted(t) - cal.searchsorted(c)) >= GAP for c in chosen):
        chosen.append(t)
    if len(chosen) == N_Q:
        break
chosen = sorted(chosen)

D = "research/factors/data"
pxa = pd.concat([pd.read_csv(f"{D}/prices.csv.gz", parse_dates=["Date"]), pd.read_csv(f"{D}/prices_removed.csv.gz", parse_dates=["Date"])])
pxa = pxa.drop_duplicates(["Date", "Ticker"])
sc = pd.read_parquet("research/fullflow/data/scores.parquet")
COLS = {"sector": "板块", "price": "当天收盘价", "score": "财务分(10分制)", "RSI14": "RSI14", "离20日线": "离20日线",
        "离50日线": "离50日线", "离200日线": "离200日线", "1月涨幅": "1月涨幅", "3月涨幅": "3月涨幅", "动量12-1": "动量12-1",
        "离52周高点": "离52周高点", "60日波动率": "60日波动率", "贝塔": "贝塔", "过去触及率(+3%,10日)": "过去1年10日内碰到+3%的比例",
        "margin": "净利润率", "de": "负债/权益", "roic": "ROIC", "cr": "流动比率", "int_cov": "利息覆盖倍数"}
if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)
index = []
for qn, t in enumerate(chosen, 1):
    qd = OUT / f"第{qn:02d}题_{t.date()}"
    qd.mkdir()
    five = allp[allp["date"] == t].sample(frac=1, random_state=qn)  # shuffled: order carries no information
    tk = five["ticker"].tolist()
    c = five[["ticker", *COLS]].rename(columns=COLS)
    c.insert(1, "来源", np.where(five["src"] == "main", "主规则", "补选：最近1个月刚发完财报、反应平稳"))
    c["上次财报日"] = five["edate"].dt.date.to_numpy()
    c["上次财报次日相对财报前涨跌"] = five["reaction"].round(4).to_numpy()
    c["交易日数_距上次财报"] = five["since"].to_numpy()
    c.round(4).to_csv(qd / "候选五只.csv", index=False, encoding="utf-8-sig")
    lo = cal[max(cal.searchsorted(t) - 260, 0)]
    p = pxa[pxa["Ticker"].isin(tk + ["SPY", "MDY"]) & (pxa["Date"] >= lo) & (pxa["Date"] <= t)]
    p.sort_values(["Ticker", "Date"]).round(4).to_csv(qd / "日线价格_近1年.csv", index=False)
    e = ev[ev["ticker"].isin(tk) & (ev["edate"] <= t)].groupby("ticker").tail(8)
    e.rename(columns={"edate": "财报日", "pre_close": "财报前收盘", "reaction": "次日相对财报前涨跌"}).round(4).to_csv(
        qd / "历史财报_最近8次.csv", index=False, encoding="utf-8-sig")
    f = sc[sc["ticker"].isin(tk) & (sc["filed"] <= t)].sort_values("filed").groupby("ticker").tail(8)
    f.round(4).to_csv(qd / "财报数据_SEC_最近8期.csv", index=False)
    index.append((qn, t.date(), ", ".join(sorted(tk))))
pd.DataFrame(index, columns=["题号", "日期", "五只（按字母）"]).to_csv(OUT / "题目列表.csv", index=False, encoding="utf-8-sig")
pd.DataFrame({"题号": [i[0] for i in index], "日期": [i[1] for i in index], "选择": "", "信心(1-5)": "", "理由（一句话）": ""}).to_csv(
    OUT / "答题卡.csv", index=False, encoding="utf-8-sig")
print("\n".join(f"{a} {b} {c}" for a, b, c in index))
