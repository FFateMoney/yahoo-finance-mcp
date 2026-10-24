"""Full method + fallback: on days with no pick, choose among stocks that reported in the last ~1 month (21 trading days)
with a calm reaction (close the day after vs the close before within ±5%), all other R1 conditions unchanged
(no earnings in the next 28 days, score >= 8, ex IT/communication, rising 200-day, below 20-day, RSI 30-50, <= $180).
Ties by lowest RSI. Exits: +3% limit, -3% from day 11, day-20 close. Last 1 year / 3 months / 1 month vs all stocks and SPY.
Usage: python research/fullflow/fallback.py
"""
exec(open("research/fullflow/full_set.py", encoding="utf-8").read().split("lines = [")[0])  # with_spy, seq, picks, d, WIN

recent = d["since"] <= 21
primary = picks(recent, 180, "RSI_neg")
fb_skip = ~recent | ~(d["reaction"].abs() <= 0.05) | (d["since"] < 1)  # since >= 1: the reaction (next-day close) is known
fallback = picks(fb_skip, 180, "RSI_neg")
both = pd.concat([primary, fallback[~fallback["date"].isin(primary["date"])]]).sort_values("date")

lines = [f"信号截至 {last.date()}。每笔平均，扣成本；SPY = 同样买卖日期拿 SPY；$180 一次一只连续做"]
al = with_spy(d[(d["date"] >= WIN["最近 1 年"]) & d["ret_3%"].notna()][["date", "ticker"]])
for name, tp in [("原方法（空仓就等）", primary), ("原方法 + 空缺时选财报平稳的", both), ("只看补上的那些", fallback[~fallback["date"].isin(primary["date"])])]:
    x1 = with_spy(tp[tp["date"] >= WIN["最近 1 年"]])
    lines.append(f"\n{name}")
    for w, t0 in WIN.items():
        x = x1[x1["date"] >= t0]
        s = seq(x, t0) if len(x) else (np.nan, 0, 0)
        days_cov = x["date"].nunique() / d.loc[d["date"] >= t0, "date"].nunique()
        lines.append(f"  {w}: 每笔 {x['ret'].mean():+.2%}（SPY {x['spy'].mean():+.2%}，亏>5% {(x['ret'] < -0.05).mean():.0%}，有股票的天 {days_cov:.0%}）"
                     f" | $180 连续做 {s[0]:+.1%}，{s[1]} 笔赚 {s[2]}")
lines.append("\n全部股票平均（同样卖法）")
for w, t0 in WIN.items():
    x = al[al["date"] >= t0]
    lines.append(f"  {w}: 每笔 {x.groupby('date')['ret'].mean().mean():+.2%}；一直拿 SPY "
                 f"{spyC[last + pd.Timedelta(days=28):].iloc[0] / spyO[spyO.index >= t0].iloc[0] - 1:+.1%}")
open("research/fullflow/fallback.txt", "w").write("\n".join(lines))
print("\n".join(lines))
