"""The 10-17 process (when MNST/FAST were picked) by year: avoid earnings in the next 28 days, score >= 8, ex IT/communication,
rising 200-day, trend (above 20/50-day, RSI 40-70) or pullback (below 20-day, RSI 30-50); buy the highest score (ties by past
+3% touch rate, or by lowest RSI), price <= $180. Old exit: +3% limit, re-posted once, sell at the day-20 close.
Compared with the whole group's average, all stocks, SPY over the same days, buy-and-hold SPY and random picks ($180 runs).
Usage: python research/fullflow/old_by_year.py
"""
exec(open("research/fullflow/by_year.py", encoding="utf-8").read().split("\nlines = [")[0])  # d, seq, cal, spyO, spyC


def with_spy_old(rows):
    ret, days = outcome(rows, None)
    i = cal.searchsorted(rows["date"].to_numpy())
    spy = spyC.to_numpy()[i + days] / spyO.to_numpy()[i + 1] - 1
    return rows.assign(ret=ret, days=days, spy=spy)


ok = d["ret_3%"].notna() & (d["price"] <= 180)
grp = d[d["full"] & ok].copy()
grp["touch"], grp["RSI_neg"] = grp["过去触及率(+3%,10日)"], -grp["RSI14"]
top = {k: grp.assign(s=grp["score"] * 10 + grp[k]).sort_values("s", ascending=False).groupby("date").head(1) for k in ("touch", "RSI_neg")}
alld = d[ok][["date", "ticker"]]
PER = {"2022": ("2022-01-01", "2022-12-31"), "2023": ("2023-01-01", "2023-12-31"), "2024": ("2024-01-01", "2024-12-31"),
       "2025": ("2025-01-01", "2025-12-31"), "最近 1 年": (str((last - pd.DateOffset(years=1)).date()), str(last.date()))}
lines = [f"信号截至 {last.date()}。旧卖法：+3% 限价，续挂一次，第 20 个交易日收盘卖。每笔扣成本"]
for p, (a, b) in PER.items():
    a, b = pd.Timestamp(a), pd.Timestamp(b)
    cut = lambda x: x[(x["date"] >= a) & (x["date"] <= b)]
    lines.append(f"\n== {p} ==")
    for name, tp in [("旧流程第 1 名（财务分最高，同分看碰到 +3% 比例）", top["touch"]), ("旧流程第 1 名（财务分最高，同分按 RSI 最低）", top["RSI_neg"])]:
        x = with_spy_old(cut(tp))
        s = seq(x, a)
        lines.append(f"  {name}: 每笔 {x['ret'].mean():+.2%}（SPY 同日 {x['spy'].mean():+.2%}）| $180 连续做 {s[0]:+.1%}，{s[1]} 笔赚 {s[2]}")
    g = with_spy_old(cut(grp[["date", "ticker"]]))
    lines.append(f"  旧流程整组平均: 每笔 {g.groupby('date')['ret'].mean().mean():+.2%}（SPY 同日 {g.groupby('date')['spy'].mean().mean():+.2%}），每天约 {g.groupby('date').size().mean():.0f} 只")
    sims = [seq(with_spy_old(cut(grp).groupby("date").sample(1, random_state=int(rng.integers(1 << 31)))[["date", "ticker"]]), a)[0] for _ in range(100)]
    lines.append(f"  旧流程里随便挑一只，$180（100 次）: 中位 {np.median(sims):+.1%}，5%–95% {np.percentile(sims, 5):+.1%} ~ {np.percentile(sims, 95):+.1%}")
    al = with_spy_old(cut(alld))
    lines.append(f"  全部股票平均: 每笔 {al.groupby('date')['ret'].mean().mean():+.2%}")
    lines.append(f"  一直拿 SPY: {spyC[:b].iloc[-1] / spyO[spyO.index >= a].iloc[0] - 1:+.1%}")
open("research/fullflow/old_by_year.txt", "w").write("\n".join(lines))
print("\n".join(lines))
