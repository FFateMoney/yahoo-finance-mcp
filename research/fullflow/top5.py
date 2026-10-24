"""Top 5 of the current method (±1 month earnings, <= $180, ties by lowest RSI; days with fewer than 5 are filled from the
7b calm post-earnings pool). How much could a perfect / random / worst choice among the 5 add vs the #1 pick?
Exits: +3% limit, -3% from day 11, day-20 close. Last 1 year and 2023 / 2024 / 2025.
Usage: python research/fullflow/top5.py
"""
exec(open("research/fullflow/by_year.py", encoding="utf-8").read().split("\nlines = [")[0])  # d, with_spy, cal, ...


def topn(skip, n):
    m = d["base"] & ~skip & (d["price"] <= 180)
    q = d[m].copy()
    q["s"] = q["score"] * 10 + q["RSI_neg"]
    return q.sort_values("s", ascending=False).groupby("date").head(n)


recent = d["since"] <= 21
p1 = topn(recent, 5).assign(src="main")
fb = topn(~recent | ~(d["reaction"].abs() <= 0.05), 5).assign(src="fallback")
fb = fb[~fb.set_index(["date", "ticker"]).index.isin(p1.set_index(["date", "ticker"]).index)]
allp = pd.concat([p1, fb])
allp["rank"] = allp.groupby("date").cumcount() + (allp["src"] == "fallback") * 100
allp = allp.sort_values(["date", "rank"]).groupby("date").head(5)
x = with_spy(allp[["date", "ticker"]].reset_index(drop=True)).assign(src=allp["src"].to_numpy())
x["pos"] = x.groupby("date").cumcount() + 1
alld = with_spy(d[d["ret_3%"].notna() & (d["price"] <= 180)][["date", "ticker"]])
PER = {"最近 1 年": (last - pd.DateOffset(years=1), last), "2023": ("2023-01-01", "2023-12-31"),
       "2024": ("2024-01-01", "2024-12-31"), "2025": ("2025-01-01", "2025-12-31")}
lines = [f"每笔平均，扣成本，信号截至 {last.date()}"]
for p, (a, b) in PER.items():
    y = x[(x["date"] >= pd.Timestamp(a)) & (x["date"] <= pd.Timestamp(b))]
    g = y.groupby("date")["ret"]
    n5 = g.size()
    full = n5[n5 == 5].index
    z = y[y["date"].isin(full)].groupby("date")["ret"]
    al = alld[(alld["date"] >= pd.Timestamp(a)) & (alld["date"] <= pd.Timestamp(b))]
    lines.append(f"\n== {p} ==（有 5 只的天 {len(full) / max(len(n5), 1):.0%}，补选占 {(y['src'] == 'fallback').mean():.0%}）")
    lines.append("  按名次每笔: " + "，".join(f"第{k}名 {y.loc[y['pos'] == k, 'ret'].mean():+.2%}" for k in range(1, 6)))
    lines.append(f"  五只里：最好那只 {z.max().mean():+.2%}，随便挑一只（=五只平均）{z.mean().mean():+.2%}，最差那只 {z.min().mean():+.2%}")
    lines.append(f"  五只里能赚钱的比例 {(y['ret'] > 0).mean():.0%}；五只全亏的天 {(z.max() <= 0).mean():.0%}")
    lines.append(f"  同样日期拿 SPY {y['spy'].mean():+.2%}；全部股票平均 {al.groupby('date')['ret'].mean().mean():+.2%}")
open("research/fullflow/top5.txt", "w").write("\n".join(lines))
print("\n".join(lines))
