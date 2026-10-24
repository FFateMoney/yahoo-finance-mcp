"""The full current method (process doc 2026-10-24: R1, ±1 month earnings rule, <= $180, ties by lowest RSI, +3% limit,
-3% exit from day 11, day-20 close, plus the 7b fallback to calm post-earnings stocks) by calendar year 2023 / 2024 / 2025.
Compared with all stocks under the same exits, SPY over the same days, buy-and-hold SPY, and random picks ($180 runs).
Usage: python research/fullflow/by_year.py   (needs flow_panel_daily.parquet from `explore.py 1 2021-11-01` + `flow.py --daily`)
"""
exec(open("research/fullflow/full_set.py", encoding="utf-8").read().split("lines = [")[0])  # with_spy, seq, picks, d, cal

YEARS = [2023, 2024, 2025]
recent = d["since"] <= 21
primary = picks(recent, 180, "RSI_neg")
fallback = picks(~recent | ~(d["reaction"].abs() <= 0.05), 180, "RSI_neg")
both = pd.concat([primary, fallback[~fallback["date"].isin(primary["date"])]]).sort_values("date")
touch = picks(recent, 180, "touch")
rng = np.random.default_rng(0)


def year_rows(x, y):
    return x[x["date"].dt.year == y]


def seq_year(x, y):
    """$180 run inside calendar year y (positions opened in y; the last one may close in early y+1)."""
    t0 = pd.Timestamp(f"{y}-01-01")
    x = year_rows(x, y)
    return seq(x, t0) if len(x) else (np.nan, 0, 0)


lines = ["每笔平均扣成本；SPY 同日 = 同样买卖日期拿 SPY；$180 连续做 = 一次一只、整股、复利，只在当年开仓"]
alld = d[d["ret_3%"].notna() & (d["price"] <= 180)][["date", "ticker"]]
for y in YEARS:
    lines.append(f"\n== {y} ==")
    spy_y = spyC[spyC.index.year == y].iloc[-1] / spyO[spyO.index.year == y].iloc[0] - 1
    for name, tp in [("现行方法（含空缺补选）", both), ("现行方法（空仓就等）", primary), ("文档原排序（空仓就等）", touch)]:
        x = with_spy(year_rows(tp, y))
        s = seq_year(x, y)
        cov = x["date"].nunique() / d.loc[d["date"].dt.year == y, "date"].nunique()
        lines.append(f"  {name}: 每笔 {x['ret'].mean():+.2%}（SPY 同日 {x['spy'].mean():+.2%}，亏>5% {(x['ret'] < -0.05).mean():.0%}，"
                     f"有股票的天 {cov:.0%}）| $180 连续做 {s[0]:+.1%}，{s[1]} 笔赚 {s[2]}")
    al = with_spy(year_rows(alld, y))
    lines.append(f"  全部股票平均（≤$180，同样卖法）: 每笔 {al.groupby('date')['ret'].mean().mean():+.2%}（SPY 同日 {al.groupby('date')['spy'].mean().mean():+.2%}）")
    g = al.groupby("date")
    sims = []
    for _ in range(200):
        r = g.sample(1, random_state=int(rng.integers(1 << 31)))
        sims.append(seq_year(r, y)[0])
    lines.append(f"  全部股票里随便挑一只，$180 连续做（200 次）: 中位 {np.median(sims):+.1%}，5%–95% {np.percentile(sims, 5):+.1%} ~ {np.percentile(sims, 95):+.1%}")
    lines.append(f"  一直拿 SPY（全年）: {spy_y:+.1%}")
open("research/fullflow/by_year.txt", "w").write("\n".join(lines))
print("\n".join(lines))
