"""Stepped exit (Jasper 2026-10-24): calendar days 0-28 (trading days 1-20) wait for +3%; days 28-48 (trading days 21-34)
sell as soon as the price gets back to the buy price (0%); after day 48 (trading day 35 on) sell as soon as it gets back to
-3%; still unsold at trading day 48 (~68 calendar days) -> sell at the close. Compared with the current exit (+3% limit,
-3% from trading day 11, day-20 close). Picks = current full method (±1 month earnings, <= $180, lowest RSI, 7b fallback).
Usage: python research/fullflow/ladder.py
"""
exec(open("research/fullflow/by_year.py", encoding="utf-8").read().split("\nlines = [")[0])  # both, with_spy, seq, ...

LMAX = 48
STEPS = {"现行：+3%，第 11 天起 −3%，第 20 天卖": [(0, 0.03), (10, -0.03)],
         "阶梯：+3% → 第 21 天起 0% → 第 35 天起 −3%，第 48 天卖": [(0, 0.03), (20, 0.0), (34, -0.03)]}
LEN = {"现行：+3%，第 11 天起 −3%，第 20 天卖": 20, "阶梯：+3% → 第 21 天起 0% → 第 35 天起 −3%，第 48 天卖": LMAX}


def outcome_steps(rows, steps, L):
    """Sell at the first day whose high reaches the current level (at the open if it opens above). Level = last step <= t."""
    i = cal.searchsorted(rows["date"].to_numpy())
    j = O.columns.get_indexer(rows["ticker"])
    k = np.arange(1, L + 1)
    op, hi = O.to_numpy()[i[:, None] + k, j[:, None]], H.to_numpy()[i[:, None] + k, j[:, None]]
    entry = op[:, 0]
    out = Cl.to_numpy()[i + L, j] / entry - 1
    days = np.full(len(i), L)
    done = np.zeros(len(i), bool)
    for t in range(L):
        lv = entry * (1 + [s for d0, s in steps if d0 <= t][-1])
        hit = ~done & (hi[:, t] >= lv)
        px_ = np.maximum(lv, op[:, t]) if t else lv
        out[hit] = px_[hit] / entry[hit] - 1
        days[hit], done = t + 1, done | hit
    cost = np.where(rows["ticker"].map(cons["Universe"]).eq("SP500"), 0.0005, 0.001)
    return out - 2 * cost, days


def add(rows, name):
    rows = rows[cal.searchsorted(rows["date"].to_numpy()) + LMAX < len(cal)]  # same signals for both exits
    ret, days = outcome_steps(rows, STEPS[name], LEN[name])
    i = cal.searchsorted(rows["date"].to_numpy())
    spy = spyC.to_numpy()[i + days] / spyO.to_numpy()[i + 1] - 1
    return rows.assign(ret=ret, days=days, spy=spy)


last_ok = cal[len(cal) - 1 - LMAX]
PER = {"最近 1 年": (last_ok - pd.DateOffset(years=1), last_ok), "2023": ("2023-01-01", "2023-12-31"),
       "2024": ("2024-01-01", "2024-12-31"), "2025": ("2025-01-01", "2025-12-31")}
lines = [f"信号截至 {last_ok.date()}（要留 48 个交易日看结果）。格式：每笔 / 同样日期 SPY / 亏>5% / 平均持有交易日 / 每 20 个交易日折算 | $180 连续做（笔数）"]
for name in STEPS:
    x_all = add(both, name)
    lines.append(f"\n== {name} ==")
    for p, (a, b) in PER.items():
        x = x_all[(x_all["date"] >= pd.Timestamp(a)) & (x_all["date"] <= pd.Timestamp(b))]
        s = seq(x, pd.Timestamp(a))
        lines.append(f"  {p}: {x['ret'].mean():+.2%} / SPY {x['spy'].mean():+.2%} / {(x['ret'] < -0.05).mean():.0%} / "
                     f"{x['days'].mean():.1f} 天 / {x['ret'].mean() / x['days'].mean() * 20:+.2%} | $180 {s[0]:+.1%}（{s[1]} 笔，赚 {s[2]}）")
    late = x_all[x_all["days"] > 20]
    lines.append(f"  拖过第 20 天的比例 {len(late) / len(x_all):.0%}，这些每笔 {late['ret'].mean():+.2%}，平均持有 {late['days'].mean():.1f} 天")
lines.append("\n一直拿 SPY")
for p, (a, b) in PER.items():
    s0 = spyO[spyO.index >= pd.Timestamp(a)].iloc[0]
    e = min(pd.Timestamp(b), last_ok)
    lines.append(f"  {p}: {spyC[:e].iloc[-1] / s0 - 1:+.1%}（{pd.Timestamp(a).date()} 到 {e.date()}）")
open("research/fullflow/ladder.txt", "w").write("\n".join(lines))
print("\n".join(lines))
