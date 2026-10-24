"""Walk-forward strategy selection (pre-registered in 回测/滚动选策略_预登记.md): every H months pick, among candidates A-F and
SPY, the one with the best per-trade average over the past L months (only trades already closed by the decision date), use it
for the next H months. L in 1,2,3,6,12 months; H in 1,2,3 months; evaluation 2023-01 to 2026-09.
Usage: python research/fullflow/walk.py
"""
exec(open("research/fullflow/by_year.py", encoding="utf-8").read().split("\nlines = [")[0])  # d, picks, both, primary, seq, ...


def res(rows, stop):
    rows = rows[["date", "ticker"]].copy()
    ret, days = outcome(rows, stop)
    i = cal.searchsorted(rows["date"].to_numpy())
    rows["ret"], rows["days"] = ret, days
    rows["exit"] = cal[i + days]
    rows["spy"] = spyC.to_numpy()[i + days] / spyO.to_numpy()[i + 1] - 1
    return rows.set_index("date")


ok = d["ret_3%"].notna() & (d["price"] <= 180)
grp = d[d["full"] & ok].copy()
grp["touch"], grp["RSI_neg"] = grp["过去触及率(+3%,10日)"], -grp["RSI14"]
old = lambda k: grp.assign(s=grp["score"] * 10 + grp[k]).sort_values("s", ascending=False).groupby("date").head(1)
C = {"A": res(picks(None, 180, "touch"), None), "B": res(picks(None, 180, "RSI_neg"), None),
     "C": res(both, -0.03), "D": res(picks(d["since"] <= 21, 180, "touch"), -0.03),
     "E": res(old("touch"), None), "F": res(old("RSI_neg"), None)}
dates = cal[(cal >= d["date"].min()) & (cal <= last)]
i = cal.searchsorted(dates)
C["S"] = pd.DataFrame({"ticker": "SPY", "ret": spyC.to_numpy()[i + 20] / spyO.to_numpy()[i + 1] - 1 - 0.0002, "days": 20,
                       "exit": cal[i + 20], "spy": spyC.to_numpy()[i + 20] / spyO.to_numpy()[i + 1] - 1}, index=dates)
import sys
NOS = "--no-spy" in sys.argv     # post-hoc: candidates A-F only
PERDAY = "--per-day" in sys.argv  # post-hoc: score and evaluate by return per trading day held (sum ret / sum days * 20)
if NOS:
    C.pop("S")
NAMES = list(C)
avg = (lambda x: x["ret"].sum() / x["days"].sum() * 20) if PERDAY else (lambda x: x["ret"].mean())
E0, E1 = pd.Timestamp("2023-01-01"), last


def seq2(x, t0):
    """$180 one at a time; whole shares for stocks, fractional for SPY."""
    cap, k, n = 180.0, cal.searchsorted(t0), 0
    while k < len(cal) and cal[k] <= E1:
        t = cal[k]
        if t in x.index:
            r = x.loc[t]
            p = O.at[cal[k + 1], r["ticker"]]
            sh = cap / p if r["ticker"] == "SPY" else np.floor(cap / p)
            if sh > 0:
                cap += sh * p * r["ret"]
                n += 1
                k += int(r["days"])
                continue
        k += 1
    return cap / 180 - 1, n


def run(choice):
    """choice: list of (start, end, name) -> combined per-day picks."""
    parts = [C[nm][(C[nm].index >= a) & (C[nm].index < b)].assign(src=nm) for a, b, nm in choice]
    return pd.concat(parts).sort_index()


def schedule(L, H, chooser):
    out, t = [], E0
    while t <= E1:
        t1 = t + pd.DateOffset(months=H)
        out.append((t, t1, chooser(t, L)))
        t = t1
    return out


def best(t, L):
    lo = t - pd.DateOffset(months=L)
    sc = {nm: (avg(x[(x.index >= lo) & (x["exit"] < t)]) if ((x.index >= lo) & (x["exit"] < t)).any() else np.nan) for nm, x in C.items()}
    return max(sc, key=lambda k: -np.inf if np.isnan(sc[k]) else sc[k])


def summ(x):
    y = {yr: avg(x[x.index.year == yr]) for yr in (2023, 2024, 2025, 2026)}
    sp = x["spy"].sum() / x["days"].sum() * 20 if PERDAY else x["spy"].mean()
    return avg(x), sp, y, seq2(x, E0)


rng = np.random.default_rng(1)
lines = [("[事后补充] 不含 SPY 候选；" if NOS else "") + ("[事后补充] 按每 20 个交易日折算收益打分和评估；" if PERDAY else "") + f"评估 {E0.date()} 至 {E1.date()}，每笔平均扣成本；SPY 同日 = 同样买卖日期拿 SPY；$180 连续做（股票整股，SPY 碎股）"]
fmt = lambda s: (f"每笔 {s[0]:+.2%}（SPY 同日 {s[1]:+.2%}）| 2023 {s[2][2023]:+.2%} 2024 {s[2][2024]:+.2%} 2025 {s[2][2025]:+.2%} 2026 {s[2][2026]:+.2%}"
                 f" | $180 {s[3][0]:+.1%}（{s[3][1]} 笔）")
lines.append("\n== 一直用同一个 ==")
static = {}
for nm in NAMES:
    s = summ(run([(E0, E1 + pd.Timedelta(days=1), nm)]))
    static[nm] = s[0]
    lines.append(f"  {nm}: {fmt(s)}")
spy_bh = spyC[:E1].iloc[-1] / spyO[spyO.index >= E0].iloc[0] - 1
lines.append(f"  一直拿 SPY（买入持有）: {spy_bh:+.1%}")

lines.append("\n== 按近期挑（L = 回看月数，H = 往后用几个月）==")
wins = 0
grid = []
for H in (1, 2, 3):
    rnd = []
    for _ in range(200):
        sch = schedule(1, H, lambda t, L: NAMES[rng.integers(len(NAMES))])
        rnd.append(avg(run(sch)))
    rmed = np.median(rnd)
    lines.append(f"  H={H}：每次随机选一个，每笔中位 {rmed:+.2%}（5%–95% {np.percentile(rnd, 5):+.2%} ~ {np.percentile(rnd, 95):+.2%}）")
    for L in (1, 2, 3, 6, 12):
        sch = schedule(L, H, best)
        x = run(sch)
        s = summ(x)
        picked = pd.Series([c for _, _, c in sch]).value_counts().to_dict()
        good = s[0] > static["A"] and s[0] > rmed and s[0] > s[1]
        wins += good
        grid.append((L, H, s[0]))
        lines.append(f"    L={L:>2} H={H}: {fmt(s)} | 选中次数 {picked} {'✓' if good else ''}")
lines.append(f"\n判定：{wins}/15 个组合同时超过“一直用 A”、随机选中位数和同日 SPY（预登记门槛 ≥ 10）")
open("research/fullflow/walk" + ("_nospy" if NOS else "") + ("_perday" if PERDAY else "") + ".txt", "w").write("\n".join(lines))
print("\n".join(lines))
