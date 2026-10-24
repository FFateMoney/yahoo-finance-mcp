"""$180 one-at-a-time runs with fractional shares (Jasper 2026-10-24: buys like 1.23 shares are allowed; >= 1 share can carry a
limit order, so all cash is invested as long as the price <= capital). Variant 'S&P 500 only fractional': Schwab Stock Slices
covers S&P 500 companies, so S&P 400 picks are whole shares. Versions: current method (C), early R1 (A, B); vs buy-and-hold SPY.
Usage: python research/fullflow/frac.py
"""
exec(open("research/fullflow/walk.py", encoding="utf-8").read().split("\nrng = np.random")[0])  # C (candidates), cal, O, cons


def seq3(x, t0, t1, mode):
    cap, k, n, w = 180.0, cal.searchsorted(t0), 0, 0
    while k < len(cal) and cal[k] <= t1:
        t = cal[k]
        if t in x.index:
            r = x.loc[t]
            p = O.at[cal[k + 1], r["ticker"]]
            frac = mode == "all" or (mode == "sp500" and cons["Universe"].get(r["ticker"]) == "SP500")
            sh = cap / p if frac else np.floor(cap / p)
            if sh >= 1:
                cap += sh * p * r["ret"]
                n, w = n + 1, w + (r["ret"] > 0)
                k += int(r["days"])
                continue
        k += 1
    return cap / 180 - 1, n, w


PER = {"最近 1 年": (last - pd.DateOffset(years=1), last), "2023": (pd.Timestamp("2023-01-01"), pd.Timestamp("2023-12-31")),
       "2024": (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31")), "2025": (pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31"))}
lines = [f"$180 一次一只、复利。信号截至 {last.date()}"]
for nm, label in [("C", "现行方法"), ("A", "早期 R1（原排序）"), ("B", "早期 R1（同分按 RSI 最低）")]:
    lines.append(f"\n== {label} ==")
    for p, (a, b) in PER.items():
        x = C[nm][(C[nm].index >= a) & (C[nm].index <= b)]
        r = {m: seq3(x, a, b, m) for m in ("whole", "sp500", "all")}
        starts = [seq3(x, cal[cal.searchsorted(a) + s], b, "all")[0] for s in range(0, 20, 2)]
        lines.append(f"  {p}: 整股 {r['whole'][0]:+.1%} | 只有标普 500 能碎股 {r['sp500'][0]:+.1%} | 全部可碎股 {r['all'][0]:+.1%}"
                     f"（{r['all'][1]} 笔赚 {r['all'][2]}；换起步日 {min(starts):+.1%} ~ {max(starts):+.1%}）")
lines.append("\n一直拿 SPY")
for p, (a, b) in PER.items():
    lines.append(f"  {p}: {spyC[:b].iloc[-1] / spyO[spyO.index >= a].iloc[0] - 1:+.1%}")
open("research/fullflow/frac.txt", "w").write("\n".join(lines))
print("\n".join(lines))
