"""Exit variants on the daily top-5 (last two years, same rows as 排雷 dataset). Jasper 2026-10-24:
'+3% for the first 10 trading days (~14 days); from day 11 sell as soon as back to 0%; from day 21 (~28 days) also stop out
at -3%' (true stop: sell when the low touches -3%, at the open if it gaps below). Unsold at day 48 -> close.
Usage: python research/fullflow/exit_bs.py
"""
import io
import sys
from contextlib import redirect_stdout

with redirect_stdout(io.StringIO()):
    sys.argv = ["x", "/tmp/exit_bs_tmp"]
    exec(open("research/blindtest/avoid_data.py", encoding="utf-8").read().split("\nis_train = ")[0])

Dd = "research/factors/data"
lo = pd.concat([pd.read_csv(f"{Dd}/prices.csv.gz", parse_dates=["Date"], usecols=["Date", "Ticker", "Low"]),
                pd.read_csv(f"{Dd}/prices_removed.csv.gz", parse_dates=["Date"], usecols=["Date", "Ticker", "Low"])])
Lw = lo.drop_duplicates(["Date", "Ticker"]).pivot(index="Date", columns="Ticker", values="Low").reindex(index=O.index, columns=O.columns)


def sim(rows, lim_steps, stop_from=None, stop=-0.03, L=48):
    i = cal.searchsorted(rows["date"].to_numpy())
    j = O.columns.get_indexer(rows["ticker"])
    k = np.arange(1, L + 1)
    op, hi, lw = (M.to_numpy()[i[:, None] + k, j[:, None]] for M in (O, H, Lw))
    entry = op[:, 0]
    out, days, done = Cl.to_numpy()[i + L, j] / entry - 1, np.full(len(i), L), np.zeros(len(i), bool)
    for t in range(L):
        if stop_from is not None and t >= stop_from:  # stop-loss first if it gapped below at the open
            sl = entry * (1 + stop)
            s = ~done & (lw[:, t] <= sl)
            out[s] = np.minimum(sl, op[:, t])[s] / entry[s] - 1
            days[s], done = t + 1, done | s
        lv = entry * (1 + [v for d0, v in lim_steps if d0 <= t][-1])
        h = ~done & (hi[:, t] >= lv)
        out[h] = (np.maximum(lv, op[:, t]) if t else lv)[h] / entry[h] - 1
        days[h], done = t + 1, done | h
    cost = np.where(rows["ticker"].map(cons["Universe"]).eq("SP500"), 0.0005, 0.001)
    return out - 2 * cost, days


rows = feat[["date", "ticker"]].assign(date=pd.to_datetime(feat["date"]))
V = {"现行：+3% → 第 11 天起回到 −3% 卖 → 第 20 天收盘卖": dict(lim_steps=[(0, .03), (10, -.03)], L=20),
     "阶梯：+3% → 第 21 天起 0% → 第 35 天起 −3% → 第 48 天收盘卖": dict(lim_steps=[(0, .03), (20, 0.), (34, -.03)]),
     "新：+3% → 第 11 天起回到 0% 就卖 → 第 21 天起跌到 −3% 止损 → 第 48 天收盘卖": dict(lim_steps=[(0, .03), (10, 0.)], stop_from=20),
     "新（20 天收尾）：同上，但第 21 天起跌到 −3% 止损、第 30 天收盘卖": dict(lim_steps=[(0, .03), (10, 0.)], stop_from=20, L=30)}
spy = Cl["SPY"]
lines = []
for half, m in [("训练年", rows["date"] <= MID), ("测试年", rows["date"] > MID)]:
    x0 = rows[m]
    a, b = x0["date"].min(), x0["date"].max()
    sr = spy[a:b]
    lines.append(f"\n== {half} {a.date()} ~ {b.date()}（{len(x0)} 笔）；一直拿 SPY 折合每 20 交易日 {(sr.iloc[-1] / sr.iloc[0]) ** (20 / (len(sr) - 1)) - 1:+.2%} ==")
    for nm, kw in V.items():
        r, dd = sim(x0, **kw)
        w = r > 0
        lines.append(f"  {nm}\n    每笔 {r.mean():+.2%}，平均持有 {dd.mean():.1f} 天，折合每 20 交易日 {r.mean() / dd.mean() * 20:+.2%} | 赚钱 {w.mean():.0%}（平均 {r[w].mean():+.2%}）"
                     f"，亏钱平均 {r[~w].mean():+.2%}，亏 >10% {(r < -0.10).mean():.1%}，最差 {r.min():+.1%}，五只里平均赚钱 {pd.Series(w).groupby(x0['date'].to_numpy()).sum().mean():.2f} 只")
open("research/fullflow/exit_bs.txt", "w").write("\n".join(lines))
print("\n".join(lines))
