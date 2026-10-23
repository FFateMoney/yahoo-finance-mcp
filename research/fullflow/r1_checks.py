"""Checks on candidate rule R1 (资料/短线选股分析流程.md, third version):
1 market filter, 2 parameter robustness, 3 affordable stocks with $180, 4 one-position-at-a-time simulation vs SPY.
Usage: python research/fullflow/r1_checks.py   (needs flow_panel_daily.parquet from `explore.py 1 2021-11-01` + `flow.py --daily`)
"""
import numpy as np
import pandas as pd

D = "research/factors/data"
d = pd.read_parquet("research/fullflow/flow_panel_daily.parquet")
cons = pd.read_csv(f"{D}/constituents.csv").set_index("yf")
px = pd.read_csv(f"{D}/prices.csv.gz", parse_dates=["Date"], usecols=["Date", "Ticker", "Open", "Close"])
rem = pd.read_csv(f"{D}/prices_removed.csv.gz", parse_dates=["Date"], usecols=["Date", "Ticker", "Open", "Close"])
px = pd.concat([px, rem[~rem["Ticker"].isin(cons.index)]])
C = px.pivot(index="Date", columns="Ticker", values="Close").sort_index()
spy_c, spy_o = C["SPY"].ffill(), px[px["Ticker"] == "SPY"].set_index("Date")["Open"].sort_index()
C = C.drop(columns=["SPY", "MDY"], errors="ignore")
for n in (10, 30):
    x = (C / C.rolling(n).mean() - 1).stack().rename(f"离{n}日线").reset_index().rename(columns={"Date": "date", "Ticker": "ticker"})
    d = d.merge(x, on=["date", "ticker"], how="left")
regime = pd.DataFrame({"spy_above200": spy_c > spy_c.rolling(200).mean(), "spy_above50": spy_c > spy_c.rolling(50).mean()})
d = d.merge(regime, left_on="date", right_index=True, how="left")
d["base"] = (d["no_earn"] & (d["score"] >= 8) & ~d["sector"].isin(["Information Technology", "Communication Services"])
             & (d["离200日线"] > 0) & d["ma200_up"].fillna(False).astype(bool))
d["touch"] = d["过去触及率(+3%,10日)"]
last = d["date"].max()
WIN = {"1个月": last - pd.DateOffset(months=1), "3个月": last - pd.DateOffset(months=3), "1年": last - pd.DateOffset(years=1)}
YEARS = [2022, 2023, 2024, 2025]


def picks(ma="离20日线", rsi=(30, 50), cap=None, reg=None, rank="touch"):
    m = d["base"] & (d[ma] < 0) & d["RSI14"].between(*rsi)
    if cap:
        m &= d["price"] <= cap
    if reg:
        m &= d[reg].fillna(False).astype(bool)
    q = d[m].copy()
    q["s"] = q["score"] * 10 + (q[rank] if rank != "touch" else q["touch"])
    return q.sort_values("s", ascending=False).groupby("date").head(1)


def row(top, n="3%"):
    out = {}
    for w, t0 in WIN.items():
        out[w] = top.loc[top["date"] >= t0, f"ret_{n}"].mean()
    for y in YEARS:
        out[str(y)] = top.loc[top["date"].dt.year == y, f"ret_{n}"].mean()
    out["有候选股的天数占比"] = top["date"].nunique() / d["date"].nunique()
    return out


def fmt(rows):
    t = pd.DataFrame(rows).T
    return t.map(lambda v: f"{v:+.2%}" if isinstance(v, float) and not np.isnan(v) else "—").to_string()


lines = [f"每个交易日一个信号，{d['date'].min().date()} 至 {last.date()}。数字是第 1 名每笔平均收益（扣成本，没成交按第 28 天收盘价）",
         "全部股票平均 n=3%：" + "，".join(f"{w} {d.loc[d['date'] >= t0].groupby('date')['ret_3%'].mean().mean():+.2%}" for w, t0 in WIN.items())
         + "，" + "，".join(f"{y} {d.loc[d['date'].dt.year == y].groupby('date')['ret_3%'].mean().mean():+.2%}" for y in YEARS)]

lines.append("\n== 1 大盘过滤 ==")
lines.append(fmt({"R1（不过滤）": row(picks()), "SPY 在 200 日线上方才买": row(picks(reg="spy_above200")),
                  "SPY 在 50 日线上方才买": row(picks(reg="spy_above50"))}))
lines.append("SPY 在 200 日线下方的交易日占比：" + "，".join(
    f"{y} {1 - regime.loc[regime.index.year == y, 'spy_above200'].mean():.0%}" for y in YEARS + [2026]))

lines.append("\n== 2 参数稳健性（n=3% 除非注明）==")
rob = {"R1：20日线下方，RSI 30-50": row(picks())}
for r in [(25, 50), (30, 55), (25, 55), (35, 50), (30, 45)]:
    rob[f"RSI {r[0]}-{r[1]}"] = row(picks(rsi=r))
for ma in ["离10日线", "离30日线", "离50日线"]:
    rob[f"{ma[1:]}下方，RSI 30-50"] = row(picks(ma=ma))
top = picks()
for n in ["2%", "5%"]:
    rob[f"目标 +{n}"] = row(top, n)
lines.append(fmt(rob))

lines.append("\n== 3 资金 $180：只看股价 ≤ $180 的股票 ==")
d["离20日线_neg"] = -d["离20日线"]
d["RSI_neg"] = -d["RSI14"]
aff = {"R1 不限股价": row(picks()), "R1 股价 ≤ $180": row(picks(cap=180)),
       "≤$180 + SPY 在 200 日线上方": row(picks(cap=180, reg="spy_above200")),
       "≤$180，同分按跌破 20 日线最深排": row(picks(cap=180, rank="离20日线_neg")),
       "≤$180，同分按 RSI 最低排": row(picks(cap=180, rank="RSI_neg"))}
lines.append(fmt(aff))


def simulate(top, start, end, capital=180.0):
    """One position at a time: buy the day's pick at next open (whole shares), hold until the limit fills or day 20,
    then look for the next pick from the exit day. Returns final capital, number of trades, wins."""
    top = top.set_index("date").sort_index()
    cal = d["date"].drop_duplicates().sort_values().reset_index(drop=True)
    pos = {t: k for k, t in enumerate(cal)}
    k = cal.searchsorted(start)
    trades = wins = 0
    while k < len(cal) and cal[k] <= end:
        t = cal[k]
        if t not in top.index:
            k += 1
            continue
        r = top.loc[t]
        shares = np.floor(capital / (r["price"] * 1.01))  # next open ~ close; small buffer
        if shares < 1:
            k += 1
            continue
        invested = shares * r["price"]
        capital += invested * r["ret_3%"]
        trades += 1
        wins += r["ret_3%"] > 0
        k += int(r["days_3%"])  # exit day; next signal at that day's close
    return capital, trades, wins


lines.append("\n== 4 一次只拿一只、$180 起步的模拟（整股，资金复利），和同期一直拿 SPY 比 ==")
for name, tp in [("R1 股价 ≤ 资金", picks(cap=180)), ("R1 ≤ 资金 + SPY 在 200 日线上方", picks(cap=180, reg="spy_above200"))]:
    for label, s, e in [(str(y), pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y}-12-31")) for y in YEARS] + \
                       [("最近 1 年", WIN["1年"], last), ("2022-01 至今", pd.Timestamp("2022-01-01"), last)]:
        tpc = tp.copy()
        cap_end, n_tr, n_win = simulate(tpc, s, e)
        sd = spy_c.index[spy_c.index.searchsorted(s)]
        ed = spy_c.index[min(spy_c.index.searchsorted(e + pd.Timedelta(days=28)), len(spy_c) - 1)]
        spy_ret = spy_c[ed] / spy_c[sd] - 1
        lines.append(f"{name:<26} {label:<12} $180 → ${cap_end:7.2f}（{cap_end / 180 - 1:+.1%}），{n_tr} 笔，赚钱 {n_win} 笔 | 同期 SPY {spy_ret:+.1%}")

lines.append("\n== 5 模拟靠不靠运气：换起步日、和随机挑选比（2022-01 至今，$180，整股）==")
cal = d["date"].drop_duplicates().sort_values().reset_index(drop=True)
k0 = cal.searchsorted(pd.Timestamp("2022-01-01"))
tp = picks(cap=180)
offs = [simulate(tp.copy(), cal[k0 + o], last)[0] / 180 - 1 for o in range(20)]
lines.append(f"R1 从 2022 年头 20 个交易日分别起步：总收益 最低 {min(offs):+.0%}，中位 {np.median(offs):+.0%}，最高 {max(offs):+.0%}")
rng = np.random.default_rng(0)
for name, pool in [("全部股价 ≤ $180 的股票里随便挑", d[(d["price"] <= 180) & d["ret_3%"].notna()]),
                   ("通过 R1 全部条件的股票里随便挑（不排序）",
                    d[d["base"] & (d["离20日线"] < 0) & d["RSI14"].between(30, 50) & (d["price"] <= 180) & d["ret_3%"].notna()])]:
    res = []
    for _ in range(200):
        r = pool.assign(u=rng.random(len(pool)))
        res.append(simulate(r.loc[r.groupby("date")["u"].idxmax()], cal[k0], last)[0] / 180 - 1)
    lines.append(f"{name}（200 次）：总收益 5% 分位 {np.percentile(res, 5):+.0%}，中位 {np.median(res):+.0%}，95% 分位 {np.percentile(res, 95):+.0%}")
lines.append("\n== 6 只看最近：1 年、3 个月、1 个月（$180，整股，一次一只）==")
d["RSI_neg"] = -d["RSI14"]
cands = d[d["base"] & (d["离20日线"] < 0) & d["RSI14"].between(30, 50) & (d["price"] <= 180) & d["ret_3%"].notna()]
allc = d[(d["price"] <= 180) & d["ret_3%"].notna()]
for w, t0 in WIN.items():
    k = cal.searchsorted(t0)
    out = []
    for name, tp in [("R1 原排序", picks(cap=180)), ("同分按 RSI 最低", picks(cap=180, rank="RSI_neg"))]:
        cap_end, n_tr, n_win = simulate(tp.copy(), cal[k], last)
        offs = [simulate(tp.copy(), cal[k + o], last)[0] / 180 - 1 for o in range(min(10, len(cal) - k - 1))]
        out.append(f"{name} {cap_end / 180 - 1:+.1%}（{n_tr} 笔，赚 {n_win}；换起步日 {min(offs):+.1%}~{max(offs):+.1%}）")
    for name, pool in [("条件内随便挑", cands), ("全部随便挑", allc)]:
        res = []
        for _ in range(200):
            r = pool.assign(u=rng.random(len(pool)))
            res.append(simulate(r.loc[r.groupby("date")["u"].idxmax()], cal[k], last)[0] / 180 - 1)
        out.append(f"{name} 中位 {np.median(res):+.1%}（{np.percentile(res, 5):+.1%}~{np.percentile(res, 95):+.1%}）")
    spy = spy_c[spy_c.index[-1]] / spy_c[spy_c.index[spy_c.index.searchsorted(t0)]] - 1
    lines.append(f"{w}（{t0.date()} 起）：" + "；".join(out) + f"；SPY {spy:+.1%}")
open("research/fullflow/r1_checks.txt", "w").write("\n".join(lines))
print("\n".join(lines))
