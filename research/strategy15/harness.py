"""Shared, fixed test bench for the 15-strategy exploration (2026-10-24). Every strategy is scored the same way.

A strategy is a function  signal(ctx) -> DataFrame (dates x tickers) of scores computed from data known at the close of
each date (higher = better, NaN = not eligible). The bench adds the universe filter, buys at the NEXT day's open,
and sells by the chosen exit:
    exit="hold", hold=H   sell at the close H trading days after buying (H in 5, 10, 20)
    exit="r1"             the current process: +3% limit for 20 days, from day 11 also sell at -3%, day-20 close
Costs: 0.05% per side for S&P 500 stocks, 0.10% per side for the rest.

Universe (point in time): S&P 500 or 400 member that day (research/factors/pit.py, includes 132 later-removed
tickers), price > $5, 21-day average dollar volume > $5M, at least 252 days of history.

Periods (signal dates):
    TRAIN  2021-11-01 .. 2025-09-22   use this to design / choose variants
    1Y     2025-09-23 .. 2026-09-23   (= the "last 1 year" Jasper looks at; only look at it after choosing on TRAIN)
    3M / 1M  the last 91 / 30 days of 1Y
Signals end ~20 trading days before the last price date so every trade has a full outcome.

Usage from a strategy file:
    from harness import load, evaluate
    ctx = load()
    res = evaluate("A1 momentum 12-1", my_signal(ctx), ctx, exit="hold", hold=20)
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(ROOT))
D = f"{REPO}/research/factors/data"
FF = f"{REPO}/research/fullflow/data"
CACHE = f"{ROOT}/data"
sys.path.insert(0, f"{REPO}/research/factors")

TRAIN_START = pd.Timestamp("2021-11-01")
CAPITAL = 180.0


class Ctx(dict):
    __getattr__ = dict.__getitem__


def _prices():
    cons = pd.read_csv(f"{D}/constituents.csv").set_index("yf")
    px = pd.read_csv(f"{D}/prices.csv.gz", parse_dates=["Date"])
    rem = pd.read_csv(f"{D}/prices_removed.csv.gz", parse_dates=["Date"])
    px = pd.concat([px, rem[~rem["Ticker"].isin(cons.index)]]).drop_duplicates(["Date", "Ticker"])
    P = {k: px.pivot(index="Date", columns="Ticker", values=k).sort_index() for k in ["Open", "High", "Low", "Close", "Volume"]}
    return cons, P


def load():
    """Everything a strategy may use. All frames are dates x tickers on the same calendar."""
    os.makedirs(CACHE, exist_ok=True)
    pk = f"{CACHE}/ctx.pkl"
    if os.path.exists(pk):
        return Ctx(pd.read_pickle(pk))
    from pit import membership
    cons, P = _prices()
    O, H, L, C, V = (P[k] for k in ["Open", "High", "Low", "Close", "Volume"])
    spy = pd.DataFrame({k.lower(): P[k]["SPY"] for k in ["Open", "Close"]})
    for X in (O, H, L, C, V):
        X.drop(columns=[c for c in ["SPY", "MDY"] if c in X], inplace=True)
    Cf = C.ffill(limit=5)
    dvol = (C * V).rolling(21, min_periods=15).mean()
    mem = membership(C, dvol)
    univ = mem & (C > 5) & (dvol > 5e6) & (C.notna().rolling(252, min_periods=252).sum() >= 250)
    tick = C.columns
    sector = pd.Series(cons["Sector"].reindex(tick).fillna("Unknown").values, index=tick)
    sp500 = pd.Series(cons["Universe"].reindex(tick).eq("SP500").values, index=tick)
    cost = pd.Series(np.where(sp500, 0.0005, 0.001), index=tick)

    # earnings dates: SEC 8-K item 2.02 + Yahoo calendar (covers removed tickers); merge dates within 5 days
    e = pd.concat([pd.read_csv(f"{FF}/{f}", parse_dates=["filed"])[["ticker", "filed"]] for f in ("earnings.csv.gz", "earnings_yf.csv.gz")])
    e = e[e["ticker"].isin(tick)].sort_values(["ticker", "filed"])
    e = e[e.groupby("ticker")["filed"].diff().dt.days.fillna(99) > 5].rename(columns={"filed": "date"}).reset_index(drop=True)
    cal = C.index
    days_since, days_to = pd.DataFrame(np.nan, index=cal, columns=tick), pd.DataFrame(np.nan, index=cal, columns=tick)
    cv = cal.values
    for tk, g in e.groupby("ticker"):
        ed = g["date"].values
        p = np.searchsorted(ed, cv, side="right")  # events on or before each day
        prev = np.where(p > 0, ed[np.maximum(p - 1, 0)], np.datetime64("NaT"))
        nxt = np.where(p < len(ed), ed[np.minimum(p, len(ed) - 1)], np.datetime64("NaT"))
        days_since[tk] = (cv - prev) / np.timedelta64(1, "D")
        days_to[tk] = (nxt - cv) / np.timedelta64(1, "D")
    # earnings-announcement 3-day return (close day-2 .. close day+1), known from the close of day+1
    ear = []
    ci = {t: i for i, t in enumerate(tick)}
    Cv = Cf.to_numpy()
    for r in e.itertuples():
        i = cal.searchsorted(r.date)
        if i < 2 or i + 1 >= len(cal) or r.ticker not in ci:
            continue
        j = ci[r.ticker]
        ear.append((r.ticker, r.date, cal[i + 1], Cv[i + 1, j] / Cv[i - 2, j] - 1))
    ear = pd.DataFrame(ear, columns=["ticker", "event", "known", "ear"])

    # point-in-time fundamentals: a filing counts from the trading day AFTER its filing date
    fund = pd.read_parquet(f"{CACHE}/fund.parquet")
    sc = pd.read_parquet(f"{FF}/scores.parquet")[["ticker", "filed", "score"]]
    fund = fund.merge(sc, on=["ticker", "filed"], how="left")
    ctx = dict(O=O, H=H, L=L, C=C, Cf=Cf, V=V, dvol=dvol, univ=univ, sector=sector, sp500=sp500, cost=cost, spy=spy,
               days_since_earn=days_since, days_to_earn=days_to, earn=e, ear=ear, fund_rows=fund, cal=cal)
    pd.to_pickle(ctx, pk)
    return Ctx(ctx)


def fund_daily(ctx, col, stale=200):
    """Daily dates x tickers frame of a fundamental column (see fund_panel.py; also 'score'), forward-filled from the
    trading day after each filing, blank once older than `stale` calendar days."""
    f = ctx.fund_rows.dropna(subset=[col]).sort_values("filed")
    f = f[f["ticker"].isin(ctx.C.columns)].assign(avail=lambda x: x["filed"] + pd.Timedelta(days=1))
    w = f.pivot_table(index="avail", columns="ticker", values=col, aggfunc="last")
    fd = f.assign(fn=f["avail"].astype("datetime64[ns]").astype("int64")).pivot_table(index="avail", columns="ticker", values="fn", aggfunc="last")
    ix = w.index.union(ctx.cal)
    val = w.reindex(ix).ffill().reindex(ctx.cal)
    last = fd.reindex(ix).ffill().reindex(ctx.cal)
    age = (ctx.cal.values.astype("datetime64[ns]").astype("int64")[:, None] - last.to_numpy()) / 86400e9
    return val.mask(age > stale).reindex(columns=ctx.C.columns)


def rsi(C, n=14):
    d = C.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


# ---------------------------------------------------------------- outcomes
def _outcomes(ctx, exit, hold):
    """Per (signal date, ticker) trade return after costs and trading days held. Cached."""
    key = f"{CACHE}/out_{exit}_{hold}.pkl"
    if os.path.exists(key):
        return pd.read_pickle(key)
    O, Hh, Cf = ctx.O.to_numpy(), ctx.H.to_numpy(), ctx.Cf.to_numpy()
    T, N = O.shape
    L = 20 if exit == "r1" else hold
    ret = np.full((T, N), np.nan)
    days = np.full((T, N), np.nan)
    for i in range(T - L - 1):
        entry = O[i + 1]
        if exit == "hold":
            ret[i] = Cf[i + L] / entry - 1
            days[i] = L
            continue
        out = Cf[i + L] / entry - 1
        dd = np.full(N, float(L))
        done = ~np.isfinite(entry)
        for t in range(L):
            op, hi = O[i + 1 + t], Hh[i + 1 + t]
            lim = entry * 1.03
            hit = ~done & (hi >= lim)
            out[hit] = (np.maximum(lim, op) if t else lim)[hit] / entry[hit] - 1
            dd[hit], done = t + 1, done | hit
            if t >= 10:
                lv = entry * 0.97
                be = ~done & (hi >= lv)
                out[be] = np.maximum(lv, op)[be] / entry[be] - 1
                dd[be], done = t + 1, done | be
        ret[i], days[i] = out, dd
    ret -= 2 * ctx.cost.to_numpy()[None, :]
    r = (pd.DataFrame(ret, index=ctx.cal, columns=ctx.C.columns), pd.DataFrame(days, index=ctx.cal, columns=ctx.C.columns))
    pd.to_pickle(r, key)
    return r


def windows(ctx):
    end = ctx.cal[-22]
    y1 = end - pd.Timedelta(days=365)
    return end, {"训练段 2021-11~2025-09": (TRAIN_START, y1 - pd.Timedelta(days=1)), "最近1年": (y1, end),
                 "最近3个月": (end - pd.Timedelta(days=91), end), "最近1个月": (end - pd.Timedelta(days=30), end)}


def nw_t(x, lag):
    x = pd.Series(x).dropna()
    if len(x) < 10:
        return np.nan
    e = (x - x.mean()).to_numpy()
    T = len(e)
    v = e @ e / T
    for k in range(1, min(lag, T - 1) + 1):
        v += 2 * (1 - k / (lag + 1)) * (e[k:] @ e[:-k]) / T
    return x.mean() / np.sqrt(v / T)


def _seq(ctx, picks, ret, days, t0, t1):
    """$180, one stock at a time, whole shares, compounding. picks: Series date -> ticker (top-1 with price <= capital)."""
    cal, O = ctx.cal, ctx.O
    k, cap, n, w = cal.searchsorted(t0), CAPITAL, 0, 0
    while k < len(cal) and cal[k] <= t1:
        t = cal[k]
        tk = picks.get(t)
        if isinstance(tk, str):
            px = O.at[cal[k + 1], tk]
            sh = np.floor(cap / px)
            if sh >= 1 and np.isfinite(ret.at[t, tk]):
                cap += sh * px * ret.at[t, tk]
                n, w = n + 1, w + (ret.at[t, tk] > 0)
                k += int(days.at[t, tk]) + 1  # next pick at the close of the exit day
                continue
        k += 1
    return cap / CAPITAL - 1, n, w


def evaluate(name, score, ctx, exit="hold", hold=20, k=5, quiet=False, start_offsets=10, with_luck=True):
    """Score a strategy. Returns a dict of results and prints a compact table."""
    ret, days = _outcomes(ctx, exit, hold)
    end, W = windows(ctx)
    s = score.reindex(index=ctx.cal, columns=ctx.C.columns).where(ctx.univ)
    s = s.where(ret.notna())
    s.loc[(s.index < TRAIN_START) | (s.index > end)] = np.nan
    allm = ret.where(ctx.univ & ret.notna()).loc[TRAIN_START:end].mean(axis=1)
    spy = ctx.spy
    L = 20 if exit == "r1" else hold
    rk = s.rank(axis=1, ascending=False, method="first")
    topk = ret.where(rk <= k).mean(axis=1)
    cnt = s.notna().sum(axis=1)
    cap = s.where(ctx.O.shift(-1) <= CAPITAL)  # whole share affordable at tomorrow's open
    rk1 = cap.rank(axis=1, ascending=False, method="first")
    top1 = ret.where(rk1 == 1)
    p1 = top1.stack().dropna()
    picks = pd.Series(p1.index.get_level_values(1), index=p1.index.get_level_values(0))
    r1 = top1.mean(axis=1)
    # SPY over the same holding days (plain hold of L days, or the median r1 days)
    so, sc = spy["open"].to_numpy(), spy["close"].ffill().to_numpy()
    idx = np.arange(len(ctx.cal))
    spy_ret = pd.Series(np.where(idx + L < len(idx), sc[np.minimum(idx + L, len(idx) - 1)] / so[np.minimum(idx + 1, len(idx) - 1)] - 1, np.nan), index=ctx.cal)
    res = {"name": name, "exit": exit, "hold": hold, "rows": {}}
    for w, (a, b) in W.items():
        sl = slice(a, b)
        ex = (topk - allm).loc[sl]
        seqs = [_seq(ctx, picks, ret, days, ctx.cal[ctx.cal.searchsorted(a) + o], b) for o in range(start_offsets)]
        tot = [x[0] for x in seqs]
        ia, ib = ctx.cal.searchsorted(a), min(ctx.cal.searchsorted(b) + L, len(ctx.cal) - 1)
        spy_bh = sc[ib] / so[ia + 1] - 1
        m0 = ctx.univ.iloc[ia]
        ew = (ctx.Cf.iloc[ib][m0] / ctx.O.iloc[ia + 1][m0] - 1).mean()
        res["rows"][w] = dict(
            n_days=int((cnt.loc[sl] > 0).sum()), avg_names=float(cnt.loc[sl][cnt.loc[sl] > 0].mean()) if (cnt.loc[sl] > 0).any() else 0,
            top1=r1.loc[sl].mean(), topk=topk.loc[sl].mean(), all=allm.loc[sl].mean(), spy=spy_ret.loc[sl].mean(),
            excess_k=ex.mean(), t_k=nw_t(ex, L), hit1=(top1.loc[sl].stack().dropna() > 0).mean(),
            seq_med=float(np.median(tot)), seq_lo=float(np.min(tot)), seq_hi=float(np.max(tot)),
            seq_n=int(np.median([x[1] for x in seqs])), seq_win=int(np.median([x[2] for x in seqs])),
            spy_bh=spy_bh, ew_bh=ew)
    if with_luck:
        p = placebo(ctx, exit, hold, k)
        for w, r in res["rows"].items():
            q = p[p["window"] == w]
            for m in ("top1", "topk", "excess_k", "seq_med"):
                r[f"pct_{m}"] = float((q[m] < r[m]).mean())  # share of random pickers this strategy beats
                r[f"rand90_{m}"] = (float(q[m].quantile(0.05)), float(q[m].quantile(0.95)))
    if not quiet:
        print(report(res))
    return res


def placebo(ctx, exit="hold", hold=20, k=5, n=100):
    """Random pickers (uniform among the universe) scored exactly like a strategy: the luck range."""
    key = f"{CACHE}/placebo_{exit}_{hold}_{k}.pkl"
    if os.path.exists(key):
        return pd.read_pickle(key)
    rows = []
    for sd in range(n):
        r = evaluate("random", ctx.C * 0 + np.random.default_rng(1000 + sd).random(ctx.C.shape), ctx, exit, hold, k,
                     quiet=True, start_offsets=3, with_luck=False)
        rows += [dict(window=w, seed=sd, **v) for w, v in r["rows"].items()]
    p = pd.DataFrame(rows)
    pd.to_pickle(p, key)
    return p


def report(res):
    lines = [f"\n=== {res['name']}  (exit={res['exit']}, hold={res['hold'] if res['exit'] == 'hold' else '≤20'}) ==="]
    lines.append("区间 | 有信号天数 | 平均候选数 | 每笔:第1名(≤$180) | 每笔:前5名 | 每笔:全部股票平均 | 每笔:同期SPY | 前5名超额 t | "
                 "$180连续做 中位(范围) 笔数 | 全部股票等权买入持有 | SPY买入持有")
    for w, r in res["rows"].items():
        lines.append(f"{w} | {r['n_days']} | {r['avg_names']:.0f} | {r['top1']:+.2%} | {r['topk']:+.2%} | {r['all']:+.2%} | {r['spy']:+.2%} | "
                     f"{r['excess_k']:+.2%} t={r['t_k']:.2f} | {r['seq_med']:+.1%} ({r['seq_lo']:+.1%}~{r['seq_hi']:+.1%}) {r['seq_n']}笔赚{r['seq_win']} | "
                     f"{r['ew_bh']:+.1%} | {r['spy_bh']:+.1%}")
        if "pct_topk" in r:
            lines.append(f"    胜过随机挑的比例：第1名每笔 {r['pct_top1']:.0%}，前5名每笔 {r['pct_topk']:.0%}，$180连续做 {r['pct_seq_med']:.0%}"
                         f"（随机挑 90% 范围：第1名每笔 {r['rand90_top1'][0]:+.2%}~{r['rand90_top1'][1]:+.2%}，"
                         f"$180连续做 {r['rand90_seq_med'][0]:+.1%}~{r['rand90_seq_med'][1]:+.1%}）")
    return "\n".join(lines)


def save(res, path):
    import json
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(res, ensure_ascii=False, default=float) + "\n")
