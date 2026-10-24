"""Regime-conditional stock selection (pre-registration: /mnt/project-files/回测/市场状态选股_预登记.md).

Usage:  python research/regime/regime_select.py train        # 2019-2020: pick A/B per state
        python research/regime/regime_select.py val          # 2021
        python research/regime/regime_select.py test         # 2025-12-31 .. 2026-09-30 rebalances (run once)
        python research/regime/regime_select.py extra        # post-test checks: 2022..2025, 2005-2018
Monthly rebalance: signal at month-end close, trade next open, 10 names equal weight, 0.05%/side.
"""
import sys, json, os
import numpy as np, pandas as pd
sys.path.insert(0, "research/pricedb")
import pricedb

HERE = os.path.dirname(os.path.abspath(__file__))
N, COST = 10, 0.0005
CHOICE_FILE = os.path.join(HERE, "choice.json")
STATES = ["rate_shock", "v_rebound", "bear", "narrow_bull", "broad_bull"]
CN = {"rate_shock": "加息冲击下跌", "v_rebound": "V 形反弹", "bear": "熊市",
      "narrow_bull": "窄牛市", "broad_bull": "宽牛市"}
CANDS = {"broad_bull": ("mom", "beta_hi"), "narrow_bull": ("big_mom", "big"),
         "v_rebound": ("beta_hi", "loser"), "bear": ("beta_lo", "vol_lo"),
         "rate_shock": ("tlt_lo", "beta_lo")}


def load(start):
    px = pricedb.load(start=start)
    C = px.pivot(index="Date", columns="Ticker", values="Close").sort_index()
    O = px.pivot(index="Date", columns="Ticker", values="Open").reindex(C.index)
    V = px.pivot(index="Date", columns="Ticker", values="Volume").reindex(C.index)
    return C, O, V


def membership():
    h = pd.read_csv("research/pricedb/data/membership/sp500_history_fja.csv", parse_dates=["date"])
    return h.set_index("date")["tickers"].str.split(",")


def members_on(hist, d):
    i = hist.index.searchsorted(d, side="right") - 1
    return set(t.replace(".", "-") for t in hist.iloc[i])


def macro():
    m = pd.read_csv(os.path.join(HERE, "data/macro.csv"), parse_dates=["Date"], index_col="Date")
    return m


def states(C, M):
    spy, rsp = C["SPY"], C["RSP"]
    ma200 = spy.rolling(200).mean()
    dd = spy / spy.rolling(252, min_periods=200).max() - 1
    dd126 = (spy / spy.rolling(126).max() - 1).rolling(126).min()
    low126 = spy.rolling(126).min()
    B = rsp.pct_change(126) - spy.pct_change(126)
    tnx = M["^TNX"].reindex(C.index).ffill()
    dY = tnx - tnx.shift(63)
    s = pd.Series("broad_bull", index=C.index)
    s[(spy >= ma200) & (B <= -0.03)] = "narrow_bull"
    s[spy < ma200] = "bear"
    s[(dd126 <= -0.15) & (spy >= low126 * 1.10)] = "v_rebound"
    s[(dY >= 0.50) & (dd <= -0.05)] = "rate_shock"
    s[ma200.isna()] = np.nan
    return s, pd.DataFrame({"dd": dd, "dd126": dd126, "B": B, "dY": dY, "above200": spy >= ma200})


def month_ends(idx, a, b):
    s = pd.Series(idx, index=idx)
    me = s.groupby([idx.year, idx.month]).max().values
    return [pd.Timestamp(d) for d in me if pd.Timestamp(a) <= d <= pd.Timestamp(b)]


def features(C, V, R, i, tickers, tlt_r):
    """Features at row i (signal date) for the given tickers."""
    w = R.iloc[i - 251:i + 1][tickers]
    ok = w.notna().sum() >= 200
    tk = [t for t in tickers if ok[t]]
    w = w[tk]
    m = R["SPY"].iloc[i - 251:i + 1]
    mom = C[tk].iloc[i - 21] / C[tk].iloc[i - 252] - 1
    vol = w.std() * np.sqrt(252)
    wc = w.sub(w.mean())
    mc = m - m.mean()
    beta = (wc.mul(mc, axis=0)).sum() / (mc ** 2).sum()
    dv = (C[tk] * V[tk]).iloc[i - 62:i + 1].mean()
    # two-factor regression on SPY and TLT, TLT coefficient
    tl = tlt_r.iloc[i - 251:i + 1].values
    X = np.column_stack([np.ones(252), m.values, tl])
    Y = w.values
    tlt_b = pd.Series(np.nan, index=tk)
    good = ~np.isnan(X).any(1)
    for j, t in enumerate(tk):
        y = Y[:, j]
        g = good & ~np.isnan(y)
        if g.sum() >= 200 and not np.isnan(tl).all():
            tlt_b[t] = np.linalg.lstsq(X[g], y[g], rcond=None)[0][2]
    return pd.DataFrame({"mom": mom, "vol": vol, "beta": beta, "dv": dv, "tlt_b": tlt_b})


def pick(f, style):
    f = f.dropna(subset=["mom", "beta", "vol", "dv"])
    if style == "mom":     return f.nlargest(N, "mom").index
    if style == "beta_hi": return f.nlargest(N, "beta").index
    if style == "beta_lo": return f.nsmallest(N, "beta").index
    if style == "vol_lo":  return f.nsmallest(N, "vol").index
    if style == "loser":   return f.nsmallest(N, "mom").index
    if style == "big":     return f.nlargest(N, "dv").index
    if style == "big_mom": return f.nlargest(100, "dv").nlargest(N, "mom").index
    if style == "tlt_lo":  return f.dropna(subset=["tlt_b"]).nsmallest(N, "tlt_b").index
    raise ValueError(style)


ALL_STYLES = ["mom", "beta_hi", "beta_lo", "vol_lo", "loser", "big", "big_mom", "tlt_lo"]


def run(a, b, last_exit=None):
    start = (pd.Timestamp(a) - pd.DateOffset(years=2)).strftime("%Y-%m-%d")
    C, O, V = load(start)
    M = macro()
    st, aux = states(C, M)
    R = C.pct_change(fill_method=None)
    tlt = M["TLT"].reindex(C.index)
    tlt_r = tlt.pct_change(fill_method=None)
    hist = membership()
    idx = C.index
    mes = month_ends(idx, a, b)
    rows = []
    for d in mes:
        i = idx.get_loc(d)
        if i + 1 >= len(idx):
            break
        nxt = [x for x in month_ends(idx, d + pd.Timedelta(days=1), idx[-1])]
        j_entry = i + 1
        if nxt and idx.get_loc(nxt[0]) + 1 < len(idx):
            j_exit = idx.get_loc(nxt[0]) + 1
        else:
            j_exit = len(idx) - 1  # partial last month: value at last close
        mem = [t for t in members_on(hist, d) if t in C.columns and pd.notna(C[t].iloc[i])]
        f = features(C, V, R, i, mem, tlt_r)
        partial = not (nxt and idx.get_loc(nxt[0]) + 1 < len(idx))
        def ret(tk):
            p0 = O[tk].iloc[j_entry]
            p1 = C[tk].iloc[j_exit] if partial else O[tk].iloc[j_exit]
            # missing exit (delisted): use last available close before exit
            last = C[tk].iloc[j_entry:j_exit + 1].ffill().iloc[-1]
            p1 = p1.fillna(last)
            return (p1 / p0 - 1).dropna()
        r_all = ret(list(f.index))
        row = {"date": d, "state": st.iloc[i], "entry": idx[j_entry], "exit": idx[j_exit],
               "n": len(r_all), "EW": r_all.mean(), **{k: aux[k].iloc[i] for k in aux}}
        spy = ret(["SPY"]).iloc[0]
        row["SPY"] = spy
        for s in ALL_STYLES:
            tk = list(pick(f, s))
            r = ret(tk)
            row[s] = r.mean()
            row[s + "_tk"] = ",".join(tk)
        rows.append(row)
    return pd.DataFrame(rows).set_index("date")


def turnover_cost(df, col):
    """Approximate cost: COST per side x fraction of names changed (+ full in/out)."""
    prev, c = set(), []
    for _, r in df.iterrows():
        cur = set(r[col].split(",")) if isinstance(r[col], str) and r[col] else set()
        ch = len(cur - prev) / max(len(cur), 1)
        c.append(COST * (2 * ch if prev else 2))  # buy new + sell old
        prev = cur
    return pd.Series(c, index=df.index)


def strategy(df, choice):
    st_col = df["state"].map(lambda s: choice[s])
    r = pd.Series([df.loc[d, st_col[d]] for d in df.index], index=df.index)
    tk = pd.Series([df.loc[d, st_col[d] + "_tk"] for d in df.index], index=df.index)
    tmp = pd.DataFrame({"tk": tk})
    return r - turnover_cost(tmp, "tk"), st_col


def net(df, s):
    return df[s] - turnover_cost(df, s + "_tk")


def summary(series_dict):
    out = []
    for k, r in series_dict.items():
        r = r.dropna()
        tot = (1 + r).prod() - 1
        out.append((k, len(r), r.mean() * 100, tot * 100))
    return pd.DataFrame(out, columns=["series", "months", "mean%/mo", "total%"]).set_index("series").round(2)


def tstat(x):
    x = x.dropna()
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 2 else np.nan


def report(df, choice, label):
    strat, used = strategy(df, choice)
    mom = net(df, "mom")
    d = {"分状态策略": strat, "一直用动量": mom, "全部股票等权": df["EW"] - 0.0, "SPY": df["SPY"]}
    print(f"\n=== {label}: {df.index[0].date()} .. {df.index[-1].date()} (last exit {df['exit'].iloc[-1].date()})")
    print(summary(d).to_string())
    ex = strat - df["EW"]
    print(f"分状态 − 等权 每月 {ex.mean()*100:+.2f}% t {tstat(ex):.2f};  分状态 − 动量 每月 {(strat-mom).mean()*100:+.2f}% t {tstat(strat-mom):.2f}")
    print("\n每月明细:")
    show = pd.DataFrame({"state": df["state"], "used": used, "strat%": strat * 100, "mom%": mom * 100,
                         "EW%": df["EW"] * 100, "SPY%": df["SPY"] * 100}).round(2)
    print(show.to_string())
    return strat, mom


def by_state(df):
    """Per state: mean excess over EW of every style (net of cost), months."""
    out = {}
    for s in STATES:
        sub = df[df["state"] == s]
        if not len(sub):
            continue
        row = {"months": len(sub)}
        for sty in ALL_STYLES:
            ex = net(df, sty)[sub.index] - sub["EW"]
            row[sty] = round(ex.mean() * 100, 2)
        row["SPY"] = round((sub["SPY"] - sub["EW"]).mean() * 100, 2)
        out[CN[s]] = row
    return pd.DataFrame(out).T


if __name__ == "__main__":
    mode = sys.argv[1]
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    if mode == "train":
        df = run("2019-01-01", "2020-12-31")
        df.to_csv(os.path.join(HERE, "out/train.csv"))
        print("state counts:\n", df["state"].value_counts().to_string())
        print("\n各状态里每类股票比等权多赚（每月，扣成本）:")
        print(by_state(df).to_string())
        choice = {}
        for s in STATES:
            a, b = CANDS[s]
            sub = df[df["state"] == s]
            if len(sub):
                ea = (net(df, a)[sub.index] - sub["EW"]).mean()
                eb = (net(df, b)[sub.index] - sub["EW"]).mean()
                choice[s] = a if ea >= eb else b
            else:
                choice[s] = a
        json.dump(choice, open(CHOICE_FILE, "w"), indent=1)
        print("\nchoice:", choice)
        report(df, choice, "train (in-sample)")
    elif mode in ("val", "test"):
        choice = json.load(open(CHOICE_FILE))
        a, b = ("2021-01-01", "2021-12-31") if mode == "val" else ("2025-12-01", "2026-09-30")
        df = run(a, b)
        df.to_csv(os.path.join(HERE, f"out/{mode}.csv"))
        print("choice:", choice)
        report(df, choice, mode)
        print("\n各状态里每类股票比等权多赚（每月，扣成本）:")
        print(by_state(df).to_string())
    elif mode == "extra":
        choice = json.load(open(CHOICE_FILE))
        for a, b in [("2022-01-01", "2022-12-31"), ("2023-01-01", "2023-12-31"),
                     ("2024-01-01", "2024-12-31"), ("2025-01-01", "2025-11-30"),
                     ("2005-01-01", "2018-12-31")]:
            df = run(a, b)
            df.to_csv(os.path.join(HERE, f"out/extra_{a[:4]}_{b[:4]}.csv"))
            report(df, choice, f"extra {a[:4]}-{b[:4]}")
            print(by_state(df).to_string())
    elif mode == "states":
        C, O, V = load("2003-01-01")
        st, aux = states(C, macro())
        me = month_ends(C.index, "2005-01-01", C.index[-1])
        print(st[me].groupby(pd.DatetimeIndex(me).year).value_counts().unstack(fill_value=0).to_string())
