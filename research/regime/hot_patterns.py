"""Hot-stock entry/exit patterns (pre-registration: /mnt/project-files/回测/热点股形态_预登记.md).

Usage: python research/regime/hot_patterns.py dev      # 2005-2018 discovery + 2019-2021 validation
       python research/regime/hot_patterns.py recent   # last 1y / 3m / 1m (run once)
Hot = top 10% of PIT S&P 500 members by 12-1 momentum, weekly signal dates.
Entry next open, forward 20 / 60 trading days (exit at open).
"""
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from regime_select import load, membership, members_on, macro, states

PATS = ["H0", "E1", "E2", "E3", "E4", "E5", "X1", "X2", "X3", "X4"]
CN = {"H0": "全部热点股", "E1": "创52周新高", "E2": "浅回调(-5~-15%,50日线上)", "E3": "跌破50日线、在200日线上",
      "E4": "涨过头(>50日线x1.25)", "E5": "5天急跌>=5%", "X1": "距高点回撤>=20%", "X2": "波动放大>=1.5倍",
      "X3": "3个月涨超50%", "X4": "大盘V形反弹期"}
HS = (20, 60)


def build(start, end):
    C, O, V = load((pd.Timestamp(start) - pd.DateOffset(years=2)).strftime("%Y-%m-%d"))
    st, _ = states(C, macro())
    hist = membership()
    idx = C.index
    R = C.pct_change(fill_method=None)
    mom = C.shift(21) / C.shift(252) - 1
    hi252 = C.rolling(252, min_periods=200).max()
    ma50, ma200 = C.rolling(50).mean(), C.rolling(200).mean()
    r5, r63 = C / C.shift(5) - 1, C / C.shift(63) - 1
    v20 = R.rolling(20).std()
    v252 = R.rolling(252, min_periods=200).std()
    s = pd.Series(idx, index=idx)
    weeks = s.groupby([idx.isocalendar().year.values, idx.isocalendar().week.values]).max().values
    weeks = [pd.Timestamp(d) for d in weeks if pd.Timestamp(start) <= d <= pd.Timestamp(end)]
    rows = []
    for d in weeks:
        i = idx.get_loc(d)
        mem = [t for t in members_on(hist, d) if t in C.columns and pd.notna(C[t].iloc[i])]
        fwd = {}
        for h in HS:
            if i + 1 + h >= len(idx):
                fwd[h] = None
                continue
            p0, p1 = O[mem].iloc[i + 1], O[mem].iloc[i + 1 + h]
            last = C[mem].iloc[i + 1:i + 1 + h].ffill().iloc[-1]
            fwd[h] = (p1.fillna(last) / p0 - 1)
        m = mom.iloc[i][mem].dropna()
        if len(m) < 100:
            continue
        hot = m[m >= m.quantile(0.9)].index
        c = C.iloc[i][hot]
        f = pd.DataFrame(index=hot)
        f["H0"] = True
        f["E1"] = c >= hi252.iloc[i][hot] * 0.9999
        dd = c / hi252.iloc[i][hot] - 1
        f["E2"] = (dd <= -0.05) & (dd >= -0.15) & (c > ma50.iloc[i][hot])
        f["E3"] = (c < ma50.iloc[i][hot]) & (c > ma200.iloc[i][hot])
        f["E4"] = c > ma50.iloc[i][hot] * 1.25
        f["E5"] = r5.iloc[i][hot] <= -0.05
        f["X1"] = dd <= -0.20
        f["X2"] = v20.iloc[i][hot] >= v252.iloc[i][hot] * 1.5
        f["X3"] = r63.iloc[i][hot] >= 0.50
        f["X4"] = st.iloc[i] == "v_rebound"
        for h in HS:
            if fwd[h] is None:
                f[f"r{h}"] = np.nan
                f[f"ew{h}"] = np.nan
            else:
                f[f"r{h}"] = fwd[h].reindex(hot)
                f[f"ew{h}"] = fwd[h].mean()
        f["date"] = d
        rows.append(f.reset_index(names="ticker"))
    return pd.concat(rows, ignore_index=True)


def block_t(x, n):
    """x: per-date series; average into non-overlapping blocks of n dates, t over blocks."""
    x = x.dropna()
    b = x.groupby(np.arange(len(x)) // n).mean()
    return b.mean() / (b.std(ddof=1) / np.sqrt(len(b))) if len(b) > 2 else np.nan


def table(df, h):
    blk = 4 if h == 20 else 12
    r, ew = f"r{h}", f"ew{h}"
    d = df.dropna(subset=[r])
    h0 = d.groupby("date")[r].mean()
    out = []
    for p in PATS:
        sub = d[d[p]]
        if not len(sub):
            out.append((CN[p], 0, 0) + (np.nan,) * 6)
            continue
        g = sub.groupby("date")
        pr = g[r].mean()
        ex_ew = pr - g[ew].first()
        ex_h0 = pr - h0[pr.index]
        out.append((CN[p], len(sub), pr.size, sub[r].mean() * 100, (sub[r] > 0).mean() * 100,
                    ex_ew.mean() * 100, block_t(ex_ew, blk), ex_h0.mean() * 100, block_t(ex_h0, blk)))
    return pd.DataFrame(out, columns=["形态", "次数", "信号周数", f"平均{h}天%", "赚钱比例%",
                                      "比等权%", "t", "比全部热点%", "t "]).set_index("形态").round(2)


def spy_line(df, start, end, h):
    C, O, _ = load((pd.Timestamp(start) - pd.DateOffset(days=10)).strftime("%Y-%m-%d"))
    idx = C.index
    ds = sorted(df.dropna(subset=[f"r{h}"])["date"].unique())
    v = []
    for d in ds:
        i = idx.get_loc(pd.Timestamp(d))
        v.append(O["SPY"].iloc[i + 1 + h] / O["SPY"].iloc[i + 1] - 1)
    ew = df.dropna(subset=[f"r{h}"]).groupby("date")[f"ew{h}"].first().mean()
    return f"同期基准（每个信号日买入持有 {h} 天的平均）：SPY {np.mean(v)*100:+.2f}%，全部股票等权 {ew*100:+.2f}%"


if __name__ == "__main__":
    H = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(f"{H}/out", exist_ok=True)
    mode = sys.argv[1]
    if mode == "dev":
        segs = [("发现 2005-2018", "2005-01-01", "2018-12-31"), ("验证 2019-2021", "2019-01-01", "2021-12-31")]
    else:
        segs = [("最近1年", "2025-09-25", "2026-10-23"), ("最近3个月", "2026-06-25", "2026-10-23"),
                ("最近1个月", "2026-08-25", "2026-10-23")]
    pd.set_option("display.width", 250)
    for lab, a, b in segs:
        df = build(a, b)
        df.to_csv(f"{H}/out/hot_{mode}_{a[:7]}.csv.gz", index=False)
        for h in HS:
            if df[f"r{h}"].notna().sum() == 0:
                continue
            print(f"\n=== {lab}  持有 {h} 天（信号 {df.dropna(subset=[f'r{h}'])['date'].min().date()} .. {df.dropna(subset=[f'r{h}'])['date'].max().date()}）")
            print(table(df, h).to_string())
            print(spy_line(df, a, b, h))
