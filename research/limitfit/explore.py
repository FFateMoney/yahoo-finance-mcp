"""Exploratory: which stock features go with a good result under Jasper's limit-sell + keep-holding plan.

Not a pre-registered test. Data: research/factors/data (S&P 500 + 400, point-in-time membership, plus removed
tickers), 2020-10 to 2026-10. Signal at close t (every 5 trading days), buy open t+1, limit sell at entry*(1+n);
if not filled in 10 trading days (~14 calendar days) keep the order 10 more, then sell at close (R=2).
Outcome = plan return after costs minus SPY over the same days held. Features are compared within each date.
Usage: python research/limitfit/explore.py [step] [first_signal_date]
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "research/factors")
from pit import membership  # noqa: E402

D = "research/factors/data"
STEP = int(sys.argv[1]) if len(sys.argv) > 1 else 5
FIRST = pd.Timestamp(sys.argv[2]) if len(sys.argv) > 2 else None
ROUND, R = 10, 2
L = ROUND * R
NS = {"2%": 0.02, "3%": 0.03, "5%": 0.05, "7%": 0.07, "10%": 0.10, "1σ": None, "hold": 99.0}  # hold = never fills, sell day 20
OUT = "research/limitfit/out"

px = pd.read_csv(f"{D}/prices.csv.gz", parse_dates=["Date"])
cons = pd.read_csv(f"{D}/constituents.csv").set_index("yf")
rem = pd.read_csv(f"{D}/prices_removed.csv.gz", parse_dates=["Date"])
rem = rem[~rem["Ticker"].isin(cons.index)]
px = pd.concat([px, rem])
cons = pd.concat([cons, pd.DataFrame({"Sector": "Unknown", "Universe": "SP400"}, index=rem["Ticker"].unique())])
P = {k: px.pivot(index="Date", columns="Ticker", values=k).sort_index() for k in ["Open", "High", "Low", "Close", "Volume"]}
spy_o, spy_c = P["Open"]["SPY"], P["Close"]["SPY"]
for k in P:
    P[k] = P[k].drop(columns=["SPY", "MDY"], errors="ignore")
O, Hi, Lo, C, V = P["Open"], P["High"], P["Low"], P["Close"], P["Volume"]
cons = cons.reindex(C.columns)
sector = cons["Sector"].fillna("Unknown")
cost = cons["Universe"].map({"SP500": 0.0005, "SP400": 0.001}).fillna(0.001).to_numpy()
dvol = C * V
member = membership(C, dvol)
valid = member & (C > 5) & (dvol.rolling(21).mean() > 5e6)

# ---------- features (all known at close t) ----------
r1 = C.pct_change(fill_method=None)
mkt = spy_c.pct_change()
ma = lambda n: C.rolling(n).mean()  # noqa: E731
vol20 = r1.rolling(20).std()
beta60 = r1.rolling(60).cov(mkt).div(mkt.rolling(60).var(), axis=0)
resid = r1 - beta60.mul(mkt, axis=0)
tr = pd.concat([Hi - Lo, (Hi - C.shift()).abs(), (Lo - C.shift()).abs()]).groupby(level=0).max()
up = r1.clip(lower=0).rolling(14).mean()
dn = (-r1.clip(upper=0)).rolling(14).mean()
hi252, lo252 = C.rolling(252).max(), C.rolling(252).min()
ret = lambda n: C / C.shift(n) - 1  # noqa: E731
sec_mean = lambda df: df.T.groupby(sector).transform("mean").T  # noqa: E731
overnight = O / C.shift() - 1
intraday = C / O - 1

F = {
    "1周涨幅": ret(5),
    "1月涨幅": ret(21),
    "3月涨幅": ret(63),
    "6月涨幅": ret(126),
    "动量12-1": C.shift(21) / C.shift(252) - 1,
    "离20日线": C / ma(20) - 1,
    "离50日线": C / ma(50) - 1,
    "离200日线": C / ma(200) - 1,
    "离52周高点": C / hi252 - 1,
    "离52周低点": C / lo252 - 1,
    "RSI14": 100 - 100 / (1 + up / dn),
    "20日波动率": vol20,
    "60日波动率": r1.rolling(60).std(),
    "残差波动率": resid.rolling(21).std(),
    "贝塔": beta60,
    "ATR占价格": tr.rolling(14).mean() / C,
    "21日最大单日涨幅": r1.rolling(21).max(),
    "21日最大单日跌幅": r1.rolling(21).min(),
    "60日偏度": r1.rolling(60).skew(),
    "5日/60日成交量": V.rolling(5).mean() / V.rolling(60).mean(),
    "成交额(规模)": np.log(dvol.rolling(21).mean()),
    "Amihud非流动性": (r1.abs() / dvol).rolling(21).mean(),
    "隔夜收益(21日)": overnight.rolling(21).sum(),
    "日内收益(21日)": intraday.rolling(21).sum(),
    "上涨天数占比(21日)": (r1 > 0).astype(float).rolling(21).mean().where(r1.notna()),
    "行业1月涨幅": sec_mean(ret(21)),
    "相对行业1月": ret(21) - sec_mean(ret(21)),
    "布林位置": (C - ma(20)) / (2 * C.rolling(20).std()),
    "20日区间位置": (C - Lo.rolling(20).min()) / (Hi.rolling(20).max() - Lo.rolling(20).min()),
    "20日振幅": (Hi.rolling(20).max() - Lo.rolling(20).min()) / C,
    "过去触及率(+3%,10日)": None,  # filled below
}
# stock's own past touch rate of +3% within 10 days (completed windows only)
fwd_hi = Hi[::-1].rolling(10).max()[::-1].shift(-1)
touched = (fwd_hi >= C * 1.03).astype(float).where(fwd_hi.notna())
F["过去触及率(+3%,10日)"] = touched.shift(11).rolling(250, min_periods=120).mean()

# ---------- plan outcomes ----------
cal = C.index
Oa, Ha, Ca = O.to_numpy(), Hi.to_numpy(), C.ffill().to_numpy()
SO, SC = spy_o.to_numpy(), spy_c.ffill().to_numpy()
v20 = vol20.to_numpy()
start = 260
if FIRST is not None:
    start = max(start, cal.searchsorted(FIRST))
sig = list(range(start, len(cal) - 1 - L, STEP))
rows = []
for i in sig:
    ok = valid.iloc[i].to_numpy() & (Oa[i + 1] > 0)
    if ok.sum() < 100:
        continue
    entry = Oa[i + 1]
    hi, op = Ha[i + 1: i + 1 + L], Oa[i + 1: i + 1 + L]
    rec = {"i": np.full(ok.sum(), i), "tk": np.where(ok)[0]}
    for name, n in NS.items():
        nn = np.full(len(entry), n) if n is not None else np.clip(v20[i] * np.sqrt(10), 0.02, 0.25)
        lim = entry * (1 + nn)
        hit = np.zeros(len(entry), bool)
        fill = np.full(len(entry), np.nan)
        days = np.full(len(entry), L)
        with np.errstate(invalid="ignore"):
            for k in range(L):
                new = ~hit & (hi[k] >= lim)
                fill[new] = lim[new] if k == 0 else np.maximum(lim[new], op[k][new])
                days[new] = k + 1
                hit |= new
        r = np.where(hit, fill, Ca[i + L]) / entry - 1 - 2 * cost
        spy = SC[i + days] / SO[i + 1] - 1
        rec[f"ret_{name}"] = r[ok]
        rec[f"ex_{name}"] = (r - spy)[ok]
        rec[f"hit_{name}"] = hit[ok]
        rec[f"days_{name}"] = days[ok]
        rec[f"n_{name}"] = nn[ok]
    for f, df in F.items():
        rec[f] = df.iloc[i].to_numpy()[ok]
    rows.append(pd.DataFrame(rec))
d = pd.concat(rows, ignore_index=True)
d["date"] = cal[d["i"]]
d["ticker"] = C.columns[d["tk"]]
d.to_parquet(f"{OUT}/panel.parquet" if FIRST is None else f"{OUT}/panel_daily.parquet")
print("panel", d.shape, d["date"].min().date(), d["date"].max().date(), d["date"].nunique(), "dates")
