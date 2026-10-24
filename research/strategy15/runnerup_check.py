"""Same champion checks for the runners-up: A3 industry momentum (sec 12-1, hold 20) and C5 Buffett-style (hold 20).
Neighbour parameters, S&P 500 vs 400, and top-1 concentration in the last year."""
import sys
sys.path.insert(0, "research/strategy15")
import numpy as np, pandas as pd
from harness import load, evaluate, save, fund_daily, _outcomes, windows, CAPITAL

ctx = load(); C = ctx.Cf
OUT = "research/strategy15/results_check.jsonl"
sp5 = pd.DataFrame(np.repeat(ctx.sp500.reindex(C.columns).values[None, :], len(ctx.cal), 0), index=ctx.cal, columns=C.columns)
sec = ctx.sector.reindex(C.columns)

def a3(form=252, skip=21, mask=None):
    r = (C.shift(skip) / C.shift(form) - 1).where(ctx.univ)
    sr = r.T.groupby(sec.values).mean().T.drop(columns=["Unknown"], errors="ignore").rank(axis=1)
    srk = pd.DataFrame(sr.reindex(columns=sec.values).to_numpy(), index=C.index, columns=C.columns)
    pct = (C.shift(21) / C.shift(252) - 1).where(ctx.univ).T.groupby(sec.values).rank(pct=True).T
    s = srk + pct
    return s.where(mask) if mask is not None else s

g = lambda c: fund_daily(ctx, c)
ni, cfo, ta, rev, op, tl, sh = [g(c) for c in "ni cfo ta rev opinc tl shares".split()]
u = ctx.univ
zs = lambda df: df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0).clip(-3, 3)
r_ = ctx.C.pct_change(fill_method=None)
spy = ctx.spy["close"].reindex(ctx.C.index).pct_change()
def c5(win=252, mask=None):
    beta = r_.rolling(win, min_periods=int(win * .8)).cov(spy).div(spy.rolling(win, min_periods=int(win * .8)).var(), axis=0)
    vol = r_.rolling(win, min_periods=int(win * .8)).std()
    mc = ctx.C * sh
    prof = (zs((ni / ta).where(u)) + zs((cfo / ta).where(u)) + zs((op / rev).where(u))) / 3
    safe = (zs((-beta).where(u)) + zs((-vol).where(u)) + zs((-tl / ta).where(u))) / 3
    cheap = (zs((ni / mc).where(u)) + zs((cfo / mc).where(u))) / 2
    s = (prof + safe + cheap) / 3
    return s.where(mask) if mask is not None else s

def conc(score, hold, tag):
    ret, _ = _outcomes(ctx, "hold", hold)
    s = score.where(ctx.univ).where(ret.notna())
    cap = s.where(ctx.O.shift(-1) <= CAPITAL)
    top1 = ret.where(cap.rank(axis=1, ascending=False, method="first") == 1).stack().dropna()
    end, W = windows(ctx)
    d = top1.index.get_level_values(0)
    y = top1[(d >= W["最近1年"][0]) & (d <= end)]
    gg = y.groupby(level=1).agg(["count", "mean", "sum"]).sort_values("sum", ascending=False)
    print(f"CONC {tag}: {len(gg)} tickers; top3 share of summed ret {gg['sum'].head(3).sum() / gg['sum'].sum():.2f}")
    print(gg.head(6).round(3).to_string())

runs = [("A3 原样", a3(), 20), ("A3 持有10", a3(), 10), ("A3 板块用6-1", a3(126), 20), ("A3 板块用9-1", a3(189), 20),
        ("A3 只S&P500", a3(mask=sp5), 20), ("A3 只S&P400", a3(mask=~sp5), 20),
        ("C5 原样", c5(), 20), ("C5 持有10", c5(), 10), ("C5 风险用126天", c5(126), 20),
        ("C5 只S&P500", c5(mask=sp5), 20), ("C5 只S&P400", c5(mask=~sp5), 20)]
for lab, sc, h in runs:
    r = evaluate(lab, sc, ctx, exit="hold", hold=h)
    r["attempt"], r["variant"] = lab[:2] + "chk", lab
    save(r, OUT)
    if lab.endswith("原样"):
        conc(sc, h, lab)
