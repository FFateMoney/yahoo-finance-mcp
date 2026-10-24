"""Champion checks for B5 (earnings + price momentum, hold 10): neighbour parameters, S&P 500 vs 400, top-1 trade
concentration. These are robustness checks, not new attempts (but are counted in the registry).
"""
import os, sys
sys.path.insert(0, "research/strategy15")
import numpy as np, pandas as pd
from B_common import ear_frame, fund_fresh
from harness import load, evaluate, save, _outcomes, windows, CAPITAL

ctx = load()
OUT = "research/strategy15/results_check.jsonl"
pr = lambda x: x.where(ctx.univ).rank(axis=1, pct=True)
def b5(sue_age=100, ear_win=60, mom_from=126, parts=("sue", "ear", "mom"), mask=None):
    sue = fund_fresh(ctx, "sue", sue_age); ear = ear_frame(ctx, ear_win)
    mom = ctx.Cf.shift(21) / ctx.Cf.shift(mom_from) - 1
    p = {"sue": sue, "ear": ear, "mom": mom}
    comb = sum(pr(p[k]) for k in parts) / len(parts)
    ok = ctx.univ
    for k in parts:
        ok = ok & p[k].notna()
    if mask is not None:
        ok = ok & mask
    return comb.where(ok)
sp5 = pd.DataFrame(np.repeat(ctx.sp500.reindex(ctx.C.columns).values[None, :], len(ctx.cal), 0), index=ctx.cal, columns=ctx.C.columns)
runs = [("champ", {}, 10), ("hold20", {}, 20), ("hold5", {}, 5), ("mom12-1", {"mom_from": 252}, 10), ("ear30", {"ear_win": 30}, 10),
        ("ear90", {"ear_win": 90}, 10), ("sue60", {"sue_age": 60}, 10), ("no_sue", {"parts": ("ear", "mom")}, 10),
        ("mom_only", {"parts": ("mom",)}, 10), ("SP500 only", {"mask": sp5}, 10), ("SP400 only", {"mask": ~sp5}, 10)]
for lab, kw, h in runs:
    r = evaluate(f"B5 check [{lab}]", b5(**kw), ctx, exit="hold", hold=h)
    r["attempt"], r["variant"] = "B5chk", lab
    save(r, OUT)
    if lab == "champ":
        ret, _ = _outcomes(ctx, "hold", 10)
        s = b5().where(ret.notna())
        cap = s.where(ctx.O.shift(-1) <= CAPITAL)
        top1 = ret.where(cap.rank(axis=1, ascending=False, method="first") == 1).stack().dropna()
        end, W = windows(ctx)
        y = top1[top1.index.get_level_values(0) >= W["最近1年"][0]]
        y = y[y.index.get_level_values(0) <= end]
        g = y.groupby(level=1).agg(["count", "mean", "sum"]).sort_values("sum", ascending=False)
        print("1y top-1 by ticker (days picked, mean ret, sum):\n", g.head(10).round(3).to_string(), "\n ... n tickers", len(g))
        print("share of total from top 3 tickers:", round(g["sum"].head(3).sum() / g["sum"].sum(), 2))
        pd.DataFrame(y.rename("ret")).to_csv("research/strategy15/champ_top1_1y.csv")
