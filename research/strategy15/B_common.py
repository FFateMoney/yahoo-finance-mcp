import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from harness import load, evaluate, fund_daily, save
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_B.jsonl")

def ear_frame(ctx, limit):
    """EAR placed on the `known` row (close of day+1), forward-filled `limit` extra trading days. No look-ahead."""
    e = ctx.ear.dropna(subset=["ear"]).sort_values("known")
    w = e.pivot_table(index="known", columns="ticker", values="ear", aggfunc="last")
    w = w.reindex(ctx.cal.union(w.index)).reindex(ctx.cal)  # known dates are trading days
    return (w.ffill(limit=limit) if limit > 0 else w).reindex(columns=ctx.C.columns)

def run(ctx, label, attempt, variant, score, nparams, exit="hold", hold=20):
    res = evaluate(f"{label} [{variant}]", score, ctx, exit=exit, hold=hold)
    res["attempt"], res["variant"], res["n_params"] = attempt, variant, nparams
    save(res, OUT)
    return res

def fund_fresh(ctx, col, stale):
    """Like harness.fund_daily but with a working staleness cut (harness's age calc is off by 1e3 under pandas 3 datetime64[us])."""
    f = ctx.fund_rows.dropna(subset=[col]).sort_values("filed")
    f = f[f["ticker"].isin(ctx.C.columns)].assign(avail=lambda x: x["filed"] + pd.Timedelta(days=1))
    w = f.pivot_table(index="avail", columns="ticker", values=col, aggfunc="last")
    ix = w.index.union(ctx.cal)
    val = w.reindex(ix).ffill().reindex(ctx.cal)
    dts = pd.DataFrame(np.where(w.notna(), w.index.to_numpy()[:, None], np.datetime64("NaT")), index=w.index, columns=w.columns)
    last = dts.reindex(ix).ffill().reindex(ctx.cal)
    age = (pd.Series(ctx.cal, index=ctx.cal).to_numpy()[:, None] - last.to_numpy()) / np.timedelta64(1, "D")
    return val.mask(age > stale).reindex(columns=ctx.C.columns)
