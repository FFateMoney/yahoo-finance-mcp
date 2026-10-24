from B_common import *
ctx = load()
e = ctx.ear.dropna(subset=["ear"]).sort_values(["ticker", "event"]).copy()
e["m4"] = e.groupby("ticker")["ear"].transform(lambda s: s.rolling(4, min_periods=3).mean())
w = e.pivot_table(index="known", columns="ticker", values="m4", aggfunc="last")
m4 = w.reindex(ctx.cal.union(w.index)).ffill().reindex(ctx.cal).reindex(columns=ctx.C.columns)  # known only at/after each event's known date
dte = ctx.days_to_earn
def sc(lo, hi): return m4.where((dte >= lo) & (dte <= hi))
for var, lo, hi, ex, h in [("main", 3, 9, "hold", 10), ("v1_h5", 1, 4, "hold", 5), ("v2_r1", 3, 9, "r1", 20)]:
    run(ctx, "B3 EA premium", "B3", var, sc(lo, hi), 3, ex, h)
