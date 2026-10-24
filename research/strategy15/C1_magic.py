from C_common import *
ctx = load(); C = ctx.C
op = fund_daily(ctx, "opinc"); sh = fund_daily(ctx, "shares"); debt = fund_daily(ctx, "debt").fillna(0)
ta, cl, eq = fund_daily(ctx, "ta"), fund_daily(ctx, "cl"), fund_daily(ctx, "eq")
ok = ctx.univ & ~ctx.sector.isin(["Financials", "Utilities"]).reindex(C.columns).values[None, :]
ev = C * sh + debt
ey = (op / ev).where(ev > 0)
def mf(cap):
    roc = (op / cap).where(cap > 0)
    return rk(ey, ok) + rk(roc, ok)  # higher = better (fixed 2026-10-24: sign was reversed)
run(ctx, "C1", "Magic Formula", [("main", mf(ta - cl), "hold", 20), ("roc_eq+debt", mf(eq + debt), "hold", 20), ("r1", mf(ta - cl), "r1", 20)])
