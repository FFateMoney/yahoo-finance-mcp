from B_common import *
ctx = load()
sue = fund_fresh(ctx, "sue", 100); ear = ear_frame(ctx, 60)
mom = ctx.Cf.shift(21) / ctx.Cf.shift(126) - 1
pr = lambda x: x.where(ctx.univ).rank(axis=1, pct=True)
comb = (pr(sue) + pr(ear) + pr(mom)) / 3
comb = comb.where(sue.notna() & ear.notna() & mom.notna() & ctx.univ)
for var, ex, h in [("main", "hold", 20), ("v1_h10", "hold", 10), ("v2_r1", "r1", 20)]:
    run(ctx, "B5 earnings+price momentum", "B5", var, comb, 2, ex, h)
