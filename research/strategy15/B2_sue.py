from B_common import *
ctx = load()
for var, st, ex in [("main", 10, "hold"), ("v1_stale5", 5, "hold"), ("v2_r1", 10, "r1")]:
    run(ctx, "B2 SUE drift", "B2", var, fund_fresh(ctx, "sue", st), 2, ex, 20)
