from B_common import *
ctx = load()
for var, lim, ex, h in [("v1_N1", 0, "hold", 20), ("v2_r1", 4, "r1", 20)]:
    run(ctx, "B1 EAR drift", "B1", var, ear_frame(ctx, lim), 2, ex, h)
