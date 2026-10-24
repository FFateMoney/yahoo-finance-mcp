from B_common import *
ctx = load()
sh = fund_fresh(ctx, "shares", 200); s1 = fund_fresh(ctx, "shares_1y", 200)
chg = sh / s1 - 1
def sc(floor): return (-chg).where((chg <= 0) & (chg >= floor))
for var, fl, ex in [("main", -0.15, "hold"), ("v1_floor10", -0.10, "hold"), ("v2_r1", -0.15, "r1")]:
    run(ctx, "B4 net buyback", "B4", var, sc(fl), 2, ex, 20)
