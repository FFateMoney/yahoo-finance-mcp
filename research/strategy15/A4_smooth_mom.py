from _common import *
ctx = load(); C = ctx.Cf
mom = C.shift(21) / C.shift(252) - 1
d = np.sign(C.pct_change(fill_method=None))
pos = (d > 0).astype(float).where(d.notna()); neg = (d < 0).astype(float).where(d.notna())
w = lambda x: x.shift(21).rolling(231, min_periods=200).mean()   # days t-251..t-21
ID = np.sign(mom) * (w(neg) - w(pos))
rk = mom.where(ctx.univ).rank(axis=1, pct=True)
def sc(cut): return (-ID).where(rk >= cut)
run("A4", [("main", sc(0.8), "hold", 20, 3), ("decile", sc(0.9), "hold", 20, 3), ("hold10", sc(0.8), "hold", 10, 3)], ctx)
