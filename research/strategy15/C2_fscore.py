from C_common import *
ctx = load(); C = ctx.C
g = lambda c: fund_daily(ctx, c)
ni, cfo, ta, rev, op, debt, ca, cl, sh, sh1, eq = [g(c) for c in "ni cfo ta rev opinc debt ca cl shares shares_1y eq".split()]
L = 252
roa = ni / ta
f = ((roa > 0).astype(float) + (cfo > 0) + (roa > roa.shift(L)) + (cfo > ni) + ((debt.fillna(0) / ta) < (debt.fillna(0) / ta).shift(L))
     + ((ca / cl) > (ca / cl).shift(L)) + (sh <= sh1 * 1.005) + ((op / rev) > (op / rev).shift(L)) + ((rev / ta) > (rev / ta).shift(L))).astype(float)
valid = roa.notna() & cfo.notna() & roa.shift(L).notna() & ta.notna() & sh.notna() & sh1.notna()
bm = (eq / (C * sh)).where(eq > 0)
bmr = rk(bm, ctx.univ)
def fs(q):
    return (f + bmr / 10).where(valid & (bmr >= q))
run(ctx, "C2", "Piotroski F-score", [("main", fs(0.8), "hold", 20), ("bm_tercile", fs(2 / 3), "hold", 20), ("r1", fs(0.8), "r1", 20)])
