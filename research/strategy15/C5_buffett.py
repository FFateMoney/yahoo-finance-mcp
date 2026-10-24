from C_common import *
ctx = load(); C = ctx.C
g = lambda c: fund_daily(ctx, c)
ni, cfo, ta, rev, op, tl, sh = [g(c) for c in "ni cfo ta rev opinc tl shares".split()]
u = ctx.univ
r = C.pct_change(fill_method=None)
spy = ctx.spy["close"].reindex(C.index).pct_change()
beta = r.rolling(252, min_periods=200).cov(spy).div(spy.rolling(252, min_periods=200).var(), axis=0)
vol = r.rolling(252, min_periods=200).std()
mc = C * sh
prof = (zs((ni / ta).where(u)) + zs((cfo / ta).where(u)) + zs((op / rev).where(u))) / 3
safe = (zs((-beta).where(u)) + zs((-vol).where(u)) + zs((-tl / ta).where(u))) / 3
cheap = (zs((ni / mc).where(u)) + zs((cfo / mc).where(u))) / 2
full = (prof + safe + cheap) / 3
noprice = (prof + safe) / 2
run(ctx, "C5", "Buffett QMJ low-risk cheap", [("main", full, "hold", 20), ("no_price", noprice, "hold", 20), ("r1", full, "r1", 20)])
