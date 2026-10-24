from C_common import *
ctx = load(); C = ctx.C
e, e1 = fund_daily(ctx, "eps_q"), fund_daily(ctx, "eps_q_1y")
g = (e / e1 - 1).where(e1 > 0)
hi = C.rolling(252, min_periods=240).max()
rs = rk(C / C.shift(252) - 1, ctx.univ)
spy = ctx.spy["close"].reindex(C.index)
mkt = (spy > spy.rolling(200).mean()).astype(float).replace(0, np.nan)
sc = rs.where((g >= 0.25) & (C >= 0.85 * hi) & (rs >= 0.8)).mul(mkt, axis=0)
run(ctx, "C3", "CAN SLIM", [("main", sc, "hold", 20), ("hold10", sc, "hold", 10), ("r1", sc, "r1", 20)])
