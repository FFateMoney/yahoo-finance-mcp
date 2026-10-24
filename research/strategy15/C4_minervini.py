from C_common import *
ctx = load(); C = ctx.C
m50, m150, m200 = [C.rolling(n).mean() for n in (50, 150, 200)]
lo, hi = C.rolling(252, min_periods=240).min(), C.rolling(252, min_periods=240).max()
rs = rk(C / C.shift(252) - 1, ctx.univ)
ok = (C > m150) & (C > m200) & (m150 > m200) & (m200 > m200.shift(22)) & (m50 > m150) & (m50 > m200) & (C > m50) & (C >= 1.3 * lo) & (C >= 0.75 * hi) & (rs >= 0.7)
sc = rs.where(ok)
run(ctx, "C4", "Minervini Trend Template", [("main", sc, "hold", 20), ("hold10", sc, "hold", 10), ("r1", sc, "r1", 20)])
