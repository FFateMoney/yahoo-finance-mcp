from _common import *
ctx = load(); C = ctx.Cf
s = C / C.rolling(252, min_periods=240).max()
run("A2", [("main", s, "hold", 20, 2), ("hold10", s, "hold", 10, 2), ("r1", s, "r1", 20, 2)], ctx)
