from _common import *
ctx = load(); C = ctx.Cf
m121 = C.shift(21) / C.shift(252) - 1
m61 = C.shift(21) / C.shift(126) - 1
run("A1", [("main", m121, "hold", 20, 3), ("hold10", m121, "hold", 10, 3), ("form6-1", m61, "hold", 20, 3)], ctx)
