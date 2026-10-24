from _common import *
ctx = load(); C = ctx.Cf
sec = ctx.sector.reindex(C.columns)
stock = C.shift(21) / C.shift(252) - 1
def build(form):
    r = C.shift(21) / C.shift(form) - 1
    r = r.where(ctx.univ)
    sr = r.T.groupby(sec.values).mean().T            # date x sector
    sr = sr.drop(columns=["Unknown"], errors="ignore")
    srank = sr.rank(axis=1)                           # higher=better sector
    srk = pd.DataFrame(srank.reindex(columns=sec.values).to_numpy(), index=C.index, columns=C.columns)
    pct = stock.where(ctx.univ).T.groupby(sec.values).rank(pct=True).T
    return srk + pct
s6 = build(126); s12 = build(252)
run("A3", [("main", s6, "hold", 20, 2), ("sec12-1", s12, "hold", 20, 2), ("hold10", s6, "hold", 10, 2)], ctx)
