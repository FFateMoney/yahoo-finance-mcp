from _common import *
ctx = load(); C = ctx.Cf
sec = ctx.sector.reindex(C.columns)
r5 = C / C.shift(5) - 1
adj = r5 - r5.where(ctx.univ).T.groupby(sec.values).transform("mean").T
spy = ctx.spy["close"].ffill().pct_change()
vol = spy.rolling(21).std()
stress = vol > vol.rolling(252, min_periods=200).quantile(0.75)
print("stress days share:", stress.mean())
base = -adj
s = base.where(pd.DataFrame(np.repeat(stress.values[:, None], base.shape[1], 1), index=base.index, columns=base.columns))
run("A5", [("main", s, "hold", 5, 3), ("hold10", s, "hold", 10, 3), ("allday", base, "hold", 5, 2)], ctx)
