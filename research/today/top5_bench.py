"""Returns of today's five vs pool average and SPY over 1m / 3m / 1y, plus past-year +3%-in-10-days touch rate."""
import sys

import pandas as pd

sys.path.insert(0, "research/pricedb")
import pricedb as P

tk = [l.strip() for l in open("research/news/tickers.txt") if l.strip() and not l.startswith("#")]
cons = pd.read_csv(f"{P.MEMBERS}/constituents.csv")
C = P.wide("Close", start="2025-01-01")
H = P.wide("High", start="2025-01-01")
mem = [c for c in cons["yf"] if c in C]


def rets(s):
    s = s.dropna()
    return [s.iat[-1] / s.iat[-k] - 1 if len(s) >= k else float("nan") for k in (22, 64, 253)]


fwd = H[::-1].rolling(10).max()[::-1].shift(-1)
touch = (fwd >= C * 1.03).astype(float).where(fwd.notna()).iloc[-262:-11].mean()
rows = {t: rets(C[t]) + [touch.get(t)] for t in tk if t in C}
pool = pd.DataFrame({c: rets(C[c]) for c in mem}).T
rows["池平均"] = list(pool.mean()) + [touch[mem].mean()]
rows["SPY"] = rets(C["SPY"]) + [touch["SPY"]]
print(pd.DataFrame(rows, index=["近1月", "近3月", "近1年", "过去1年10日内碰+3%比例"]).T.round(4).to_string())
