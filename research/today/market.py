"""Market state and benchmark returns: SPY / MDY / equal-weight pool, VIX, breadth, sector moves."""
import sys

import pandas as pd
import yfinance as yf

sys.path.insert(0, "research/pricedb")
import pricedb as P

cons = pd.read_csv(f"{P.MEMBERS}/constituents.csv")
C = P.wide("Close", start="2025-01-01")
t = C.index[-1]
print("数据日期", t.date())
mem = [c for c in cons["yf"] if c in C]
r = C[mem].pct_change(fill_method=None)
ew = (1 + r.mean(1).fillna(0)).cumprod()
for name, s in [("SPY", C["SPY"]), ("MDY", C["MDY"] if "MDY" in C else None), ("池等权", ew)]:
    if s is None:
        continue
    s = s.dropna()
    ma20, ma50, ma200 = s.rolling(20).mean().iat[-1], s.rolling(50).mean().iat[-1], s.rolling(200).mean().iat[-1]
    hi = s.tail(252).max()
    print(f"{name}: 1月 {s.iat[-1]/s.iat[-22]-1:+.2%} 3月 {s.iat[-1]/s.iat[-64]-1:+.2%} 1年 {s.iat[-1]/s.iat[-253]-1:+.2%} | "
          f"离20日线 {s.iat[-1]/ma20-1:+.2%} 50日线 {s.iat[-1]/ma50-1:+.2%} 200日线 {s.iat[-1]/ma200-1:+.2%} 离一年高点 {s.iat[-1]/hi-1:+.2%}")
print("\n最近 15 个交易日：")
d = pd.DataFrame({"SPY": C["SPY"], "SPY日涨跌": C["SPY"].pct_change(), "池等权日涨跌": r.mean(1),
                  "上涨家数比例": (r > 0).mean(1)}).tail(15)
print(d.round(4).to_string())
cm = C[mem]
print(f"\n成分股在 50 日线上方比例 {(cm.iloc[-1] > cm.rolling(50).mean().iloc[-1]).mean():.1%}，"
      f"200 日线上方 {(cm.iloc[-1] > cm.rolling(200).mean().iloc[-1]).mean():.1%}")
sec = pd.DataFrame({"sector": cons.set_index("yf")["Sector"], "m1": cm.iloc[-1] / cm.iloc[-22] - 1, "w1": cm.iloc[-1] / cm.iloc[-6] - 1})
print("\n行业（等权）近 1 周 / 1 月：")
print(sec.groupby("sector")[["w1", "m1"]].mean().sort_values("m1").round(4).to_string())
try:
    v = yf.download(["^VIX"], period="3mo", progress=False, auto_adjust=True)["Close"].squeeze()
    print(f"\nVIX 最新 {v.iat[-1]:.2f}（{v.index[-1].date()}），1 月前 {v.iat[-22]:.2f}，3 月均值 {v.mean():.2f}，3 月最高 {v.max():.2f}")
    print(v.tail(10).round(2).to_string())
except Exception as ex:
    print("VIX 获取失败", ex)
