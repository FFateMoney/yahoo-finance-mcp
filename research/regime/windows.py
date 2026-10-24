"""1y / 3m / 1m windows ending at the last bar, from out/extra_2025_2025.csv + out/test.csv."""
import json, sys, os
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from regime_select import strategy, net, CHOICE_FILE
H = os.path.dirname(os.path.abspath(__file__))
df = pd.concat([pd.read_csv(f"{H}/out/extra_2025_2025.csv", index_col=0, parse_dates=[0, 3, 4]),
                pd.read_csv(f"{H}/out/test.csv", index_col=0, parse_dates=[0, 3, 4])])
df = df[~df.index.duplicated(keep="last")].sort_index()
choice = json.load(open(CHOICE_FILE))
s, _ = strategy(df, choice)
m = net(df, "mom")
for lab, k in [("最近1年", 12), ("最近3个月", 3), ("最近1个月", 1)]:
    sub = df.index[-k:]
    f = lambda r: ((1 + r[sub]).prod() - 1) * 100
    print(f"{lab} {sub[0].date()}..{df['exit'].iloc[-1].date()}: 分状态 {f(s):+.1f}%  一直动量 {f(m):+.1f}%  等权 {f(df['EW']):+.1f}%  SPY {f(df['SPY']):+.1f}%")
