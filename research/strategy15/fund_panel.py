"""Point-in-time fundamental LEVELS from SEC XBRL (reuses research/fullflow/fundamentals.py helpers).
One row per ticker and filing date: TTM revenue / net income / operating income / CFO / capex, balance-sheet items,
shares outstanding, latest-quarter EPS vs a year earlier, and SUE (standardised unexpected earnings, seasonal random walk).
A value is known only from its `filed` date. Output: research/strategy15/data/fund.parquet
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "research/fullflow")
import fundamentals as F  # noqa: E402


def one(item):
    tk, g = item
    flows = {}
    for name, tags in [("rev", F.REV), ("ni", ["NetIncomeLoss"]), ("cfo", ["NetCashProvidedByUsedInOperatingActivities"]),
                       ("capex", ["PaymentsToAcquirePropertyPlantAndEquipment"]), ("opinc", ["OperatingIncomeLoss"]),
                       ("eps", ["EarningsPerShareDiluted"])]:
        x = F.pick(g, tags)
        flows[name] = F.quarters(x) if len(x) else pd.DataFrame(columns=["end", "val", "filed"])
    inst = {k: F.instant(g, t) for k, t in [("eq", ["StockholdersEquity"]), ("ltd", ["LongTermDebtNoncurrent", "LongTermDebt"]),
                                            ("std", ["DebtCurrent", "LongTermDebtCurrent"]), ("ca", ["AssetsCurrent"]),
                                            ("cl", ["LiabilitiesCurrent"]), ("ta", ["Assets"]), ("tl", ["Liabilities"])]}
    shares = g[g["tag"] == "EntityCommonStockSharesOutstanding"].sort_values("filed")[["filed", "val"]]
    T = {k: F.ttm(v) for k, v in flows.items() if k != "eps" and len(v)}
    Y = F.yoy_q(flows["eps"]) if len(flows["eps"]) else pd.DataFrame(columns=["end", "now", "before", "filed"])
    if len(Y):
        Y = Y.sort_values("end").reset_index(drop=True)
        Y["diff"] = Y["now"] - Y["before"]
        Y["sd8"] = Y["diff"].shift(1).rolling(8, min_periods=4).std()
        Y["sue"] = Y["diff"] / Y["sd8"]
    rows = []
    for d in sorted(x for x in g["filed"].unique() if x >= pd.Timestamp("2019-06-01")):
        def last(df, col="val"):
            x = df[df["filed"] <= d]
            return x.sort_values("end")[col].iloc[-1] if len(x) else np.nan
        r = {"ticker": tk, "filed": d}
        for k in ("rev", "ni", "cfo", "capex", "opinc"):
            r[k] = last(T[k]) if k in T else np.nan
        for k in ("eq", "ca", "cl", "ta", "tl"):
            r[k] = last(inst[k])
        r["debt"] = np.nansum([last(inst["ltd"]), last(inst["std"])])
        sh = shares[shares["filed"] <= d]["val"]
        r["shares"] = sh.iloc[-1] if len(sh) else np.nan
        r["shares_1y"] = sh[shares["filed"] <= d - pd.Timedelta(days=330)].iloc[-1] if (shares["filed"] <= d - pd.Timedelta(days=330)).any() else np.nan
        if len(Y):
            y = Y[Y["filed"] <= d]
            if len(y):
                y = y.iloc[-1]
                r.update(eps_q=y["now"], eps_q_1y=y["before"], sue=y["sue"], eps_q_end=y["end"], eps_q_filed=y["filed"])
        rows.append(r)
    return rows


if __name__ == "__main__":
    from multiprocessing import Pool
    os.makedirs("research/strategy15/data", exist_ok=True)
    with Pool(4) as pool:
        rows = [r for rs in pool.map(one, list(F.f.groupby("ticker")), chunksize=8) for r in rs]
    s = pd.DataFrame(rows)
    s.to_parquet("research/strategy15/data/fund.parquet")
    print(s.shape, s["ticker"].nunique(), "tickers")
    print(s.describe().T)
