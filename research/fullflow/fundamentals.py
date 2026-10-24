"""Point-in-time financial health score (the process's Finviz-style 10-point screen) from SEC XBRL facts.
A value counts on a date only after the filing that reported it (the fact's `filed` date).
Output: research/fullflow/data/scores.parquet with one row per ticker and filing date.
"""
import os

import numpy as np
import pandas as pd

OUT = "research/fullflow/data"
f = pd.read_csv(f"{OUT}/facts.csv.gz", parse_dates=["start", "end", "filed"])
f = f[f["unit"].isin(["USD", "USD/shares", "shares"])]
if os.environ.get("ONLY"):  # quick check on a few tickers, e.g. ONLY=AAPL,JNJ
    f = f[f["ticker"].isin(os.environ["ONLY"].split(","))]
REV = ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet"]
INT = ["InterestExpense", "InterestExpenseNonoperating", "InterestExpenseDebt"]


def first_known(g):
    """One value per period: the earliest filing that reported it."""
    return g.sort_values("filed").drop_duplicates(["start", "end"], keep="first")


def quarters(g):
    """Quarterly values (end -> (val, filed)) from 3-month facts, or YTD/annual minus the previous YTD with the same start."""
    g = first_known(g).dropna(subset=["start"])
    g = g.assign(dur=(g["end"] - g["start"]).dt.days)
    q = {}
    for r in g[g["dur"].between(80, 100)].itertuples():
        q[r.end] = (r.val, r.filed)
    long = g[g["dur"] > 100]
    for r in long.itertuples():
        if r.end in q:
            continue
        prev = long[(long["start"] - r.start).abs().dt.days.le(5) & ((r.end - long["end"]).dt.days.between(80, 100))]
        if len(prev) == 0 and r.dur <= 200:  # 6-month YTD minus the 3-month first quarter
            prev = g[g["dur"].between(80, 100) & (g["start"] - r.start).abs().dt.days.le(5)]
        if len(prev):
            p = prev.iloc[0]
            q[r.end] = (r.val - p.val, max(r.filed, p.filed))
    return pd.DataFrame([(e, v, fl) for e, (v, fl) in q.items()], columns=["end", "val", "filed"]).sort_values("end")


def ttm(qs):
    """Trailing-12-month sums at each quarter end, known from the latest of the four filings."""
    rows = []
    e = qs.set_index("end")
    for end in e.index:
        need = [end - pd.DateOffset(months=m) for m in (0, 3, 6, 9)]
        got = [e[(abs((e.index - d).days) <= 12)] for d in need]
        if all(len(x) for x in got):
            rows.append((end, sum(x["val"].iloc[0] for x in got), max(x["filed"].iloc[0] for x in got)))
    return pd.DataFrame(rows, columns=["end", "val", "filed"])


def yoy_q(qs):
    """Latest-quarter value vs the same quarter a year earlier."""
    e = qs.set_index("end")
    rows = []
    for end, r in e.iterrows():
        prev = e[abs((e.index - (end - pd.DateOffset(years=1))).days) <= 12]
        if len(prev):
            rows.append((end, r["val"], prev["val"].iloc[0], max(r["filed"], prev["filed"].iloc[0])))
    return pd.DataFrame(rows, columns=["end", "now", "before", "filed"])


def pick(g, tags):
    """Rows of the first tag in `tags` that the company uses (per period, earlier tags win)."""
    out = []
    for t in tags:
        x = g[g["tag"] == t]
        if len(x):
            out.append(x)
    if not out:
        return g.iloc[0:0]
    x = pd.concat(out)
    x["pri"] = x["tag"].map({t: k for k, t in enumerate(tags)})
    return x.sort_values("pri").drop_duplicates(["start", "end", "filed"], keep="first")


def instant(g, tags):
    x = pick(g, tags)
    x = x[x["start"].isna()]
    return x.sort_values("filed").drop_duplicates("end", keep="first")[["end", "val", "filed"]]


PRETAX = "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest"


def combine(a, b, sign):
    """Quarter-by-quarter a + sign * b (same period end), known once both are filed."""
    m = a.merge(b, on="end", suffixes=("", "_b"))
    return pd.DataFrame({"end": m["end"], "val": m["val"] + sign * m["val_b"], "filed": m[["filed", "filed_b"]].max(axis=1)})


def score_one(item):
    tk, g = item
    rows = []
    flows = {}
    for name, tags in [("rev", REV), ("ni", ["NetIncomeLoss"]), ("cfo", ["NetCashProvidedByUsedInOperatingActivities"]),
                       ("capex", ["PaymentsToAcquirePropertyPlantAndEquipment"]), ("opinc", ["OperatingIncomeLoss"]),
                       ("int", INT), ("eps", ["EarningsPerShareDiluted"]), ("pretax", [PRETAX]),
                       ("tax", ["IncomeTaxExpenseBenefit"])]:
        x = pick(g, tags)
        flows[name] = quarters(x) if len(x) else pd.DataFrame(columns=["end", "val", "filed"])
    # fallbacks for companies that do not use the main tags
    if not len(flows["ni"]) and len(flows["pretax"]) and len(flows["tax"]):
        flows["ni"] = combine(flows["pretax"], flows["tax"], -1)
    if not len(flows["opinc"]) and len(flows["pretax"]):
        flows["opinc"] = combine(flows["pretax"], flows["int"], +1) if len(flows["int"]) else flows["pretax"]
    if not len(flows["eps"]):
        flows["eps"] = flows["ni"]  # growth of net income instead of EPS
    del flows["pretax"], flows["tax"]
    inst = {k: instant(g, t) for k, t in [("eq", ["StockholdersEquity"]), ("ltd", ["LongTermDebtNoncurrent", "LongTermDebt"]),
                                          ("std", ["DebtCurrent", "LongTermDebtCurrent"]), ("ca", ["AssetsCurrent"]),
                                          ("cl", ["LiabilitiesCurrent"]), ("ta", ["Assets"]), ("tl", ["Liabilities"]),
                                          ("re", ["RetainedEarningsAccumulatedDeficit"])]}
    shares = g[(g["tag"] == "EntityCommonStockSharesOutstanding")].sort_values("filed")[["filed", "val"]]
    T = {k: ttm(v) for k, v in flows.items() if k != "eps" and len(v)}
    Y = {k: yoy_q(flows[k]) for k in ("rev", "eps") if len(flows[k])}
    if "rev" not in T or "ni" not in T:
        return rows
    # evaluate at every filing date of this company
    for d in sorted(x for x in g["filed"].unique() if x >= pd.Timestamp(os.environ.get("SCORE_FROM", "2020-06-01"))):  # SCORE_FROM=2018-03-01 for the 2018-21 check
        def last(df, col="val"):
            x = df[df["filed"] <= d]
            return x.sort_values("end")[col].iloc[-1] if len(x) else np.nan

        def last_end(df):
            x = df[df["filed"] <= d]
            return x["end"].max() if len(x) else pd.NaT
        rev, ni = last(T["rev"]), last(T["ni"])
        if not np.isfinite(rev) or rev <= 0:
            continue
        cfo = last(T["cfo"]) if "cfo" in T else np.nan
        capex = last(T["capex"]) if "capex" in T else 0.0
        opinc = last(T["opinc"]) if "opinc" in T else np.nan
        intexp = last(T["int"]) if "int" in T else np.nan
        eq, ca, cl, ta, tl, re_ = (last(inst[k]) for k in ("eq", "ca", "cl", "ta", "tl", "re"))
        debt = np.nansum([last(inst["ltd"]), last(inst["std"])])
        if not np.isfinite(tl) and np.isfinite(ta) and np.isfinite(eq):
            tl = ta - eq
        ry = Y["rev"][Y["rev"]["filed"] <= d].sort_values("end").tail(1) if "rev" in Y else pd.DataFrame()
        ey = Y["eps"][Y["eps"]["filed"] <= d].sort_values("end").tail(1) if "eps" in Y else pd.DataFrame()
        margin = ni / rev
        fcf = cfo - (capex if np.isfinite(capex) else 0)
        de = debt / eq if eq and eq > 0 else np.nan
        roic = opinc * 0.79 / (debt + eq) if eq and np.isfinite(opinc) and (debt + eq) > 0 else np.nan
        cr = ca / cl if cl and cl > 0 else np.nan
        s = 0
        s += (margin > 0) + (margin >= 0.10)
        s += 2 * (fcf > 0)
        s += bool(de < 1)
        s += 2 * bool(roic >= 0.10) + bool(0.05 <= roic < 0.10)
        s += bool(len(ey) and ey["now"].iloc[0] > ey["before"].iloc[0])
        s += bool(len(ry) and ry["now"].iloc[0] > ry["before"].iloc[0])
        s += bool(cr >= 1)
        sh = shares[shares["filed"] <= d]["val"]
        rows.append({"ticker": tk, "filed": d, "score": int(s), "margin": margin, "fcf": fcf, "de": de, "roic": roic, "cr": cr,
                     "int_cov": opinc / intexp if intexp and intexp > 0 else np.inf,
                     "wc_ta": (ca - cl) / ta if ta else np.nan, "re_ta": re_ / ta if ta else np.nan,
                     "ebit_ta": opinc / ta if ta else np.nan, "sales_ta": rev / ta if ta else np.nan,
                     "tl": tl, "shares": sh.iloc[-1] if len(sh) else np.nan, "period_end": last_end(T["rev"])})
    return rows


if __name__ == "__main__":
    from multiprocessing import Pool
    with Pool(4) as pool:
        rows = [r for rs in pool.map(score_one, list(f.groupby("ticker")), chunksize=8) for r in rs]
    s = pd.DataFrame(rows)
    s.to_parquet(f"{OUT}/scores.parquet" if not os.environ.get("ONLY") else f"{OUT}/scores_check.parquet")
    print(s.shape, s["ticker"].nunique(), "tickers")
    print(s.groupby(s["filed"].dt.year)["score"].describe())
