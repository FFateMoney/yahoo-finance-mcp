"""Point-in-time S&P 500 + S&P 400 membership mask, built from Wikipedia change tables.

- S&P 400: add/remove events since 2012 (tickers with no add event were members before 2012).
- S&P 500: current members count from their "Date added".
- A ticker added to the S&P 400 is treated as a member before its add date only if it was larger
  than the median member then (i.e. it most likely moved down from the S&P 500, not up from the 600).
"""
import pandas as pd

M = "research/factors/data/membership"


def membership(C, dvol):
    t = pd.read_csv(f"{M}/sp400_table1.csv")
    t["d"] = pd.to_datetime(t["Date"].str.replace(r"\[.*?\]", "", regex=True), errors="coerce")
    t = t.dropna(subset=["d"])
    ev = pd.concat([
        t[["d", "Added | Ticker"]].set_axis(["d", "s"], axis=1).assign(e=1),
        t[["d", "Removed | Ticker"]].set_axis(["d", "s"], axis=1).assign(e=-1),
    ]).dropna(subset=["s"])
    ev["s"] = ev["s"].str.replace(".", "-", regex=False)
    sp500 = pd.read_csv(f"{M}/sp500_table0.csv")
    add500 = pd.Series(pd.to_datetime(sp500["Date added"], errors="coerce").values,
                       index=sp500["Symbol"].str.replace(".", "-", regex=False))
    idx = C.index
    mask = pd.DataFrame(False, index=idx, columns=C.columns)
    for s in C.columns:
        m = pd.Series(False, index=idx)
        e = ev[ev["s"] == s].sort_values("d")
        if len(e):
            state = e["e"].iloc[0] == -1  # first event is a removal -> was a member before
            if not state:
                d0 = e["d"].iloc[0]
                before = dvol.loc[(idx < d0) & (idx >= d0 - pd.Timedelta(days=90))]
                if len(before) and before[s].mean() > before.stack().median():
                    state = True  # moved down from the S&P 500
            last = idx[0]
            for d, x in zip(e["d"], e["e"]):
                m[(idx >= last) & (idx < d)] = state
                state, last = x == 1, d
            m[idx >= last] = state
        elif s not in add500.index:
            m[:] = True  # current S&P 400 member since before 2012
        if s in add500.index and pd.notna(add500[s]):
            m |= idx >= add500[s]
        mask[s] = m
    return mask
