"""Shared settings and per-stock price arrays for the chart-image CNN."""
import numpy as np
import pandas as pd

WORK = "research/cnn/work"
TRAIN_START, TRAIN_END, TEST_START = "2001-01-01", "2010-12-31", "2011-01-01"
STEP, H = 5, 5


def load(quick=False):
    px = pd.read_parquet(f"{WORK}/prices.parquet").sort_values(["Ticker", "Date"])
    cal = px.loc[px["Ticker"] == "SPY", "Date"].sort_values()
    px = px[px["Ticker"] != "SPY"]
    test_dates = cal[cal >= TEST_START].iloc[::STEP].reset_index(drop=True)
    if quick:
        px = px[px["Ticker"].isin(px["Ticker"].drop_duplicates().sample(60, random_state=0))]
        test_dates = test_dates.iloc[:40]
    return px, test_dates


def stocks(px):
    for tk, g in px.groupby("Ticker", sort=False):
        o, h, l, c, v = (g[k].to_numpy(float) for k in ("Open", "High", "Low", "Close", "Volume"))
        ma = pd.Series(c).rolling(20).mean().to_numpy()
        yield tk, g["Date"].to_numpy(), (o, h, l, c, v, ma)
