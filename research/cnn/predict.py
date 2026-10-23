"""Predict P(up over next 5 days) with a trained CNN on a chunk of test dates.

Usage: python research/cnn/predict.py SEED CHUNK NCHUNKS [--quick]
Test dates: every 5th trading day from 2011-01; chunk k takes dates k, k+NCHUNKS, ...
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
from data import WORK, load, stocks  # noqa: E402
from images import stock_images  # noqa: E402
from model import CNN20, DEVICE, batches  # noqa: E402

SEED, CHUNK, N = (int(a) for a in sys.argv[1:4])
px, test_dates = load("--quick" in sys.argv)
want = set(test_dates.iloc[CHUNK::N])
model = CNN20()
model.load_state_dict(torch.load(f"{WORK}/model_{SEED}.pt", map_location="cpu"))
model.to(DEVICE)
model.eval()
rows = []
for tk, d, arr in stocks(px):
    if len(d) < 20:  # too short for a 20-day image (and negative indices would go out of bounds)
        continue
    te = np.where(pd.Series(d).isin(want).to_numpy())[0]
    if not len(te):
        continue
    im, ok = stock_images(*arr, te)
    if not ok.any():
        continue
    im = im[ok]
    with torch.no_grad():
        p = np.concatenate([torch.softmax(model(xb), 1)[:, 1].cpu().numpy()
                            for _, xb in batches(im, np.arange(len(im)), 2048)])
    rows.append(pd.DataFrame({"Date": d[te[ok]], "Ticker": tk, "p": p.astype(np.float32)}))
out = pd.concat(rows)
out.to_parquet(f"{WORK}/pred_{SEED}_{CHUNK}.parquet", index=False)
print("saved", len(out), "predictions for", out["Date"].nunique(), "dates")
