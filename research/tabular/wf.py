"""Shared walk-forward split for every model (numpy only, so the GPU machine needs nothing else from the repo).
Expanding window, retrain every 63 trading days from 2023-01-03; training rows end 15 trading days before the
prediction block (10-day labels cannot leak); the last 20% of training dates (after a 10-day gap) is the
early-stopping validation set.
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OOS_START = "2023-01-03"
STEP, EMBARGO, GAP = 63, 15, 10


def load_panel(path=None):
    path = path or f"{HERE}/data/panel.npz"
    if os.path.exists(path):
        z = np.load(path)
        P = {k: z[k] for k in z.files}
    else:  # split copy (each part < 30 MB) made by split_panel.py
        P = {}
        for part in ("panel_X.npz", "panel_meta.npz"):
            z = np.load(os.path.join(os.path.dirname(path), part))
            P.update({k: z[k] for k in z.files})
    X = P["X"].astype(np.float32)
    X[P["X"] == 255] = np.nan
    P["Xf"] = X / 254.0  # 0..1 percentile, NaN = missing
    return P


def folds(P, window=None):
    """Yields (train_rows, val_rows, pred_rows). window = rolling length in trading days (None = expanding)."""
    di, dates = P["di"], P["dates"]
    used = np.unique(di)
    start = used[np.searchsorted(dates[used], OOS_START)]
    k0 = np.searchsorted(used, start)
    for k in range(k0, len(used), STEP):
        p0, p1 = used[k], used[min(k + STEP, len(used)) - 1]
        tr_dates = used[(used <= p0 - EMBARGO) & ((used > p0 - EMBARGO - window) if window else True)]
        cut = tr_dates[int(len(tr_dates) * 0.8)]
        tr = np.nonzero((di >= tr_dates[0]) & (di < cut - GAP))[0]
        va = np.nonzero((di >= cut) & (di <= tr_dates[-1]))[0]
        pr = np.nonzero((di >= p0) & (di <= p1))[0]
        yield tr, va, pr


def daily_ic(pred, y, di):
    """Mean of per-date Pearson correlation between a score and the rank label (both near-uniform -> ~RankIC)."""
    ok = np.isfinite(pred)
    out = []
    for d in np.unique(di[ok]):
        m = ok & (di == d)
        if m.sum() > 30:
            out.append(np.corrcoef(pred[m], y[m])[0, 1])
    return np.array(out)


def save_pred(name, pred):
    os.makedirs(f"{HERE}/out", exist_ok=True)
    np.save(f"{HERE}/out/pred_{name}.npy", pred.astype(np.float32))
