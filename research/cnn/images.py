"""20-day OHLC chart images (Jiang, Kelly & Xiu 2023, "I20"): 64x60 binary pixels, 3 columns per day.

Column 3i: open tick, 3i+1: high-low bar (+ volume bar), 3i+2: close tick. Price area is rows 13..63
(min-max scaled over the window's highs, lows and 20-day MA), volume area rows 0..11 (scaled by the
window's max volume). Row 0 is the bottom of the chart.
"""
import numpy as np

W, HGT, WID = 20, 64, 60
P0, PH, VH = 13, 51, 12  # price area starts at row 13 and spans 51 rows; volume spans 12 rows


def stock_images(o, h, l, c, v, ma, ends):
    """Images for windows ending at each index in `ends`. Returns (images uint8 [n,64,60], valid bool [n])."""
    ends = np.asarray(ends, dtype=int)
    n = len(ends)
    if n == 0:
        return np.zeros((0, HGT, WID), np.uint8), np.zeros(0, bool)
    idx = ends[:, None] + np.arange(-W + 1, 1)
    O, H, L, C, V, M = (x[idx] for x in (o, h, l, c, v, ma))
    valid = (ends >= W - 1) & np.isfinite(np.stack([O, H, L, C, V, M])).all(axis=(0, 2))
    lo = np.fmin(L.min(1), M.min(1))
    hi = np.fmax(H.max(1), M.max(1))
    rng = hi - lo
    vmax = V.max(1)
    valid &= (rng > 0) & (vmax > 0)
    lo, rng, vmax = lo[:, None], np.where(rng > 0, rng, 1)[:, None], np.where(vmax > 0, vmax, 1)[:, None]

    def row(x):
        r = np.nan_to_num((x - lo) / rng * (PH - 1), nan=0.0)
        return np.clip(np.rint(r), 0, PH - 1).astype(int) + P0

    img = np.zeros((n, HGT, WID), np.uint8)
    nn = np.repeat(np.arange(n), W)
    days = np.tile(np.arange(W), n)
    img[nn, row(O).ravel(), 3 * days] = 1
    img[nn, row(C).ravel(), 3 * days + 2] = 1

    # high-low bars and volume bars in the middle column, via a difference array along rows
    d = np.zeros((n, HGT + 1, W), np.int16)
    rl, rh = row(L).ravel(), row(H).ravel()
    np.add.at(d, (nn, np.minimum(rl, rh), days), 1)
    np.add.at(d, (nn, np.maximum(rl, rh) + 1, days), -1)
    vh = np.clip(np.rint(np.nan_to_num(V / vmax) * VH), 0, VH).astype(int).ravel()
    np.add.at(d, (nn, np.zeros_like(vh), days), (vh > 0).astype(np.int16))
    np.add.at(d, (nn, vh, days), -(vh > 0).astype(np.int16))
    mid = np.cumsum(d, axis=1)[:, :HGT, :] > 0
    img[:, :, 1::3] |= mid.astype(np.uint8)

    # moving-average line, linearly interpolated between day midpoints across all 60 columns
    pos = np.clip((np.arange(WID) - 1) / 3, 0, W - 1)
    k = np.minimum(pos.astype(int), W - 2)
    f = pos - k
    mrow = (M - lo) / rng * (PH - 1)
    mline = mrow[:, k] * (1 - f) + mrow[:, k + 1] * f
    mr = np.clip(np.rint(np.nan_to_num(mline)), 0, PH - 1).astype(int) + P0
    img[np.repeat(np.arange(n), WID), mr.ravel(), np.tile(np.arange(WID), n)] = 1

    img[~valid] = 0
    return img, valid
