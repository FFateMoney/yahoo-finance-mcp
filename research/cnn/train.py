"""Train one I20/R5 CNN (Jiang, Kelly & Xiu 2023). Usage: python research/cnn/train.py SEED [--quick]

Pre-registered design: /mnt/project-files/回测/图像形态模型_预登记.md (F127). Do not tune on test results.
Train 2001-2010 (labels end by 2010-12-31), one image per stock every 5 days, random 70/30 train/validation
split, Adam lr 1e-5, batch 128, early stopping after 2 epochs without validation improvement (max 20 epochs,
and no new epoch once the time budget would be exceeded).
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from data import STEP, H, TRAIN_END, TRAIN_START, WORK, load, stocks  # noqa: E402
from images import stock_images  # noqa: E402
from model import CNN20, batches  # noqa: E402

SEED = int(sys.argv[1])
QUICK = "--quick" in sys.argv
BUDGET = float(os.environ.get("TRAIN_BUDGET_MIN", "290")) * 60
T0 = time.time()
torch.manual_seed(SEED)

px, _ = load(QUICK)
rng = np.random.default_rng(0)  # same training sample and split for every seed; seeds differ only in init/order
X, y = [], []
for tk, d, arr in stocks(px):
    c = arr[3]
    if len(d) <= H + 20:
        continue
    lab_ok = np.zeros(len(d), bool)
    lab_ok[: len(d) - H] = d[H:] <= np.datetime64(TRAIN_END)
    cand = np.where((d >= np.datetime64(TRAIN_START)) & lab_ok)[0]
    if not len(cand):
        continue
    cand = cand[int(rng.integers(STEP))::STEP]
    im, ok = stock_images(*arr, cand)
    ret = c[cand + H] / c[cand] - 1
    ok &= np.isfinite(ret)
    X.append(im[ok])
    y.append((ret[ok] > 0).astype(np.int64))
X, y = np.concatenate(X), np.concatenate(y)
perm = rng.permutation(len(X))
cut = int(0.7 * len(X))
tr, va = perm[:cut], perm[cut:]
print(f"train {len(tr)} val {len(va)} images, up share {y.mean():.3f}, built in {time.time() - T0:.0f}s", flush=True)

model = CNN20()
opt = torch.optim.Adam(model.parameters(), lr=1e-5)
lossf = nn.CrossEntropyLoss()
best, bad, state, last = np.inf, 0, None, 0.0
for ep in range(2 if QUICK else 20):
    if time.time() - T0 + last > BUDGET:
        print("stopping: time budget", flush=True)
        break
    t0 = time.time()
    model.train()
    order = tr[torch.randperm(len(tr), generator=torch.Generator().manual_seed(SEED * 100 + ep)).numpy()]
    for j, xb in batches(X, order, 128):
        opt.zero_grad()
        lossf(model(xb), torch.from_numpy(y[j])).backward()
        opt.step()
    model.eval()
    tot, acc = 0.0, 0
    with torch.no_grad():
        for j, xb in batches(X, va, 1024):
            out = model(xb)
            yb = torch.from_numpy(y[j])
            tot += lossf(out, yb).item() * len(j)
            acc += (out.argmax(1) == yb).sum().item()
    vl = tot / len(va)
    last = time.time() - t0
    print(f"epoch {ep} val loss {vl:.5f} acc {acc / len(va):.4f} ({last:.0f}s)", flush=True)
    if vl < best - 1e-6:
        best, bad, state = vl, 0, {k: t.clone() for k, t in model.state_dict().items()}
    else:
        bad += 1
        if bad >= 2:
            break
torch.save(state, f"{WORK}/model_{SEED}.pt")
print("saved model; best val loss", round(best, 5))
