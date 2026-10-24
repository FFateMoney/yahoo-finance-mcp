"""Score every out/pred_<name>.npy on the shared bench (hold 10 = decision; R1 exit = extra) and the dev-period RankIC.
Usage: python research/tabular/evaluate_preds.py [name ...]   -> out/results.txt
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "strategy15"))
import harness  # noqa: E402
from harness import evaluate, load, nw_t, report  # noqa: E402
from wf import load_panel  # noqa: E402
from cpu_models import daily_ic_fast  # noqa: E402

DEV = ("2023-01-03", "2025-09-22")
ctx = load()
# out-of-sample predictions start 2023-01-03, so the bench's first window (and its random-picker range) starts there too
harness.TRAIN_START = pd.Timestamp(DEV[0])
harness.CACHE = os.path.join(HERE, "data", "bench")
os.makedirs(harness.CACHE, exist_ok=True)
P = load_panel()
di, ti, y = P["di"], P["ti"], P["y"]
names = sys.argv[1:] or sorted(os.path.basename(p)[5:-4] for p in glob.glob(f"{HERE}/out/pred_*.npy"))
lines = []
for n in names:
    pred = np.load(f"{HERE}/out/pred_{n}.npy")
    S = np.full(ctx.C.shape, np.nan)
    S[di, ti] = pred
    S = pd.DataFrame(S, index=ctx.cal, columns=ctx.C.columns)
    ok = np.isfinite(pred)
    ic = pd.Series(daily_ic_fast(pred[ok], y[ok], di[ok]), index=pd.DatetimeIndex(P["dates"][np.unique(di[ok])]))
    dev_ic = ic.loc[DEV[0]:DEV[1]]
    head = f"\n##### {n}: dev RankIC {dev_ic.mean():+.4f} (NW t {nw_t(dev_ic, 10):.2f}, {len(dev_ic)} days); " \
           f"last 1y RankIC {ic.loc['2025-09-23':].mean():+.4f} (t {nw_t(ic.loc['2025-09-23':], 10):.2f})"
    print(head, flush=True)
    lines.append(head)
    for ex, h in (("hold", 10), ("r1", 20)):
        r = evaluate(f"{n} exit={ex}", S, ctx, exit=ex, hold=h, quiet=True)
        rep = report(r).replace("训练段 2021-11~2025-09", "开发段 2023-01~2025-09")
        print(rep, flush=True)
        lines.append(rep)
with open(f"{HERE}/out/results_{'_'.join(names) if sys.argv[1:] else 'all'}.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
