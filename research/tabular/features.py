"""Feature panel for the tabular-model test (pre-registration: 回测/表格模型_预登记.md, F138-F142).
30 numeric features known at each date's close + GICS sector, turned into within-universe percentile ranks each day
and stored as uint8 (0..254, 255 = missing). Label: 10-day trade return (bench outcome, after costs), its daily
percentile rank minus 0.5, plus the raw return.
Output: research/tabular/data/panel.npz  (portable: the GPU machine only needs this file + numpy/torch)
Usage: python research/tabular/features.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "strategy15"))
from B_common import ear_frame, fund_fresh  # noqa: E402
from harness import TRAIN_START, _outcomes, load, rsi, windows  # noqa: E402

HOLD = 10

ctx = load()
C, Cf, H, V, O = ctx.C, ctx.Cf, ctx.H, ctx.V, ctx.O
r1 = Cf.pct_change(fill_method=None)
F = {}
# price (15)
F["mom12_1"] = Cf.shift(21) / Cf.shift(252) - 1
F["mom6_1"] = Cf.shift(21) / Cf.shift(126) - 1
F["ret1m"] = Cf / Cf.shift(21) - 1
F["ret1w"] = Cf / Cf.shift(5) - 1
F["to_high52"] = Cf / H.rolling(252, min_periods=200).max() - 1
F["vol20"] = r1.rolling(20, min_periods=15).std()
F["vol60"] = r1.rolling(60, min_periods=45).std()
F["max21"] = r1.rolling(21, min_periods=15).max()
F["log_dvol"] = np.log(ctx.dvol)
F["amihud"] = (r1.abs() / (C * V)).rolling(21, min_periods=15).mean()
F["vol_ratio"] = V.rolling(5).mean() / V.rolling(60, min_periods=45).mean()
F["rsi14"] = rsi(C)
ma20, ma200 = C.rolling(20).mean(), C.rolling(200).mean()
F["c_ma20"] = C / ma20 - 1
F["c_ma200"] = C / ma200 - 1
F["ma200_slope"] = ma200 / ma200.shift(20) - 1
# fundamentals (12), point in time, stale after 200 days
fd = {k: fund_fresh(ctx, k, 200) for k in ["ni", "eq", "rev", "cfo", "opinc", "ta", "debt", "shares", "shares_1y",
                                            "sue", "eps_q", "eps_q_1y", "score"]}
mcap = Cf * fd["shares"]
F["ep"] = fd["ni"] / mcap
F["bm"] = fd["eq"] / mcap
F["sp"] = fd["rev"] / mcap
F["cfp"] = fd["cfo"] / mcap
F["roe"] = (fd["ni"] / fd["eq"]).where(fd["eq"] > 0)
F["op_margin"] = (fd["opinc"] / fd["rev"]).where(fd["rev"] > 0)
F["accruals"] = (fd["ni"] - fd["cfo"]) / fd["ta"]
F["leverage"] = fd["debt"] / fd["ta"]
F["issuance"] = fd["shares"] / fd["shares_1y"] - 1
F["sue"] = fd["sue"]
F["eps_yoy"] = (fd["eps_q"] - fd["eps_q_1y"]) / fd["eps_q_1y"].abs().clip(lower=0.05)
F["fin_score"] = fd["score"]
# earnings events (3)
F["days_since_earn"] = ctx.days_since_earn
F["days_to_earn"] = ctx.days_to_earn
F["ear"] = ear_frame(ctx, 60)
assert len(F) == 30, len(F)

ret, _ = _outcomes(ctx, "hold", HOLD)
end, _ = windows(ctx)
ok = ctx.univ & ret.notna()
ok.loc[(ok.index < TRAIN_START) | (ok.index > end)] = False
mask = ok.to_numpy()
di, ti = np.nonzero(mask)
X = np.full((len(di), len(F)), 255, np.uint8)
for j, (k, f) in enumerate(F.items()):
    f = f.reindex(index=ctx.cal, columns=C.columns).replace([np.inf, -np.inf], np.nan).where(ok)
    rk = f.rank(axis=1, pct=True).to_numpy()[mask]
    X[:, j] = np.where(np.isfinite(rk), np.clip(np.floor(rk * 254.999), 0, 254), 255).astype(np.uint8)
    print(f"{k:16s} coverage {np.isfinite(rk).mean():.2f}")
y_raw = ret.to_numpy()[mask].astype(np.float32)
y = (ret.where(ok).rank(axis=1, pct=True).to_numpy()[mask] - 0.5).astype(np.float32)
sec_names = sorted(ctx.sector.unique())
sec = np.array([sec_names.index(s) for s in ctx.sector.to_numpy()], np.int8)[ti]
os.makedirs(f"{HERE}/data", exist_ok=True)
np.savez_compressed(f"{HERE}/data/panel.npz", X=X, y=y, y_raw=y_raw, sec=sec, di=di.astype(np.int32), ti=ti.astype(np.int32),
                    dates=ctx.cal.strftime("%Y-%m-%d").to_numpy().astype("U10"), tickers=C.columns.to_numpy().astype("U12"),
                    feat=np.array(list(F), dtype="U20"), sec_names=np.array(sec_names, dtype="U30"), hold=HOLD)
print(X.shape, "rows;", len(np.unique(di)), "dates;", os.path.getsize(f"{HERE}/data/panel.npz") / 1e6, "MB")
