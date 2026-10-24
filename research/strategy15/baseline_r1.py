"""Baseline: the current process rule R1 (短线选股分析流程, 2026-10-24) on the shared bench, and buy-everything.
Usage: python research/strategy15/baseline_r1.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import evaluate, fund_daily, load, rsi, save  # noqa: E402

ctx = load()
C = ctx.C
ma20, ma200 = C.rolling(20).mean(), C.rolling(200).mean()
r = rsi(C)
score = fund_daily(ctx, "score")
ok = ((ctx.days_since_earn > 30) & (ctx.days_to_earn > 28) & (score >= 8)
      & ~ctx.sector.isin(["Information Technology", "Communication Services"]).reindex(C.columns).values[None, :]
      & (C > ma200) & (ma200 > ma200.shift(20)) & (C < ma20) & (r >= 30) & (r <= 50))
sig = (score * 1000 - r).where(ok)
out = f"{os.path.dirname(os.path.abspath(__file__))}/results.jsonl"
save(evaluate("R0 现行规则 R1（≤$180，同分 RSI 最低，+3% 限价续挂）", sig, ctx, exit="r1"), out)
import numpy as np  # noqa: E402
rand = C * 0 + np.random.default_rng(0).random(C.shape)
save(evaluate("对照：每天随机挑（种子 0）", rand, ctx, exit="hold", hold=20), out)
