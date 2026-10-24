import os, sys; sys.path.insert(0, "research/strategy15")
import numpy as np, pandas as pd
from harness import load, evaluate, save
OUT = "research/strategy15/results_A.jsonl"
def run(att, versions, ctx):
    """versions: list of (variant, score, exit, hold, n_params)"""
    best = None
    for v, sc, ex, h, npar in versions:
        r = evaluate(f"{att} [{v}]", sc, ctx, exit=ex, hold=h)
        r["attempt"], r["variant"], r["n_params"] = att, v, npar
        save(r, OUT)
        tr = r["rows"]["训练段 2021-11~2025-09"]["excess_k"]
        print(f"   -> {att}/{v} train top5 excess {tr:+.3%}")
        if best is None or tr > best[1]: best = (v, tr)
    print("CHOSEN BY TRAIN:", att, best)
