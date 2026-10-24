import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from harness import load, evaluate, fund_daily, rsi, save
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_C.jsonl")
def rk(df, mask=None):
    if mask is not None: df = df.where(mask)
    return df.rank(axis=1, pct=True)
def zs(df):
    m, s = df.mean(axis=1), df.std(axis=1)
    return df.sub(m, axis=0).div(s, axis=0).clip(-3, 3)
def run(ctx, attempt, name, variants):
    """variants: list of (label, score, exit, hold)"""
    for i, (label, sc, ex, h) in enumerate(variants):
        res = evaluate(f"{attempt} {name} [{label}]", sc, ctx, exit=ex, hold=h)
        res["attempt"], res["variant"], res["n_params"] = attempt, label, 3
        save(res, OUT)
