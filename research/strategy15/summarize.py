"""Collect all results, pick each attempt's version by TRAIN top-5 excess, apply the pre-registered gate."""
import glob, json
import pandas as pd
rows = []
for f in sorted(glob.glob("research/strategy15/results*.jsonl")):
    for line in open(f, encoding="utf-8"):
        r = json.loads(line)
        rr = {"attempt": r.get("attempt", r["name"][:2]), "variant": r.get("variant", ""), "name": r["name"], "exit": r["exit"], "hold": r["hold"]}
        for w, v in r["rows"].items():
            k = w[:4]
            for m in ("top1", "topk", "all", "spy", "excess_k", "t_k", "seq_med", "seq_lo", "seq_hi", "seq_n", "pct_topk", "pct_excess_k", "pct_seq_med", "spy_bh", "ew_bh", "avg_names"):
                rr[f"{k}|{m}"] = v.get(m)
        rows.append(rr)
d = pd.DataFrame(rows)
d.to_csv("research/strategy15/all_versions.csv", index=False)
tr = "训练段 |excess_k"
sel = d[d["attempt"].str.match(r"^[ABC]\d$")].sort_values(tr, ascending=False).groupby("attempt").head(1).sort_values("attempt")
sel["gate_raw"] = (sel["训练段 |pct_topk"] >= 0.9) & (sel["最近1年|pct_topk"] >= 0.9)
sel["gate_excess"] = (sel["训练段 |pct_excess_k"] >= 0.9) & (sel["最近1年|pct_excess_k"] >= 0.9)
sel.to_csv("research/strategy15/selected.csv", index=False)
pd.set_option("display.width", 250)
cols = ["attempt", "variant", "exit", "hold"] + [f"{w}|{m}" for w in ["训练段 ", "最近1年", "最近3个", "最近1个"] for m in ("top1", "topk", "excess_k", "seq_med")] + ["训练段 |pct_topk", "最近1年|pct_topk", "训练段 |pct_excess_k", "最近1年|pct_excess_k", "gate_raw", "gate_excess"]
print(sel[cols].round(4).to_string())
print(d[~d["attempt"].str.match(r"^[ABC]\d$")][["name"] + cols[4:-2]].round(4).to_string())
