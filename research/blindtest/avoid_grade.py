"""Grade an exclusion list on the test year: python research/blindtest/avoid_grade.py EXCLUDE_CSV
EXCLUDE_CSV has columns date,ticker: the stocks you would NOT buy (1-2 per day; days you leave out exclude nothing).
Per test day: average of the remaining stocks (= buying one of them at random) vs the five-stock average, both exits.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

tmp = Path(tempfile.mkdtemp())
subprocess.run([sys.executable, "research/blindtest/avoid_data.py", str(tmp), "--answers", str(tmp / "ans.csv")],
               check=True, capture_output=True)
a = pd.read_csv(tmp / "ans.csv", encoding="utf-8-sig", dtype={"date": str})
ex = pd.read_csv(sys.argv[1], encoding="utf-8-sig", dtype=str)
ex["date"] = pd.to_datetime(ex["date"]).dt.strftime("%Y-%m-%d")
a["排除"] = a.set_index(["date", "ticker"]).index.isin(ex.set_index(["date", "ticker"]).index)
for lab in ("阶梯", "现行"):
    r = f"结果_{lab}"
    day = a.groupby("date").apply(lambda g: pd.Series({"五只": g[r].mean(), "剩下": g.loc[~g["排除"], r].mean(),
                                                        "排除的": g.loc[g["排除"], r].mean(), "n": g["排除"].sum()}))
    hit = a[a["排除"]][f"当天最差2只_{lab}"].mean()
    diff = day["剩下"] - day["五只"]
    blocks = diff.groupby(np.arange(len(diff)) // 20).mean()  # ~independent 20-day blocks
    print(f"[{lab}] 测试年 {len(day)} 天，有排除的 {(day['n'] > 0).sum()} 天")
    print(f"  五只平均 {day['五只'].mean():+.2%}，剩下的平均 {day['剩下'].mean():+.2%}，被排除的平均 {day['排除的'].mean():+.2%}")
    print(f"  剩下 − 五只 {diff.mean():+.2%}（按 20 天分块 t {blocks.mean() / blocks.std(ddof=1) * np.sqrt(len(blocks)):+.2f}，{len(blocks)} 块）")
    print(f"  被排除的里真的是当天最差 2 只的比例 {hit:.0%}（随便排除约 40%）；被排除的亏钱比例 {a[a['排除']][f'亏钱_{lab}'].mean():.0%}，全部 {a[f'亏钱_{lab}'].mean():.0%}")
