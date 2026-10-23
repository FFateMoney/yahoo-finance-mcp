"""Dump index membership-change tables (S&P 500 / S&P 400) for point-in-time universes (run in GitHub Actions)."""
import io
import urllib.parse
import urllib.request

import pandas as pd

UA = {"User-Agent": "Mozilla/5.0"}
OUT = "research/factors/data/membership"
PAGES = {
    "sp500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "sp400": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
}
for key, url in PAGES.items():
    html = urllib.request.urlopen(urllib.request.Request(url, headers=UA)).read().decode()
    for i, t in enumerate(pd.read_html(io.StringIO(html))):
        if isinstance(t.columns, pd.MultiIndex):
            t.columns = [" | ".join(dict.fromkeys(map(str, c))) for c in t.columns]
        t.to_csv(f"{OUT}/{key}_table{i}.csv", index=False)
        print(key, i, t.shape, list(t.columns)[:8])

# community-maintained daily S&P 500 membership history
src = "https://raw.githubusercontent.com/fja05680/sp500/master/"
try:
    idx = urllib.request.urlopen(urllib.request.Request("https://api.github.com/repos/fja05680/sp500/contents", headers=UA)).read().decode()
    import json
    names = [f["name"] for f in json.loads(idx) if f["name"].startswith("S&P 500 Historical Components")]
    print("fja05680 files:", names)
    name = sorted(names)[-1]
    data = urllib.request.urlopen(urllib.request.Request(src + urllib.parse.quote(name), headers=UA)).read()
    open(f"{OUT}/sp500_history_fja05680.csv", "wb").write(data)
except Exception as e:
    print("fja05680 failed:", e)
