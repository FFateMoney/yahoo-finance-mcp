"""Fetch earnings-release dates (8-K item 2.02) and point-in-time financial facts (XBRL, with filing dates) from SEC EDGAR
for every ticker in research/factors/data. Runs on GitHub Actions (the cloud container cannot reach sec.gov).
Output: research/fullflow/data/earnings.csv.gz, facts.csv.gz, cik_map.csv
"""
import json
import os
import time

import pandas as pd
import requests

# SEC requires "name + real contact email"; set SEC_UA in the environment so the email is not committed
UA = {"User-Agent": os.environ["SEC_UA"], "Accept-Encoding": "gzip, deflate"}
OUT = "research/fullflow/data"
TAGS = {
    "us-gaap": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet", "NetIncomeLoss",
                "NetCashProvidedByUsedInOperatingActivities", "PaymentsToAcquirePropertyPlantAndEquipment",
                "Liabilities", "StockholdersEquity", "LongTermDebtNoncurrent", "LongTermDebt", "DebtCurrent",
                "LongTermDebtCurrent", "AssetsCurrent", "LiabilitiesCurrent", "Assets", "InterestExpense",
                "InterestExpenseNonoperating", "InterestExpenseDebt", "OperatingIncomeLoss", "EarningsPerShareDiluted",
                "RetainedEarningsAccumulatedDeficit", "IncomeTaxExpenseBenefit",
                "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest"],
    "dei": ["EntityCommonStockSharesOutstanding"],
}
s = requests.Session()
s.headers.update(UA)


def get(url):
    for k in range(5):
        r = s.get(url, timeout=60)
        if r.status_code == 200:
            time.sleep(0.12)
            return r.json()
        if r.status_code in (403, 404):
            if r.status_code == 403:
                print("HTTP 403", url, flush=True)
            return None
        print("HTTP", r.status_code, url, r.text[:200], flush=True)
        time.sleep(2 ** k)
    return None


ok = {}
for url in ["https://data.sec.gov/submissions/CIK0000320193.json",
            "https://efts.sec.gov/LATEST/search-index?keysTyped=AAPL",
            "https://www.sec.gov/files/company_tickers.json"]:
    r = s.get(url, timeout=60)
    print("probe", r.status_code, url, flush=True)
    ok[url.split("/")[2]] = r.status_code == 200
    time.sleep(0.2)
cik = {}
LOCAL = f"{OUT}/company_tickers.json"  # a copy saved from a browser also works
if os.path.exists(LOCAL):
    j = json.load(open(LOCAL, encoding="utf-8"))
else:
    j = get("https://www.sec.gov/files/company_tickers.json") if ok["www.sec.gov"] else None
if j:
    tick = pd.DataFrame(j).T
    cik = dict(zip(tick["ticker"].str.replace(".", "-", regex=False).str.upper(), tick["cik_str"].astype(int)))
w = pd.read_csv("research/factors/data/membership/sp500_table0.csv")  # Wikipedia S&P 500 list carries CIKs
for t, c in zip(w["Symbol"].str.replace(".", "-", regex=False), w["CIK"]):
    cik.setdefault(t, int(c))
print("CIK map", len(cik), flush=True)
ours = set(pd.read_csv("research/factors/data/constituents.csv")["yf"])
rem = pd.read_csv("research/factors/data/prices_removed.csv.gz", usecols=["Ticker"])["Ticker"].unique()
ours |= set(rem)
ours -= {"SPY", "MDY"}
mp = pd.DataFrame({"ticker": sorted(ours)})
mp["cik"] = mp["ticker"].map(cik)
for k, t in enumerate(mp.loc[mp["cik"].isna(), "ticker"] if ok["efts.sec.gov"] else []):  # look up the rest by ticker in EDGAR full-text search
    j = get(f"https://efts.sec.gov/LATEST/search-index?keysTyped={t}")
    for h in (j or {}).get("hits", {}).get("hits", []):
        src = h.get("_source", {})
        if t in [x.upper() for x in src.get("tickers", "").split(",")] or src.get("ticker", "").upper() == t:
            mp.loc[mp["ticker"] == t, "cik"] = int(h["_id"])
            break
mp.to_csv(f"{OUT}/cik_map.csv", index=False)
print("tickers", len(mp), "with CIK", mp["cik"].notna().sum(), flush=True)
if not ok["data.sec.gov"]:
    raise SystemExit("data.sec.gov is not reachable from this runner")

earn, facts = [], []
for k, (t, c) in enumerate(mp.dropna().itertuples(index=False)):
    c = int(c)
    sub = get(f"https://data.sec.gov/submissions/CIK{c:010d}.json")
    if sub:
        blocks = [sub["filings"]["recent"]]
        for f in sub["filings"].get("files", []):
            if f["filingFrom"] >= "2019-01-01":
                more = get(f"https://data.sec.gov/submissions/{f['name']}")
                if more:
                    blocks.append(more)
        for b in blocks:
            df = pd.DataFrame({k2: b[k2] for k2 in ["form", "filingDate", "acceptanceDateTime", "items"]})
            df = df[df["form"].isin(["8-K", "8-K/A"]) & df["items"].str.contains("2.02", regex=False)]
            for r in df.itertuples(index=False):
                earn.append((t, r.filingDate, r.acceptanceDateTime))
    cf = get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{c:010d}.json")
    if cf:
        for ns, tags in TAGS.items():
            for tag in tags:
                node = cf.get("facts", {}).get(ns, {}).get(tag)
                if not node:
                    continue
                for unit, rows in node["units"].items():
                    for r in rows:
                        if r.get("filed", "") < "2018-01-01":
                            continue
                        facts.append((t, tag, unit, r.get("start"), r["end"], r["val"], r["filed"], r.get("form"), r.get("fp")))
    if k % 100 == 0:
        print(k, t, len(earn), len(facts), flush=True)

pd.DataFrame(earn, columns=["ticker", "filed", "accepted"]).drop_duplicates().to_csv(f"{OUT}/earnings.csv.gz", index=False)
pd.DataFrame(facts, columns=["ticker", "tag", "unit", "start", "end", "val", "filed", "form", "fp"]).drop_duplicates() \
    .to_csv(f"{OUT}/facts.csv.gz", index=False)
print("done", len(earn), len(facts))
