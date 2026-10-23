"""Fetch earnings-release dates (8-K item 2.02) and point-in-time financial facts (XBRL, with filing dates) from SEC EDGAR
for every ticker in research/factors/data. Runs on GitHub Actions (the cloud container cannot reach sec.gov).
Output: research/fullflow/data/earnings.csv.gz, facts.csv.gz, cik_map.csv
"""
import time

import pandas as pd
import requests

UA = {"User-Agent": "FFateMoney Research ffatemoney-research@users.noreply.github.com", "Accept-Encoding": "gzip, deflate"}
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
        if r.status_code == 404:
            return None
        print("HTTP", r.status_code, url, r.text[:200], flush=True)
        time.sleep(2 ** k)
    return None


j = get("https://www.sec.gov/files/company_tickers.json")
if j:
    tick = pd.DataFrame(j).T
    cik = dict(zip(tick["ticker"].str.replace(".", "-", regex=False).str.upper(), tick["cik_str"].astype(int)))
else:  # plain-text fallback: "ticker\tcik" per line
    txt = s.get("https://www.sec.gov/include/ticker.txt", timeout=60).text
    cik = {a.replace(".", "-").upper(): int(b) for a, b in (ln.split("\t") for ln in txt.splitlines() if "\t" in ln)}
print("CIK map", len(cik), flush=True)
ours = set(pd.read_csv("research/factors/data/constituents.csv")["yf"])
rem = pd.read_csv("research/factors/data/prices_removed.csv.gz", usecols=["Ticker"])["Ticker"].unique()
ours |= set(rem)
ours -= {"SPY", "MDY"}
mp = pd.DataFrame({"ticker": sorted(ours)})
mp["cik"] = mp["ticker"].map(cik)
mp.to_csv(f"{OUT}/cik_map.csv", index=False)
print("tickers", len(mp), "with CIK", mp["cik"].notna().sum(), flush=True)

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
