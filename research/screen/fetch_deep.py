"""Deep-dive inputs for shortlisted tickers: statements, scores, news, StockAnalysis text."""
import json
import re
import sys

import requests
import yfinance as yf

TICKERS = ["KMI", "FAST", "CSX"]
OUT = "research/screen/out"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"}


def g(df, *names, col=0):
    for n in names:
        if n in df.index:
            v = df.loc[n].iloc[col]
            if v == v:
                return float(v)
    return None


res = {}
for t in TICKERS:
    tk = yf.Ticker(t)
    r = {}
    try:
        inc, bal, cf = tk.income_stmt, tk.balance_sheet, tk.cashflow
        r["annual_years"] = [str(c.date()) for c in inc.columns]
        for name, df, keys in [
            ("income", inc, ["Total Revenue", "Net Income", "EBIT", "Interest Expense", "Operating Income", "Gross Profit"]),
            ("balance", bal, ["Total Assets", "Current Assets", "Current Liabilities", "Retained Earnings",
                              "Total Liabilities Net Minority Interest", "Long Term Debt", "Total Debt",
                              "Stockholders Equity", "Ordinary Shares Number"]),
            ("cashflow", cf, ["Operating Cash Flow", "Free Cash Flow", "Capital Expenditure"]),
        ]:
            r[name] = {k: [None if v != v else float(v) for v in df.loc[k].tolist()] for k in keys if k in df.index}
        info = tk.info
        mcap = info.get("marketCap")
        ta, ca, cl = g(bal, "Total Assets"), g(bal, "Current Assets"), g(bal, "Current Liabilities")
        re_, tl = g(bal, "Retained Earnings"), g(bal, "Total Liabilities Net Minority Interest")
        ebit, rev = g(inc, "EBIT", "Operating Income"), g(inc, "Total Revenue")
        if None not in (ta, ca, cl, re_, tl, ebit, rev, mcap):
            r["altman_z"] = (1.2 * (ca - cl) + 1.4 * re_ + 3.3 * ebit + 1.0 * rev) / ta + 0.6 * mcap / tl
        ie = g(inc, "Interest Expense")
        if ie and ebit:
            r["interest_coverage"] = ebit / abs(ie)
        # Piotroski (annual, latest vs prior)
        try:
            ni0, ni1 = g(inc, "Net Income"), g(inc, "Net Income", col=1)
            ta0, ta1 = ta, g(bal, "Total Assets", col=1)
            ocf0 = g(cf, "Operating Cash Flow")
            ltd0, ltd1 = g(bal, "Long Term Debt") or 0, g(bal, "Long Term Debt", col=1) or 0
            cr0 = ca / cl
            cr1 = g(bal, "Current Assets", col=1) / g(bal, "Current Liabilities", col=1)
            sh0, sh1 = g(bal, "Ordinary Shares Number"), g(bal, "Ordinary Shares Number", col=1)
            gm0 = g(inc, "Gross Profit") / rev if g(inc, "Gross Profit") else None
            rev1 = g(inc, "Total Revenue", col=1)
            gm1 = g(inc, "Gross Profit", col=1) / rev1 if g(inc, "Gross Profit", col=1) else None
            f = [ni0 > 0, ocf0 > 0, ni0 / ta0 > ni1 / ta1, ocf0 > ni0, ltd0 / ta0 <= ltd1 / ta1, cr0 > cr1,
                 sh0 <= sh1 if sh0 and sh1 else False, (gm0 > gm1) if gm0 and gm1 else False, rev / ta0 > rev1 / ta1]
            r["piotroski"] = int(sum(bool(x) for x in f))
            r["piotroski_parts"] = [bool(x) for x in f]
        except Exception as e:  # noqa: BLE001
            r["piotroski_err"] = str(e)[:100]
        r["info"] = {k: info.get(k) for k in ["exDividendDate", "dividendRate", "dividendYield", "trailingPE",
                                                "forwardPE", "pegRatio", "marketCap", "targetMeanPrice",
                                                "recommendationKey", "fiftyTwoWeekHigh", "fiftyTwoWeekLow"]}
        try:
            r["calendar"] = {k: str(v) for k, v in (tk.calendar or {}).items()}
        except Exception:  # noqa: BLE001
            pass
        news = []
        for n in (tk.news or [])[:12]:
            c = n.get("content", n)
            news.append({"title": c.get("title"), "date": c.get("pubDate") or c.get("providerPublishTime"),
                         "summary": (c.get("summary") or "")[:300]})
        r["news"] = news
    except Exception as e:  # noqa: BLE001
        r["err"] = str(e)[:200]
    try:
        html = requests.get(f"https://stockanalysis.com/stocks/{t.lower()}/statistics/", headers=UA, timeout=20).text
        text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html))
        snip = {}
        for f in ["Altman Z-Score", "Piotroski F-Score", "Interest Coverage", "Earnings Date", "Ex-Dividend Date"]:
            i = text.find(f)
            if i >= 0:
                snip[f] = text[i:i + 60]
        r["stockanalysis"] = snip or text[:300]
    except Exception as e:  # noqa: BLE001
        r["stockanalysis"] = str(e)[:100]
    res[t] = r
json.dump(res, open(f"{OUT}/deep.json", "w"), indent=1, default=str, ensure_ascii=False)
print(json.dumps({t: {k: v for k, v in r.items() if k in ("altman_z", "interest_coverage", "piotroski")} for t, r in res.items()}))
