"""Recent news and company filings for a few tickers (search by company name, not only ticker).
Sources, all free and keyless: Google News RSS (company name and ticker), SEC EDGAR filings (8-K items etc.).
GDELT answers 429 to GitHub runners, so it is off unless GDELT=1. Optional: FINNHUB_KEY env adds Finnhub company news.
Usage: python research/news/fetch_news.py [TICKER ...] [--days 14]   (no tickers: read research/news/tickers.txt)
Writes research/news/out/news_<date>.md and .json. Names come from SEC; research/news/aliases.csv (ticker,query) overrides.
"""
import json, os, re, sys, time, urllib.parse, urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

HERE = os.path.dirname(os.path.abspath(__file__))
UA = {"User-Agent": "finance-research cx1670252948@gmail.com"}  # SEC requires a contact UA
args = sys.argv[1:]
DAYS = 14
if "--days" in args:
    i = args.index("--days"); DAYS = int(args[i + 1]); del args[i:i + 2]
TICKERS = [a.upper() for a in args] or [l.strip().upper() for l in open(f"{HERE}/tickers.txt") if l.strip() and not l.startswith("#")]
NOW = datetime.now(timezone.utc)
SINCE = NOW - timedelta(days=DAYS)
# words worth a second look before buying
RISK = re.compile(r"lawsuit|sues|sued|class action|investigat|probe|subpoena|fraud|restat|downgrad|cut[s]? (?:to|price target)|"
                  r"guidance (?:cut|lower)|lowers? (?:guidance|outlook)|miss|resign|steps down|ousted|departure|layoff|recall|"
                  r"offering|dilut|delist|bankrupt|default|short seller|halt|antitrust|breach|hack|outage|warning", re.I)
EVENT = re.compile(r"acqui|merger|buyout|takeover|deal|upgrad|raises? (?:guidance|outlook)|buyback|repurchase|dividend|"
                   r"earnings|results|beat|partnership|contract|activist|stake|index", re.I)
ITEMS = {"1.01": "签重大协议", "1.02": "终止重大协议", "1.03": "破产", "2.01": "完成收购/处置", "2.02": "业绩公告",
         "2.03": "新增债务", "2.05": "重组/裁员成本", "2.06": "资产减值", "3.01": "退市/不符上市标准", "3.02": "增发股票",
         "4.01": "更换审计师", "4.02": "财报不可依赖(重述)", "5.01": "控制权变更", "5.02": "高管/董事变动",
         "5.03": "章程变更", "5.07": "股东大会投票", "7.01": "FD 披露", "8.01": "其他重要事项", "9.01": "附件"}


def get(url, tries=3):
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
                return r.read()
        except Exception as e:
            err = e; time.sleep(2 * (k + 1))
    raise err


def clean_name(n):
    n = re.sub(r"[,.]", " ", n)
    n = re.sub(r"\b(inc|corp|corporation|co|company|ltd|plc|holdings?|group|/de|/md|/new|the|class [a-z])\b", " ", n, flags=re.I)
    return " ".join(w.capitalize() if w.isupper() and len(w) > 3 else w for w in n.split())


def google(q):
    url = "https://news.google.com/rss/search?" + urllib.parse.urlencode({"q": f"{q} when:{DAYS}d", "hl": "en-US", "gl": "US", "ceid": "US:en"})
    out = []
    for it in ET.fromstring(get(url)).iter("item"):
        d = parsedate_to_datetime(it.findtext("pubDate"))
        src = it.find("source")
        out.append({"date": d.isoformat(), "title": it.findtext("title"), "source": src.text if src is not None else "",
                    "url": it.findtext("link"), "via": "google"})
    return out


def gdelt(q):
    url = "https://api.gdeltproject.org/api/v2/doc/doc?" + urllib.parse.urlencode(
        {"query": f"{q} sourcelang:english", "mode": "artlist", "format": "json", "maxrecords": 50, "timespan": f"{DAYS}d", "sort": "datedesc"})
    raw = get(url)
    try:
        arts = json.loads(raw).get("articles", [])
    except ValueError:
        return []
    return [{"date": datetime.strptime(a["seendate"], "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc).isoformat(),
             "title": a["title"], "source": a.get("domain", ""), "url": a["url"], "via": "gdelt"} for a in arts]


def finnhub(t):
    key = os.environ.get("FINNHUB_KEY")
    if not key:
        return []
    url = f"https://finnhub.io/api/v1/company-news?symbol={t}&from={SINCE.date()}&to={NOW.date()}&token={key}"
    return [{"date": datetime.fromtimestamp(a["datetime"], timezone.utc).isoformat(), "title": a["headline"],
             "source": a.get("source", ""), "url": a["url"], "via": "finnhub", "summary": a.get("summary", "")}
            for a in json.loads(get(url))]


def sec_filings(cik):
    s = \
        json.loads(get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json"))["filings"]["recent"]
    out = []
    for i, form in enumerate(s["form"]):
        d = s["filingDate"][i]
        if d < str((NOW - timedelta(days=max(DAYS, 30))).date()):
            continue
        if form in ("4", "3", "144", "424B2", "FWP"):  # insider forms / note offerings are noise here (Form 4 counted below)
            continue
        items = [f"{x} {ITEMS.get(x, '')}" for x in s["items"][i].split(",") if x and x != "9.01"]
        acc = s["accessionNumber"][i].replace("-", "")
        out.append({"date": d, "form": form, "items": items, "desc": s["primaryDocDescription"][i],
                    "url": f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{s['primaryDocument'][i]}"})
    n4 = sum(1 for i, f in enumerate(s["form"]) if f == "4" and s["filingDate"][i] >= str(SINCE.date()))
    return out, n4


def norm(t):
    return re.sub(r"\W+", " ", re.sub(r" - [^-]+$", "", t or "")).lower().strip()[:80]


tick = {v["ticker"]: v for v in json.loads(get("https://www.sec.gov/files/company_tickers.json")).values()}
alias = {}
if os.path.exists(f"{HERE}/aliases.csv"):
    alias = dict(l.strip().split(",", 1) for l in open(f"{HERE}/aliases.csv") if "," in l and not l.startswith("#"))
res = {}
for t in TICKERS:
    info = tick.get(t.replace(".", "-")) or tick.get(t)
    name = alias.get(t) or (clean_name(info["title"]) if info else t)
    r = {"ticker": t, "name": name, "sec_title": info["title"] if info else None, "errors": []}
    arts = []
    words = [w for w in re.findall(r"[A-Za-z0-9&']+", name) if len(w) > 2][:1] or [name]
    rel = re.compile(r"\b(" + "|".join(map(re.escape, words)) + r")\b", re.I)
    rel_t = re.compile(r"(?<![A-Za-z])\$?" + re.escape(t) + r"(?![A-Za-z])")  # ticker in caps, e.g. (P), NYSE:P, $P
    for label, f in [("google name", lambda: google(f'"{name}"')), ("google ticker", lambda: google(f'"{t}" stock')),
                     ("gdelt", lambda: gdelt(f'"{name}"') if os.environ.get("GDELT") else []), ("finnhub", lambda: finnhub(t))]:
        try:
            arts += f()
        except Exception as e:
            r["errors"].append(f"{label}: {e}")
        time.sleep(6 if label == "gdelt" else 1)  # GDELT allows one request per 5 s
    seen, keep = set(), []
    for a in sorted(arts, key=lambda a: a["date"], reverse=True):
        k = norm(a["title"])
        # keep only headlines that name the company (first words of its name), or the ticker for tickers of 3+ letters
        if k in seen or a["date"] < SINCE.isoformat() or not (rel.search(a["title"]) or (len(t) > 2 and rel_t.search(a["title"]))):
            continue
        seen.add(k)
        a["flag"] = "risk" if RISK.search(a["title"]) else ("event" if EVENT.search(a["title"]) else "")
        keep.append(a)
    r["articles"] = keep
    try:
        r["filings"], r["form4"] = sec_filings(int(info["cik_str"])) if info else ([], 0)
    except Exception as e:
        r["filings"], r["form4"] = [], 0; r["errors"].append(f"sec: {e}")
    res[t] = r
    time.sleep(1)

os.makedirs(f"{HERE}/out", exist_ok=True)
stamp = NOW.strftime("%Y-%m-%d")
json.dump(res, open(f"{HERE}/out/news_{stamp}.json", "w"), ensure_ascii=False, indent=1)
L = [f"# 候选股新闻与公告 {stamp}（近 {DAYS} 天）", "", "⚠ = 标题含风险词（诉讼、调查、下调、高管离职、增发等），★ = 重大事件（并购、上调、回购、业绩等）。", ""]
for t, r in res.items():
    risk = sum(a["flag"] == "risk" for a in r["articles"])
    L += [f"## {t}  {r['name']}", f"新闻 {len(r['articles'])} 条，其中风险词 {risk} 条；SEC 公告 {len(r['filings'])} 份；内部人交易 Form 4 {r['form4']} 份", ""]
    if r["filings"]:
        L.append("**SEC 公告**")
        L += [f"- {f['date']} {f['form']} {'；'.join(f['items']) or f['desc']}  [链接]({f['url']})" for f in r["filings"]]
        L.append("")
    L.append("**新闻**")
    for a in r["articles"][:40]:
        mark = {"risk": "⚠ ", "event": "★ "}.get(a["flag"], "")
        L.append(f"- {a['date'][:10]} {mark}{a['title']} — {a['source']} [{a['via']}]({a['url']})")
    if r["errors"]:
        L += ["", "抓取出错：" + "；".join(r["errors"])]
    L.append("")
open(f"{HERE}/out/news_{stamp}.md", "w").write("\n".join(L))
print("\n".join(L))
