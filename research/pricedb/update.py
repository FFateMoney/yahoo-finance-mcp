"""Build or incrementally update the price store in research/pricedb/data.

    python research/pricedb/update.py                 # incremental: only days after each ticker's last date
    python research/pricedb/update.py --no-earnings   # skip the earnings-date refresh (~15 min)
    python research/pricedb/update.py --full          # refetch everything since 2000 (first build)
    python research/pricedb/update.py --refetch AAPL,MSFT

Universe: every ticker that was in the S&P 500 (fja05680 history since 2000 + Wikipedia changes) or
the S&P 400 (current + Wikipedia change table), plus benchmark ETFs. Needs Yahoo access: run it on
GitHub Actions (workflow pricedb-update.yml) or a PC with yfinance; the cloud container is blocked.

Incremental logic per ticker: refetch the last ~10 trading days, convert to raw prices, and compare
the overlap with what is stored. If it disagrees (a split Yahoo recorded late, a data correction) the
ticker is refetched in full; otherwise only the new days and any new splits/dividends are appended.
Only month files that actually changed are rewritten, so a daily update touches ~1 file.
"""
import argparse
import datetime as dt
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pricedb as P  # noqa: E402

UA = {"User-Agent": "Mozilla/5.0"}
START = "2000-01-01"
ETFS = ["SPY", "MDY", "QQQ", "IWM", "RSP"]
OVERLAP_DAYS = 14          # calendar days refetched before each ticker's last stored date
STALE_DAYS = 30            # no new bar for this long -> treat as delisted, stop updating
RETRY_NODATA_DAYS = 30     # how often to retry tickers Yahoo had no data for
TOL = 0.01                 # overlap mismatch (relative) that triggers a full refetch
META = os.path.join(P.ROOT, "meta.json")


# ---------- universe ----------

def wiki_tables(url):
    html = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read().decode()
    out = []
    for t in pd.read_html(io.StringIO(html)):
        if isinstance(t.columns, pd.MultiIndex):
            t.columns = [" | ".join(dict.fromkeys(map(str, c))) for c in t.columns]
        out.append(t)
    return out


def pick(tables, need):
    for t in tables:
        if all(any(n in c for c in t.columns) for n in need):
            return t
    raise RuntimeError(f"no table with columns {need}")


def refresh_membership():
    """Write membership/*.csv; return the set of Yahoo tickers ever in either index."""
    M = P.MEMBERS
    os.makedirs(M, exist_ok=True)
    t5 = wiki_tables("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    t4 = wiki_tables("https://en.wikipedia.org/wiki/List_of_S%26P_400_companies")
    cur5 = pick(t5, ["Symbol", "GICS Sector"])
    cur4 = pick(t4, ["Symbol", "GICS Sector"])
    cur5.to_csv(f"{M}/sp500_current.csv", index=False)
    cur4.to_csv(f"{M}/sp400_current.csv", index=False)
    changes = []
    for key, t in (("sp500", t5), ("sp400", t4)):
        try:  # the S&P 500 page has not carried its change table lately; fja05680 covers that history
            ch = pick(t, ["Added", "Removed"])
            ch.to_csv(f"{M}/{key}_changes.csv", index=False)
            changes.append(ch)
        except RuntimeError as e:
            print(key, "change table not found:", e)
    cons = pd.concat([cur5[["Symbol", "GICS Sector"]].assign(Universe="SP500"),
                      cur4[["Symbol", "GICS Sector"]].assign(Universe="SP400")]).drop_duplicates("Symbol")
    cons = cons.rename(columns={"GICS Sector": "Sector"})
    cons["yf"] = cons["Symbol"].str.replace(".", "-", regex=False)
    cons.to_csv(f"{M}/constituents.csv", index=False)  # same columns as research/factors/data/constituents.csv

    # fja05680 daily S&P 500 membership since 1996 (git clone: the GitHub API rate-limits shared runners)
    tmp = tempfile.mkdtemp()
    try:
        subprocess.run(["git", "clone", "-q", "--depth", "1", "https://github.com/fja05680/sp500.git", tmp],
                       check=True, timeout=300)
        files = [n for n in os.listdir(tmp) if n.startswith("S&P 500 Historical Components") and n.endswith(".csv")]
        latest = next((n for n in files if "Updated" in n), sorted(files)[-1])
        shutil.copy(os.path.join(tmp, latest), f"{M}/sp500_history_fja.csv")
    except Exception as e:  # keep the previous copy
        print("fja05680 refresh failed:", e)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    names = set(cons["Symbol"])
    for t in changes:
        for c in t.columns:
            if "Ticker" in c:
                names |= set(t[c].dropna().astype(str))
    if os.path.exists(f"{M}/sp500_history_fja.csv"):
        fja = pd.read_csv(f"{M}/sp500_history_fja.csv")
        for x in fja.loc[fja["date"] >= START, "tickers"]:
            names |= {s for s in x.split(",") if "-" not in s}  # "-YYYYMM" = delisted/renamed, not on Yahoo
    names = {s.strip().replace(".", "-") for s in names if s and s.strip() and s.strip() != "nan"}
    print("universe:", len(cons), "current members,", len(names), "ever-members")
    return names, cons


# ---------- fetching ----------

def ny_now():
    return dt.datetime.now(ZoneInfo("America/New_York"))


def to_raw(d):
    """One ticker's yfinance frame (auto_adjust=False, actions=True) -> raw rows + action rows.

    Yahoo's OHLC/Volume/Dividends are already adjusted for splits up to today, so undo every split
    that happened after each row's date (within this window; nothing later exists).
    """
    d = d.dropna(subset=["Close"]).sort_index()
    if len(d) == 0:
        return None, None
    split = d.get("Stock Splits", pd.Series(0.0, index=d.index)).fillna(0.0).to_numpy(float)
    div = d.get("Dividends", pd.Series(0.0, index=d.index)).fillna(0.0).to_numpy(float)
    r = np.where(split > 0, split, 1.0)
    after = np.concatenate([np.cumprod(r[::-1])[::-1][1:], [1.0]])  # product of splits strictly after each row
    raw = pd.DataFrame({"Date": pd.to_datetime(d.index).tz_localize(None).normalize()})
    for c in ("Open", "High", "Low", "Close"):
        raw[c] = d[c].to_numpy(float) * after
    raw["Volume"] = d["Volume"].fillna(0).to_numpy(float) / after
    ev = (split > 0) | (div > 0)
    prev = np.concatenate([[np.nan], raw["Close"].to_numpy()[:-1]])
    acts = pd.DataFrame({"Date": raw["Date"][ev].to_numpy(), "Dividend": (div * after)[ev],
                         "Split": split[ev], "PrevClose": prev[ev]})
    return raw, acts


def download(tickers, start):
    """{ticker: (raw, acts)} for those Yahoo returned data for."""
    out = {}
    for i in range(0, len(tickers), 50):
        batch = tickers[i:i + 50]
        for attempt in range(3):
            try:
                df = yf.download(batch, start=start, interval="1d", auto_adjust=False, actions=True,
                                 group_by="ticker", threads=True, progress=False, multi_level_index=True)
                break
            except Exception as e:
                print("download error, retrying:", e)
                time.sleep(10 * (attempt + 1))
        else:
            continue
        if df is None or len(df) == 0:
            continue
        have = set(df.columns.get_level_values(0))
        for t in batch:
            if t in have:
                raw, acts = to_raw(df[t])
                if raw is not None and len(raw):
                    out[t] = (raw, acts)
        print(f"  {min(i + 50, len(tickers))}/{len(tickers)} tickers, {len(out)} with data", flush=True)
    return out


def drop_unfinished(raw):
    """Drop today's bar while the US session is still open (it would be a partial day)."""
    now = ny_now()
    if now.weekday() < 5 and now.time() < dt.time(16, 30):
        return raw[raw["Date"] < pd.Timestamp(now.date())]
    return raw


# ---------- store ----------

def month_key(d):
    return d.dt.strftime("%Y-%m")


def write_months(px, months):
    for m in sorted(months):
        part = px[px["_m"] == m].drop(columns="_m").sort_values(["Ticker", "Date"])
        path = os.path.join(P.PRICES, m[:4], f"{m}.parquet")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if len(part) == 0:
            if os.path.exists(path):
                os.remove(path)
            continue
        part = part.astype({"Open": "float64", "High": "float64", "Low": "float64", "Close": "float64",
                            "Volume": "float64"})
        part["Date"] = part["Date"].dt.date
        part.to_parquet(path, index=False, compression="zstd")


def changed_months(old, new):
    """Months where a ticker's rows differ beyond rounding (both frames: one ticker)."""
    o = old.set_index("Date")[["Open", "High", "Low", "Close", "Volume"]]
    n = new.set_index("Date")[["Open", "High", "Low", "Close", "Volume"]]
    idx = o.index.union(n.index)
    o, n = o.reindex(idx), n.reindex(idx)
    diff = ~np.isclose(o.to_numpy(float), n.to_numpy(float), rtol=1e-4, atol=1e-6, equal_nan=True).all(axis=1)
    return set(pd.Series(idx[diff]).dt.strftime("%Y-%m"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="refetch every ticker since 2000")
    ap.add_argument("--refetch", default="", help="comma-separated tickers to refetch in full")
    ap.add_argument("--no-earnings", action="store_true")
    ap.add_argument("--no-membership", action="store_true", help="reuse stored membership files")
    args = ap.parse_args()
    os.makedirs(P.PRICES, exist_ok=True)
    today = pd.Timestamp(ny_now().date())

    if args.no_membership and os.path.exists(f"{P.MEMBERS}/constituents.csv"):
        cons = pd.read_csv(f"{P.MEMBERS}/constituents.csv")
        names = set(P.tickers()["Ticker"]) if os.path.exists(P.TICKERS) else set(cons["yf"])
    else:
        names, cons = refresh_membership()
    universe = sorted(names | set(ETFS))
    current = set(cons["yf"]) | set(ETFS)

    print("loading store...", flush=True)
    px = P.load_raw()
    acts = P.actions()
    reg = (pd.read_csv(P.TICKERS, parse_dates=["first", "last", "tried"]) if os.path.exists(P.TICKERS)
           else pd.DataFrame(columns=["Ticker", "first", "last", "rows", "status", "current", "tried"]))
    reg = reg.set_index("Ticker")
    print(f"store: {px['Ticker'].nunique() if len(px) else 0} tickers, {len(px)} rows,"
          f" last {px['Date'].max().date() if len(px) else None}")

    last = px.groupby("Ticker")["Date"].max() if len(px) else pd.Series(dtype="datetime64[ns]")
    refetch = set(t for t in args.refetch.split(",") if t)
    full, inc = [], []
    for t in universe:
        if args.full or t in refetch:
            full.append(t)
        elif t in last.index:
            if (today - last[t]).days <= STALE_DAYS or t in current:
                inc.append(t)
        else:
            tried = reg["tried"].get(t) if "tried" in reg else None
            if tried is None or pd.isna(tried) or (today - tried).days >= RETRY_NODATA_DAYS:
                full.append(t)
    print(f"plan: {len(full)} full fetches, {len(inc)} incremental", flush=True)

    dirty = set()
    new_rows, new_acts, replace = [], [], {}

    # incremental, grouped by start date so a batch shares one request window
    if inc:
        starts = {t: (last[t] - pd.Timedelta(days=OVERLAP_DAYS)).strftime("%Y-%m-%d") for t in inc}
        for s in sorted(set(starts.values())):
            batch = [t for t in inc if starts[t] == s]
            got = download(batch, s)
            for t in batch:
                if t not in got:
                    continue
                raw, a = got[t]
                raw = drop_unfinished(raw)
                if len(raw) == 0:
                    continue
                old = px[(px["Ticker"] == t) & (px["Date"] >= raw["Date"].min())]
                ov = raw.merge(old[["Date", "Close"]], on="Date", suffixes=("", "_old"))
                if len(ov) == 0 or (np.abs(ov["Close"] / ov["Close_old"] - 1) > TOL).any():
                    print(f"  {t}: overlap mismatch -> full refetch")
                    full.append(t)
                    continue
                add = raw[raw["Date"] > last[t]]
                if len(add):
                    new_rows.append(add.assign(Ticker=t))
                a = a[a["Date"] <= raw["Date"].max()]
                if len(a):  # PrevClose of an event on the window's first day comes from the store
                    a["PrevClose"] = a["PrevClose"].fillna(old["Close"].iloc[0] if len(old) else np.nan)
                    new_acts.append(a.assign(Ticker=t))

    if full:
        got = download(full, START)
        for t in full:
            if t in got:
                raw, a = got[t]
                replace[t] = (drop_unfinished(raw).assign(Ticker=t), a.assign(Ticker=t))
            reg.loc[t, "tried"] = today

    # apply full replacements
    if replace:
        for t, (raw, _) in replace.items():
            old = px[px["Ticker"] == t]
            dirty |= changed_months(old, raw) if len(old) else set(month_key(raw["Date"]))
        px = pd.concat([px[~px["Ticker"].isin(list(replace))]] + [r for r, _ in replace.values()],
                       ignore_index=True)
        acts = pd.concat([acts[~acts["Ticker"].isin(list(replace))]] + [a for _, a in replace.values()],
                         ignore_index=True)
    if new_rows:
        add = pd.concat(new_rows, ignore_index=True)
        dirty |= set(month_key(add["Date"]))
        px = pd.concat([px, add], ignore_index=True)
    if new_acts:
        acts = pd.concat([acts] + new_acts, ignore_index=True)
    acts["Date"] = pd.to_datetime(acts["Date"])
    acts = (acts.dropna(subset=["Date"]).drop_duplicates(["Ticker", "Date"], keep="first")  # stored rows win
            .sort_values(["Ticker", "Date"], ignore_index=True))

    px["Date"] = pd.to_datetime(px["Date"])
    acts["Date"] = pd.to_datetime(acts["Date"])
    if dirty:
        px["_m"] = month_key(px["Date"])
        write_months(px, dirty)
        px = px.drop(columns="_m")
    acts[["Ticker", "Date", "Dividend", "Split", "PrevClose"]].assign(Date=acts["Date"].dt.date).to_parquet(
        P.ACTIONS, index=False, compression="zstd")

    # ticker registry
    g = px.groupby("Ticker")["Date"]
    stats = pd.DataFrame({"first": g.min(), "last": g.max(), "rows": g.size()})
    for t in universe:
        if t not in reg.index:
            reg.loc[t, "tried"] = today
    reg = reg.drop(columns=[c for c in ("first", "last", "rows") if c in reg.columns]).join(stats, how="outer")
    reg["status"] = np.where(reg["rows"].fillna(0) > 0, "ok", "nodata")
    reg["current"] = reg.index.isin(list(current))
    reg.index.name = "Ticker"
    reg = reg.reset_index()[["Ticker", "first", "last", "rows", "status", "current", "tried"]]
    for c in ("first", "last", "tried"):
        reg[c] = pd.to_datetime(reg[c]).dt.strftime("%Y-%m-%d")
    reg["rows"] = reg["rows"].fillna(0).astype(int)
    reg.sort_values("Ticker").to_csv(P.TICKERS, index=False)

    meta = json.load(open(META)) if os.path.exists(META) else {}
    meta.update({"updated_utc": dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
                 "last_bar": str(px["Date"].max().date()), "tickers_with_data": int(px["Ticker"].nunique()),
                 "rows": int(len(px)), "months_rewritten": len(dirty)})
    print(f"prices: {meta['tickers_with_data']} tickers, {meta['rows']} rows, last bar {meta['last_bar']},"
          f" {len(dirty)} month files rewritten, {len(new_rows)} tickers appended, {len(replace)} refetched")

    if not args.no_earnings:
        refresh_earnings(sorted(current - set(ETFS)), today)
        meta["earnings_refreshed_utc"] = meta["updated_utc"]
    json.dump(meta, open(META, "w"), indent=1)


def refresh_earnings(tickers, today):
    """Yahoo earnings calendar for current members: replace each ticker's upcoming dates, keep history."""
    path = P.EARN_YF
    os.makedirs(os.path.dirname(path), exist_ok=True)
    old = pd.read_csv(path, parse_dates=["date"]) if os.path.exists(path) else pd.DataFrame(columns=["ticker", "date"])
    rows, done = [], set()
    for k, t in enumerate(tickers):
        try:
            e = yf.Ticker(t).get_earnings_dates(limit=12)
            if e is not None and len(e):
                done.add(t)
                for ts in e.index:
                    rows.append((t, pd.Timestamp(ts).tz_localize(None).normalize()))
        except Exception:
            pass
        if k % 100 == 0:
            print(f"  earnings {k}/{len(tickers)}", flush=True)
        time.sleep(0.2)
    new = pd.DataFrame(rows, columns=["ticker", "date"])
    stale = old["ticker"].isin(done) & (old["date"] >= today - pd.Timedelta(days=3))  # upcoming dates can move
    out = pd.concat([old[~stale], new]).drop_duplicates().sort_values(["ticker", "date"])
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out.to_csv(path, index=False, compression="gzip")
    print(f"earnings: {len(done)}/{len(tickers)} tickers refreshed, {len(out)} dates stored")


if __name__ == "__main__":
    main()
