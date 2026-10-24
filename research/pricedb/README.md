# pricedb: the project's canonical daily price store

One place for daily OHLCV of every ticker that has been in the S&P 500 (since 2000) or S&P 400,
including removed ones (for point-in-time backtests), plus SPY, MDY, QQQ, IWM, RSP. Updated
incrementally: each run only fetches the days after each ticker's last stored bar.

## Read it

```python
import sys; sys.path.insert(0, "research/pricedb")
import pricedb
px  = pricedb.load(start="2020-01-01")            # Date, Ticker, Open, High, Low, Close, Volume (adjusted)
C   = pricedb.wide("Close", start="2020-01-01")   # Date x Ticker
raw = pricedb.load(adjust_prices=False)           # as traded
cal = pricedb.calendar()                          # SPY trading days
e   = pricedb.earnings()                          # Yahoo calendar incl. upcoming dates; earnings("sec") = 8-K 2.02
reg = pricedb.tickers()                           # registry: first/last date, rows, status, current member
```

Adjusted prices match `yf.download(..., auto_adjust=True)` as of the latest update. The old
`research/factors/data/prices.csv.gz` frames are `pricedb.load(start=...)` (same columns).

## Layout (`data/`)

| path | content |
|---|---|
| `prices/YYYY/YYYY-MM.parquet` | raw daily OHLCV (not adjusted for later splits/dividends) |
| `actions.parquet` | splits and dividends: Ticker, Date, Dividend, Split, PrevClose (append-only) |
| `tickers.csv` | ticker registry |
| `membership/` | Wikipedia S&P 500/400 current lists and change tables, fja05680 daily S&P 500 history, `constituents.csv` |
| `earnings/yahoo.csv.gz`, `earnings/sec_8k.csv.gz` | earnings dates |
| `meta.json` | last update time, last bar |

Storing raw prices means a split or dividend never rewrites old months; adjustment is applied on read.

## Update it

- **GitHub Actions** (`.github/workflows/pricedb-update.yml`): runs weekdays 22:30 UTC once the
  workflow is on the default branch; or edit the first line of `research/pricedb/RUN` (update.py
  arguments, may be empty) and push; or dispatch it manually. The bot commits `data: pricedb update <date>`.
- **Locally** (needs Yahoo access, `pip install yfinance pandas pyarrow lxml`):
  `git pull` then `python research/pricedb/update.py` (add `--no-earnings` to skip the ~15 min
  earnings refresh). Commit and push `research/pricedb/data` if others should see it; otherwise
  `git checkout -- research/pricedb/data` before the next pull.

Options: `--full` (refetch all since 2000), `--refetch AAPL,MSFT`, `--no-earnings`, `--no-membership`.

Each incremental run re-downloads the last ~2 weeks and compares them with the store; a mismatch
(late split, data correction) triggers a full refetch of that ticker. Tickers with no new bar for
30 days and not in an index today are treated as delisted and skipped; tickers Yahoo has no data for
are retried monthly.

Caveats: Yahoo has no data for most tickers delisted long ago, and reused symbols return the current
company's history. Use the membership files (and `research/cnn/universe.py` / `research/factors/pit.py`)
to restrict to members on each date.
