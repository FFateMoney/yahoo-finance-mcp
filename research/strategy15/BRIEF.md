# Brief for strategy sub-agents (read fully before starting)

Goal: Jasper (beginner, $180 account, buys 1 stock at a time, whole shares, US S&P 500/400) wants to know which
stock-selection strategy earns more over the last 1 year / 3 months / 1 month. You design and test FIVE distinct
strategies in your assigned theme. Each must be grounded in published literature, an established methodology, or a
famous investor's documented strategy — cite it (author, year, title/journal or book).

Pre-registration (rules you must follow): /mnt/project-files/回测/策略探索15_预登记.md (read it; do NOT edit it).

## Bench (fixed, do not edit harness.py — if you find a bug, stop and describe it in your final answer)
Work dir /home/user/yahoo-finance-mcp. Run scripts as `python research/strategy15/<file>.py`.
```python
import os, sys; sys.path.insert(0, "research/strategy15")
import numpy as np, pandas as pd
from harness import load, evaluate, fund_daily, rsi, save
ctx = load()          # ~5 s, cached
# ctx.O/H/L/C/V: dates x tickers DataFrames (split-adjusted daily). ctx.Cf = C forward-filled.
# ctx.univ: bool universe mask (the bench applies it for you). ctx.dvol: 21d avg dollar volume.
# ctx.sector: Series ticker -> GICS sector ("Unknown" for removed tickers). ctx.sp500: Series bool.
# ctx.spy: DataFrame open/close of SPY. ctx.cal: trading calendar.
# ctx.days_since_earn / ctx.days_to_earn: dates x tickers, calendar days since last / until next earnings date.
# ctx.earn: DataFrame ticker, date (earnings dates). ctx.ear: ticker, event, known, ear  (3-day earnings return
#     close(day-2)->close(day+1); usable only from the `known` date onwards).
# fund_daily(ctx, col): point-in-time daily frame of a fundamental, cols: rev ni cfo capex opinc (TTM, USD),
#     eq ca cl ta tl debt (balance sheet), shares, shares_1y (shares ~1y earlier), eps_q, eps_q_1y (latest quarter
#     diluted EPS and same quarter a year before), sue (seasonal-random-walk SUE), score (project's 10-pt health score)
# rsi(C, n=14)
res = evaluate("A1 momentum 12-1 [main]", score_df, ctx, exit="hold", hold=20)   # prints a table
res["attempt"], res["variant"], res["n_params"] = "A1", "main", 2
save(res, "research/strategy15/results_<GROUP>.jsonl")
```
`score_df`: dates x tickers, higher = better, NaN = not eligible, computed ONLY from data known at that date's close
(no look-ahead: never use shift(-k), future earnings returns, etc.; days_to_earn is OK — dates are pre-announced).
exit="hold" with hold in {5,10,20}, or exit="r1" (Jasper's +3% limit sell). Pick the holding period the literature
suggests (most anomalies are monthly -> 20). Placebo caches exist only for hold 5/10/20 and r1 with k=5 (default).
Each evaluate takes ~5-20 s. Watch out for pandas 3: DataFrame.stack() keeps NaN (use .dropna()).

## Rules
1. Before running an attempt, write its spec (hypothesis, citation, formula, parameters = literature defaults,
   holding period) into your notes file research/strategy15/notes_<GROUP>.md. Then run.
2. Per attempt at most 3 versions: "main" + up to 2 variants changing only 1-2 parameters (or exit r1 vs hold).
   Variants do NOT use up quota; a genuinely different idea does. Max 3 tunable parameters per strategy.
   The version that goes forward is chosen by TRAIN-period top-5 excess only (I will also do this automatically).
3. Prefer simple rules. No ML, no parameter grids, no peeking at the 1-year window to redesign. If an attempt fails,
   say so and move on — failures are valuable. You may use the result of an earlier attempt to motivate the next
   one only via the TRAIN window.
4. Be open-minded: you may replace the suggested ideas with other well-grounded ones in your theme (explain why).
5. Do not commit/push git, do not write outside research/strategy15/ (except reading). Name scripts <GROUP><n>_<short>.py.
6. Final answer (concise, <= 60 lines): per attempt: citation, rule in one line, version chosen by train,
   table row for train / 1y / 3m / 1m (top1 per trade, top5 per trade, all-stock avg, SPY same days, % random beaten
   for top5, $180 seq median), and a 1-line verdict. Plus anything suspicious you noticed.
