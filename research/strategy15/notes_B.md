# Group B notes (earnings / corporate events). Specs written BEFORE running.

## B1 EAR drift
- Cite: Chan, Jegadeesh & Lakonishok (1996) "Momentum Strategies", J. Finance; Brandt, Kishore, Santa-Clara & Venkatachalam (2008) "Earnings announcements are full of surprises".
- Rule: score = 3-day earnings-announcement return (ctx.ear), placed on `known` row, held as signal for N=5 trading days after known (ffill limit 4). Top-5 highest. Hold 20 (papers: 6-12m drift; 20 is bench max).
- Params: N (5), hold (20). Variants: v1 N=1 (enter next open after known); v2 exit r1.

## B2 SUE drift
- Cite: Bernard & Thomas (1989) "Post-earnings-announcement drift: delayed price response or risk premium?", J. Acc. Research; Foster, Olsen & Shevlin (1984) Accounting Review.
- Rule: score = fund_daily sue, only within 10 calendar days after filing availability (stale=10). Hold 20.
- Params: stale window (10), hold (20). Variants: stale=5; exit r1.

## B3 Earnings announcement premium
- Cite: Frazzini & Lamont (2007) "The earnings announcement premium and trading volume" (NBER 13090); Barber, De George, Lehavy & Trueman (2013) "The earnings announcement premium around the globe", JFE.
- Rule: eligible if next earnings 3..9 calendar days away (so a 10-day hold spans it). Rank by mean of last 4 announcement 3-day returns (known at the time) = earnings-seasonality / ann-return (Frazzini-Lamont: stocks with high past announcement premium). Hold 10.
- Params: window (3-9), hold(10), lookback quarters (4). Variants: hold 5 with window 1-4; hold 10 with random-like rank = no (skip). v2: exit r1.

## B4 Net share issuance / buyback
- Cite: Pontiff & Woodgate (2008) "Share issuance and cross-sectional returns", J. Finance; Ikenberry, Lakonishok & Vermaelen (1995) "Market underreaction to open market share repurchases", JFE.
- Rule: score = -(shares/shares_1y - 1); eligible only if change in [-15%, 0] (larger drops are likely reverse splits / data errors). Hold 20.
- Params: floor(-15%), hold. Variants: floor -10%; exit r1.

## B5 Earnings momentum + price momentum
- Cite: Chan, Jegadeesh & Lakonishok (1996) "Momentum Strategies", J. Finance (SUE, EAR, 6m price momentum combined).
- Rule: score = mean of cross-sectional percentile ranks of SUE (age<=100d), EAR (last 60 trading days), 6-month return skipping 1 month (C[-21]/C[-126]). Hold 20.
- Params: hold(20), mom window(6m). Variants: hold 10; exit r1.

## Results / bench bug
- BUG in harness.fund_daily: staleness cut is a no-op (ctx.cal / filed are datetime64[us] under pandas 3, so astype('int64') is microseconds but divided by 86400e9). Values never go stale (sue present for ~812 names every day). I used B_common.fund_fresh (correct age) for B2/B4/B5. First (buggy) B2 runs were deleted from results_B.jsonl.
- B1 variants: N=1 uses ffill none. B3 ranking by mean of last 4 EARs is my choice (needed to rank eligible set).
- Chosen by train top-5 excess: B1 main, B2 v1_stale5, B3 main, B4 v1_floor10, B5 v1_h10. Only B5 v1_h10 passes the 90% gate in both train and 1y.
