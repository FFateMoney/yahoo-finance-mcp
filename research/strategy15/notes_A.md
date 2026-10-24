# Group A notes (price behaviour / momentum) - specs written BEFORE running

Common: score from close data only; universe applied by bench; version chosen by TRAIN top-5 excess (excess_k).

## A1 Cross-sectional momentum 12-1
Jegadeesh & Titman (1993), "Returns to Buying Winners and Selling Losers", J. Finance. Score = C[t-21]/C[t-252]-1 (skip last month). Main hold 20. Variants: V1 hold 10; V2 formation 6-1 (C[t-21]/C[t-126]) hold 20. Params: formation, skip, hold (3).

## A2 52-week-high proximity
George & Hwang (2004), "The 52-Week High and Momentum Investing", J. Finance. Score = C / max(C over past 252 days). Main hold 20. V1 hold 10; V2 exit r1. Params: window 252, hold.

## A3 Industry momentum -> strongest stocks in strongest sectors
Moskowitz & Grinblatt (1999), "Do Industries Explain Momentum?", J. Finance. Only 11 GICS sectors available (coarse). Sector return = equal-weighted mean of its universe stocks' 6-1 month return; score = sector rank (1..11, higher=better sector) + within-sector percentile of stock 12-1 momentum (0..1) => best sector first, best stock within. Main hold 20. V1 sector formation 12-1; V2 hold 10. Params: sector formation, hold.

## A4 Smooth ("frog-in-the-pan") momentum
Da, Gurun & Warachka (2014), "Frog in the Pan: Continuous Information and Momentum", RFS. ID = sign(PRET)*(%days negative - %days positive) over the 12-1 window (252..21 days back); low ID = smooth. Rule: among top-quintile 12-1 momentum stocks (each day, cross-section) pick lowest ID; score = -ID. Main hold 20. V1 top-decile momentum filter (instead of quintile); V2 hold 10. Params: momentum cut, window(252 fixed), hold.

## A5 Short-term reversal in market stress
Nagel (2012), "Evaluating Reversal Strategies / Short-term reversals as compensation for liquidity provision", RFS. Score = -(5-day return minus sector-mean 5-day return) [Da, Liu & Schaumburg 2014 style industry-adjusted], only on days of market stress: SPY 21d realised vol above its 75th pct of the trailing 252 days (VIX not available). Else NaN. Main hold 5. V1 hold 10; V2 stress threshold dropped (all days) hold 5 as unconditional comparator. Params: lookback 5, stress pct 75, hold.
