# Group C notes (famous investors / fundamental styles). Specs written BEFORE running.

Common: universe via bench; fundamentals point-in-time (fund_daily). Market cap = C*shares. Sector "Unknown" (132 removed tickers) cannot be
excluded as financial/utility -> kept (caveat). Cash not available -> EV = mktcap + debt (overstates EV for cash-rich firms).

## C1 Greenblatt Magic Formula (The Little Book That Beats the Market, 2005)
Rank universe (ex Financials/Utilities) on earnings yield = opinc_TTM / (C*shares + debt) and return on capital = opinc_TTM / (ta - cl)
(proxy of NWC + net fixed assets; cash not removed). Score = -(rank_EY + rank_ROC). Book holds 1 yr; we use hold 20 (bench max; monthly-rebalanced proxy).
Variants: V1 exit r1; V2 ROC denominator eq+debt (tangible-ish capital instead of ta-cl).

## C2 Piotroski F-score (Piotroski 2000, JAR "Value Investing: The Use of Historical Financial Statement Information...")
Cheap = book/market (eq/(C*shares)) in top quintile of universe (daily cross-section). F-score (0-9) built from: ROA>0 (ni/ta), CFO>0, dROA>0 (vs 252d ago),
CFO>NI, dLeverage<0 (debt/ta), dCurrent ratio>0, no net share issuance (shares<=shares_1y), dOperating margin>0 (opinc/rev; gross margin unavailable),
dAsset turnover>0 (rev/ta). Score = F + BM-percentile/10 tie-break among top-quintile BM. Hold 20.
Variants: V1 BM top tercile; V2 exit r1.

## C3 O'Neil CAN SLIM (How to Make Money in Stocks, 1988; simplified)
C: eps_q/eps_q_1y-1 >= 25% (eps_q_1y>0). N: close within 15% of 252d high. L: 12m return pct-rank >= 80. M: SPY close > its 200d MA (else no signals).
S/I/A omitted. Score = 12m return pct rank (RS). Hold 20. Variants: V1 hold 10; V2 exit r1.

## C4 Minervini Trend Template (Trade Like a Stock Market Wizard, 2013)
C>150MA & C>200MA; 150MA>200MA; 200MA rising vs 22d ago; 50MA>150MA & >200MA; C>50MA; C>=1.30*52w low; C>=0.75*52w high; RS (12m return pct-rank)>=70.
Score = RS rank. Hold 20. Variants: V1 hold 10; V2 exit r1.

## C5 Buffett-style quality + low risk + reasonable price (Frazzini-Kabiller-Pedersen 2018 "Buffett's Alpha", FAJ; Asness-Frazzini-Pedersen 2019 QMJ, RAS)
Components (z-scores cross-section, equal weights, no tunables): profitability (ni/ta, cfo/ta, opinc/rev), safety (-beta 252d vs SPY, -vol 252d, -tl/ta),
price (earnings yield ni/mktcap, cfo/mktcap). Score = mean of three group scores. Hold 20. Variants: V1 no price component (pure QMJ+low risk); V2 exit r1.

## Results / selection (by TRAIN top-5 excess per trade)
C1 main (+0.79% vs 0.29/0.14), C2 r1 (-0.04% vs -0.12/-0.23; all negative), C3 main (+1.28%), C4 main (+1.79%), C5 main (+0.57%).
All 15 evaluations saved in results_C.jsonl (5 attempts x 3 versions). Caveats: Unknown-sector removed tickers not excludable from financials/utilities; EV lacks cash;
$180 sequential results are extremely noisy (11-47 trades); t-stats on 1m window use overlapping 20d trades, ignore.
