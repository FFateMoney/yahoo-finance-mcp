Similar-years check (2026-10-24, pre-registration /mnt/project-files/回测/相似年份_预登记.md).
How it was run (in a separate worktree, nothing here overwrites the 2020-26 data):
1. research/pricedb from branch claude/project-thread-ilfdjl; prices 2016-06..2022-06 written as research/factors/data/prices.csv.gz, empty prices_removed
2. pit.py uses the fja05680 S&P 500 history when research/pricedb/data/membership is present
3. SCORE_FROM=2018-03-01 python research/fullflow/fundamentals.py   (scores complete only from 2018 Q4)
4. python research/limitfit/explore.py 1 2018-06-01 ; python research/fullflow/flow.py --daily
5. AV_START=... AV_END=... python research/blindtest/oos_year.py   -> h_<start>.txt
