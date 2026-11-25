# Walkthrough: one seed, from targets to a real book

This follows development seed 1 through every step. All output below is pasted from `make demo` (`python -m targetbook.cli demo --seed 1`) and `python -m targetbook.cli demo --seed 1 --signal control`, run with the versions in `constraints.txt`. Seed 1 is a development seed, so nothing here touches the held-out results.

## 1. Install and run

```bash
pip install -c constraints.txt -e ".[dev]"
python -m targetbook.cli demo --seed 1
```

```
seed 1, model signal, desk schedule 2 tranches x 2 weeks, band 0.08 (edge)
stage       return %  Sharpe  vol %  gross  turn/yr  delta %
paper          13.95    1.81   7.70   2.50     83.3
mandate        11.85    2.02   5.85   1.96     65.9    -2.10
vol_cap        11.02    2.03   5.44   1.87     63.0    -0.83
schedule       10.11    1.99   5.07   1.77     40.5    -0.90
costs           8.98    1.77   5.08   1.77     40.5    -1.13
carry           8.31    1.64   5.08   1.77     40.5    -0.67
bands           7.32    1.65   4.44   1.60     21.1    -1.00
single book every 4 weeks, Sharpe by phase: 1.31, 1.38, 1.12, 1.28; staggered: 1.42
```

Seed 1 is a lucky seed: its paper Sharpe of 1.81 is the highest of the five development seeds (the others are 0.76 to 1.07). Read the shape, not the level.

## 2. What happens at each row

**paper.** Every Friday-equivalent (every fifth day) the research team's target is held exactly: z-score of the signal, divided by trailing vol, gross 2.5, no costs. Traded notional is 83 times the NAV per year, because the signal noise has a 10-day half-life and the team re-forms targets from scratch each week.

**mandate.** Every one of these targets breaks the mandate (gross cap 2.0, group net limits, net cap 0.25). The projection finds the closest feasible book; gross falls to 1.96. Return falls by 2.10 points but Sharpe rises: the projection removes net and group exposure that was mostly risk.

**vol_cap.** Ex-ante vol of the projected book, from a shrunk 252-day covariance, is scaled down to 6% when above it. Return falls again (-0.83), Sharpe is flat.

**schedule.** The desk runs 2 tranches, each rebalanced every two weeks on alternate weeks. Turnover drops from 63 to 40.5. Return falls by 0.90 points: half the book is a week staler, and the two tranches' targets partly net out (gross 1.87 to 1.77).

**costs.** Trading cost on the trade from the drifted weights to the new book: 1.13 points a year at this turnover.

**carry.** Financing on long and short positions, shorts at a higher rate: 0.67 points a year.

**bands.** A no-trade zone of 0.08 around each target, trading only to its edge. Turnover halves (40.5 to 21.1), cost drag falls, but the book now lags its targets and holds less (gross 1.60). On this seed that costs 1.00 point of return and leaves Sharpe flat (1.64 to 1.65). Across the ten held-out seeds the same step costs 0.35 points and adds 0.11 Sharpe in the median, with a loss of return in 7 of 10 seeds (README).

**timing luck.** The last line runs the real-book settings as one book rebalanced every four weeks, once for each of the four possible weeks, and as four staggered tranches. On this seed the four single books score 1.12 to 1.38; the staggered book scores 1.42, above all four.

## 3. The control

```
seed 1, control signal, desk schedule 2 tranches x 2 weeks, band 0.08 (edge)
stage       return %  Sharpe  vol %  gross  turn/yr  delta %
paper          -1.90   -0.26   7.39   2.50     80.7
mandate        -2.39   -0.42   5.74   1.94     64.3    -0.48
vol_cap        -2.25   -0.42   5.39   1.86     61.3    +0.14
schedule       -2.33   -0.46   5.03   1.77     40.0    -0.08
costs          -3.47   -0.69   5.03   1.77     40.0    -1.14
carry          -4.14   -0.82   5.03   1.77     40.0    -0.67
bands          -3.15   -0.70   4.47   1.59     20.4    +0.99
single book every 4 weeks, Sharpe by phase: -0.73, -0.70, -0.49, -0.47; staggered: -0.67
```

The control signal trades as much as the model signal (80.7 against 83.3 a year on paper) and knows nothing. Costs and carry take almost exactly what they take from the model book (1.14 and 0.67 points). With no edge, bands only save cost, so they add 0.99 points here. The phase spread (-0.73 to -0.47) is pure timing luck.

## 4. The full benchmark

```bash
make bench     # development seeds 1-5 choose; held-out seeds 101-110 reported with frozen choices; about a minute with 5 processes
make verify    # rerun every seed and compare with results/runs.json
```

`results/RESULTS.md` holds every table; `results/manifest.json` holds the frozen configuration and the SHA-256 of each results file.

## 5. Using it on your own targets

The simulator only needs arrays:

```python
import numpy as np
from targetbook.book import Schedule, simulate
from targetbook.mandate import Mandate, Repairer
from targetbook.metrics import stats

# returns: (T, N) daily simple returns; days: (K,) hand-over day indices; targets: (K, N)
m = Mandate(groups=(0, 0, 1, 1, 2, 2), gross_cap=2.0, net_cap=0.25,
            group_lo=(-0.3, -0.3, -0.3), group_hi=(0.4, 0.4, 0.4), asset_cap=0.5)
rep = Repairer(m)
books = np.array([rep.project(w) for w in targets])
res = simulate(returns, days, books, Schedule(2, 2), band=0.05, band_mode="edge",
               cost=np.full(6, 3e-4), carry_long=np.full(6, 0.002 / 252),
               carry_short=np.full(6, 0.006 / 252), repairer=rep)
print(stats(res))
```

Run the same with `books = targets`, `Schedule(1, 1)` and no frictions to get your own paper line, and take the difference step by step as `pipeline.run_signal` does. Make the band and schedule choices on one part of your history, or on a set of simulated paths, and report on another.
