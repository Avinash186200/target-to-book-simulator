# Choosing turnover bands

A turnover band (no-trade zone) of half-width h leaves a position alone while it is within h of its target. When it drifts or the target moves further, the desk trades: either to the near edge of the zone (**edge mode**) or all the way to the target (**target mode**). This page is about how wide h should be and which mode to use, using the numbers in `results/RESULTS.md` (development seeds for the choices, held-out seeds for the checks).

## What a band buys and what it costs

A band does three things at once:

1. **It saves trading cost.** Small trades are skipped. With the chosen schedule, turnover falls from 38.7 to 19.8 a year at h = 0.08 in edge mode, and cost drag from 1.09% to 0.51% a year (held-out medians).
2. **It makes the book lag its target.** In edge mode a position that should go from 0 to 0.20 stops at 0.12. The book holds less of the signal than intended (gross 1.72 without bands, 1.55 with), so return per year falls even as Sharpe can rise.
3. **It can break the mandate.** The untouched positions are old ones. If the new target sits on the gross cap, keeping some old positions pushes the book over it, and the desk must trade again to repair. Here that happens at 226 of 503 tranche rebalances in the real book (median), and the minimal-trade repair takes back part of the turnover saved.

Without costs, bands bring no clear benefit: on the weekly paper book with no costs, the band alone changed Sharpe by +0.04 [-0.20, 0.24] and lost return in 7 of 10 held-out seeds. Bands are a cost tool.

## Edge mode or target mode

Edge mode won on development seeds at every band up to 0.12 for every schedule, by up to 0.27 Sharpe (weekly schedule, h = 0.12). Beyond that the picture is mixed: at 0.16 edge mode loses to target mode for 4 tranches x 4 weeks. The reason is the classic one for proportional costs: once a trade is worth making, the cheapest move back inside the zone is to its edge. Trading to the target pays to cross the whole zone and invites the next trade the other way.

At very wide bands (0.24) target mode wins. An edge-mode book with a wide zone almost never trades (5.5 a year on held-out seeds) and drifts far from its targets; target mode still makes full corrections when it does trade.

## How wide

On development seeds, with the chosen schedule (2 tranches x 2 weeks), mean net Sharpe in edge mode:

| cost x | no band | 0.02 | 0.04 | 0.08 | 0.12 | 0.16 | 0.24 | chosen |
|---|---|---|---|---|---|---|---|---|
| 0.5 | 0.898 | 0.954 | 0.997 | 1.049 | 1.011 | 0.926 | 0.713 | 0.08 |
| 1 | 0.784 | 0.856 | 0.913 | 0.987 | 0.964 | 0.889 | 0.691 | 0.08 |
| 2 | 0.556 | 0.661 | 0.746 | 0.862 | 0.869 | 0.815 | 0.648 | 0.12 |
| 4 | 0.100 | 0.272 | 0.412 | 0.614 | 0.679 | 0.667 | 0.562 | 0.12 |

The best width grows with the cost level, and slowly: an eightfold rise in cost moves it from 0.08 to 0.12. Theory for a single asset with proportional costs says the optimal zone grows roughly with the cube root of the cost; on this coarse grid I cannot tell that apart from other slow growth, so treat it as context, not as something this repo tested.

The curve is flat near the top. At the scenario cost level, widths from 0.04 to 0.12 are all within 0.08 Sharpe of the best on development seeds, smaller than one standard error (about 0.2).

## Held-out check

Held-out seeds, chosen schedule, scenario costs: change in Sharpe against no band, paired per seed.

| band (edge) | delta Sharpe | seeds better |
|---|---|---|
| 0.02 | 0.034 [0.011, 0.099] | 10/10 |
| 0.04 | 0.067 [-0.008, 0.184] | 8/10 |
| 0.08 (chosen) | 0.108 [-0.100, 0.291] | 7/10 |
| 0.12 | 0.044 [-0.240, 0.402] | 6/10 |
| 0.16 | -0.059 [-0.479, 0.503] | 4/10 |
| 0.24 | -0.275 [-0.475, 0.214] | 2/10 |

Two things to note. Wider bands give larger median gains but less consistent ones: 0.02 helped in every seed by a little; 0.08 helped more in the median but hurt in 3 of 10. And the chosen 0.08 was the best median, not the most reliable. Ten seeds cannot separate 0.04 from 0.08.

At other cost levels, with the band chosen on development seeds: +0.058 Sharpe at half the cost (6/10 seeds better), +0.108 at the scenario cost (7/10), +0.168 at twice (7/10) and +0.414 at four times (10/10).

## A recipe

1. **Use edge mode** unless the band is so wide that the book barely trades.
2. **Pick h on data you will not report on**, at your own cost level, with the rest of the book (mandate, vol cap, schedule) already in place. Bands interact with all three.
3. **Prefer the narrow end of the flat region** when results are close. It gives up a little median gain for a lot of consistency, lags the target less, and needs fewer mandate repairs.
4. **Count turnover on drifted weights.** With bands the holdings are not the last target. Counting turnover between consecutive targets would report 38.7 a year for a book that actually trades 19.8 (ratio 0.510 [0.490, 0.532]); a cost budget built on target changes would be twice too high.
5. **Revisit h when costs change.** Here, quadrupling costs moved the best band from 0.08 to 0.12 and turned the band from a coin flip into a win in every seed.

## What this page does not cover

- Bands in units of risk (wider for low-vol assets) or per-asset costs; here h is one number in weight units for every asset.
- Optimal partial trading (trade a fixed fraction of the gap toward the target each period), which is the closest alternative to bands for quadratic costs.
- Market impact, where large trades cost more per unit and the band should depend on trade size.
