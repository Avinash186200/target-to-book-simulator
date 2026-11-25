# Methodology

The rules behind every number in the README, and what was decided where. Every figure quoted here is a field in `results/*.json` and appears in `results/RESULTS.md`.

## Market

`data.py` draws one independent panel per seed: 11 years of 252 trading days, 12 assets in 4 groups of 3 (named equity, rates, commodities and credit only to give the groups different volatilities: 16%, 6%, 20% and 8% a year).

- **Shocks.** `eps = sqrt(0.15) m + sqrt(0.35) g_group + sqrt(0.5) e`: one market factor, one factor per group, one idiosyncratic term, so assets in a group correlate at 0.5 and across groups at 0.15. Each term is Student-t with 5 degrees of freedom scaled to unit variance.
- **Stochastic volatility.** A common multiplier `v_t = exp(x_t - s^2)`, with x an AR(1) in logs (half-life 60 days, sd 0.30), so E[v^2] = 1 and the volatility cap has regimes to react to.
- **Planted edge.** A true alpha `a[t, i]`, AR(1) with a 60-day half-life and stationary sd 0.03, in units of the asset's daily vol. The drift of day t is `sigma_i * a[t - 1, i]`: the alpha known at the previous close. Returns are `r[t] = sigma_i * a[t - 1] + sigma_i * v_t * eps[t]`.
- **Model signal.** `a[t] + n[t]`, with n an AR(1) noise (10-day half-life, sd 0.04), known at the close of day t.
- **Control signal.** An independent AR(1) with the alpha's parameters plus an independent noise with the noise's parameters. It has the same persistence and the same turnover as the model signal and no information about returns.

## From signal to book

| step | rule | data used |
|---|---|---|
| research target | z-score the signal across assets, divide by each asset's trailing 63-day vol, scale to gross 2.5 | signal and returns up to the hand-over day |
| mandate | gross <= 2.0; abs(net) <= 0.25; group nets in [-0.30, 0.40], [-0.20, 0.50], [-0.30, 0.30], [-0.15, 0.30]; abs(position) <= 0.5 | - |
| repair (default) | Euclidean projection onto the mandate (QP, Clarabel) | the target |
| repair (compared) | the largest uniform multiple s in [0, 1] that is feasible | the target |
| vol cap | if ex-ante vol > 6% a year, scale the book to 6%; never scale up | trailing 252-day returns up to the hand-over day |
| covariance | sample correlations shrunk toward zero with the Ledoit-Wolf (2004) intensity on standardised returns, rescaled by sample vols | same window |
| schedule | tranche j of N rebalances in week k when `k mod period = offset_j`, offsets spread evenly over the period | - |
| bands | positions within h of the target are left alone; outside, trade to the edge (`target -/+ h`) or to the target | drifted holdings |
| band repair | if the banded book breaks the mandate: minimise `sum abs(x - b) + ||x - b||^2` over the mandate | the banded book |
| trading cost | `sum_i c_i abs(w_new - w_drifted)` per tranche, c = 3, 1, 5, 4 bps by group | drifted holdings |
| carry | per day `(sum long_i * cl_i + sum short_i * cs_i) / 252`, cl = 20, 10, 30, 25 and cs = 60, 30, 80, 70 bps a year by group | beginning-of-day positions |

Every QP is solved by Clarabel directly (`mandate.Repairer`), single-threaded, with tolerances 1e-9, from matrices built once per seed. The solution is then multiplied by its own max-feasible scale, so a solver-tolerance breach never reaches the simulator. `tests/test_mandate.py` solves the same two problems in cvxpy and checks the answers agree to 1e-4. Over the held-out runs: 68,033 solves, none with reduced accuracy, none falling back to scaling (`repair.json`).

BLAS and OpenMP threads are pinned to 1 when the package is imported, so the numbers do not depend on the machine's core count.

## Simulation

`book.py`, per tranche, per day (cash earns zero, so returns are excess returns):

1. Earn `w . r_t`, pay carry on beginning-of-day positions.
2. Drift: `w <- w * (1 + r_t) / (1 + tranche return)`.
3. On the tranche's rebalance day (at the close): apply the band, repair if needed, pay the trading cost from the tranche NAV, set the new weights.

The combined book is the NAV-weighted sum of the tranches. All tranches start at the first target with no build cost, so every configuration starts from the same book. The evaluated window is the ten years after the warm-up year.

## Waterfall

The ladder adds one step at a time. Each stage differs from the one before in exactly one setting:

| stage | books | schedule | costs | carry | bands |
|---|---|---|---|---|---|
| paper | targets | weekly | no | no | no |
| mandate | projected | weekly | no | no | no |
| vol cap | projected, capped | weekly | no | no | no |
| schedule | projected, capped | desk | no | no | no |
| costs | projected, capped | desk | yes | no | no |
| carry | projected, capped | desk | yes | yes | no |
| bands (real book) | projected, capped | desk | yes | yes | yes |

Deltas are taken per seed and then summarised (median, min, max over the 10 held-out seeds). Return deltas use the arithmetic annualised return (mean daily return x 252), so per seed the stage deltas add up exactly to real minus paper. Medians of the deltas do not have to add up to the median of the total.

Because the size of a step depends on what came before it, each step is also run **alone** on top of the paper book: projected targets, capped raw targets, the desk schedule, costs, carry, or bands on the raw targets (with no mandate there is nothing to repair). The README shows both.

The **control** runs the same ladder with the control signal. Costs and carry should lose about the same with or without an edge; anything the control "earns" is noise of the 10-year sample.

## Development vs held-out

| | seeds | used for |
|---|---|---|
| development | 1-5 | generator calibration by hand, the schedule, the band width and mode, the band per cost level |
| held-out | 101-110 | every reported table, with the frozen choices; held-out results never used for any choice (held-out sections first read after the third full run; a fourth run only added diagnostic fields; see below) |

Made by code from development records (`results/dev_choices.json`):

- **Schedule, band, mode.** Every combination of 3 schedules (weekly; 2 tranches x 2 weeks; 4 tranches x 4 weeks) and 13 band settings (no band, and 6 widths from 0.02 to 0.24 in two modes), with every friction on at the scenario cost level, scored by mean net Sharpe over the 5 development seeds. Highest mean wins: 2 tranches x 2 weeks, band 0.08, edge mode. The best band for each schedule scored 0.971 to 0.987 with standard errors near 0.17: the schedule choice is a tie.
- **Band per cost level.** With the chosen schedule, the best band at 0.5x, 1x, 2x and 4x the scenario costs: 0.08, 0.08, 0.12 and 0.12, all edge mode.

Made by hand on development seeds while building, before any held-out run:

- The edge size. First tried alpha sd 0.04 on 6-year panels: paper Sharpe 1.36 and 1.61 on two development seeds, more than "modest". Lowered to 0.03, then lengthened the panels to 11 years (10 evaluated) to cut the sampling noise of each seed's Sharpe.
- The vol cap level. At 10% it never bound on the repaired book (development seed 1), so it was lowered to 6%, where it binds in a fraction of weeks (40% median on held-out seeds).
- The minimal-trade repair objective (L1 plus an L2 tie-break term for a unique answer), and replacing the first cvxpy implementation of both QPs with direct Clarabel calls for speed (the cvxpy versions stay as the reference in the tests).

**History of the held-out runs.** The first full run used a band grid that stopped at 0.08, and the development choice landed on that largest band at every cost level. That run also wrote held-out results; I read only the console line with the development choice and the development section of RESULTS.md, not the held-out sections. I widened the grid to 0.24 (a test now fails if the choice is at the grid edge) and reran everything. In the second run Clarabel returned 5 reduced-accuracy solutions out of about 68,000 with tolerances of 1e-10; they had been sent to the scaling fallback. I set the tolerances to 1e-9, made a reduced-accuracy answer usable (it is polished to feasibility anyway, and counted), and reran. That rerun had no reduced-accuracy answers and no fallbacks. After that, only summaries and report text were added (the turnover-measure comparison with bands, the development cost grid table); they compute nothing new from data. After an external review, two diagnostic fields were added to every configuration (maximum gross and the share of days with gross above the cap, both measured on the existing simulations) and the benchmark was rerun to record them; every previously recorded field came out identical. No held-out result was used to change a method, a setting or a seed.

## Leakage and look-ahead

- Targets, vol estimates and the covariance at a hand-over use data up to and including that day; trades happen at its close and earn from the next day. `tests/test_book.py::test_no_look_ahead_in_targets_covariance_and_book` replaces every return after a day d with noise and checks that targets, covariance and book returns up to d are bit-for-bit unchanged.
- The drift of day t depends on the alpha at t - 1 only (`tests/test_data.py`).
- Choices are made on development seeds, which are disjoint from held-out seeds (tested), and the library defaults equal the frozen choices (tested).

## Reproducing

`make bench` rewrites `results/`. `pytest` recomputes every summary and RESULTS.md from `runs.json`, checks `manifest.json` (configuration and SHA-256 of each results file) and reruns held-out seed 101 and development seed 1 from scratch; `make verify` reruns every seed. Floats are compared to 1e-6, integer counts to within 1 or 1%.
