# Changelog

All notable changes are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
uses [Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-11-25

### Added

- Synthetic market: 12 assets in 4 groups, market and group factors, Student-t shocks, stochastic volatility, a planted slow alpha driving next-day drift, a model signal (alpha plus noise) and a no-edge control signal with the same dynamics.
- Research targets (signal z-score over trailing vol, fixed gross) formed point-in-time each week.
- Mandate with gross, net, group-net and per-position limits; uniform max-feasible scaling; Euclidean projection and minimal-trade repair as Clarabel QPs, cross-checked against cvxpy in the tests.
- Ex-ante volatility cap on a Ledoit-Wolf shrunk covariance.
- Daily simulator with staggered tranches, drift, turnover bands (edge and target modes), trading cost on drifted weights and carry on long and short exposure at separate rates.
- Attribution waterfall from paper target to real book, each step alone, projection vs scaling, rebalance-timing luck, band and cost sweeps, over 10 held-out seeds with choices made on 5 development seeds; no-edge control throughout.
- Results tests that recompute every summary from per-seed records, check a manifest of configuration and SHA-256 checksums, rerun one held-out and one development seed (all with `FULL=1`), and check README tables against RESULTS.md.
- `constraints.txt` with exact versions, CI on ubuntu (Python 3.11) with a secret scan.
- Docs: methodology, walkthrough, choosing turnover bands.
