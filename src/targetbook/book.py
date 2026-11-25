"""Daily book simulator: tranches, drift, turnover bands, trading cost, carry.

Accounting, per tranche j with NAV share ``nav_j`` and weights ``w_j`` (fractions
of the tranche's NAV; cash earns 0, so returns are excess returns):

* day t return: ``w_j . r_t - carry_j``, carry charged on beginning-of-day
  positions at the long or the short rate of each asset (a short is not the
  negative of a long: it pays a different financing rate);
* weights then drift: ``w_j <- w_j * (1 + r_t) / (1 + tranche return)``;
* on a tranche's rebalance day (at the close, after the day's return) it trades
  from its **drifted** weights to the new book; the cost
  ``sum_i c_i |w_new_i - w_drift_i|`` is a fraction of the tranche NAV.

The combined book is the NAV-weighted sum of the tranches. All tranches start at
the first target with no build cost, so every configuration starts from the same
book.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import TRADING_DAYS


@dataclass(frozen=True)
class Schedule:
    """``n_tranches`` sub-books, each rebalanced every ``period_weeks`` weeks.

    Tranche j trades in week k when ``k % period_weeks == offsets[j]``; the
    offsets are spread evenly over the period, shifted by ``phase``."""

    n_tranches: int = 1
    period_weeks: int = 1
    phase: int = 0

    def __post_init__(self):
        if self.period_weeks % self.n_tranches:
            raise ValueError("period_weeks must be a multiple of n_tranches")

    def offsets(self) -> list[int]:
        step = self.period_weeks // self.n_tranches
        return [(self.phase + j * step) % self.period_weeks for j in range(self.n_tranches)]

    @property
    def label(self) -> str:
        if self.n_tranches == 1:
            base = "weekly" if self.period_weeks == 1 else f"every {self.period_weeks} weeks"
            return base + (f" (phase {self.phase})" if self.phase else "")
        return f"{self.n_tranches} tranches x {self.period_weeks} weeks"


@dataclass
class SimResult:
    returns: np.ndarray  # daily net return of the combined book
    cost: np.ndarray  # daily trading cost (fraction of NAV)
    carry: np.ndarray  # daily carry (fraction of NAV)
    gross: np.ndarray  # end-of-day gross exposure of the combined book
    turnover: float  # traded notional per year, measured on drifted weights
    turnover_vs_targets: float  # the same, measured between consecutive targets
    n_repairs: int  # rebalances where the banded book needed a mandate repair
    n_tranche_rebalances: int


def apply_band(current: np.ndarray, target: np.ndarray, band: float, mode: str) -> np.ndarray:
    """No-trade zone of half-width ``band`` (in weight) around each target.

    Positions inside the zone are left alone. Outside it, ``mode="edge"`` trades
    to the nearest edge of the zone and ``mode="target"`` trades all the way."""
    if band <= 0:
        return target.copy()
    diff = target - current
    out = current.copy()
    move = np.abs(diff) > band
    if mode == "edge":
        out[move] = target[move] - np.sign(diff[move]) * band
    elif mode == "target":
        out[move] = target[move]
    else:
        raise ValueError(f"unknown band mode {mode!r}")
    return out


def simulate(
    returns: np.ndarray,
    rebalance_days: np.ndarray,
    books: np.ndarray,
    schedule: Schedule = Schedule(),
    *,
    band: float = 0.0,
    band_mode: str = "edge",
    cost: np.ndarray | None = None,
    carry_long: np.ndarray | None = None,
    carry_short: np.ndarray | None = None,
    repairer=None,
) -> SimResult:
    """Simulate from rebalance_days[0] to the last day of ``returns``.

    ``books[k]`` is the book handed over at the close of ``rebalance_days[k]``.
    ``cost`` is per unit of traded notional; ``carry_long`` / ``carry_short`` are
    daily rates per unit of long / short exposure. ``repairer`` (with
    ``min_trade`` and ``m.is_feasible``) is applied when a banded book breaks the
    mandate."""
    T, N = returns.shape
    m = schedule.n_tranches
    p = schedule.period_weeks
    offsets = schedule.offsets()
    cost = np.zeros(N) if cost is None else np.asarray(cost, float)
    cl = np.zeros(N) if carry_long is None else np.asarray(carry_long, float)
    cs = np.zeros(N) if carry_short is None else np.asarray(carry_short, float)

    start = int(rebalance_days[0])
    week_of = {int(d): k for k, d in enumerate(rebalance_days)}
    W = np.tile(books[0], (m, 1))
    last_target = np.zeros(m, dtype=int)
    nav = np.full(m, 1.0 / m)

    n_out = T - start - 1
    out_ret = np.empty(n_out)
    out_cost = np.zeros(n_out)
    out_carry = np.empty(n_out)
    out_gross = np.empty(n_out)
    turn = 0.0
    turn_naive = 0.0
    n_repairs = 0
    n_reb = 0

    for i, t in enumerate(range(start + 1, T)):
        r = returns[t]
        navb = nav.copy()
        nav0 = navb.sum()
        gross_ret = W @ r
        carry = np.maximum(W, 0.0) @ cl + np.maximum(-W, 0.0) @ cs
        tr_ret = gross_ret - carry
        W = W * (1.0 + r) / (1.0 + tr_ret)[:, None]
        nav = nav * (1.0 + tr_ret)
        out_carry[i] = float(navb @ carry / nav0)

        k = week_of.get(t)
        if k is not None:
            for j in range(m):
                if k % p != offsets[j]:
                    continue
                target = books[k]
                new = apply_band(W[j], target, band, band_mode)
                if band > 0 and repairer is not None and not repairer.m.is_feasible(new):
                    new = repairer.min_trade(new)
                    n_repairs += 1
                traded = np.abs(new - W[j])
                c = float(traded @ cost)
                out_cost[i] += nav[j] * c / nav0  # exact share of the day's return
                share = nav[j] / nav.sum()
                turn += share * float(traded.sum())
                turn_naive += share * float(np.abs(target - books[last_target[j]]).sum())
                nav[j] *= 1.0 - c
                W[j] = new
                last_target[j] = k
                n_reb += 1

        out_ret[i] = nav.sum() / nav0 - 1.0
        combined = (nav[:, None] * W).sum(axis=0) / nav.sum()
        out_gross[i] = float(np.abs(combined).sum())

    years = n_out / TRADING_DAYS
    return SimResult(
        returns=out_ret,
        cost=out_cost,
        carry=out_carry,
        gross=out_gross,
        turnover=turn / years,
        turnover_vs_targets=turn_naive / years,
        n_repairs=n_repairs,
        n_tranche_rebalances=n_reb,
    )
