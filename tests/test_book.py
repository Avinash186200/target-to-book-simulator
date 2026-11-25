import numpy as np
import pytest

from targetbook.book import Schedule, apply_band, simulate
from targetbook.data import make_panel
from targetbook.mandate import Mandate, Repairer
from targetbook.risk import shrunk_cov
from targetbook.targets import weekly_targets


def _short_panel(seed, T):
    p = make_panel(seed)
    days = p.rebalance_days[p.rebalance_days < T - 1]
    return p, p.returns[:T], days


def test_schedule_offsets():
    assert Schedule(4, 4).offsets() == [0, 1, 2, 3]
    assert Schedule(2, 4).offsets() == [0, 2]
    assert Schedule(1, 4, phase=3).offsets() == [3]
    with pytest.raises(ValueError):
        Schedule(3, 4)


def test_apply_band_modes():
    cur = np.array([0.10, 0.10, 0.10])
    tgt = np.array([0.12, 0.30, -0.10])
    assert np.allclose(apply_band(cur, tgt, 0.05, "edge"), [0.10, 0.25, -0.05])
    assert np.allclose(apply_band(cur, tgt, 0.05, "target"), [0.10, 0.30, -0.10])
    assert np.array_equal(apply_band(cur, tgt, 0.0, "edge"), tgt)


def test_fully_invested_single_asset_earns_the_asset_return():
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.01, (60, 1))
    days = np.arange(10, 59, 5)
    res = simulate(r, days, np.ones((len(days), 1)))
    assert np.allclose(res.returns, r[11:, 0])
    assert res.turnover == 0.0


def test_trading_cost_is_measured_on_drifted_weights():
    r = np.zeros((8, 2))
    r[3] = [0.10, -0.10]  # one move between the two hand-overs
    days = np.array([1, 4])
    books = np.array([[0.5, 0.5], [0.5, 0.5]])
    res = simulate(r, days, books, cost=np.array([0.01, 0.01]))
    # after day 3 the weights are 0.55 / 0.45 of an unchanged NAV: trading back costs 1% of 0.10
    assert np.isclose(res.cost.sum(), 0.01 * 0.10)
    assert res.turnover_vs_targets == 0.0  # identical consecutive targets: the naive measure sees no trade


def test_carry_charges_longs_and_shorts_at_their_own_rates():
    r = np.zeros((20, 2))
    days = np.arange(2, 19, 5)
    books = np.tile([1.0, -1.0], (len(days), 1))
    res = simulate(r, days, books, carry_long=np.array([1e-4, 0.0]), carry_short=np.array([0.0, 3e-4]))
    assert np.isclose(res.returns[0], -4e-4) and np.isclose(res.carry[0], 4e-4)
    # carry is paid from NAV, so exposure per unit of NAV creeps up until the next hand-over
    assert np.allclose(res.returns, -4e-4, rtol=2e-3, atol=0)
    assert np.allclose(res.carry, -res.returns)


def test_staggered_book_is_the_average_of_its_single_phase_books():
    """With N tranches, combined NAV = mean of N single books at each phase (exact,
    with costs, carry, bands and mandate repairs on)."""
    p, r, days = _short_panel(2, 700)
    m = Mandate(groups=tuple(int(g) for g in p.groups))
    rep = Repairer(m)
    books = np.array([rep.project(w) for w in weekly_targets(r, p.signal, days)])
    kw = dict(band=0.04, cost=np.full(12, 3e-4), carry_long=np.full(12, 1e-5),
              carry_short=np.full(12, 3e-5), repairer=rep)
    stag = simulate(r, days, books, Schedule(4, 4), **kw)
    singles = [simulate(r, days, books, Schedule(1, 4, ph), **kw) for ph in range(4)]
    nav_stag = np.cumprod(1 + stag.returns)
    nav_mean = np.mean([np.cumprod(1 + s.returns) for s in singles], axis=0)
    assert np.allclose(nav_stag, nav_mean, rtol=0, atol=1e-12)


def test_a_huge_band_never_trades_after_the_start_and_zero_band_is_no_band():
    p, r, days = _short_panel(3, 400)
    books = weekly_targets(r, p.signal, days)
    assert simulate(r, days, books, band=10.0).turnover == 0.0
    a = simulate(r, days, books, band=0.0)
    b = simulate(r, days, books)
    assert np.array_equal(a.returns, b.returns)


def test_banded_books_that_break_the_mandate_are_repaired():
    p, r, days = _short_panel(4, 600)
    m = Mandate(groups=tuple(int(g) for g in p.groups))
    rep = Repairer(m)
    books = np.array([rep.project(w) for w in weekly_targets(r, p.signal, days)])
    res = simulate(r, days, books, Schedule(1, 1), band=0.08, repairer=rep)
    assert res.n_repairs > 0
    # right after each hand-over the book is inside the mandate, so its gross is within the cap
    handover = np.isin(np.arange(int(days[0]) + 1, r.shape[0]), days)
    assert res.gross[handover].max() <= m.gross_cap + 1e-9


def test_no_look_ahead_in_targets_covariance_and_book():
    """Changing returns after day d changes nothing decided at or before d."""
    p = make_panel(6)
    d = int(p.rebalance_days[30])
    future = p.returns.copy()
    future[d + 1:] = np.random.default_rng(1).normal(0, 0.05, future[d + 1:].shape)
    k = np.searchsorted(p.rebalance_days, d, side="right")
    a = weekly_targets(p.returns, p.signal, p.rebalance_days)
    b = weekly_targets(future, p.signal, p.rebalance_days)
    assert np.array_equal(a[:k], b[:k])
    assert np.array_equal(shrunk_cov(p.returns, d)[0], shrunk_cov(future, d)[0])
    ra = simulate(p.returns, p.rebalance_days, a)
    rb = simulate(future, p.rebalance_days, b)
    n = d - int(p.rebalance_days[0])  # simulated days up to and including d
    assert np.array_equal(ra.returns[:n], rb.returns[:n])
