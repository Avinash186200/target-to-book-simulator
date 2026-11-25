import numpy as np
import pytest

from targetbook.mandate import Mandate, Repairer, blend, max_feasible_scale, scale_to_feasible

M = Mandate(groups=tuple(int(g) for g in np.repeat(np.arange(4), 3)))


def _books(n_books, seed=0, scale=0.35):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_books, 12)) * scale


@pytest.fixture(scope="module")
def rep():
    return Repairer(M)


def test_breaches_report_each_constraint_family():
    w = np.zeros(12)
    assert M.is_feasible(w)
    w[0] = 0.6  # asset cap 0.5, equity group cap 0.4, net cap 0.25
    b = M.breaches(w)
    assert np.isclose(b["asset"], 0.1) and np.isclose(b["group"], 0.2) and np.isclose(b["net"], 0.35)
    assert b["gross"] == 0.0


def test_uniform_scale_is_feasible_and_maximal():
    for w in _books(200, seed=1):
        s = max_feasible_scale(w, M)
        assert M.is_feasible(s * w)
        if s < 1.0:
            assert not M.is_feasible(s * 1.001 * w)


def test_feasible_books_are_left_alone(rep):
    for w in _books(50, seed=2, scale=0.02):
        assert M.is_feasible(w)
        assert np.array_equal(scale_to_feasible(w, M), w)
        assert np.array_equal(rep.project(w), w)
        assert np.array_equal(rep.min_trade(w), w)


def test_projection_is_feasible_and_never_farther_than_scaling(rep):
    for w in _books(200, seed=3):
        x = rep.project(w)
        assert M.is_feasible(x)
        assert np.linalg.norm(x - w) <= np.linalg.norm(scale_to_feasible(w, M) - w) + 1e-9


def test_min_trade_is_feasible_and_trades_less_than_scaling(rep):
    rng = np.random.default_rng(4)
    checked = 0
    for _ in range(100):
        b = rep.project(rng.standard_normal(12) * 0.4)
        i = rng.integers(12)
        b[i] += 0.3 if b[i] >= 0 else -0.3  # push one position outside the mandate
        if M.is_feasible(b):
            continue
        y = rep.min_trade(b)
        assert M.is_feasible(y)
        assert np.abs(y - b).sum() <= np.abs(scale_to_feasible(b, M) - b).sum() + 1e-9
        checked += 1
    assert checked > 50


def _cvxpy_problems():
    cp = pytest.importorskip("cvxpy")
    A = M.group_matrix()

    def cons(x):
        return [cp.norm1(x) <= M.gross_cap, cp.abs(cp.sum(x)) <= M.net_cap,
                A @ x >= np.array(M.group_lo), A @ x <= np.array(M.group_hi), cp.abs(x) <= M.asset_cap]

    w, x = cp.Parameter(12), cp.Variable(12)
    proj = cp.Problem(cp.Minimize(cp.sum_squares(x - w)), cons(x))
    b, y = cp.Parameter(12), cp.Variable(12)
    mt = cp.Problem(cp.Minimize(cp.norm1(y - b) + cp.sum_squares(y - b)), cons(y))
    return (proj, w, x), (mt, b, y)


def test_projection_and_min_trade_match_other_solvers_via_cvxpy(rep):
    """Same problems, independent solvers: OSQP for the projection, SCS for the minimal trade."""
    (proj, w, x), (mt, b, y) = _cvxpy_problems()
    for v in _books(40, seed=5):
        w.value = v
        proj.solve(solver="OSQP", eps_abs=1e-10, eps_rel=1e-10, max_iter=200000)
        assert np.abs(rep.project(v) - x.value).max() < 1e-4
        b.value = v
        mt.solve(solver="SCS", eps_abs=1e-9, eps_rel=1e-9, max_iters=200000)
        assert np.abs(rep.min_trade(v) - y.value).max() < 1e-4


def test_blending_feasible_books_stays_feasible_but_loses_gross(rep):
    weights = [0.4, 0.3, 0.2, 0.1]
    books = np.array([rep.project(w) for w in _books(4, seed=6, scale=1.0)])
    assert all(M.is_feasible(w) for w in books)
    mixed = blend(books, weights)
    assert M.is_feasible(mixed)
    assert np.abs(mixed).sum() < np.average(np.abs(books).sum(axis=1), weights=weights)
