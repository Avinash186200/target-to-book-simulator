"""The desk's mandate and the ways to make a book satisfy it.

Constraints (all homogeneous, and zero is always feasible):

* gross exposure ``sum |w| <= gross_cap``
* net exposure ``|sum w| <= net_cap``
* each group's net exposure in ``[group_lo[g], group_hi[g]]`` (a negative lower
  bound means the group may be held net short)
* each position ``|w_i| <= asset_cap``

Repairs:

* ``scale_to_feasible``: the largest uniform multiplier s in [0, 1] that makes
  ``s * w`` feasible. Closed form, keeps the shape exactly, can under-invest a lot
  because one breached constraint shrinks every position.
* ``Repairer.project``: the closest feasible book in Euclidean distance, a
  small QP solved with Clarabel (pinned, single-threaded; cross-checked against
  the same problem in cvxpy in the tests). By construction it is never farther from the
  target than the scaled book.
* ``Repairer.min_trade``: used after turnover bands, when keeping some positions
  untouched leaves the book outside the mandate. Finds the feasible book with the
  smallest L1 change from the banded book (plus an L2 tie-break term so the answer is
  unique and does not depend on the solver's path).

Solver output is polished with ``scale_to_feasible`` so that tiny solver
tolerance breaches never reach the simulator.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

SOLVER = "CLARABEL"
FEAS_TOL = 1e-9


@dataclass(frozen=True)
class Mandate:
    groups: tuple[int, ...]
    gross_cap: float = 2.0
    net_cap: float = 0.25
    group_lo: tuple[float, ...] = (-0.30, -0.20, -0.30, -0.15)
    group_hi: tuple[float, ...] = (0.40, 0.50, 0.30, 0.30)
    asset_cap: float = 0.50

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def n_groups(self) -> int:
        return len(self.group_lo)

    def group_matrix(self) -> np.ndarray:
        g = np.asarray(self.groups)
        return (g[None, :] == np.arange(self.n_groups)[:, None]).astype(float)

    def breaches(self, w: np.ndarray) -> dict[str, float]:
        """Amount by which each constraint family is exceeded (0 when satisfied)."""
        gnet = self.group_matrix() @ w
        lo, hi = np.asarray(self.group_lo), np.asarray(self.group_hi)
        return {
            "gross": max(0.0, float(np.abs(w).sum()) - self.gross_cap),
            "net": max(0.0, abs(float(w.sum())) - self.net_cap),
            "group": float(max(0.0, np.max(gnet - hi), np.max(lo - gnet))),
            "asset": max(0.0, float(np.abs(w).max()) - self.asset_cap),
        }

    def is_feasible(self, w: np.ndarray, tol: float = FEAS_TOL) -> bool:
        return all(v <= tol for v in self.breaches(w).values())


def max_feasible_scale(w: np.ndarray, m: Mandate) -> float:
    """Largest s in [0, 1] with s * w feasible (every constraint is s * a(w) <= b)."""
    s = 1.0
    gross = np.abs(w).sum()
    if gross > 0:
        s = min(s, m.gross_cap / gross)
    net = abs(w.sum())
    if net > 0:
        s = min(s, m.net_cap / net)
    amax = np.abs(w).max()
    if amax > 0:
        s = min(s, m.asset_cap / amax)
    for gnet, lo, hi in zip(m.group_matrix() @ w, m.group_lo, m.group_hi):
        if gnet > 0:
            s = min(s, hi / gnet)
        elif gnet < 0:
            s = min(s, lo / gnet)
    return float(max(s, 0.0))


def scale_to_feasible(w: np.ndarray, m: Mandate) -> np.ndarray:
    s = max_feasible_scale(w, m)
    return w * s if s < 1.0 else w.copy()


class Repairer:
    """The two repair problems, solved with Clarabel directly.

    Both are small QPs built once as sparse matrices; only the linear term and
    the right-hand side change per book. Variables (n assets):

    * projection ``z = [x, s]``: minimise ``||x - w||^2`` with ``s >= |x|``;
    * minimal trade ``z = [x, t, s]``: minimise ``sum t + l2 ||x - b||^2`` with
      ``t >= |x - b|`` and ``s >= |x|``.

    The mandate is ``sum s <= gross_cap``, ``|sum x| <= net_cap``, group nets in
    their bounds and ``|x_i| <= asset_cap``. ``tests/test_mandate.py`` checks
    both against the same problems written in cvxpy. Clarabel runs single-threaded
    with fixed settings, so results do not depend on the machine's core count."""

    def __init__(self, m: Mandate, l2_weight: float = 1.0):
        import clarabel
        from scipy import sparse

        self._clarabel = clarabel
        self._sparse = sparse
        self.m = m
        self.l2 = float(l2_weight)
        n = len(m.groups)
        self.n = n
        A = m.group_matrix()
        I, Z = np.eye(n), np.zeros((n, n))
        one, zero = np.ones((1, n)), np.zeros((1, n))
        lo, hi = np.asarray(m.group_lo), np.asarray(m.group_hi)

        # mandate rows on x and s (shared by both problems)
        Mx = np.vstack([I, -I, zero, one, -one, A, -A, I, -I])
        Ms = np.vstack([-I, -I, one, zero, zero, np.zeros_like(A), np.zeros_like(A), Z, Z])
        mb = np.concatenate([np.zeros(2 * n), [m.gross_cap, m.net_cap, m.net_cap], hi, -lo,
                             np.full(2 * n, m.asset_cap)])
        self._mb = mb
        # projection: z = [x, s]
        self._A_proj = sparse.csc_matrix(np.hstack([Mx, Ms]))
        self._P_proj = sparse.csc_matrix(np.diag(np.concatenate([np.full(n, 2.0), np.zeros(n)])))
        # minimal trade: z = [x, t, s]; extra rows x - t <= b and -x - t <= -b
        top = np.hstack([np.vstack([I, -I]), np.vstack([-I, -I]), np.zeros((2 * n, n))])
        mid = np.hstack([Mx, np.zeros((Mx.shape[0], n)), Ms])
        self._A_mt = sparse.csc_matrix(np.vstack([top, mid]))
        self._P_mt = sparse.csc_matrix(np.diag(np.concatenate([np.full(n, 2.0 * self.l2), np.zeros(2 * n)])))
        settings = clarabel.DefaultSettings()
        settings.verbose = False
        settings.max_threads = 1
        settings.tol_gap_abs = settings.tol_gap_rel = settings.tol_feas = 1e-9
        self._settings = settings
        self.n_solves = 0
        self.n_almost = 0
        self.n_fallbacks = 0

    def _qp(self, P, q, A, b, w_fallback: np.ndarray) -> np.ndarray:
        cl = self._clarabel
        self.n_solves += 1
        solver = cl.DefaultSolver(P, q, A, b, [cl.NonnegativeConeT(A.shape[0])], self._settings)
        sol = solver.solve()
        if sol.status == cl.SolverStatus.AlmostSolved:  # reduced accuracy; the polish below keeps it feasible
            self.n_almost += 1
        elif sol.status != cl.SolverStatus.Solved:  # pragma: no cover - never seen on the committed runs
            self.n_fallbacks += 1
            return scale_to_feasible(w_fallback, self.m)
        x = np.asarray(sol.x[: self.n], dtype=float)
        return scale_to_feasible(x, self.m)  # polish away solver-tolerance breaches

    def project(self, w: np.ndarray) -> np.ndarray:
        """Closest feasible book to w in Euclidean distance."""
        if self.m.is_feasible(w):
            return w.copy()
        q = np.concatenate([-2.0 * w, np.zeros(self.n)])
        return self._qp(self._P_proj, q, self._A_proj, self._mb, w)

    def min_trade(self, b: np.ndarray) -> np.ndarray:
        """Feasible book with the smallest L1 change from b (plus an L2 tie-break term)."""
        if self.m.is_feasible(b):
            return b.copy()
        n = self.n
        q = np.concatenate([-2.0 * self.l2 * b, np.ones(n), np.zeros(n)])
        rhs = np.concatenate([b, -b, self._mb])
        return self._qp(self._P_mt, q, self._A_mt, rhs, b)


def blend(books: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Convex blend of candidate books (rows). Feasible books blend to a feasible
    book under these constraints (they are all convex); what blending does break
    is a gross *target*: netting across books lowers the gross."""
    weights = np.asarray(weights, dtype=float)
    return weights @ books / weights.sum()
