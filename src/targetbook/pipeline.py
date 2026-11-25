"""One seed, end to end: targets -> repaired -> vol-capped books, then every
simulated configuration (the waterfall ladder, each step alone, the repair
comparison, rebalance-timing luck, band and cost sweeps)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np

from .book import Schedule, simulate
from .data import TRADING_DAYS, MarketConfig, Panel, make_panel
from .mandate import Mandate, Repairer, scale_to_feasible
from .metrics import stats
from .risk import shrunk_cov, vol_cap
from .targets import weekly_targets

BAND_GRID = (0.0, 0.02, 0.04, 0.08, 0.12, 0.16, 0.24)
BAND_MODES = ("edge", "target")
SCHEDULES = ((1, 1), (2, 2), (4, 4))
COST_MULTS = (0.5, 1.0, 2.0, 4.0)
LUCK_PERIOD = 4


def band_configs() -> list[tuple[float, str]]:
    """(band, mode) pairs; a zero band is the same in both modes."""
    return [(0.0, "edge")] + [(b, m) for b in BAND_GRID if b > 0 for m in BAND_MODES]


@dataclass(frozen=True)
class DeskConfig:
    """Desk frictions (fixed by the scenario) and implementation choices.

    The last four fields are the development-seed choices (see bench.py and
    results/dev_choices.json); the defaults are those frozen choices."""

    vol_cap: float = 0.06
    cost_bps: tuple[float, ...] = (3.0, 1.0, 5.0, 4.0)  # one-way, per group
    carry_long_bps: tuple[float, ...] = (20.0, 10.0, 30.0, 25.0)  # per year, per group
    carry_short_bps: tuple[float, ...] = (60.0, 30.0, 80.0, 70.0)  # per year, per group
    n_tranches: int = 2
    period_weeks: int = 2
    band: float = 0.08
    band_mode: str = "edge"

    def to_dict(self) -> dict:
        return asdict(self)

    def choices(self) -> dict:
        return {k: getattr(self, k) for k in ("n_tranches", "period_weeks", "band", "band_mode")}

    @property
    def schedule(self) -> Schedule:
        return Schedule(self.n_tranches, self.period_weeks)


@dataclass
class Prepared:
    panel: Panel
    mandate: Mandate
    repairer: Repairer
    covs: np.ndarray  # (K, N, N) shrunk covariance at each rebalance day
    shrinkage: np.ndarray  # (K,)
    cost: np.ndarray  # (N,) per unit traded
    carry_long: np.ndarray  # (N,) daily
    carry_short: np.ndarray  # (N,) daily


def prepare(seed: int, market: MarketConfig | None = None, desk: DeskConfig | None = None) -> Prepared:
    market = market or MarketConfig()
    desk = desk or DeskConfig()
    panel = make_panel(seed, market)
    mandate = Mandate(groups=tuple(int(g) for g in panel.groups))
    covs, shrink = [], []
    for d in panel.rebalance_days:
        c, delta = shrunk_cov(panel.returns, int(d))
        covs.append(c)
        shrink.append(delta)
    g = panel.groups
    return Prepared(
        panel=panel,
        mandate=mandate,
        repairer=Repairer(mandate),
        covs=np.asarray(covs),
        shrinkage=np.asarray(shrink),
        cost=np.asarray(desk.cost_bps)[g] / 1e4,
        carry_long=np.asarray(desk.carry_long_bps)[g] / 1e4 / TRADING_DAYS,
        carry_short=np.asarray(desk.carry_short_bps)[g] / 1e4 / TRADING_DAYS,
    )


def build_books(prep: Prepared, signal_name: str, desk: DeskConfig) -> tuple[dict, dict]:
    """Weekly books at each stage of construction, plus diagnostics."""
    p = prep.panel
    sig = p.signal if signal_name == "model" else p.control
    raw = weekly_targets(p.returns, sig, p.rebalance_days)
    proj = np.array([prep.repairer.project(w) for w in raw])
    scaled = np.array([scale_to_feasible(w, prep.mandate) for w in raw])

    def capped(books):
        out, hits = [], 0
        for w, c in zip(books, prep.covs):
            v, hit = vol_cap(w, c, desk.vol_cap)
            out.append(v)
            hits += hit
        return np.asarray(out), hits / len(books)

    raw_c, raw_hit = capped(raw)
    proj_c, proj_hit = capped(proj)
    scaled_c, _ = capped(scaled)
    books = {"raw": raw, "proj": proj, "scaled": scaled,
             "raw_capped": raw_c, "proj_capped": proj_c, "scaled_capped": scaled_c}
    infeasible = np.array([not prep.mandate.is_feasible(w) for w in raw])
    diag = {
        "share_targets_outside_mandate": float(infeasible.mean()),
        "gross_target": float(np.abs(raw).sum(axis=1).mean()),
        "gross_projected": float(np.abs(proj).sum(axis=1).mean()),
        "gross_scaled": float(np.abs(scaled).sum(axis=1).mean()),
        "distance_projected": float(np.linalg.norm(proj - raw, axis=1).mean()),
        "distance_scaled": float(np.linalg.norm(scaled - raw, axis=1).mean()),
        "share_vol_cap_binding_projected": float(proj_hit),
        "share_vol_cap_binding_target": float(raw_hit),
        "mean_shrinkage": float(prep.shrinkage.mean()),
    }
    return books, diag


def _sim(prep: Prepared, books: dict, key: str, schedule: Schedule, *, band=0.0, mode="edge",
         costs=False, carry=False, cost_mult=1.0, repair=True):
    p = prep.panel
    return simulate(
        p.returns, p.rebalance_days, books[key], schedule,
        band=band, band_mode=mode,
        cost=prep.cost * cost_mult if costs else None,
        carry_long=prep.carry_long if carry else None,
        carry_short=prep.carry_short if carry else None,
        repairer=prep.repairer if repair else None,
    )


LADDER = ("paper", "mandate", "vol_cap", "schedule", "costs", "carry", "bands")
ALONE = ("mandate", "vol_cap", "schedule", "costs", "carry", "bands")


def run_signal(prep: Prepared, signal_name: str, desk: DeskConfig, *, sweeps: bool,
               cost_bands: dict | None = None) -> dict:
    books, diag = build_books(prep, signal_name, desk)
    weekly = Schedule(1, 1)
    ds = desk.schedule
    full = dict(costs=True, carry=True)
    real = dict(band=desk.band, mode=desk.band_mode, **full)

    sims = {}
    # Waterfall ladder: each stage adds exactly one step to the one before.
    sims["ladder/paper"] = _sim(prep, books, "raw", weekly)
    sims["ladder/mandate"] = _sim(prep, books, "proj", weekly)
    sims["ladder/vol_cap"] = _sim(prep, books, "proj_capped", weekly)
    sims["ladder/schedule"] = _sim(prep, books, "proj_capped", ds)
    sims["ladder/costs"] = _sim(prep, books, "proj_capped", ds, costs=True)
    sims["ladder/carry"] = _sim(prep, books, "proj_capped", ds, **full)
    sims["ladder/bands"] = _sim(prep, books, "proj_capped", ds, **real)
    # Each step alone on top of the paper book (no mandate, so no repair after bands).
    sims["alone/vol_cap"] = _sim(prep, books, "raw_capped", weekly)
    sims["alone/schedule"] = _sim(prep, books, "raw", ds)
    sims["alone/costs"] = _sim(prep, books, "raw", weekly, costs=True)
    sims["alone/carry"] = _sim(prep, books, "raw", weekly, carry=True)
    sims["alone/bands"] = _sim(prep, books, "raw", weekly, band=desk.band, mode=desk.band_mode, repair=False)
    # Repair method: uniform scaling instead of the projection.
    sims["repair/scale_only"] = _sim(prep, books, "scaled", weekly)
    sims["repair/scale_real"] = _sim(prep, books, "scaled_capped", ds, **real)
    # Rebalance-timing luck: one book every LUCK_PERIOD weeks at each phase vs staggered tranches.
    for ph in range(LUCK_PERIOD):
        sims[f"luck/single_phase{ph}"] = _sim(prep, books, "proj_capped", Schedule(1, LUCK_PERIOD, ph), **real)
    sims["luck/staggered"] = _sim(prep, books, "proj_capped", Schedule(LUCK_PERIOD, LUCK_PERIOD), **real)
    sims["luck/weekly"] = _sim(prep, books, "proj_capped", weekly, **real)

    if sweeps:
        for b, m in band_configs():
            sims[f"bands/{b:g}/{m}"] = _sim(prep, books, "proj_capped", ds, band=b, mode=m, **full)
        for mult in COST_MULTS:
            b, m = cost_bands[f"{mult:g}"]
            sims[f"cost/{mult:g}/chosen"] = _sim(prep, books, "proj_capped", ds, band=b, mode=m,
                                                 cost_mult=mult, **full)
            sims[f"cost/{mult:g}/no_band"] = _sim(prep, books, "proj_capped", ds, cost_mult=mult, **full)

    paper = sims["ladder/paper"].returns
    cap = prep.mandate.gross_cap
    configs = {}
    for k, v in sims.items():
        s = stats(v, paper)
        # the mandate is enforced at rebalances only; between them, drift can lift gross above the cap
        s["max_gross"] = float(v.gross.max())
        s["share_days_over_gross_cap"] = float((v.gross > cap + 1e-9).mean())
        configs[k] = s
    return {"diag": diag, "configs": configs}


def heldout_task(seed: int, desk: DeskConfig, cost_bands: dict, market: MarketConfig | None = None) -> dict:
    prep = prepare(seed, market, desk)
    out = {"seed": seed}
    out["model"] = run_signal(prep, "model", desk, sweeps=True, cost_bands=cost_bands)
    out["control"] = run_signal(prep, "control", desk, sweeps=False)
    out["solver"] = {"solves": prep.repairer.n_solves, "almost_solved": prep.repairer.n_almost,
                     "fallbacks": prep.repairer.n_fallbacks}
    return out


def dev_task(seed: int, market: MarketConfig | None = None) -> dict:
    """Every (schedule, band, mode, cost multiplier) on one development seed, model signal."""
    desk = DeskConfig()
    prep = prepare(seed, market, desk)
    books, diag = build_books(prep, "model", desk)
    paper = _sim(prep, books, "raw", Schedule(1, 1))
    rows = []
    for n, per in SCHEDULES:
        sch = Schedule(n, per)
        for mult in COST_MULTS:
            for b, m in band_configs():
                s = stats(_sim(prep, books, "proj_capped", sch, band=b, mode=m,
                               costs=True, carry=True, cost_mult=mult))
                rows.append({"n_tranches": n, "period_weeks": per, "band": b, "band_mode": m,
                             "cost_mult": mult, "sharpe": s["sharpe"], "ann_return": s["ann_return"],
                             "turnover": s["turnover"]})
    return {"seed": seed, "paper_sharpe": stats(paper)["sharpe"], "diag": diag, "rows": rows}


def with_choices(desk: DeskConfig, choices: dict) -> DeskConfig:
    return replace(desk, **choices)
