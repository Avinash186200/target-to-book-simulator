"""Performance statistics of a simulated book. Returns are daily excess returns."""
from __future__ import annotations

import numpy as np

from .book import SimResult
from .data import TRADING_DAYS


def sharpe(r: np.ndarray) -> float:
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else 0.0


def max_drawdown(r: np.ndarray) -> float:
    nav = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(np.concatenate([[1.0], nav]))[1:]
    return float((1.0 - nav / peak).max())


def stats(res: SimResult, paper: np.ndarray | None = None) -> dict:
    r = res.returns
    out = {
        "ann_return": float(r.mean() * TRADING_DAYS),  # arithmetic, so stage deltas add up
        "ann_vol": float(r.std(ddof=1) * np.sqrt(TRADING_DAYS)),
        "sharpe": sharpe(r),
        "max_drawdown": max_drawdown(r),
        "avg_gross": float(res.gross.mean()),
        "turnover": res.turnover,
        "turnover_vs_targets": res.turnover_vs_targets,
        "cost_drag": float(res.cost.mean() * TRADING_DAYS),
        "carry_drag": float(res.carry.mean() * TRADING_DAYS),
        "n_repairs": res.n_repairs,
        "n_tranche_rebalances": res.n_tranche_rebalances,
    }
    if paper is not None:
        d = r - paper
        out["tracking_error"] = float(d.std(ddof=1) * np.sqrt(TRADING_DAYS))
        out["corr_to_paper"] = float(np.corrcoef(r, paper)[0, 1])
    return out
