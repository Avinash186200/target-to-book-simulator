"""The research team's target weights: what they hand to the desk each week.

The team does not know the mandate. Their convention: cross-sectional z-score of
the signal, divided by each asset's trailing 63-day volatility (risk-weighting),
scaled to a fixed gross exposure. Everything uses data up to and including the
hand-over day only.
"""
from __future__ import annotations

import numpy as np

VOL_LOOKBACK = 63
PAPER_GROSS = 2.5


def trailing_vol(returns: np.ndarray, day: int, lookback: int = VOL_LOOKBACK) -> np.ndarray:
    """Sample sd of daily returns over days (day - lookback, day], inclusive of day."""
    window = returns[day - lookback + 1 : day + 1]
    return window.std(axis=0, ddof=1)


def target_weights(signal_row: np.ndarray, vol: np.ndarray, gross: float = PAPER_GROSS) -> np.ndarray:
    z = signal_row - signal_row.mean()
    sd = z.std()
    if sd <= 0:
        return np.zeros_like(signal_row)
    raw = (z / sd) / vol
    return gross * raw / np.abs(raw).sum()


def weekly_targets(returns: np.ndarray, signal: np.ndarray, rebalance_days: np.ndarray,
                   gross: float = PAPER_GROSS) -> np.ndarray:
    """(K, N) targets, row k formed at the close of rebalance_days[k]."""
    out = np.empty((len(rebalance_days), returns.shape[1]))
    for k, d in enumerate(rebalance_days):
        out[k] = target_weights(signal[d], trailing_vol(returns, d), gross)
    return out
