"""Shrunk covariance and the ex-ante volatility cap.

Covariance: sample correlations shrunk toward zero with the Ledoit-Wolf (2004)
intensity computed on standardised returns, then rescaled by the sample
volatilities. Window: the trailing ``COV_LOOKBACK`` days up to and including the
rebalance day, so nothing after the decision is used.
"""
from __future__ import annotations

import numpy as np

from .data import TRADING_DAYS

COV_LOOKBACK = 252


def ledoit_wolf_corr(x: np.ndarray) -> tuple[np.ndarray, float]:
    """Shrink the correlation matrix of x (T, N) toward the identity.

    Returns (shrunk correlation, intensity delta in [0, 1])."""
    T = x.shape[0]
    z = x - x.mean(axis=0)
    z = z / z.std(axis=0)
    S = z.T @ z / T  # sample correlation (unit diagonal)
    n = S.shape[0]
    F = np.eye(n)
    d2 = float(((S - F) ** 2).sum())
    if d2 <= 0:
        return S, 0.0
    # b_bar^2 = (1 / T^2) * sum_t ||z_t z_t' - S||_F^2
    #         = (1 / T^2) * (sum_t ||z_t||^4 - T ||S||_F^2), since sum_t z_t z_t' = T S
    b2 = float((np.einsum("ti,ti->t", z, z) ** 2).sum() - T * (S**2).sum())
    b2 = min(max(b2, 0.0) / T**2, d2)
    delta = b2 / d2
    return delta * F + (1.0 - delta) * S, float(delta)


def shrunk_cov(returns: np.ndarray, day: int, lookback: int = COV_LOOKBACK) -> tuple[np.ndarray, float]:
    window = returns[day - lookback + 1 : day + 1]
    sd = window.std(axis=0)
    C, delta = ledoit_wolf_corr(window)
    return C * np.outer(sd, sd), delta


def ex_ante_vol(w: np.ndarray, cov: np.ndarray) -> float:
    return float(np.sqrt(max(w @ cov @ w, 0.0) * TRADING_DAYS))


def vol_cap(w: np.ndarray, cov: np.ndarray, cap: float) -> tuple[np.ndarray, bool]:
    """Scale the book down (never up) so its ex-ante annual vol is at most cap."""
    v = ex_ante_vol(w, cov)
    if v > cap:
        return w * (cap / v), True
    return w.copy(), False
